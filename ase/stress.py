"""Off-script stress set: hand-written messy conversations, played against the same v0 prompt and retrieval as the eval.

The benchmark customer only speaks in templates. These cases (stress/stress_cases.jsonl, made by
stress/build_cases.py) add typos, slang, self-corrections, damage in passing, two issues at once, devices or
problems the KB does not cover, "the fix didn't work", anger, mid-chat requests for a human, and vague devices.
The customer is still scripted: each case has a reply per slot, and a fixed line once a goal is reached.

Scores per case:
  strict      every goal reached with an accepted article / escalation reason
  tolerant    strict, or the only miss is a "tolerant" escalate goal (not_covered / fix_failed) that the agent
              escalated with a different reason. The default prompt has no not_covered / fix_failed reason, so
              this is the fair score for harnesses that lack them.
  no_wrong_resolve   the agent never resolved with an article outside the accepted ones
  first_ok    the first move matches the case's "first" spec (only cases that have one)

Usage (v0 mode):
  ASE_RETRIEVAL=fixed python -m ase.stress --model accounts/fireworks/models/gpt-oss-120b --harness h2 --run 1
  ASE_RETRIEVAL=fixed python -m ase.stress --report        # table over every runs/stress_* folder
Logs follow the ase.episodes format (case_id, steps, outcome, asks, transcript, ...) plus stress fields.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from .env import RETRIEVER, format_results, parse_action, system_prompt
from .harness import HARNESSES, HarnessPolicy
from .world import FIXED

CASES = Path(__file__).resolve().parent.parent / "stress" / "stress_cases.jsonl"
RESULTS_K = 5
MAX_STEPS = 10
# $ per 1M tokens (input, output). Estimates for the cost line only; override with --price.
PRICES = {"gpt-oss-120b": (0.15, 0.60), "deepseek-v4p1-flash": (0.30, 1.20)}


class LLM:
    """Same call as ase.episodes.LLMPolicy, but keeps finish_reason so empty outputs can be explained."""

    def __init__(self, model, temperature, max_tokens):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"),
                                  api_key=os.environ["FIREWORKS_API_KEY"], max_retries=8)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.last_finish = None

    async def __call__(self, messages, gt):
        t0 = time.perf_counter()
        r = await self.client.chat.completions.create(model=self.model, messages=messages,
                                                      temperature=self.temperature, max_tokens=self.max_tokens)
        self.last_finish = r.choices[0].finish_reason
        u = r.usage
        return (r.choices[0].message.content or ""), time.perf_counter() - t0, (u.prompt_tokens if u else 0), \
            (u.completion_tokens if u else 0)


def load_cases(path=CASES):
    return [json.loads(line) for line in open(path)]


class Convo:
    """The conversation as the agent sees it: customer lines carry the top-5 results for everything said so far."""

    def __init__(self):
        self.messages, self.said, self.seen = [], [], set()

    def customer(self, text):
        self.said.append(text)
        res = RETRIEVER.search(" ".join(self.said), k=RESULTS_K)
        self.seen |= {a.doc_id for a in res}
        self.messages.append({"role": "user", "content": "[CUSTOMER] " + text + "\n\n" + format_results(res)})

    def agent(self, action):
        self.messages.append({"role": "assistant", "content": json.dumps(action)})


def reply_for(case, slot, used):
    r = case["replies"].get(slot, case["replies"]["other"])
    if isinstance(r, list):
        i = used[slot]
        used[slot] += 1
        return r[min(i, len(r) - 1)]
    return r


async def run_case(case, policy, llm) -> dict:
    cv = Convo()
    for h in case.get("history", []):
        cv.customer(h["customer"]) if "customer" in h else cv.agent(h["agent"])
    cv.customer(case["opening"])
    goals = [dict(g) for g in case["goals"]]
    done_goals, used = [], defaultdict(int)
    log = {"case_id": case["id"], "category": case["category"], "escalation": None, "steps": [], "lat": [],
           "ptok": 0, "ctok": 0, "goals": case["goals"], "first": case.get("first")}
    outcome, asks, wrong_resolve, tolerant_hit, empty, first_ok = "max_steps", 0, False, False, False, None
    for i in range(MAX_STEPS):
        prompt = [{"role": "system", "content": system_prompt()}] + cv.messages
        text, lat, pt, ct = await policy(prompt, None)
        log["lat"].append(lat)
        log["ptok"] += pt
        log["ctok"] += ct
        action = parse_action(text)
        if action is None:
            empty = not text.split("</think>")[-1].strip()
            outcome = "invalid"
            log["steps"].append({"raw": text[-300:], "empty": empty, "finish_reason": llm.last_finish})
            break
        log["steps"].append(action)
        kind = action.get("action")
        if i == 0 and case.get("first"):
            spec = case["first"]
            first_ok = kind in spec and (kind != "ask" or action.get("slot") in spec["ask"])
        open_goals = goals if case.get("unordered") else goals[:1]
        if kind == "ask":
            asks += 1
            cv.agent(action)
            cv.customer(reply_for(case, str(action.get("slot", "")), used))
            continue
        if kind == "resolve":
            cv.agent(action)
            hit = next((g for g in open_goals if action.get("doc_id") in g.get("resolve", [])), None)
            if hit is None:
                wrong_resolve, outcome = True, "resolved_wrong"
                break
            goals.remove(hit)
            done_goals.append(hit)
            if not goals:
                outcome = "resolved_correct"
                break
            cv.customer(hit.get("then", "Thanks. Anything else?"))
            continue
        if kind == "escalate":
            cv.agent(action)
            hit = next((g for g in open_goals if action.get("reason") in g.get("escalate", [])), None)
            if hit is not None and len(goals) == 1:
                done_goals.append(hit)
                goals.remove(hit)
                outcome = "escalated_correct"
            else:
                outcome = "escalated_wrong"
                tolerant_hit = len(goals) == 1 and goals[0].get("tolerant", False) and "escalate" in goals[0]
            break
        outcome = "invalid"  # search or unknown action: not available in v0
        break
    strict = not goals and outcome in ("resolved_correct", "escalated_correct")
    shown = {d: d in cv.seen for g in case["goals"] for d in g.get("resolve", [])}
    log.update(outcome=outcome, asks=asks, wasted_asks=0, searches=0, strict=strict,
               tolerant=strict or tolerant_hit, no_wrong_resolve=not wrong_resolve, first_ok=first_ok, empty=empty,
               goals_met=len(done_goals), n_goals=len(case["goals"]), expected_shown=shown, transcript=cv.messages)
    return log


def rate(xs, key):
    xs = [x for x in xs if x.get(key) is not None]
    return round(sum(bool(x[key]) for x in xs) / len(xs), 3) if xs else None


def summarize(logs):
    ok = [x for x in logs if x["outcome"] != "api_error"]
    by_cat = defaultdict(list)
    for x in ok:
        by_cat[x["category"]].append(x)
    return {
        "n": len(ok), "api_errors": len(logs) - len(ok),
        "strict": rate(ok, "strict"), "tolerant": rate(ok, "tolerant"),
        "no_wrong_resolve": rate(ok, "no_wrong_resolve"), "first_ok": rate(ok, "first_ok"),
        "invalid": sum(x["outcome"] == "invalid" for x in ok), "empty": sum(bool(x.get("empty")) for x in ok),
        "max_steps": sum(x["outcome"] == "max_steps" for x in ok),
        "ptok": sum(x["ptok"] for x in ok), "ctok": sum(x["ctok"] for x in ok),
        "by_category": {c: {"n": len(v), "strict": rate(v, "strict"), "tolerant": rate(v, "tolerant"),
                            "no_wrong_resolve": rate(v, "no_wrong_resolve")} for c, v in sorted(by_cat.items())},
    }


async def amain(args):
    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c["id"] in args.only.split(",") or c["category"] in args.only.split(",")]
    llm = LLM(args.model, args.temperature, args.max_tokens)
    policy = HarnessPolicy(llm, args.harness) if args.harness != "base" else llm
    sem = asyncio.Semaphore(args.concurrency)

    async def one(c):
        async with sem:
            try:
                return await run_case(c, policy, llm)
            except Exception as ex:  # an API failure is not a model failure
                return {"case_id": c["id"], "category": c["category"], "outcome": "api_error", "error": repr(ex)[:300],
                        "ptok": 0, "ctok": 0}

    tag = args.tag or f"{args.model.split('/')[-1]}_{args.harness}_r{args.run}"
    out = Path("runs") / f"stress_{tag}"
    prev = {}
    if args.retry_errors and (out / "logs.jsonl").exists():  # re-run only the cases that hit API errors
        prev = {x["case_id"]: x for x in map(json.loads, open(out / "logs.jsonl"))}
        cases = [c for c in cases if prev.get(c["id"], {}).get("outcome", "api_error") == "api_error"]
    new = {x["case_id"]: x for x in await asyncio.gather(*[one(c) for c in cases])}
    logs = list({**prev, **new}.values())
    summary = {"model": args.model, "harness": args.harness, "run": args.run, **summarize(logs)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "logs.jsonl", "w") as f:
        for x in logs:
            f.write(json.dumps(x) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}))


def report(prices):
    """Pool every runs/stress_* folder by model x harness."""
    groups = defaultdict(list)
    for d in sorted(Path("runs").glob("stress_*")):
        s = json.loads((d / "summary.json").read_text())
        logs = [json.loads(line) for line in open(d / "logs.jsonl")]
        groups[(s["model"].split("/")[-1], s["harness"])].append(logs)
    cats = sorted({x["category"] for runs in groups.values() for logs in runs for x in logs if "category" in x})
    total = 0.0
    print("| model | harness | runs | strict | tolerant | no wrong resolve | first move ok | invalid (empty) | cost $ |")
    print("|---|---|---|---|---|---|---|---|---|")
    pooled = {}
    for (m, h), runs in sorted(groups.items()):
        logs = [x for r in runs for x in r]
        s = summarize(logs)
        pi, po = prices.get(m, (0, 0))
        cost = (s["ptok"] * pi + s["ctok"] * po) / 1e6
        total += cost
        pooled[(m, h)] = s
        print(f"| {m} | {h} | {len(runs)} | {s['strict']} | {s['tolerant']} | {s['no_wrong_resolve']} | "
              f"{s['first_ok']} | {s['invalid']} ({s['empty']}) | {cost:.3f} |")
    print(f"\nTotal estimated cost: ${total:.3f}\n")
    keys = sorted(pooled)
    print("| category | n | " + " | ".join(f"{m} {h}" for m, h in keys) + " |")
    print("|---|---|" + "---|" * len(keys))
    for c in cats:
        n = next(pooled[k]["by_category"][c]["n"] for k in keys if c in pooled[k]["by_category"])
        cells = []
        for k in keys:
            b = pooled[k]["by_category"].get(c, {})
            cells.append(f"{b.get('strict')} / {b.get('tolerant')}")
        print(f"| {c} | {n} | " + " | ".join(cells) + " |")
    print("\n(cells: strict / tolerant; n = case-runs per model x harness)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--harness", default="base", choices=HARNESSES)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--only", default="", help="comma list of case ids or categories")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--tag", default="")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--retry-errors", action="store_true", help="re-run only api_error cases of an existing run")
    ap.add_argument("--price", action="append", default=[], help="model=in,out ($/1M tokens)")
    args = ap.parse_args()
    if not FIXED:
        raise SystemExit("set ASE_RETRIEVAL=fixed: the stress set is written for the v0 fixed-retrieval world")
    if args.report:
        prices = dict(PRICES)
        for p in args.price:
            m, v = p.split("=")
            prices[m] = tuple(float(x) for x in v.split(","))
        return report(prices)
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
