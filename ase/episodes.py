"""Held-out EVAL verifier: full multi-turn episodes against the scripted customer.

Deliberately different from the training reward (which scores one step). Measures the outcome
the business cares about: right fix, no wrong fixes, right escalations, few customer questions.

Usage:
  python -m ase.episodes --split test --model accounts/fireworks/models/<id> [--n 50] [--concurrency 8]
  python -m ase.episodes --split test --policy oracle      # no LLM, sanity baseline
  PYTHONPATH=.:../fw-cookbook python -m ase.episodes --split test --snapshot <final_sampler_snapshot>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from .env import KB, RETRIEVER, ground_truth, new_state, parse_action, step, to_prompt
from .world import BUDGET, DATA_DIR, HARD, RUN_PREFIX, Case, run_config

MAX_STEPS = 12 if HARD else 8
# ASE_BUDGET=N: the customer gives up after N agent turns. Tighter than MAX_STEPS and
# outcome-bearing -- running out is "abandoned", a failure, not a silent cap.
TURN_CAP = min(BUDGET, MAX_STEPS) if BUDGET else MAX_STEPS


def load_cases(split: str, n: int | None):
    rows = [json.loads(line) for line in open(f"{DATA_DIR}/{split}_episodes.jsonl")]
    return rows[:n] if n else rows


def make_case(e: dict) -> Case:
    return Case(e["case_id"], e["platform"], e["symptom"], e["trigger"], e["escalation"], dict(e["revealed"]),
                dict(e.get("extra", {})))


class LLMPolicy:
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float, max_tokens: int):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, max_retries=8)  # backs off on 429s
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    async def __call__(self, messages, gt):
        t0 = time.perf_counter()
        r = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens
        )
        text = r.choices[0].message.content or ""
        u = r.usage
        return text, time.perf_counter() - t0, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


async def oracle_policy(messages, gt):
    from .sanity import oracle

    return json.dumps(oracle(gt, messages, random.Random(0))), 0.0, 0, 0


async def run_episode(e: dict, policy) -> dict:
    rng = random.Random(e["seed"])
    state = new_state(make_case(e), rng)
    esc = e["escalation"] or ("physical_damage" if e.get("extra", {}).get("damaged") else None)
    log = {"case_id": e["case_id"], "escalation": esc, "steps": [], "lat": [], "ptok": 0, "ctok": 0}
    outcome, asks, wasted_asks, searches = ("abandoned" if BUDGET else "max_steps"), 0, 0, 0
    for _ in range(TURN_CAP):
        gt = ground_truth(state, KB)
        text, lat, pt, ct = await policy(to_prompt(state), gt)
        log["lat"].append(lat)
        log["ptok"] += pt
        log["ctok"] += ct
        action = parse_action(text)
        if action is None:
            outcome = "invalid"
            log["steps"].append({"raw": text[-300:]})
            break
        if action.get("action") == "ask":
            asks += 1
            wasted_asks += action.get("slot") not in gt["allowed"].get("ask", {}).get("slots", [])
        searches += action.get("action") == "search"
        log["steps"].append(action)
        done, step_outcome = step(state, action, RETRIEVER, rng)
        if done:  # only a terminal step sets the outcome; otherwise "max_steps" must survive the loop
            outcome = step_outcome
            break
    log["hidden"] = bool(e.get("extra", {}).get("damaged"))
    log.update(outcome=outcome, asks=asks, wasted_asks=wasted_asks, searches=searches)
    log["transcript"] = state.messages
    return log


def summarize(logs: list[dict]) -> dict:
    api_errors = sum(x["outcome"] == "api_error" for x in logs)
    logs = [x for x in logs if x["outcome"] != "api_error"]  # scored on the cases the model actually saw
    n = len(logs)
    if not n:  # every case errored (e.g. a bad model id): say so instead of crashing on an empty mean
        return {"n": 0, "api_errors": api_errors}
    fix = [x for x in logs if not x["escalation"]]
    esc = [x for x in logs if x["escalation"]]

    def rate(xs, *outs):
        return round(sum(x["outcome"] in outs for x in xs) / max(1, len(xs)), 3)

    lats = [lat for x in logs for lat in x["lat"]]
    return {
        "n": n,
        "api_errors": api_errors,
        "success": rate(logs, "resolved_correct", "escalated_correct"),
        "fix_rate": rate(fix, "resolved_correct"),
        "wrong_fix_rate": rate(fix, "resolved_wrong"),
        # wrong_fix_rate is a rate over ALL fixable cases, so never resolving scores a perfect 0.
        # Read it next to this: no_resolve_rate is the share of fixable cases the agent never
        # attempted to fix at all (false escalation, invalid, or ran out of turns).
        "no_resolve_rate": round(
            sum(not x["outcome"].startswith("resolved") for x in fix) / max(1, len(fix)), 3
        ),
        "skipped_check_rate": rate(logs, "resolved_skipped_check"),
        "false_escalation_rate": rate(fix, "escalated_wrong"),
        "missed_escalation_rate": round(1 - rate(esc, "escalated_correct"), 3) if esc else None,
        "hidden_damage_missed": round(1 - rate([x for x in esc if x.get("hidden")], "escalated_correct"), 3)
        if any(x.get("hidden") for x in esc) else None,
        "invalid_or_stuck": rate(logs, "invalid", "max_steps", "abandoned"),
        "abandoned_rate": rate(logs, "abandoned"),
        "customer_questions_per_case": round(statistics.mean(x["asks"] for x in logs), 2),
        "wasted_questions_per_case": round(statistics.mean(x["wasted_asks"] for x in logs), 2),
        "searches_per_case": round(statistics.mean(x["searches"] for x in logs), 2),
        "p50_step_latency_s": round(statistics.median(lats), 3) if lats else None,
        "tokens_per_case": round(statistics.mean(x["ptok"] + x["ctok"] for x in logs)),
    }


async def amain(args):
    cases = load_cases(args.split, args.n)
    if args.policy == "oracle":
        policy = oracle_policy
    elif args.snapshot:
        from .snapshot_policy import SnapshotPolicy

        policy = SnapshotPolicy(args.snapshot, temperature=args.temperature)
    else:
        policy = LLMPolicy(
            args.model,
            os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"),
            os.environ["FIREWORKS_API_KEY"],
            args.temperature,
            args.max_tokens,
        )
    sem = asyncio.Semaphore(args.concurrency)

    async def one(e):
        async with sem:
            try:
                return await run_episode(e, policy)
            except Exception as ex:  # keep the batch going; an API failure is not a model failure
                return {"case_id": e["case_id"], "escalation": e["escalation"], "outcome": "api_error", "asks": 0,
                        "wasted_asks": 0, "searches": 0, "lat": [], "ptok": 0, "ctok": 0, "error": repr(ex)[:300]}

    logs = await asyncio.gather(*[one(e) for e in cases])
    summary = summarize(logs)
    tag = args.tag or (args.policy if args.policy == "oracle" else (args.snapshot or args.model).split("/")[-1])
    out = Path("runs") / f"{RUN_PREFIX}episodes_{args.split}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"config": run_config(), **summary}, indent=2))
    with open(out / "logs.jsonl", "w") as f:
        for x in logs:
            f.write(json.dumps(x) + "\n")
    print(json.dumps({"tag": tag, **summary}, indent=2))


load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="")
    ap.add_argument("--policy", default="llm", choices=["llm", "oracle"])
    ap.add_argument("--snapshot", default="", help="serverless training snapshot path (from result.json)")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--tag", default="")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
