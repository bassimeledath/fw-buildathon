"""Harness for the hard_v2 benchmark (ase/hard_v2.py): how far do prompting and a small code rule get a cheap model?

Variants (same cases, scripted customer and grading as ase.hard_v2; only the agent side changes):
  --prompt base   the published system prompt (= ase.hard_v2's own baseline)
  --prompt h1     a call playbook: pick the current-software article, run its steps in order, skip what was
                  already tried, re-give a step the customer couldn't do, ask consent before a reset, check before
                  closing, hand off when the ladder runs out
  --guard2        two checks from the visible call: (a) if the agent repeats a fix the customer already did (said in
                  the opening, or "I already did that" / "it's already on" / "okay, done"), retry once with a nudge;
                  (b) before escalate fix_failed, one self-check call: did the problem change (e.g. SOS) or are fixes left?
  --guard         one code rule, from the visible call only: never close without a check. A resolve that does not
                  come right after a successful verify is turned into a verify. It never reads the hidden case or
                  the answer key.

Usage (tune on dev; run test once at the end):
  M=accounts/fireworks/models/deepseek-v4p1-flash
  python -m ase.harness_v2 --split dev --model $M --prompt base
  python -m ase.harness_v2 --split dev --model $M --prompt h1
  python -m ase.harness_v2 --split dev --model $M --prompt h1 --guard
  python -m ase.stats runs/hard_v2_harness_dev_base_<model>/logs.jsonl runs/hard_v2_harness_dev_h1-guard_<model>/logs.jsonl

Splits: dev / test are the frozen case files in data/hard_v2/<split>/test_cases.jsonl. Does not change ase/hard_v2.py.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
from pathlib import Path

from . import hard_v2 as hv
from .env import parse_action

H1_PROMPT = hv.SYSTEM + """

How to run the call:
1. Hand off right away if the customer mentions damage, a safety hazard or billing, or asks for a person at any point (human_request).
2. Pick the article: it must be for the customer's platform (iOS or Android) and their exact problem, and say "Last reviewed 2026" for current software. Never use a copy written for older software, an in-store/technician copy, or a how-to page. Similar problems are different articles: mobile data vs no service, texts vs picture messages, dropped calls vs Wi-Fi calling, Wi-Fi vs hotspot.
3. Run the article's fixes strictly in order: the first sections, then the "Still not working?" fixes in the order listed. One step per turn. In "step", copy the bold heading of that fix exactly.
4. Skip a fix the customer already did (in the opening, or "I already did that", or "it's already on"): treat it as done and give the next one. Never repeat a fix that was done.
5. If the customer couldn't do the step (they did something else, or can't find the menu, e.g. "Mobile Service" instead of "Cellular"), the step is NOT done: give the same step again, with clearer words.
6. Android: the article has a Samsung line and a Pixel/other line. If you don't know the brand, ask for the model first. Use path "samsung" for Galaxy phones and "pixel" for Pixel and other Android phones; if the customer says their settings look different, switch to the other path for the same step. iPhone: path null.
7. Before a fix that resets or erases anything, ask for consent first.
8. If the customer's problem turns out to be different (for example the phone now shows SOS), switch to the article the current article points to for that, and start its fixes.
9. When a step seems to have worked, verify: ask the customer to do the article's check. Only resolve after the check passes. A step can look like it worked before it really has; if the check fails, go on to the next fix.
10. When every fix in the article has been done or tried and the problem remains, escalate with fix_failed."""

PROMPTS = {"base": hv.SYSTEM, "h1": H1_PROMPT}


def _verified_last(messages: list[dict]) -> bool:
    """True if the agent's previous move was a verify (the customer's answer to it is the last message)."""
    for m in reversed(messages):
        if m["role"] == "assistant":
            a = parse_action(m["content"])
            return bool(a and a.get("action") == "verify")
    return False


# ---- guard2: two code checks from the visible call only (never the hidden case or the answer key) ----------
DONE_REPLY = ("already did that", "it's already on", "okay, i did that.", "okay, done.", "okay, i did that too.")
OPENING_TRIED = {"restarted": "restart the phone", "airplane mode": "turn airplane mode on and off"}


def _done_steps(messages: list[dict]) -> set:
    """Fix headings the customer has clearly completed or already tried, read off the transcript."""
    done = set()
    users = [m for m in messages if m["role"] == "user"]
    if users:
        opening = users[0]["content"].split("\n\n[SEARCH RESULTS]")[0].lower()
        done |= {head for cue, head in OPENING_TRIED.items() if cue in opening and "already" in opening}
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        a = parse_action(m["content"])
        if not (a and a.get("action") == "step") or i + 1 >= len(messages):
            continue
        reply = messages[i + 1]["content"].split("\n\n[SEARCH RESULTS]")[0].lower()
        if any(cue in reply for cue in DONE_REPLY) and "nothing changed" not in reply:
            done.add(hv.norm(a.get("step")))
    return done


REPEAT_NUDGE = ("[CHECK] The customer already did \"{head}\" earlier in this call. Don't repeat it. Give the next fix "
                "in the article's order that the customer hasn't done yet (ask for consent first if it resets anything), "
                "or verify if a step just seemed to work. Output one JSON action only.")
HANDOFF_NUDGE = ("[CHECK BEFORE HANDING OFF] Before you escalate with fix_failed, check in at most 5 short lines: "
                 "(1) Did the customer mention a new sign that the article says belongs to a different article "
                 "(for example SOS or No Service at the top, or regular texts failing too)? If yes, switch to that "
                 "article from the search results and give its first fix the customer hasn't done. "
                 "(2) Are there fixes in the current article, including the \"Still not working?\" section, that the "
                 "customer hasn't done? If yes, give the next one. (3) Only if every fix is done, escalate with "
                 "fix_failed. End with the final JSON action on the last line.")


def _last_json(text: str) -> str:
    end = text.rfind("}")
    start = text.rfind("{", 0, end)
    while start >= 0:
        a = parse_action(text[start:end + 1])
        if a and a.get("action"):
            return text[start:end + 1]
        start = text.rfind("{", 0, start)
    return text


class Policy:
    def __init__(self, model, prompt, guard, reasoning_effort, max_tokens, guard2=False, snapshot=None):
        from openai import AsyncOpenAI

        self.snap = None
        if snapshot:  # a qwen training snapshot, sampled token-in/token-out (needs fw-cookbook on PYTHONPATH)
            from .snapshot_policy import SnapshotPolicy

            self.snap = SnapshotPolicy(snapshot, temperature=0.0, max_tokens=512)
        self.client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"),
                                  api_key=os.environ["FIREWORKS_API_KEY"], max_retries=8)
        self.model, self.prompt, self.guard, self.max_tokens = model, PROMPTS[prompt], guard, max_tokens
        self.extra = {"reasoning_effort": reasoning_effort} if reasoning_effort not in (None, "default") else {}
        self.calls, self.ptok, self.ctok, self.guard_fired = 0, 0, 0, 0
        self.guard2, self.repeat_fired, self.handoff_fired, self.g2_changed = guard2, 0, 0, 0

    async def _call(self, messages):
        if self.snap:
            text, _lat, pt, ct = await self.snap(messages, None)
            self.calls += 1
            self.ptok += pt
            self.ctok += ct
            return text
        r = await self.client.chat.completions.create(model=self.model, messages=messages, temperature=0,
                                                      max_tokens=self.max_tokens, **self.extra)
        self.calls += 1
        if r.usage:
            self.ptok += r.usage.prompt_tokens
            self.ctok += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    async def _retry(self, messages, text, nudge):
        """One retry with a nudge; an unusable answer keeps the original move (never worse, no loops)."""
        text2 = _last_json(await self._call(messages + [{"role": "assistant", "content": text},
                                                         {"role": "user", "content": nudge}]))
        a1, a2 = parse_action(text), parse_action(text2)
        if a2 is None:
            return text
        self.g2_changed += json.dumps(a1, sort_keys=True) != json.dumps(a2, sort_keys=True)
        return text2

    async def __call__(self, messages: list[dict]) -> str:
        messages = [{"role": "system", "content": self.prompt}, *messages[1:]]  # swap the system prompt only
        text = await self._call(messages)
        if self.guard2:
            a = parse_action(text)
            if a and a.get("action") == "step" and hv.norm(a.get("step")) in _done_steps(messages):
                self.repeat_fired += 1
                text = await self._retry(messages, text, REPEAT_NUDGE.format(head=a.get("step")))
            elif a and a.get("action") == "escalate" and a.get("reason") == "fix_failed":
                self.handoff_fired += 1
                text = await self._retry(messages, text, HANDOFF_NUDGE)
        if self.guard:
            a = parse_action(text)
            if a and a.get("action") == "resolve" and not _verified_last(messages):
                self.guard_fired += 1  # close without a check -> check first (no extra model call)
                return json.dumps({"action": "verify", "say": "Before we wrap up, can you check that it's working now?"})
        return text


def load_cases(split: str) -> list:
    rows = [json.loads(line) for line in open(f"data/hard_v2/{split}/test_cases.jsonl")]
    out = []
    for r in rows:
        r = {k: tuple(v) if isinstance(v, list) else v for k, v in r.items()}
        out.append(hv.Case(**r))
    return out


def write_train(prompt: str):
    """Copy Bassim's hard_v2 train/heldout states with the system prompt swapped for this harness's prompt, so RL
    trains the model under the same prompt the harness uses at inference. Same states, same answer key."""
    for name in ("train_states", "heldout_states"):
        src = Path(f"data/hard_v2/train/{name}.jsonl")
        dst = src.with_name(f"{name}_{prompt}.jsonl")
        n = 0
        with open(src) as f, open(dst, "w") as g:
            for line in f:
                r = json.loads(line)
                assert r["messages"][0]["role"] == "system"
                r["messages"][0]["content"] = PROMPTS[prompt]
                g.write(json.dumps(r) + "\n")
                n += 1
        print(f"wrote {dst} ({n} states)")


async def amain(args):
    cases = load_cases(args.split)
    if args.n:
        cases = cases[: args.n]
    if args.write_train:
        write_train(args.prompt)
        return
    policy = Policy(args.model, args.prompt, args.guard, args.reasoning_effort, args.max_tokens, guard2=args.guard2,
                    snapshot=args.snapshot)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(c):
        async with sem:
            err = ""
            for attempt in range(4):
                try:
                    return await hv.run(c, policy)
                except Exception as ex:  # API failure: not scored (same as hard_v2)
                    err = repr(ex)[:200]
                    await asyncio.sleep(10 * (attempt + 1))
            return {"case_id": c.case_id, "kind": c.kind, "outcome": "api_error", "clean": False, "error": err}

    logs = await asyncio.gather(*(one(c) for c in cases))
    scored = [x for x in logs if x["outcome"] != "api_error"]
    for x in scored:  # ase.stats counts these outcomes as a pass; hard_v2's pass is "clean", so mirror it
        x["outcome_raw"] = x["outcome"]
        if not x["clean"] and x["outcome"] in ("resolved_correct", "escalated_correct"):
            x["outcome"] = "unclean_" + x["outcome"]
    name = args.prompt + ("-guard" if args.guard else "") + ("-guard2" if args.guard2 else "")
    who = args.snapshot.split("/")[-1] if args.snapshot else f"{args.model.split('/')[-1]}_re-{args.reasoning_effort}"
    out = Path("runs") / f"hard_v2_harness_{args.split}_{name}_{who}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs.jsonl").write_text("".join(json.dumps(x, default=str) + "\n" for x in logs))
    summary = {
        "harness": name, "model": args.model, "split": args.split, "n": len(scored),
        "api_errors": len(logs) - len(scored),
        "success": round(sum(x["clean"] for x in scored) / max(1, len(scored)), 3),
        "outcomes": dict(collections.Counter(x["outcome"] for x in scored)),
        "wasted_questions_per_case": round(sum(x.get("wasted", 0) for x in scored) / max(1, len(scored)), 2),
        "turns_per_case": round(sum(x.get("turns", 0) for x in scored) / max(1, len(scored)), 1),
        "calls": policy.calls, "tokens_per_case": round((policy.ptok + policy.ctok) / max(1, len(scored))),
        "guard_fired": policy.guard_fired, "repeat_fired": policy.repeat_fired,
        "handoff_fired": policy.handoff_fired, "guard2_changed_action": policy.g2_changed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    hv.report(scored)
    print(json.dumps(summary, indent=2))
    print(f"logs: {out / 'logs.jsonl'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accounts/fireworks/models/deepseek-v4p1-flash")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="h1")
    ap.add_argument("--guard", action="store_true", help="code rule: never resolve without a verify right before")
    ap.add_argument("--guard2", action="store_true",
                    help="code checks: don't repeat a fix the customer already did; self-check before fix_failed")
    ap.add_argument("--reasoning-effort", default="none")  # same as the published baseline
    ap.add_argument("--snapshot", default=None, help="evaluate a qwen training snapshot instead of --model")
    ap.add_argument("--write-train", action="store_true",
                    help="write data/hard_v2/train/{train,heldout}_states_<prompt>.jsonl with this prompt, then exit")
    ap.add_argument("--n", type=int, default=None, help="first N cases only (smoke test)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=1024)  # same as ase.hard_v2
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
