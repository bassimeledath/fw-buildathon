"""pass@k / pass^k on hard_v2 episodes (the deck's "measure the gap" step).

Each held-out case is played n times at temperature > 0 (at 0 every draw is the same). A draw passes if the whole
call was handled correctly (`clean` in hard_v2). From the same n draws, read at every k:
  pass@k = P(at least one of k draws passes)   (capability, the ceiling)
  pass^k = P(all k draws pass)                 (consistency)
  reliability gap = pass@k - pass^k
Both use the unbiased estimators over the n draws (pass@k = 1 - C(n-c,k)/C(n,k), pass^k = C(c,k)/C(n,k)).

  python -m ase.hard_v2_passk --model accounts/fireworks/models/deepseek-v4p1-flash --reasoning-effort none
  python -m ase.hard_v2_passk --snapshot <account/run-id/prefix-step-N-hash>      # a qwen training snapshot
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
from math import comb
from pathlib import Path

from .hard_v2 import Case, run


def pass_at(n, c, k):
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)


def pass_hat(n, c, k):
    return comb(c, k) / comb(n, k) if c >= k else 0.0


def curves(counts: list[tuple[int, int]], kmax: int) -> dict:
    out = {}
    for k in range(1, kmax + 1):
        out[k] = (sum(pass_at(n, c, k) for n, c in counts) / len(counts),
                  sum(pass_hat(n, c, k) for n, c in counts) / len(counts))
    return out


def quadrant(cv: dict) -> str:
    kmax = max(cv)
    ceiling, gap = cv[kmax][0], cv[kmax][0] - cv[kmax][1]
    shape = "STOCHASTIC" if gap >= 0.2 else "SYSTEMATIC"
    high = ceiling >= 0.5
    verdict = {("STOCHASTIC", True): "consolidate (RL / rejection sampling)",
               ("STOCHASTIC", False): "import, then consolidate",
               ("SYSTEMATIC", True): "already good: go after cost and latency",
               ("SYSTEMATIC", False): "import capability"}[(shape, high)]
    return f"{shape} . CEILING {'HIGH' if high else 'LOW'} (pass@{kmax} {ceiling:.0%}, gap {gap:.0%}) -> {verdict}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=8, help="draws per case")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (quick look)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    cases = [Case(**json.loads(l)) for l in open(f"data/hard_v2/{args.split}/test_cases.jsonl")]
    if args.limit:
        cases = cases[: args.limit]

    if args.snapshot:
        from .snapshot_policy import SnapshotPolicy

        sp = SnapshotPolicy(args.snapshot, temperature=args.temperature, max_tokens=512)

        async def policy(msgs):
            return (await sp(msgs, None))[0]
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"],
                             max_retries=8)
        extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort not in (None, "default") else {}

        async def policy(msgs):
            r = await client.chat.completions.create(model=args.model, messages=msgs, temperature=args.temperature,
                                                     max_tokens=1024, **extra)
            return r.choices[0].message.content or ""

    sem = asyncio.Semaphore(args.concurrency)

    async def draw(c):
        async with sem:
            for attempt in range(4):
                try:
                    return await run(c, policy)
                except Exception:  # API error: retry; never scored as a model failure
                    await asyncio.sleep(10 * (attempt + 1))
            return None

    async def go():
        return await asyncio.gather(*[draw(c) for c in cases for _ in range(args.n)])

    logs = asyncio.run(go())
    per = collections.defaultdict(list)
    for x in logs:
        if x is not None:
            per[x["case_id"]].append(x)
    kind = {c.case_id: c.kind for c in cases}
    counts = {cid: (len(xs), sum(x["clean"] for x in xs)) for cid, xs in per.items() if len(xs) == args.n}
    tag = args.tag or (args.snapshot or args.model).split("/")[-1]
    out = Path("runs") / f"passk_hv2_{args.split}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs.jsonl").write_text("".join(json.dumps(x, default=str) + "\n" for x in logs if x))
    cv = curves(list(counts.values()), args.n)
    summary = {"tag": tag, "split": args.split, "cases": len(counts), "draws_per_case": args.n,
               "temperature": args.temperature,
               "curve": {k: {"pass@k": round(a, 3), "pass^k": round(b, 3)} for k, (a, b) in cv.items()},
               "quadrant": quadrant(cv), "by_kind": {}}
    by = collections.defaultdict(list)
    for cid, nc in counts.items():
        by[kind[cid]].append(nc)
    for kd, ncs in sorted(by.items()):
        kc = curves(ncs, args.n)
        summary["by_kind"][kd] = {"pass@1": round(kc[1][0], 3), f"pass@{args.n}": round(kc[args.n][0], 3),
                                  f"pass^{args.n}": round(kc[args.n][1], 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{tag} on {args.split}: {len(counts)} cases x {args.n} draws, T={args.temperature}")
    print("  k  pass@k  pass^k   gap")
    for k, (a, b) in cv.items():
        print(f"  {k}  {a:6.1%}  {b:6.1%}  {a - b:5.1%}")
    print("  " + summary["quadrant"])
    print(f"  {'case type':15s} pass@1  pass@{args.n}  pass^{args.n}")
    for kd, v in summary["by_kind"].items():
        print(f"  {kd:15s} {v['pass@1']:6.0%}  {v[f'pass@{args.n}']:6.0%}  {v[f'pass^{args.n}']:6.0%}")


if __name__ == "__main__":
    main()
