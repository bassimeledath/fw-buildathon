"""Single-step baseline + difficulty calibration (Track 2, section 11).

Samples k completions per decision state from a model, scores each with the training reward,
reports per-action pass rates, and writes the 0.2-0.8 pass-rate band to data/train_rl.jsonl.

Usage:
  python -m ase.calibrate --split train --model accounts/fireworks/models/<id> --k 8 --n 300
  python -m ase.calibrate --split test  --model ... --k 1           # plain single-step baseline
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import random
import statistics
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .reward import score
from .world import DATA_DIR, RUN_PREFIX, run_config

load_dotenv()  # same as the training entry points: FIREWORKS_API_KEY may live in .env


async def amain(a):
    rows = [json.loads(line) for line in open(f"{DATA_DIR}/{a.split}_states.jsonl")]
    random.Random(0).shuffle(rows)
    rows = rows[: a.n] if a.n else rows
    client = AsyncOpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://api.fireworks.ai/inference/v1"),
                         api_key=os.environ["FIREWORKS_API_KEY"])
    sem = asyncio.Semaphore(a.concurrency)
    snap = None
    if a.snapshot:
        from .snapshot_policy import SnapshotPolicy

        snap = SnapshotPolicy(a.snapshot, temperature=a.temperature, max_tokens=a.max_tokens)

    async def sample(r):
        async with sem:
            if snap:
                try:
                    texts = [(await snap(r["messages"], None))[0] for _ in range(a.k)]
                except Exception as ex:
                    return [], repr(ex)[:200]
                return [score(t, r["ground_truth"]) | {"text": t[-400:]} for t in texts], None
            try:
                resp = await client.chat.completions.create(
                    model=a.model, messages=r["messages"], n=a.k, temperature=a.temperature, max_tokens=a.max_tokens)
                texts = [c.message.content or "" for c in resp.choices]
            except Exception as ex:
                if a.k == 1:
                    return [], repr(ex)[:200]
                texts = []  # some endpoints reject n>1: fall back to k single calls
                for _ in range(a.k):
                    try:
                        resp = await client.chat.completions.create(
                            model=a.model, messages=r["messages"], temperature=a.temperature, max_tokens=a.max_tokens)
                        texts.append(resp.choices[0].message.content or "")
                    except Exception as ex2:
                        return [], repr(ex2)[:200]
            return [score(t, r["ground_truth"]) | {"text": t[-400:]} for t in texts], None

    results = await asyncio.gather(*[sample(r) for r in rows])
    tag = a.tag or (a.snapshot or a.model).split("/")[-1]
    out = Path("runs") / f"{RUN_PREFIX}calib_{a.split}_{tag}_k{a.k}"
    out.mkdir(parents=True, exist_ok=True)
    by_kind, band, too_easy, too_hard, errors = collections.defaultdict(list), [], 0, 0, 0
    with open(out / "samples.jsonl", "w") as f:
        for r, (scored, err) in zip(rows, results):
            if err or not scored:
                errors += 1
                continue
            rate = sum(s["reward"] >= 0.5 for s in scored) / len(scored)
            kind = "+".join(sorted(r["ground_truth"]["allowed"]))
            by_kind[kind].append(statistics.mean(s["reward"] for s in scored))
            f.write(json.dumps({"gt": r["ground_truth"], "pass_rate": rate, "samples": scored}) + "\n")
            if rate >= 0.8:
                too_easy += 1
            elif rate <= 0.2:
                too_hard += 1
            else:
                band.append(r)
    allr = [x for v in by_kind.values() for x in v]
    summary = {
        "model": a.snapshot or a.model, "split": a.split, "k": a.k, "n_states": len(rows), "errors": errors,
        "mean_reward": round(statistics.mean(allr), 3) if allr else None,
        "by_allowed_action": {k: {"n": len(v), "mean_reward": round(statistics.mean(v), 3)} for k, v in by_kind.items()},
        "band_0.2_0.8": len(band), "too_easy": too_easy, "too_hard": too_hard,
    }
    (out / "summary.json").write_text(json.dumps({"config": run_config(), **summary}, indent=2))
    print(json.dumps(summary, indent=2))
    if a.write_band and band:
        with open(f"{DATA_DIR}/train_rl.jsonl", "w") as f:
            for r in band:
                f.write(json.dumps({"messages": r["messages"], "ground_truth": r["ground_truth"]}) + "\n")
        print(f"wrote {len(band)} rows to {DATA_DIR}/train_rl.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="")
    ap.add_argument("--snapshot", default="", help="serverless training snapshot to sample instead of --model")
    ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--write-band", action="store_true")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
