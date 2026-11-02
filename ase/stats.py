"""Is the gain real? Exact McNemar + case-clustered bootstrap, on two runs over the same cases.

Works on episode logs (one row per case) or calibrate samples (many states per case: we resample
whole cases, since states from one conversation are correlated).

Usage:
  python -m ase.stats runs/episodes_test_<base>/logs.jsonl runs/episodes_test_<new>/logs.jsonl
  python -m ase.stats runs/calib_test_<base>_k1/samples.jsonl runs/calib_test_<new>_k1/samples.jsonl
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict

SUCCESS = {"resolved_correct", "escalated_correct"}


def load(path: str) -> dict:
    """-> {item_key: (cluster_id, passed)}"""
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("outcome") in ("api_error", "invalid") and "error" in r:
            continue  # the API failed, the model never saw the case; pairing drops it
        if "outcome" in r:  # episode log
            out[r["case_id"]] = (r["case_id"], r["outcome"] in SUCCESS)
        else:  # calibrate sample: first draw, full credit
            gt = r["gt"]
            key = json.dumps(gt, sort_keys=True)
            out[key] = (gt["case_id"], r["samples"][0]["reward"] >= 0.99)
    return out


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def cluster_bootstrap(pairs: list[tuple[str, bool, bool]], iters: int = 5000, seed: int = 0):
    by = defaultdict(list)
    for cid, a, b in pairs:
        by[cid].append((a, b))
    clusters = list(by.values())
    rng = random.Random(seed)
    gaps = []
    for _ in range(iters):
        sample = [x for _ in clusters for x in rng.choice(clusters)]
        gaps.append(100 * (sum(b for _, b in sample) - sum(a for a, _ in sample)) / len(sample))
    gaps.sort()
    return gaps[int(0.025 * iters)], gaps[int(0.975 * iters)], sum(g > 0 for g in gaps) / iters


def main():
    base, new = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(base) & set(new))
    pairs = [(base[k][0], base[k][1], new[k][1]) for k in keys]
    n = len(pairs)
    acc_a = 100 * sum(a for _, a, _ in pairs) / n
    acc_b = 100 * sum(b for _, _, b in pairs) / n
    both = sum(a and b for _, a, b in pairs)
    only_a = sum(a and not b for _, a, b in pairs)
    only_b = sum(b and not a for _, a, b in pairs)
    neither = n - both - only_a - only_b
    lo, hi, frac = cluster_bootstrap(pairs)
    clusters = len({c for c, _, _ in pairs})
    print(f"n={n} items in {clusters} cases | base {acc_a:.1f}% -> new {acc_b:.1f}% ({acc_b - acc_a:+.1f} pts)")
    print(f"  both right {both} | only base right {only_a} | only new right {only_b} | both wrong {neither}")
    print(f"  McNemar exact p = {mcnemar_exact(only_a, only_b):.3g}")
    print(f"  case-cluster bootstrap 95% CI for the gap: [{lo:+.1f}, {hi:+.1f}] pts; new ahead in {frac:.1%} of resamples")


if __name__ == "__main__":
    main()
