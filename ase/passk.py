"""pass@k (capability ceiling) and pass^k (consistency) per category, from a calibrate samples file.

Usage: python -m ase.passk runs/calib_test_<tag>_k8/samples.jsonl [more files...]
Categories: the correct-action type, plus first-turn vs after-search, so the curve is never one number.
"""

from __future__ import annotations

import collections
import json
import sys

PASS = 0.99  # a draw passes only on full credit


def curves(rows, ks=(1, 2, 4, 8)):
    out = {}
    for k in ks:
        if all(len(r["samples"]) >= k for r in rows):
            at = sum(any(s["reward"] >= PASS for s in r["samples"][:k]) for r in rows) / len(rows)
            hat = sum(all(s["reward"] >= PASS for s in r["samples"][:k]) for r in rows) / len(rows)
            out[k] = (round(at, 3), round(hat, 3))
    return out


def quadrant(c):
    k = max(c)
    ceiling, floor = c[k]
    gap = ceiling - floor
    kind = "stochastic" if gap >= 0.2 else "systematic"
    high = ceiling >= 0.5
    verdict = {
        ("systematic", True): "already good -> efficiency",
        ("systematic", False): "import capability",
        ("stochastic", True): "consolidate (RS-SFT / RL)",
        ("stochastic", False): "import, then consolidate",
    }[(kind, high)]
    return f"ceiling {ceiling:.2f} gap {gap:.2f} -> {kind}, {verdict}"


def category(r):
    kind = "+".join(sorted(r["gt"]["allowed"]))
    stage = "after_search" if r["gt"]["seen_ids"] else "before_search"
    return f"{kind}|{stage}"


def main():
    for path in sys.argv[1:]:
        rows = [json.loads(line) for line in open(path)]
        print(f"\n== {path}  (n={len(rows)})")
        c = curves(rows)
        print(f"  {'ALL':28s} {c}  {quadrant(c)}")
        by = collections.defaultdict(list)
        for r in rows:
            by[category(r)].append(r)
        for name, rs in sorted(by.items()):
            c = curves(rs)
            print(f"  {name:28s} n={len(rs):3d} {c}  {quadrant(c)}")


if __name__ == "__main__":
    main()
