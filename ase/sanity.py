"""Checklist step 2: the reward must separate a good policy from deliberately bad ones. No LLM needed.

Usage: python -m ase.sanity --split test
"""

from __future__ import annotations

import argparse
import collections
import json
import random

from .reward import score
from .world import DATA_DIR


def oracle(gt, msgs, rng):
    a = gt["allowed"]
    if "escalate" in a:
        return {"action": "escalate", "reason": a["escalate"]["reason"], "reply": "Connecting you now."}
    if "resolve" in a:
        return {"action": "resolve", "doc_id": a["resolve"]["doc_id"], "reply": "Try this."}
    if "ask" in a:
        return {"action": "ask", "slot": a["ask"]["slots"][0], "question": "Which one?"}
    from .env import KB
    title = next(x.title for x in KB if x.doc_id == a["search"]["gold"])
    return {"action": "search", "query": title}


def always_search_raw(gt, msgs, rng):
    last = msgs[-1]["content"].removeprefix("[CUSTOMER] ")
    return {"action": "search", "query": " ".join(last.split()[:20])}


def resolve_top_hit(gt, msgs, rng):
    """The one-shot bias: search once, then resolve with whatever came back first."""
    for m in reversed(msgs):
        res = m["content"].split("[SEARCH RESULTS]")[-1] if "[SEARCH RESULTS]" in m["content"] else ""
        if "- KB-" in res:
            doc = res.split("- ")[1].split(":")[0]
            return {"action": "resolve", "doc_id": doc, "reply": "Try this."}
    return always_search_raw(gt, msgs, rng)


def always_ask(gt, msgs, rng):
    return {"action": "ask", "slot": rng.choice(["device", "symptom", "trigger"]), "question": "More detail?"}


def always_escalate(gt, msgs, rng):
    return {"action": "escalate", "reason": "human_request", "reply": "Transferring."}


POLICIES = [oracle, resolve_top_hit, always_search_raw, always_ask, always_escalate]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(f"{DATA_DIR}/{args.split}_states.jsonl")]
    rng = random.Random(0)
    for pol in POLICIES:
        rs, kinds = [], collections.Counter()
        for r in rows:
            out = score(json.dumps(pol(r["ground_truth"], r["messages"], rng)), r["ground_truth"])
            rs.append(out["reward"])
            kinds[out["why"].split(" ")[0]] += 1
        print(f"{pol.__name__:18s} mean reward {sum(rs) / len(rs):.3f}  pass@0.5 {sum(x >= 0.5 for x in rs) / len(rs):.2f}")


if __name__ == "__main__":
    main()
