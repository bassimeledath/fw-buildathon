"""Roll out a noisy oracle through the world to collect decision states, label each one, write JSONL.

Usage: python -m ase.build_dataset --n-cases 600 --seed 7 --out data
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from .env import KB, RETRIEVER, allowed_actions, ground_truth, new_state, sample_case, step, to_prompt
from .world import BACKUP_DOC, DATA_DIR, HARD, PLATFORMS, SYMPTOM_TITLE, gold_article

OOD_SYMPTOMS = {"bluetooth_pairing", "overheating"}  # never in train; tests generalisation


def oracle_query(state, rng: random.Random, noisy: bool) -> str:
    c = state.case
    opening = state.messages[0]["content"].removeprefix("[CUSTOMER] ")
    if noisy or c.escalation:
        return " ".join(opening.split()[:20])
    parts = []
    if c.revealed.get("device") == "specific":
        parts.append(PLATFORMS[c.platform]["label"])
    if c.revealed.get("symptom") == "specific":
        parts.append(SYMPTOM_TITLE[c.symptom])
    if c.revealed.get("trigger") == "specific" and c.trigger:
        parts.append(c.trigger.replace("_", " "))
    if HARD:
        if allowed_actions(state, KB).get("search", {}).get("gold") == BACKUP_DOC:
            return "back up the device before an erase"
        parts.append("advanced fix" if c.tier() == "advanced" else "basic checks")
    return " ".join(parts) or " ".join(opening.split()[:20])


def oracle_action(state, rng: random.Random, p_noise: float = 0.3) -> dict:
    allowed = allowed_actions(state, KB)
    kind = rng.choice(sorted(allowed))  # drawn even when overridden below, to keep the stream aligned
    if state.case.extra.get("shard_slot") and not state.case.extra.get("shard_fired") and "search" in allowed:
        kind = "search"  # the shard fires on the first search; get there so post-shard states exist
    if kind == "escalate":
        return {"action": "escalate", "reason": allowed[kind]["reason"], "reply": "Let me connect you."}
    if kind == "resolve":
        return {"action": "resolve", "doc_id": allowed[kind]["doc_id"], "reply": "Here's the fix."}
    if kind == "ask":
        slots = allowed[kind]["slots"]
        slot = rng.choice(["device", "symptom", "trigger"]) if rng.random() < 0.15 else rng.choice(slots)
        return {"action": "ask", "slot": slot, "question": f"Can you tell me more about the {slot}?"}
    return {"action": "search", "query": oracle_query(state, rng, noisy=rng.random() < p_noise)}


ROLLOUT_STEPS = 12 if HARD else 8  # match the episode budget so the dataset covers the same states


def rollout_states(case, rng: random.Random, max_steps: int = ROLLOUT_STEPS) -> list[dict]:
    state = new_state(case, rng)
    rows = []
    for t in range(max_steps):
        gt = ground_truth(state, KB)
        rows.append({"messages": to_prompt(state), "ground_truth": gt, "step": t,
                     "shard_fired": bool(case.extra.get("shard_fired"))})
        done, _ = step(state, oracle_action(state, rng), RETRIEVER, rng)
        if done:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cases", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=DATA_DIR)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    splits = collections.defaultdict(list)
    episodes = collections.defaultdict(list)
    dropped = 0
    for i in range(args.n_cases):
        case = sample_case(rng, f"c{i:04d}")
        if case.symptom in OOD_SYMPTOMS:
            split = "test_ood"
        else:
            split = "test" if rng.random() < 0.2 else "train"
        snap = {"case_id": case.case_id, "seed": rng.randrange(1 << 30), "revealed": dict(case.revealed),
                "extra": dict(case.extra), **{k: case.__dict__[k] for k in ("platform", "symptom", "trigger", "escalation")}}
        rows = rollout_states(case, random.Random(snap["seed"]))
        if any("retrieval_miss" in r["ground_truth"]["allowed"] for r in rows):
            dropped += 1  # fixed retrieval missed the article (~0.05% of hard cases); keep the set fair
            continue
        episodes[split].append(snap)
        splits[split] += rows

    for split, rows in splits.items():
        with open(out / f"{split}_states.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(out / f"{split}_episodes.jsonl", "w") as f:
            for e in episodes[split]:
                f.write(json.dumps(e) + "\n")
        kinds = collections.Counter("+".join(sorted(r["ground_truth"]["allowed"])) for r in rows)
        print(f"{split}: {len(episodes[split])} cases, {len(rows)} states, allowed-action mix {dict(kinds)}")
    with open(out / "kb.jsonl", "w") as f:
        for a in KB:
            f.write(json.dumps(a.__dict__) + "\n")
    print(f"kb: {len(KB)} articles; dropped {dropped} cases with a retrieval miss")


if __name__ == "__main__":
    main()
