"""A world must be fair (and the hard one harder). Every check here is code; any failure prints and exits 1.

Fair means: each fact the oracle needs is in the opening, in a customer answer (at most 2 asks per
slot), or in the text of an article the agent retrieved. And the oracle solves every case.

Usage: [ASE_WORLD=hard] [ASE_RETRIEVAL=fixed] python -m ase.fairness
"""

from __future__ import annotations

import asyncio
import collections
import json
import random
import sys

from .env import FILLER, KB, customer_answer, new_state
from .episodes import load_cases, make_case, oracle_policy, run_episode
from .world import (
    BACKUP_DOC,
    DATA_DIR,
    DESTRUCTIVE,
    DEVICE_CLUE,
    GATED,
    HARD,
    SLOTS,
    SYMPTOM_SPECIFIC,
    TRIED_TEXT,
    target_article,
)

fails = []


def check(ok: bool, msg: str):
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def main():
    if HARD:
        static_checks()
    for split in ("train", "test", "test_ood"):
        split_checks(split)
    sys.exit(1 if fails else 0)


def static_checks():
    # 1. Gates are written in the article the agent must read.
    gated = [a for a in KB if a.symptom in GATED]
    check(all("dropped or got wet" in a.body for a in gated), f"damage gate stated in all {len(gated)} gated articles")
    erase = [a for a in KB if a.tier == "advanced" and a.symptom in DESTRUCTIVE]
    check(all("backed up" in a.body and BACKUP_DOC in a.body for a in erase),
          f"backup caution + backup doc id stated in all {len(erase)} erase articles")
    # 2. Observations map to exactly one answer.
    owner = collections.defaultdict(set)
    for sym, phrases in SYMPTOM_SPECIFIC.items():
        for p in phrases:
            owner[p].add(sym)
    check(all(len(v) == 1 for v in owner.values()), "every symptom description belongs to one symptom")
    check(len(set(DEVICE_CLUE.values())) == len(DEVICE_CLUE), "every device clue names one platform")
    check(not any(t.lower() in f.lower() for t in TRIED_TEXT for f in FILLER), "filler never says basic checks were tried")


def split_checks(split: str):
    cases = load_cases(split, None)
    # 3. The tier is decidable from the opening alone.
    bad_tier = 0
    for e in cases:
        st = new_state(make_case(e), random.Random(e["seed"]))
        said = any(t.lower() in st.messages[0]["content"].lower() for t in TRIED_TEXT)
        bad_tier += said != bool(e.get("extra", {}).get("tried")) and not e["escalation"]
    check(bad_tier == 0, f"{split}: tier visible in the opening ({bad_tier} mismatches)")
    # 4. Each askable slot gives an informative answer within 2 asks.
    slow = 0
    for e in cases:
        if e["escalation"]:
            continue
        for slot in SLOTS:
            c, rng = make_case(e), random.Random(0)
            if slot in ("trigger",) and not c.trigger:
                continue
            if c.revealed.get(slot) == "specific":
                continue
            slow += not any(customer_answer(c, slot, rng)[1] for _ in range(2))
    check(slow == 0, f"{split}: every slot answered within 2 asks ({slow} slow)")
    # 5. The oracle solves every case inside the step budget.
    async def run_all():
        return await asyncio.gather(*[run_episode(e, oracle_policy) for e in cases])

    logs = asyncio.run(run_all())
    outcomes = collections.Counter(x["outcome"] for x in logs)
    steps = max(len(x["steps"]) for x in logs)
    check(set(outcomes) <= {"resolved_correct", "escalated_correct"},
          f"{split}: oracle solves all {len(cases)} cases {dict(outcomes)}, max {steps} steps")
    mix = collections.Counter()
    for e in cases:
        x = e.get("extra", {})
        c = make_case(e)
        mix["tried"] += bool(x.get("tried"))
        mix["hidden_damage"] += bool(x.get("damaged"))
        tgt = target_article(c, KB) if not e["escalation"] else None
        mix["backup_first"] += bool(tgt and tgt.doc_id == BACKUP_DOC)
        mix["vague_device"] += bool(x.get("vague_device"))
    print(f"     {split} mechanism mix: {json.dumps(dict(mix))} of {len(cases)} cases")
    states = [json.loads(line) for line in open(f"{DATA_DIR}/{split}_states.jsonl")]
    misses = sum("retrieval_miss" in r["ground_truth"]["allowed"] for r in states)
    check(misses == 0, f"{split}: no state where retrieval missed the needed article ({misses})")


if __name__ == "__main__":
    main()
