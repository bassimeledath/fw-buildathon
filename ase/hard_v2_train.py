"""Training states for hard_v2 GRPO, and the per-turn reward.

States come from rollouts on TRAIN cases (a different seed from the dev/test cases, so nothing is shared). The
rollouts are noisy on purpose: about 1 move in 5 is a mistake (a repeated step, a skipped step, a premature check),
so the data holds the recovery states real models end up in, and the answer key labels those too.

Reward per state: 1 if the move is one the answer key allows, else 0. A needless question is 0 (it isn't fatal in
the episode eval, but it isn't rewarded either).

  python -m ase.hard_v2_train            # writes data/hard_v2/train/{train,heldout}_states.jsonl
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

from .env import parse_action
from .hard_v2 import SYMS, SYSTEM, STEPS, State, allowed, is_allowed, new_state, oracle_moves, sample_cases, step

TRAIN_SEED, DEV_SEED = 101, 7  # dev/test cases use seed 7 (see data/hard_v2)


def reward(text: str, gt: dict) -> float:
    a = parse_action(text)
    if a is None:
        return 0.0
    return 1.0 if is_allowed(a, gt["allowed"]) else 0.0


def _mistake(st: State, rng: random.Random) -> dict | None:
    """A plausible wrong move that keeps the call going (so later states are recovery states)."""
    lad = SYMS[st.case.sym]["ladder"]
    al = allowed(st)
    options = []
    if "step" in al:
        cur = al["step"]
        later = [s for s in lad if STEPS[s]["head"] != cur["step"]]
        if later:  # skip ahead or repeat something
            options.append({"action": "step", "doc_id": cur["doc_id"], "step": STEPS[rng.choice(later)]["head"],
                            "path": (cur["paths"] or [None])[0], "say": "."})
    options.append({"action": "ask", "slot": "symptom", "question": "."})
    return rng.choice(options) if options else None


def rollout_states(c, rng: random.Random, p_mistake: float = 0.2) -> list[dict]:
    st, out = new_state(c), []
    for _ in range(14):
        al = allowed(st)
        out.append({"messages": [{"role": "system", "content": SYSTEM}] + [dict(m) for m in st.messages],
                    "ground_truth": {"allowed": al, "case_id": c.case_id, "kind": c.kind}})
        moves = oracle_moves(st)
        if not moves:
            break
        a = _mistake(st, rng) if rng.random() < p_mistake else rng.choice(moves)
        done, _, _ = step(st, a or moves[0])
        if done:
            break
    return out


def main():
    rng = random.Random(0)
    train_cases = sample_cases(30, TRAIN_SEED)
    dev_ids = {c.case_id for c in sample_cases(12, DEV_SEED)}
    rows = [r for c in train_cases for r in rollout_states(c, rng)]
    # held-out states for the trainer's own eval come from fresh cases on yet another seed
    held = [r for c in sample_cases(4, 202) for r in rollout_states(c, rng, p_mistake=0.0)]
    d = Path("data/hard_v2/train")
    d.mkdir(parents=True, exist_ok=True)
    (d / "train_states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "heldout_states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in held))
    kinds = collections.Counter("+".join(sorted(r["ground_truth"]["allowed"])) for r in rows)
    print(f"train cases {len(train_cases)} (seed {TRAIN_SEED}, disjoint from dev/test seed {DEV_SEED}: "
          f"{not ({c.case_id for c in train_cases} & dev_ids) or 'ids overlap by name only'}), "
          f"train states {len(rows)}, held-out states {len(held)}; correct-move mix {dict(kinds.most_common(6))}")


if __name__ == "__main__":
    main()
