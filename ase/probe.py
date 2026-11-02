"""Adversarial probe: how far do shallow, no-LLM policies get on this benchmark?

Different from ase.sanity. Sanity scores deliberately *bad* policies on single states to prove the
reward discriminates. This runs deliberately *cheap but shrewd* policies through full episodes, with
the real eval scorer, to find where the benchmark's difficulty actually lives. Every policy here
reads only `messages` -- never the ground truth -- so a high score is a real weakness, not a leak.

Usage: ASE_WORLD=hard python -m ase.probe [--split test] [--policy gate_follower]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import re

from .env import KB
from .episodes import load_cases, run_episode, summarize
from .world import BACKUP_DOC, DEVICE_CLUE, DEVICE_SPECIFIC, SYMPTOM_SPECIFIC, TRIED_TEXT, TRIGGER_TEXT

CORE = ("device", "symptom", "trigger")

# Surface cues an agent could match without understanding anything. Each maps to an escalation reason.
ESCALATION_CUES = {
    "physical_damage": ["dropped", "cracked", "spilled", "in the pool", "got a bit wet", "got wet"],
    "safety": ["swelling", "burning", "smoke", "smells like"],
    "billing": ["bill", "charge", "plan", "payment", "suspended"],
    "human_request": ["real person", "human agent", "talk to a"],
}
# Phrases the articles themselves use for each required pre-check (guaranteed present by ase.fairness).
GATE_CUE = "dropped or got wet"
ERASE_CUE = "erases the device"


def _said(msgs, *cues) -> bool:
    text = " ".join(m["content"] for m in msgs if m["role"] == "user").lower()
    return any(c in text for c in cues)


def _results(msgs) -> list[tuple[str, str]]:
    """Latest search results as [(doc_id, text)], parsed out of the rendered block."""
    for m in reversed(msgs):
        if "[SEARCH RESULTS]" not in m["content"]:
            continue
        block = m["content"].split("[SEARCH RESULTS]")[-1]
        hits = re.findall(r"- (KB-\d+): (.*?)(?=\n- KB-|\Z)", block, re.S)
        if hits:
            return [(d, t) for d, t in hits]
    return []


def _asked(msgs) -> set[str]:
    """Slots this policy already asked about, read back off its own turns."""
    out = set()
    for m in msgs:
        if m["role"] != "assistant":
            continue
        try:
            a = json.loads(m["content"])
        except Exception:
            continue
        if a.get("action") == "ask":
            out.add(a.get("slot"))
    return out


def _customer_text(msgs) -> str:
    return " ".join(m["content"] for m in msgs if m["role"] == "user" and "[SEARCH RESULTS]" not in m["content"])


def _opening(msgs) -> str:
    """The customer's first turn. NB: msgs[0] is the system prompt, so this is not msgs[0]."""
    return next((m["content"].lower() for m in msgs if m["role"] == "user"), "")


def _best(msgs, hits) -> tuple[str, str]:
    """Pick a result by token overlap with everything the customer said. Deliberately shallow."""
    said = set(re.findall(r"[a-z0-9]+", _customer_text(msgs).lower()))
    tried = any(t.lower() in _customer_text(msgs).lower() for t in TRIED_TEXT)
    best, score = hits[0], -1.0
    for doc, text in hits:
        low = text.lower()
        if ("advanced fix" in low) != tried:  # the tier is stated in the opening; match it
            continue
        s = len(said & set(re.findall(r"[a-z0-9]+", low)))
        if s > score:
            best, score = (doc, text), s
    return best


# ---- policies: (messages, gt) -> action dict. gt is accepted and ignored.


def gate_follower(msgs, gt) -> dict:
    """The key probe: never reasons about the customer, just obeys the text it retrieved.

    Escalate on surface cues, search once, pick a plausible article, then do whatever pre-checks
    that article's body names before resolving. If this scores well, the hard world's new
    mechanisms test text-following, not judgment.
    """
    first = _opening(msgs)
    for reason, cues in ESCALATION_CUES.items():  # opening-line escalations
        if any(c in first for c in cues):
            return {"action": "escalate", "reason": reason, "reply": "Let me get you to someone."}
    if _said(msgs, "did drop it", "got a bit wet"):  # the gate question came back yes
        return {"action": "escalate", "reason": "physical_damage", "reply": "That needs a technician."}
    hits = _results(msgs)
    if not hits:
        return {"action": "search", "query": " ".join(_customer_text(msgs).split()[:20])}
    doc, text = _best(msgs, hits)
    asked, low = _asked(msgs), text.lower()
    if GATE_CUE in low and "damage" not in asked:
        return {"action": "ask", "slot": "damage", "question": "Was it dropped or did it get wet?"}
    if ERASE_CUE in low and "backup" not in asked:
        return {"action": "ask", "slot": "backup", "question": "Is your data backed up?"}
    if _said(msgs, "never backed it up"):
        return {"action": "resolve", "doc_id": BACKUP_DOC, "reply": "Let's back up first."}
    return {"action": "resolve", "doc_id": doc, "reply": "Try this."}


def ask_everything(msgs, gt) -> dict:
    """Ask every slot, then resolve the top hit. Tests whether over-asking is punished enough."""
    first = _opening(msgs)
    for reason, cues in ESCALATION_CUES.items():
        if any(c in first for c in cues):
            return {"action": "escalate", "reason": reason, "reply": "Transferring."}
    if _said(msgs, "did drop it", "got a bit wet"):
        return {"action": "escalate", "reason": "physical_damage", "reply": "That needs a technician."}
    asked = _asked(msgs)
    for slot in CORE + ("damage", "backup"):
        if slot not in asked:
            return {"action": "ask", "slot": slot, "question": f"Tell me about the {slot}?"}
    hits = _results(msgs)
    if not hits:
        return {"action": "search", "query": " ".join(_customer_text(msgs).split()[:20])}
    return {"action": "resolve", "doc_id": _best(msgs, hits)[0], "reply": "Try this."}


def blind_top_hit(msgs, gt) -> dict:
    """Control: search once, resolve the top hit, ignore every gate. Should fail the checks."""
    hits = _results(msgs)
    if not hits:
        return {"action": "search", "query": " ".join(_customer_text(msgs).split()[:20])}
    return {"action": "resolve", "doc_id": hits[0][0], "reply": "Try this."}


def cue_only_escalate(msgs, gt) -> dict:
    """How many escalation cases fall to a keyword list alone? Resolves nothing."""
    first = _opening(msgs)
    for reason, cues in ESCALATION_CUES.items():
        if any(c in first for c in cues):
            return {"action": "escalate", "reason": reason, "reply": "Transferring."}
    return {"action": "ask", "slot": "device", "question": "Which device?"}


POLICIES = {p.__name__: p for p in (gate_follower, ask_everything, blind_top_hit, cue_only_escalate)}


def as_async(fn):
    async def policy(messages, gt):
        return json.dumps(fn(messages, gt)), 0.0, 0, 0

    return policy


async def amain(args):
    cases = load_cases(args.split, args.n)
    names = [args.policy] if args.policy else list(POLICIES)
    rows = {}
    for name in names:
        logs = await asyncio.gather(*[run_episode(e, as_async(POLICIES[name])) for e in cases])
        s = summarize(logs)
        rows[name] = s
        outs = collections.Counter(x["outcome"] for x in logs)
        print(f"\n== {name} ({args.split}, n={s['n']})")
        print(f"   success {s['success']:.3f}  wrong_fix {s['wrong_fix_rate']:.3f}  "
              f"skipped_check {s['skipped_check_rate']:.3f}  "
              f"missed_esc {s['missed_escalation_rate']}  hidden_damage_missed {s['hidden_damage_missed']}")
        print(f"   questions/case {s['customer_questions_per_case']}  wasted {s['wasted_questions_per_case']}  "
              f"stuck {s['invalid_or_stuck']:.3f}  no_resolve {s['no_resolve_rate']:.3f}")
        print(f"   outcomes {dict(outs)}")
    print("\n" + json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--policy", default="", choices=[""] + list(POLICIES))
    ap.add_argument("--n", type=int, default=None)
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
