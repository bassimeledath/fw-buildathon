"""Retriever, customer rendering, scripted customer, and the oracle that labels each state."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .world import (
    DEVICE_FAMILY,
    DEVICE_SPECIFIC,
    ESCALATIONS,
    PLATFORMS,
    SYMPTOM_SPECIFIC,
    SYMPTOM_VAGUE,
    SYMPTOMS,
    TRIGGER_TEXT,
    Article,
    Case,
    build_kb,
    candidates,
    discriminating_slots,
    gold_article,
)

SYSTEM_PROMPT = """You are a tech-support agent for a phone carrier, talking to a customer. Each turn, output exactly one JSON object and nothing else. The four possible actions:

{"action": "search", "query": "<KB search query>"}
{"action": "ask", "slot": "device" | "symptom" | "trigger", "question": "<one short question for the customer>"}
{"action": "resolve", "doc_id": "<KB id from search results>", "reply": "<short reply to the customer>"}
{"action": "escalate", "reason": "physical_damage" | "safety" | "billing" | "human_request", "reply": "<short reply>"}

Rules:
- Escalate, never troubleshoot: physical or liquid damage, safety hazards (swelling battery, smoke, burning smell), billing, plan or payment questions, or a customer asking for a human.
- The knowledge base has one article per exact situation. Articles differ by device type (iPhone, Android phone, Windows laptop, Mac, home router), by symptom, and often by the trigger (when, where, or under what conditions it happens).
- Only resolve with an article you have seen in search results that matches the customer's exact device, symptom and trigger.
- If you cannot tell which article applies, ask the one question that decides it. The slot says what the question is about.
- Searches are invisible to the customer. Questions cost the customer time, so do not ask what you already know."""


def _tok(s: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in s).split()


class Retriever:
    def __init__(self, kb: list[Article]):
        self.kb = kb
        self.bm25 = BM25Okapi([_tok(a.title + " " + a.body) for a in kb])

    def search(self, query: str, k: int = 3) -> list[Article]:
        scores = self.bm25.get_scores(_tok(query))
        order = sorted(range(len(self.kb)), key=lambda i: -scores[i])
        return [self.kb[i] for i in order[:k] if scores[i] > 0]

    def rank_of(self, query: str, doc_id: str) -> int | None:
        for i, a in enumerate(self.search(query, k=10)):
            if a.doc_id == doc_id:
                return i + 1
        return None


def format_results(arts: list[Article]) -> str:
    if not arts:
        return "[SEARCH RESULTS] No matching articles."
    lines = [f"- {a.doc_id}: {a.title}" for a in arts]
    return "[SEARCH RESULTS]\n" + "\n".join(lines)


FILLER = [
    "I've had it for about two years.",
    "I'm at work and really need this fixed.",
    "My daughter set it up for me so I'm not great with this stuff.",
    "I already tried turning it off and on.",
    "This is the second time this month.",
    "Honestly this is getting ridiculous.",
    "",
    "",
]


def sample_case(rng: random.Random, case_id: str, p_escalate: float = 0.2) -> Case:
    if rng.random() < p_escalate:
        return Case(case_id, None, None, None, escalation=rng.choice(list(ESCALATIONS)))
    sym = rng.choice(list(SYMPTOMS))
    spec = SYMPTOMS[sym]
    plat = rng.choice(spec["platforms"])
    trig = rng.choice(spec["triggers"]) if spec["triggers"] else None
    fam = PLATFORMS[plat]["family"]
    dev_level = rng.choices(["specific", "family", None], [0.5, 0.3, 0.2])[0]
    if fam == "router" and dev_level == "family":
        dev_level = "specific"
    revealed = {
        "device": dev_level,
        "symptom": rng.choices(["specific", "vague"], [0.65, 0.35])[0],
        "trigger": rng.choices(["specific", None], [0.45, 0.55])[0] if trig else None,
    }
    return Case(case_id, plat, sym, trig, revealed=revealed)


def render_opening(case: Case, rng: random.Random) -> str:
    if case.escalation:
        dev = rng.choice(["phone", "iPhone", "Galaxy", "laptop"])
        return rng.choice(ESCALATIONS[case.escalation]).format(dev=dev) + ". " + rng.choice(FILLER)
    r = case.revealed
    if r["device"] == "specific":
        dev = "my " + rng.choice(DEVICE_SPECIFIC[case.platform])
    elif r["device"] == "family":
        dev = "my " + rng.choice(DEVICE_FAMILY[PLATFORMS[case.platform]["family"]])
    else:
        dev = "it"
    sym = (
        rng.choice(SYMPTOM_SPECIFIC[case.symptom])
        if r["symptom"] == "specific"
        else rng.choice(SYMPTOM_VAGUE[SYMPTOMS[case.symptom]["group"]])
    )
    parts = [f"Hi, {dev}: {sym}." if dev != "it" else f"Hi, {sym}."]
    if r.get("trigger") == "specific":
        parts.append(rng.choice(TRIGGER_TEXT[case.trigger]).capitalize() + ".")
    parts.append(rng.choice(FILLER))
    return " ".join(p for p in parts if p)


def customer_answer(case: Case, slot: str, rng: random.Random) -> tuple[str, bool]:
    """Scripted customer. Returns (reply, informative). Updates what has been revealed."""
    if case.escalation:
        return "I told you what happened. Can you help or not?", False
    already = case.revealed.get(slot) == "specific"
    if slot == "device":
        text = "It's a " + rng.choice(DEVICE_SPECIFIC[case.platform]) + "."
    elif slot == "symptom":
        text = rng.choice(SYMPTOM_SPECIFIC[case.symptom]).capitalize() + "."
    elif slot == "trigger" and case.trigger:
        text = rng.choice(TRIGGER_TEXT[case.trigger]).capitalize() + "."
    else:
        return "I'm not sure, it just happens.", False
    if already:
        return "I already told you. " + text, False
    case.revealed[slot] = "specific"
    return text, True


@dataclass
class State:
    """A conversation snapshot the policy acts on, plus everything the reward needs."""

    case: Case
    messages: list[dict] = field(default_factory=list)  # user/assistant turns after the system prompt
    seen_ids: set = field(default_factory=set)
    searched_since_answer: bool = False

    def snapshot_case(self) -> Case:
        return Case(**{**self.case.__dict__, "revealed": dict(self.case.revealed)})


def allowed_actions(state: State, kb: list[Article]) -> dict:
    """The oracle. Returns the set of correct actions for this state and what each needs."""
    case = state.case
    if case.escalation:
        return {"escalate": {"reason": case.escalation}}
    cands = candidates(case, kb)
    gold = gold_article(case, kb)
    if len(cands) == 1:
        if gold.doc_id in state.seen_ids:
            return {"resolve": {"doc_id": gold.doc_id}}
        return {"search": {"gold": gold.doc_id, "candidates": [gold.doc_id]}}
    disc = discriminating_slots(cands)
    out = {"ask": {"slots": disc}}
    if not state.searched_since_answer and not state.seen_ids:
        out["search"] = {"gold": gold.doc_id, "candidates": [a.doc_id for a in cands]}
    return out


def ground_truth(state: State, kb: list[Article]) -> dict:
    c = state.case
    gold = gold_article(c, kb) if not c.escalation else None
    return {
        "case_id": c.case_id,
        "allowed": allowed_actions(state, kb),
        "gold_doc": gold.doc_id if gold else None,
        "escalation": c.escalation,
        "n_candidates": len(candidates(c, kb)),
        "seen_ids": sorted(state.seen_ids),
    }


def to_prompt(state: State) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}] + state.messages


def parse_action(text: str) -> dict | None:
    """Pull the one JSON object out of a completion (after any <think> block)."""
    text = text.split("</think>")[-1]
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "action" in obj else None


def step(state: State, action: dict, retriever: Retriever, rng: random.Random) -> tuple[bool, str]:
    """Apply an action to the environment. Returns (done, outcome)."""
    state.messages.append({"role": "assistant", "content": json.dumps(action)})
    kind = action.get("action")
    if kind == "search":
        res = retriever.search(str(action.get("query", "")))
        state.seen_ids |= {a.doc_id for a in res}
        state.searched_since_answer = True
        state.messages.append({"role": "user", "content": format_results(res)})
        return False, "search"
    if kind == "ask":
        reply, _ = customer_answer(state.case, str(action.get("slot", "")), rng)
        state.searched_since_answer = False
        state.messages.append({"role": "user", "content": "[CUSTOMER] " + reply})
        return False, "ask"
    if kind == "resolve":
        gold = gold_article(state.case, retriever.kb) if not state.case.escalation else None
        ok = gold is not None and action.get("doc_id") == gold.doc_id
        return True, "resolved_correct" if ok else "resolved_wrong"
    if kind == "escalate":
        if state.case.escalation:
            return True, "escalated_correct"
        return True, "escalated_wrong"
    return True, "invalid"


def new_state(case: Case, rng: random.Random) -> State:
    return State(case=case, messages=[{"role": "user", "content": "[CUSTOMER] " + render_opening(case, rng)}])


KB = build_kb()
RETRIEVER = Retriever(KB)
