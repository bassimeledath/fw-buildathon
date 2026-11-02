"""Retriever, customer rendering, scripted customer, and the oracle that labels each state."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .world import (
    DESTRUCTIVE,
    DEVICE_CLUE,
    DEVICE_FAMILY,
    DEVICE_SPECIFIC,
    ESCALATIONS,
    GATED,
    FIXED,
    HARD,
    NOISE_FACTS,
    SHARD,
    PLATFORMS,
    SYMPTOM_SPECIFIC,
    DISTRACTOR_TEXT,
    SHARD_FILLER_CONFLICT,
    SHARD_PREFIX,
    SYMPTOM_VAGUE,
    SYMPTOMS,
    TRIED_TEXT,
    TRIGGER_TEXT,
    BACKUP_DOC,
    Article,
    Case,
    build_kb,
    candidates,
    discriminating_slots,
    gold_article,
    target_article,
)

RESULTS_K = 5 if HARD or FIXED else 3
P_SHARD = 0.75  # share of "already tried" cases that volunteer it late instead of up front
P_NOISE = 0.4  # share of cases carrying a second, explicitly set-aside device

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

SYSTEM_PROMPT_HARD = """You are a tech-support agent for a phone carrier, talking to a customer. Each turn, output exactly one JSON object and nothing else. The four possible actions:

{"action": "search", "query": "<KB search query>"}
{"action": "ask", "slot": "device" | "symptom" | "trigger" | "damage" | "backup", "question": "<one short question for the customer>"}
{"action": "resolve", "doc_id": "<KB id from search results>", "reply": "<short reply to the customer>"}
{"action": "escalate", "reason": "physical_damage" | "safety" | "billing" | "human_request", "reply": "<short reply>"}

Rules:
- Escalate, never troubleshoot: physical or liquid damage, safety hazards (swelling battery, smoke, burning smell), billing, plan or payment questions, or a customer asking for a human.
- Articles differ by device type (iPhone, Android phone, Windows laptop, Mac, home router), by symptom, often by the trigger (when, where, or under what conditions it happens), and by tier: basic checks or advanced fix.
- Use the basic-checks article unless the customer reports they already tried a restart and the settings. Use the advanced fix only then.
- Read the article text in the search results. Do what it says to do before you resolve with it.
- Only resolve with an article you have seen in search results that matches the customer's exact situation.
- If you cannot tell which article applies, ask the one question that decides it. The slot says what the question is about: damage = was it dropped or wet, backup = is the data backed up.
- Searches are invisible to the customer. Questions cost the customer time, so do not ask what you already know."""


def system_prompt() -> str:
    p = SYSTEM_PROMPT_HARD if HARD else SYSTEM_PROMPT
    if not FIXED:
        return p
    p = p.replace("The four possible actions:", "The three possible actions:")
    p = p.replace('{"action": "search", "query": "<KB search query>"}\n', "")
    p = p.replace("Searches are invisible to the customer. Questions", "Each customer message comes with the top KB "
                  "search results for everything the customer has said so far. Questions")
    return p


def _tok(s: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in s).split()


def _aliases(a: Article) -> str:
    """Fixed retrieval searches raw customer words, so index the words customers use for each article."""
    if a.symptom == "other":
        return ""
    words = DEVICE_SPECIFIC[a.platform] + ([DEVICE_CLUE[a.platform]] if a.platform in DEVICE_CLUE else [])
    words += SYMPTOM_SPECIFIC[a.symptom] + (TRIGGER_TEXT[a.trigger] if a.trigger else [])
    if a.tier == "advanced":
        words += TRIED_TEXT
    return " ".join(words)


class Retriever:
    def __init__(self, kb: list[Article]):
        self.kb = kb
        self.bm25 = BM25Okapi([_tok(a.title + " " + a.body + (" " + _aliases(a) if FIXED else "")) for a in kb])

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
    if HARD:  # the agent must read the article text: the gates live there
        lines = [f"- {a.doc_id}: {a.title}\n  {a.body}" for a in arts]
    else:
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
if HARD:  # "already tried" is signal in the hard world, so filler must not say it; it may tease it
    FILLER[FILLER.index("I already tried turning it off and on.")] = "My friend said to restart it, but I haven't yet."


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
    extra = {}
    if HARD:
        extra = {
            "tried": rng.random() < 0.4,
            "damaged": sym in GATED and rng.random() < 0.3,  # never volunteered; only the gate question finds it
            "backed_up": rng.random() < 0.6,
            "vague_device": dev_level != "specific" and fam != "router" and rng.random() < 0.4,
        }
        if NOISE_FACTS and random.Random(case_id + "-noise").random() < P_NOISE:
            # A device from another family, so it can never be confused with the real cell.
            decoy = random.Random(case_id + "-decoy").choice(
                [p for p in PLATFORMS if PLATFORMS[p]["family"] != fam])
            extra["distractor"] = DISTRACTOR_TEXT[decoy]
        # Take the shard decision from a side RNG keyed on the case id, so the main stream is
        # untouched and data_hard_s holds the SAME cases as data_hard -- stage comparisons stay
        # paired. The fact is still true at resolve either way, so the tier mix is identical by
        # construction; only its delivery moves from the opening to mid-conversation.
        if SHARD and extra["tried"] and random.Random(case_id + "-shard").random() < P_SHARD:
            srng = random.Random(case_id + "-shard-text")
            extra["tried"] = False  # withheld from the opening; maybe_shard() restores it
            extra["shard_slot"] = "tried"
            extra["shard_value"] = True
            extra["shard_text"] = srng.choice(SHARD_PREFIX) + srng.choice(TRIED_TEXT) + "."
            extra["shard_fired"] = False
            extra["shard_turn"] = None
    return Case(case_id, plat, sym, trig, revealed=revealed, extra=extra)


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
    if case.extra.get("tried"):
        parts.append(rng.choice(TRIED_TEXT).capitalize() + ".")
    filler = rng.choice(FILLER)
    if case.extra.get("shard_slot") == "tried" and filler == SHARD_FILLER_CONFLICT:
        filler = ""  # would make the customer contradict its own shard
    filler = case.extra.get("distractor", filler)  # draw above already consumed, so cases stay paired
    parts.append(filler)
    return " ".join(p for p in parts if p)


def customer_answer(case: Case, slot: str, rng: random.Random) -> tuple[str, bool]:
    """Scripted customer. Returns (reply, informative). Updates what has been revealed."""
    if case.escalation:
        return "I told you what happened. Can you help or not?", False
    if slot in ("damage", "backup"):
        yes = case.extra.get("damaged") if slot == "damage" else case.extra.get("backed_up", True)
        if slot in case.revealed:
            return "I already told you.", False
        case.revealed[slot] = "yes" if yes else "no"
        if slot == "damage":
            return ("Well, I did drop it last week, and it got a bit wet." if yes
                    else "No, it hasn't been dropped or gotten wet."), True
        return ("Yes, it backs up to the cloud every night." if yes else "No, I've never backed it up."), True
    already = case.revealed.get(slot) == "specific"
    if slot == "device" and case.extra.get("vague_device") and not already:
        if not case.extra.get("device_asked"):  # first ask: the customer does not know the model
            case.extra["device_asked"] = True
            return "Honestly, I'm not sure what model it is.", False
        case.revealed[slot] = "specific"
        return "I don't know the name, but " + DEVICE_CLUE[case.platform] + ".", True
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
        return Case(**{**self.case.__dict__, "revealed": dict(self.case.revealed), "extra": dict(self.case.extra)})


def maybe_shard(state: State) -> bool:
    """Volunteer the withheld fact once the agent has committed to retrieved results."""
    c = state.case
    slot = c.extra.get("shard_slot")
    if not slot or c.extra.get("shard_fired") or not state.seen_ids:
        return False
    c.extra["shard_fired"] = True
    if slot in c.revealed:  # would contradict an answer already given
        c.extra["shard_defused"] = True
        return False
    c.extra[slot] = c.extra["shard_value"]  # moves the gold article
    c.extra["shard_turn"] = sum(m["role"] == "assistant" for m in state.messages)
    state.messages.append({"role": "user", "content": "[CUSTOMER] " + c.extra["shard_text"]})
    state.searched_since_answer = False  # a new fact invalidates the last search
    return True


def _allowed_hard(state: State, kb: list[Article]) -> dict:
    case = state.case
    if case.escalation:
        return {"escalate": {"reason": case.escalation}}
    if case.revealed.get("damage") == "yes":
        return {"escalate": {"reason": "physical_damage"}}
    cands = candidates(case, kb)
    if len(cands) > 1:
        slots = discriminating_slots(cands)
        if "damage" not in case.revealed and all(a.symptom in GATED for a in cands):
            slots.append("damage")  # every remaining article opens with the damage question
        out = {"ask": {"slots": slots}}
        # seen_ids only grows, so without the shard clause a post-shard re-search -- the behaviour the
        # mechanism exists to reward -- would score WRONG_TYPE_CREDIT.
        if not state.searched_since_answer and (not state.seen_ids or case.extra.get("shard_fired")):
            out["search"] = {"gold": gold_article(case, kb).doc_id, "candidates": [a.doc_id for a in cands]}
        return out
    art = cands[0]
    need = []  # checks the article says to do before resolving
    if art.symptom in GATED and "damage" not in case.revealed:
        need.append("damage")
    if art.tier == "advanced" and art.symptom in DESTRUCTIVE and "backup" not in case.revealed:
        need.append("backup")
    if need:
        out = {"ask": {"slots": need}}
        if art.doc_id not in state.seen_ids:
            out["search"] = {"gold": art.doc_id, "candidates": [art.doc_id]}
        return out
    tgt = target_article(case, kb)
    if tgt.doc_id in state.seen_ids:
        return {"resolve": {"doc_id": tgt.doc_id}}
    return {"search": {"gold": tgt.doc_id, "candidates": [tgt.doc_id]}}


def allowed_actions(state: State, kb: list[Article]) -> dict:
    """The oracle. Returns the set of correct actions for this state and what each needs."""
    out = _allowed_hard(state, kb) if HARD else _allowed_v1(state, kb)
    if FIXED:
        out.pop("search", None)
        if not out:  # retrieval missed the article the customer described; ase.fairness counts these
            out = {"retrieval_miss": {}}
    return out


def _allowed_v1(state: State, kb: list[Article]) -> dict:
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
    gold = target_article(c, kb) if not c.escalation else None
    return {
        "case_id": c.case_id,
        "allowed": allowed_actions(state, kb),
        "gold_doc": gold.doc_id if gold else None,
        "escalation": c.escalation,
        "n_candidates": len(candidates(c, kb)),
        # survives calibrate.py's band writer, which keeps only messages + ground_truth
        "post_shard": bool(c.extra.get("shard_fired")) and not c.extra.get("shard_defused"),
        "seen_ids": sorted(state.seen_ids),
    }


def to_prompt(state: State) -> list[dict]:
    return [{"role": "system", "content": system_prompt()}] + state.messages


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
    allowed = allowed_actions(state, retriever.kb)  # before this action changes the state
    state.messages.append({"role": "assistant", "content": json.dumps(action)})
    kind = action.get("action")
    if FIXED and kind == "search":
        return True, "invalid"  # not an action in the fixed-retrieval baseline
    if (HARD or FIXED) and kind == "resolve":
        if "resolve" in allowed and action.get("doc_id") == allowed["resolve"]["doc_id"]:
            return True, "resolved_correct"
        tgt = target_article(state.case, retriever.kb) if not state.case.escalation else None
        hidden_damage = state.case.extra.get("damaged")
        if tgt and action.get("doc_id") == tgt.doc_id and not hidden_damage:
            return True, "resolved_skipped_check"  # right article, but a check the article requires was skipped
        return True, "resolved_wrong"
    if (HARD or FIXED) and kind == "escalate":
        ok = "escalate" in allowed and (not FIXED or action.get("reason") == allowed["escalate"]["reason"])
        return True, "escalated_correct" if ok else "escalated_wrong"
    if kind == "search":
        res = retriever.search(str(action.get("query", "")), k=RESULTS_K)
        state.seen_ids |= {a.doc_id for a in res}
        state.searched_since_answer = True
        state.messages.append({"role": "user", "content": format_results(res)})
        maybe_shard(state)
        return False, "search"
    if kind == "ask":
        reply, _ = customer_answer(state.case, str(action.get("slot", "")), rng)
        state.searched_since_answer = False
        state.messages.append({"role": "user", "content": "[CUSTOMER] " + reply})
        maybe_shard(state)  # before auto_search, so fixed retrieval sees the new fact
        if FIXED:
            auto_search(state)
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


def customer_text(state: State) -> str:
    msgs = [m["content"].split("\n\n[SEARCH RESULTS]")[0] for m in state.messages if m["role"] == "user"]
    return " ".join(m.removeprefix("[CUSTOMER] ") for m in msgs if m.startswith("[CUSTOMER]"))


def auto_search(state: State):
    """Fixed retrieval: search everything the customer said; append results to the last customer message."""
    res = RETRIEVER.search(customer_text(state), k=RESULTS_K)
    state.seen_ids |= {a.doc_id for a in res}
    if any(BACKUP_DOC in a.body for a in res):  # the erase caution names the backup article by id
        state.seen_ids.add(BACKUP_DOC)
    state.messages[-1]["content"] += "\n\n" + format_results(res)


def new_state(case: Case, rng: random.Random) -> State:
    state = State(case=case, messages=[{"role": "user", "content": "[CUSTOMER] " + render_opening(case, rng)}])
    if FIXED:
        auto_search(state)
    return state


KB = build_kb()
RETRIEVER = Retriever(KB)
