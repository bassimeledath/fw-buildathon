"""Hard ASE evaluation: conversational state, prerequisites, and KB applicability.

The world is fully synthetic and deterministic.  The policy sees customer turns and
fixed retrieval results; hidden case fields are used only by the scripted customer.

Usage:
    python -m ase.hard_sol --oracle-only
    python -m ase.hard_sol --freeze-only
    python -m ase.hard_sol
    python -m ase.hard_sol --model accounts/fireworks/models/kimi-k3 --reasoning-effort none
    python -m ase.hard_sol --split dev --model accounts/fireworks/models/deepseek-v4p1-flash --reasoning-effort none

Splits: dev (first 40% of each case type, published in data/hard_v1/dev/) and test (the rest, held out
for final scoring; regenerate locally with --split test, don't commit it).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import copy
import itertools
import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from .env import parse_action
from .world import SYMPTOM_SPECIFIC, SYMPTOM_TITLE, TRIGGER_TEXT, TRIGGER_TITLE


PLATFORMS = ("iphone", "android")
LABEL = {"iphone": "iPhone", "android": "Android phone"}
CURRENT_OS = {"iphone": "iOS 26", "android": "Android 16"}
DEVICE_NAMES = {
    "iphone": ("iPhone", "Apple phone", "iPhone 15"),
    "android": ("Android phone", "Galaxy phone", "Pixel phone"),
}
SYMPTOMS = {
    "app_crashing": ("one_app", "all_apps"),
    "battery_drain": ("after_update", "no_change"),
    "calls_dropping": ("at_home", "everywhere"),
    "wifi_wont_connect": ("one_network", "any_network"),
}
SYMPTOM_WORDS = {
    "app_crashing": ("apps keep closing", "my apps keep crashing"),  # names apps, but not one app vs all apps
    "battery_drain": ("the battery runs down very quickly", "the charge disappears too fast"),
    "calls_dropping": ("calls keep disconnecting", "my calls cut off unexpectedly"),
    "wifi_wont_connect": ("it will not join Wi-Fi", "Wi-Fi connections keep failing"),
}
TRIGGER_WORDS = {
    key: tuple(TRIGGER_TEXT[key]) for key in set(itertools.chain.from_iterable(SYMPTOMS.values()))
}
ATTEMPT_TEXT = {
    False: (
        "I have not restarted it or checked the relevant settings yet",
        "I have not tried a restart or the settings checks",
    ),
    True: (
        "I already restarted it and checked the relevant settings",
        "I completed both a restart and the settings checks",
    ),
}

SLOTS = ("device", "symptom", "trigger", "attempts", "backup", "calling_device")
REASONS = ("physical_damage", "safety", "billing", "human_request", "no_article", "cant_identify")
MAX_STEPS = 10
RESULTS_K = 8
BACKUP_DOC = "KB-905"
MOVE_CALL_DOC = "KB-906"
DESTRUCTIVE = {"app_crashing", "battery_drain"}
DISRUPTIVE = {"app_crashing", "battery_drain", "calls_dropping", "wifi_wont_connect"}

# Deliberate holes.  Every missing article has close platform/trigger/tier neighbors.
GAP_CELLS = {
    ("iphone", "app_crashing", "all_apps", True),
    ("android", "app_crashing", "one_app", False),
    ("iphone", "battery_drain", "no_change", False),
    ("android", "battery_drain", "after_update", True),
    ("iphone", "calls_dropping", "at_home", True),
    ("android", "wifi_wont_connect", "any_network", False),
    ("iphone", "wifi_wont_connect", "one_network", True),
    ("android", "calls_dropping", "everywhere", False),
}


@dataclass(frozen=True)
class Article:
    doc_id: str
    platform: str
    symptom: str
    trigger: str | None
    tried: bool | None
    title: str
    body: str
    destructive: bool = False
    disruptive: bool = False


def build_kb() -> list[Article]:
    articles: list[Article] = []
    for platform in PLATFORMS:
        for symptom, triggers in SYMPTOMS.items():
            for trigger in triggers:
                for tried in (False, True):
                    if (platform, symptom, trigger, tried) in GAP_CELLS:
                        continue
                    tier = "advanced" if tried else "basic"
                    destructive = tried and symptom in DESTRUCTIVE
                    disruptive = symptom in DISRUPTIVE
                    body = [
                        f"Applies only to {LABEL[platform]} on {CURRENT_OS[platform]}.",
                        f"Use when the issue is {SYMPTOM_TITLE[symptom].lower()} "
                        f"{TRIGGER_TITLE[trigger].lower()}.",
                        (
                            "Use this advanced article only after the customer confirms both a restart "
                            "and the relevant settings check were already completed."
                            if tried
                            else "Use this basic article only when the customer has not completed both "
                            "a restart and the relevant settings check."
                        ),
                    ]
                    if destructive:
                        body.append(
                            f"This procedure erases user data. Before selecting it, confirm a completed "
                            f"backup. If there is no backup, select {BACKUP_DOC} first."
                        )
                    if disruptive:
                        body.append(
                            f"This procedure can interrupt the affected phone's call or chat connection. "
                            f"Before selecting it, ask whether the customer is using the affected phone "
                            f"for this support conversation. If yes, select {MOVE_CALL_DOC} first."
                        )
                    body.append(
                        "Steps: "
                        + (
                            "reset the affected service, restart, and test again."
                            if tried
                            else "review the relevant setting, restart once, and test again."
                        )
                    )
                    articles.append(
                        Article(
                            doc_id=f"KB-{100 + len(articles)}",
                            platform=platform,
                            symptom=symptom,
                            trigger=trigger,
                            tried=tried,
                            title=(
                                f"{LABEL[platform]}: {SYMPTOM_TITLE[symptom]} "
                                f"{TRIGGER_TITLE[trigger]} ({tier})"
                            ),
                            body=" ".join(body),
                            destructive=destructive,
                            disruptive=disruptive,
                        )
                    )
    articles.extend(
        [
            Article(
                BACKUP_DOC,
                "any",
                "prerequisite",
                None,
                None,
                "Back up a phone before an erase",
                "Use when the matching troubleshooting article requires an erase but the customer has "
                "not completed a backup. Steps: back up to cloud storage or a computer and verify that "
                "the backup finished before returning to the erase procedure.",
            ),
            Article(
                MOVE_CALL_DOC,
                "any",
                "prerequisite",
                None,
                None,
                "Move support to another calling device",
                "Use when a matching procedure can interrupt service and the customer is using the "
                "affected phone for this support call or chat. Move the conversation to another device "
                "before returning to the disruptive procedure.",
            ),
            Article(
                "KB-930",
                "any",
                "other",
                None,
                None,
                "Set up voicemail",
                "Synthetic guide for setting up a voicemail greeting on a supported phone.",
            ),
            Article(
                "KB-931",
                "any",
                "other",
                None,
                None,
                "Transfer contacts",
                "Synthetic guide for copying contacts from an old phone to a new phone.",
            ),
        ]
    )
    return articles


KB = build_kb()
BY_ID = {article.doc_id: article for article in KB}
CELL_TO_DOC = {
    (article.platform, article.symptom, article.trigger, article.tried): article.doc_id
    for article in KB
    if article.platform in PLATFORMS
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _index_text(article: Article) -> str:
    if article.platform not in PLATFORMS:
        return f"{article.title} {article.body}"
    aliases = list(DEVICE_NAMES[article.platform])
    aliases += list(SYMPTOM_WORDS[article.symptom])
    aliases += list(SYMPTOM_SPECIFIC.get(article.symptom, ()))
    aliases += list(TRIGGER_WORDS[article.trigger])
    aliases += list(ATTEMPT_TEXT[bool(article.tried)])
    return " ".join([article.title, article.body, *aliases])


BM25 = BM25Okapi([_tokens(_index_text(article)) for article in KB])


def search(text: str, k: int = RESULTS_K) -> list[Article]:
    scores = BM25.get_scores(_tokens(text))
    order = sorted(range(len(KB)), key=lambda index: (-scores[index], KB[index].doc_id))
    return [KB[index] for index in order[:k]]


@dataclass
class Case:
    case_id: str
    seed: int
    kind: str
    platform: str | None = None
    symptom: str | None = None
    trigger: str | None = None
    tried: bool | None = None
    backed_up: bool = True
    calling_on_affected: bool = False
    escalation: str | None = None
    open_slots: tuple[str, ...] = ()
    indirect_slot: str | None = None
    unavailable_slot: str | None = None
    correction_from: str | None = None
    human_midchat: bool = False


@dataclass
class State:
    case: Case
    known: dict[str, object] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    ask_counts: dict[str, int] = field(default_factory=dict)
    unavailable: set[str] = field(default_factory=set)
    visible_escalation: str | None = None
    correction_delivered: bool = False


@dataclass(frozen=True)
class Situation:
    platform: str
    symptom: str
    trigger: str
    tried: bool

    @property
    def doc_id(self) -> str | None:
        return CELL_TO_DOC.get((self.platform, self.symptom, self.trigger, self.tried))


ALL_SITUATIONS = tuple(
    Situation(platform, symptom, trigger, tried)
    for platform in PLATFORMS
    for symptom, triggers in SYMPTOMS.items()
    for trigger in triggers
    for tried in (False, True)
)


def possible(state: State) -> list[Situation]:
    """Return situations consistent with confirmed conversation facts only."""
    known = state.known
    out = []
    for situation in ALL_SITUATIONS:
        if "device" in known and situation.platform != known["device"]:
            continue
        if "symptom" in known and situation.symptom != known["symptom"]:
            continue
        if "trigger" in known and situation.trigger != known["trigger"]:
            continue
        if "attempts" in known and situation.tried != known["attempts"]:
            continue
        out.append(situation)
    return out


def _slot_value(situation: Situation, slot: str) -> object:
    return {
        "device": situation.platform,
        "symptom": situation.symptom,
        "trigger": situation.trigger,
        "attempts": situation.tried,
    }[slot]


def allowed(state: State) -> dict:
    """Code oracle. It intentionally does not inspect hidden case truth."""
    if state.visible_escalation:
        return {"escalate": state.visible_escalation}

    situations = possible(state)
    if not situations:
        return {"escalate": "cant_identify"}
    targets = {situation.doc_id for situation in situations}
    if targets == {None}:
        return {"escalate": "no_article"}

    if len(targets) != 1 or None in targets:
        # A prerequisite every still-possible article starts with is right before the rest is known: if the customer
        # is on the affected phone and every candidate fix would drop the call, moving the call comes first either way.
        prereq = []
        arts = [BY_ID[t] for t in targets if t]
        if None not in targets and arts:
            if state.known.get("backup") is False and all(x.destructive for x in arts) and BACKUP_DOC in state.seen:
                prereq = [BACKUP_DOC]
            elif (state.known.get("calling_device") is True and all(x.disruptive for x in arts)
                  and MOVE_CALL_DOC in state.seen):
                prereq = [MOVE_CALL_DOC]
        asks = []
        for slot in ("device", "symptom", "trigger", "attempts"):
            if slot in state.known or slot in state.unavailable:
                continue
            groups: dict[object, set[str | None]] = collections.defaultdict(set)
            for situation in situations:
                groups[_slot_value(situation, slot)].add(situation.doc_id)
            if len(groups) > 1 and len({frozenset(value) for value in groups.values()}) > 1:
                asks.append(slot)
        if prereq:
            return {"ask": asks, "resolve": prereq} if asks else {"resolve": prereq}
        return {"ask": asks} if asks else {"escalate": "cant_identify"}

    doc_id = next(iter(targets))
    article = BY_ID[doc_id]
    if article.destructive and "backup" not in state.known:
        return {"ask": ["backup"]}
    if article.destructive and state.known["backup"] is False:
        return (
            {"resolve": [BACKUP_DOC]}
            if BACKUP_DOC in state.seen
            else {"retrieval_miss": [BACKUP_DOC]}
        )
    if article.disruptive and "calling_device" not in state.known:
        return {"ask": ["calling_device"]}
    if article.disruptive and state.known["calling_device"] is True:
        return (
            {"resolve": [MOVE_CALL_DOC]}
            if MOVE_CALL_DOC in state.seen
            else {"retrieval_miss": [MOVE_CALL_DOC]}
        )
    return {"resolve": [doc_id]} if doc_id in state.seen else {"retrieval_miss": [doc_id]}


def _phrase(rng: random.Random, values: tuple[str, ...] | list[str]) -> str:
    return values[rng.randrange(len(values))]


def opening(case: Case, rng: random.Random) -> str:
    if case.escalation == "human_request":
        return "Please connect me with a human agent now. I do not want to continue with automation."
    if case.escalation == "safety":
        return "Smoke and a burning smell are coming from the phone's charging port. I need safe next steps."
    if case.escalation == "physical_damage":
        return "The phone fell into water and stopped working afterward."
    if case.escalation == "billing":
        return "There is an unfamiliar charge on my phone bill and I need billing help."

    parts = ["Hello."]
    if case.correction_from:
        parts.append(f"The affected device is definitely my {LABEL[case.correction_from]}.")
    elif case.kind == "tentative_guess":
        guessed = "android" if case.platform == "iphone" else "iphone"
        parts.append(
            f"It might be {LABEL[guessed]}, but that is only a guess; I do not know which kind it is."
        )
    elif "device" in case.open_slots:
        parts.append(f"The affected device is my {_phrase(rng, DEVICE_NAMES[case.platform])}.")
    if "symptom" in case.open_slots:
        parts.append(_phrase(rng, SYMPTOM_WORDS[case.symptom]).capitalize() + ".")
    if "trigger" in case.open_slots:
        parts.append(_phrase(rng, TRIGGER_WORDS[case.trigger]).capitalize() + ".")
    if "attempts" in case.open_slots:
        parts.append(_phrase(rng, ATTEMPT_TEXT[bool(case.tried)]).capitalize() + ".")
    if "backup" in case.open_slots:
        parts.append(
            "A current backup has finished."
            if case.backed_up
            else "There is no completed backup of this phone."
        )
    if "calling_device" in case.open_slots:
        parts.append(
            "I am using the affected phone for this conversation."
            if case.calling_on_affected
            else "I am contacting you from a different device."
        )
    return " ".join(parts)


def _initial_known(case: Case) -> dict[str, object]:
    known: dict[str, object] = {}
    for slot in case.open_slots:
        if slot == "device":
            known[slot] = case.platform
        elif slot == "symptom":
            known[slot] = case.symptom
        elif slot == "trigger":
            known[slot] = case.trigger
        elif slot == "attempts":
            known[slot] = case.tried
        elif slot == "backup":
            known[slot] = case.backed_up
        elif slot == "calling_device":
            known[slot] = case.calling_on_affected
    if case.correction_from:
        known["device"] = case.correction_from
    return known


def answer(state: State, slot: str, rng: random.Random) -> str:
    case = state.case
    count = state.ask_counts.get(slot, 0) + 1
    state.ask_counts[slot] = count

    if case.indirect_slot == slot and count == 1:
        text = "It happens while I am using the phone, but I cannot describe the condition any better yet."
    elif case.unavailable_slot == slot:
        state.unavailable.add(slot)
        text = (
            f"I checked again, but I cannot identify the {slot}. "
            "I do not have any more specific information to give you."
        )
    elif slot == "device":
        state.known["device"] = case.platform
        text = f"I checked the About screen. It is an {LABEL[case.platform]}."
    elif slot == "symptom":
        state.known["symptom"] = case.symptom
        text = _phrase(rng, SYMPTOM_WORDS[case.symptom]).capitalize() + "."
    elif slot == "trigger":
        state.known["trigger"] = case.trigger
        text = _phrase(rng, TRIGGER_WORDS[case.trigger]).capitalize() + "."
    elif slot == "attempts":
        state.known["attempts"] = case.tried
        text = _phrase(rng, ATTEMPT_TEXT[bool(case.tried)]).capitalize() + "."
    elif slot == "backup":
        state.known["backup"] = case.backed_up
        text = (
            "Yes, I verified that a current backup completed."
            if case.backed_up
            else "No, there is no completed backup."
        )
    elif slot == "calling_device":
        state.known["calling_device"] = case.calling_on_affected
        text = (
            "Yes, this support conversation is on the affected phone."
            if case.calling_on_affected
            else "No, I am using another device to contact you."
        )
    else:
        text = "I am not sure what that question is asking."

    if case.correction_from and not state.correction_delivered:
        state.known["device"] = case.platform
        state.correction_delivered = True
        text += (
            f" I also need to correct something I said earlier: I checked the label, and the "
            f"affected device is {LABEL[case.platform]}, not {LABEL[case.correction_from]}."
        )
    if case.human_midchat and not state.visible_escalation:
        state.visible_escalation = "human_request"
        text += " Stop troubleshooting now and connect me with a human agent."
    return text


def customer_text(state: State) -> str:
    chunks = []
    for message in state.messages:
        if message["role"] != "user" or not message["content"].startswith("[CUSTOMER]"):
            continue
        chunks.append(message["content"].split("\n\n[SEARCH RESULTS]", 1)[0].removeprefix("[CUSTOMER] "))
    return " ".join(chunks)


def _format_results(articles: list[Article]) -> str:
    return "[SEARCH RESULTS]\n" + "\n".join(
        f"- {article.doc_id}: {article.title}\n  {article.body}" for article in articles
    )


def auto_search(state: State) -> None:
    results = search(customer_text(state))
    ids = {article.doc_id for article in results}
    # Deterministic KB relation expansion: prerequisite IDs named by a shown article are also shown.
    for prerequisite in (BACKUP_DOC, MOVE_CALL_DOC):
        if any(prerequisite in article.body for article in results) and prerequisite not in ids:
            results.append(BY_ID[prerequisite])
            ids.add(prerequisite)
    state.seen |= ids
    state.messages[-1]["content"] += "\n\n" + _format_results(results)


def new_state(case: Case) -> State:
    rng = random.Random(case.seed)
    state = State(
        case=case,
        known=_initial_known(case),
        messages=[{"role": "user", "content": "[CUSTOMER] " + opening(case, rng)}],
        visible_escalation=case.escalation,
    )
    if not case.escalation:
        auto_search(state)
    return state


SYSTEM = """You are a phone-carrier tech-support agent. Each turn, output exactly one JSON object and nothing else. The three actions are:

{"action": "ask", "slot": "device" | "symptom" | "trigger" | "attempts" | "backup" | "calling_device", "question": "<one short question>"}
{"action": "resolve", "doc_id": "<KB id from the search results>", "reply": "<short reply>"}
{"action": "escalate", "reason": "physical_damage" | "safety" | "billing" | "human_request" | "no_article" | "cant_identify", "reply": "<short reply>"}

Slots: device = which phone it is; symptom = what is wrong; trigger = when or where it happens, or how much is affected; attempts = what the customer already tried; backup = whether the data is backed up; calling_device = whether the customer is contacting you on the affected phone.

Reasons: no_article = nothing in the knowledge base fits the customer's situation; cant_identify = you can't find out a detail you need to pick the article.

Immediately escalate physical or liquid damage, a safety hazard, billing or payment issues, or any request for a human.
Each customer message includes fixed KB search results. Customers' phones run the current software named in the articles."""


def oracle_action(state: State) -> dict:
    legal = allowed(state)
    if "resolve" in legal:
        return {"action": "resolve", "doc_id": legal["resolve"][0], "reply": "."}
    if "escalate" in legal:
        return {"action": "escalate", "reason": legal["escalate"], "reply": "."}
    if "ask" in legal and legal["ask"]:
        return {"action": "ask", "slot": legal["ask"][0], "question": "?"}
    return {"action": "escalate", "reason": "cant_identify", "reply": "."}


def allowed_actions(state: State) -> list[dict]:
    legal = allowed(state)
    actions = [
        {"action": "resolve", "doc_id": doc_id, "reply": "."}
        for doc_id in legal.get("resolve", ())
    ]
    actions += [
        {"action": "ask", "slot": slot, "question": "?"} for slot in legal.get("ask", ())
    ]
    if "escalate" in legal:
        actions.append({"action": "escalate", "reason": legal["escalate"], "reply": "."})
    return actions


def step(state: State, action: dict, rng: random.Random) -> tuple[bool, str]:
    legal = allowed(state)
    state.messages.append({"role": "assistant", "content": json.dumps(action)})
    kind = action.get("action")
    text_field = "question" if kind == "ask" else "reply"
    if kind not in {"ask", "resolve", "escalate"}:
        return True, "invalid"
    if not isinstance(action.get(text_field), str) or not action[text_field].strip():
        return True, "invalid"

    if kind == "resolve":
        if action.get("doc_id") in legal.get("resolve", ()):
            return True, "resolved_correct"
        actual_doc = CELL_TO_DOC.get(
            (state.case.platform, state.case.symptom, state.case.trigger, state.case.tried)
        )
        if action.get("doc_id") == actual_doc:
            return True, "resolved_skipped_check"
        return True, "resolved_wrong"
    if kind == "escalate":
        if action.get("reason") not in REASONS:
            return True, "invalid"
        return (
            True,
            "escalated_correct"
            if legal.get("escalate") == action.get("reason")
            else "escalated_wrong",
        )
    if action.get("slot") not in SLOTS:
        return True, "invalid"

    reply = answer(state, action["slot"], rng)
    state.messages.append({"role": "user", "content": "[CUSTOMER] " + reply})
    if not state.visible_escalation:
        auto_search(state)
    return False, "ask"


def _available_cells(predicate=lambda article: True) -> list[Article]:
    return [
        article
        for article in KB
        if article.platform in PLATFORMS and predicate(article)
    ]


CASE_KINDS = (
    "tentative_guess",
    "indirect_answer",
    "later_correction",
    "basic_untried",
    "advanced_tried",
    "backup_gate",
    "calling_gate",
    "no_article",
    "no_article_after_clarify",
    "gap_with_gate",
    "prereq_bait",
    "cant_identify",
    "human_immediate",
    "human_midchat",
    "policy_escalation",
)
CONTROL_KINDS = {"backup_gate", "calling_gate", "human_immediate"}
# Over-weight what models get wrong in real calls: hand-offs (the most common real ending) and acting on a
# "do this first" gate before the deciding fact is known.
WEIGHT = {"no_article": 3, "no_article_after_clarify": 2, "gap_with_gate": 2, "prereq_bait": 2}
SCORED_KINDS = tuple(kind for kind in CASE_KINDS if kind not in CONTROL_KINDS)


def sample_cases(
    n_per_type: int = 10, seed: int = 41, *, include_controls: bool = False
) -> list[Case]:
    """Generate the locked 120-case set, optionally including 30 easy fairness controls."""
    rng = random.Random(seed)
    pools = {
        "basic": _available_cells(lambda article: article.tried is False),
        "advanced": _available_cells(lambda article: article.tried is True),
        "destructive": _available_cells(lambda article: article.destructive),
        "disruptive": _available_cells(lambda article: article.disruptive and not article.destructive),
    }
    counters = collections.Counter()
    out: list[Case] = []

    def choose(name: str) -> Article:
        pool = pools[name]
        article = pool[counters[name] % len(pool)]
        counters[name] += 1
        return article

    for kind in CASE_KINDS:
        kind_count = WEIGHT.get(kind, 1) * n_per_type
        for index in range(kind_count):
            case_seed = rng.randrange(10**9)
            case_id = f"hard-sol-{len(out) + 1:04d}"
            if kind == "human_immediate":
                out.append(Case(case_id, case_seed, kind, escalation="human_request"))
                continue
            if kind == "policy_escalation":
                reason = ("physical_damage", "safety", "billing")[index % 3]
                out.append(Case(case_id, case_seed, kind, escalation=reason))
                continue

            if kind in {"no_article", "no_article_after_clarify", "gap_with_gate"}:
                platform, symptom, trigger, tried = sorted(GAP_CELLS)[index % len(GAP_CELLS)]
                out.append(
                    Case(
                        case_id,
                        case_seed,
                        kind,
                        platform,
                        symptom,
                        trigger,
                        tried,
                        open_slots=(
                            ("device", "symptom", "trigger", "attempts")
                            if kind == "no_article"
                            else ("device", "symptom")
                        ),
                        # gap_with_gate: the customer is on the affected phone and has no backup, so the
                        # near-match articles' "do this first" steps look applicable before the gap is clear.
                        calling_on_affected=kind == "gap_with_gate",
                        backed_up=kind != "gap_with_gate",
                    )
                )
                continue

            pool_name = {
                "basic_untried": "basic",
                "advanced_tried": "advanced",
                "backup_gate": "destructive",
                "calling_gate": "disruptive",
                "prereq_bait": "destructive" if index % 2 == 0 else "disruptive",
            }.get(kind, "basic" if index % 2 == 0 else "advanced")
            article = choose(pool_name)
            open_slots: tuple[str, ...] = ("device", "symptom", "trigger")
            kwargs = {}
            backed_up = index % 2 == 0
            calling_on = index % 2 == 0

            if kind in {"basic_untried", "advanced_tried"}:
                # The attempts question is decisive between the two visible tiers.
                open_slots = ("device", "symptom", "trigger")
            elif kind == "tentative_guess":
                # Device and trigger are both unresolved, so either useful question may come first.
                open_slots = ("symptom", "attempts")
            elif kind == "indirect_answer":
                open_slots = ("device", "symptom", "attempts")
                kwargs["indirect_slot"] = "trigger"
            elif kind == "later_correction":
                open_slots = ("symptom", "trigger")
                kwargs["correction_from"] = (
                    "android" if article.platform == "iphone" else "iphone"
                )
            elif kind == "backup_gate":
                open_slots = ("device", "symptom", "trigger", "attempts", "calling_device")
                calling_on = False
            elif kind == "calling_gate":
                open_slots = ("device", "symptom", "trigger", "attempts")
            elif kind == "human_midchat":
                # Either useful clarification triggers the customer's request to stop.
                open_slots = ("symptom", "attempts")
                kwargs["human_midchat"] = True
            elif kind == "prereq_bait":
                # Gates apply (on the affected phone, no backup) but trigger and attempts still decide the article.
                open_slots = ("device", "symptom")
                calling_on = True
                backed_up = False
            elif kind == "cant_identify":
                open_slots = ("device", "symptom", "attempts")
                kwargs["indirect_slot"] = "trigger"
                kwargs["unavailable_slot"] = "trigger"

            out.append(
                Case(
                    case_id=case_id,
                    seed=case_seed,
                    kind=kind,
                    platform=article.platform,
                    symptom=article.symptom,
                    trigger=article.trigger,
                    tried=article.tried,
                    backed_up=backed_up,
                    calling_on_affected=calling_on,
                    open_slots=open_slots,
                    **kwargs,
                )
            )
    return out if include_controls else [case for case in out if case.kind in SCORED_KINDS]


def split(cases: list[Case], name: str) -> list[Case]:
    """dev = the first 40% of each case type, test = the rest. Same seed, so the split is stable."""
    if name == "all":
        return cases
    seen = collections.Counter()
    total = collections.Counter(case.kind for case in cases)
    out = []
    for case in cases:
        in_dev = seen[case.kind] < round(0.4 * total[case.kind])
        seen[case.kind] += 1
        if in_dev == (name == "dev"):
            out.append(case)
    return out


def explore(case: Case) -> dict:
    """Walk every action path accepted by the oracle."""
    result = {"paths": 0, "bad": 0, "miss": False, "max_depth": 0}

    def walk(state: State, rng: random.Random, depth: int) -> None:
        legal = allowed(state)
        if "retrieval_miss" in legal:
            result["miss"] = True
            return
        if depth >= MAX_STEPS:
            result["bad"] += 1
            return
        actions = allowed_actions(state)
        if not actions:
            result["bad"] += 1
            return
        for action in actions:
            child = copy.deepcopy(state)
            child_rng = random.Random()
            child_rng.setstate(rng.getstate())
            done, outcome = step(child, action, child_rng)
            if done:
                result["paths"] += 1
                result["bad"] += outcome not in ("resolved_correct", "escalated_correct")
                result["max_depth"] = max(result["max_depth"], depth + 1)
            else:
                walk(child, child_rng, depth + 1)

    walk(new_state(case), random.Random(case.seed + 1), 0)
    return result


async def run(case: Case, policy) -> dict:
    state = new_state(case)
    rng = random.Random(case.seed + 1)
    outcome, asks, wasted = "max_steps", 0, 0
    steps = []
    for _ in range(MAX_STEPS):
        legal = allowed(state)
        text = await policy([{"role": "system", "content": SYSTEM}, *state.messages])
        action = parse_action(text)
        if action is None:
            outcome = "invalid"
            break
        steps.append({"action": action, "allowed": legal})
        if action.get("action") == "ask":
            asks += 1
            wasted += action.get("slot") not in legal.get("ask", ())
        done, outcome = step(state, action, rng)
        if done:
            break
    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "outcome": outcome,
        "asks": asks,
        "wasted": wasted,
        "steps": steps,
        "transcript": state.messages,
    }


async def _gather(cases: list[Case], policy) -> list[dict]:
    if policy is None:
        logs = []
        for case in cases:
            state = new_state(case)
            rng = random.Random(case.seed + 1)
            outcome = "max_steps"
            for _ in range(MAX_STEPS):
                done, outcome = step(state, oracle_action(state), rng)
                if done:
                    break
            logs.append({"case_id": case.case_id, "kind": case.kind, "outcome": outcome})
        return logs

    semaphore = asyncio.Semaphore(8)

    async def one(case: Case) -> dict:
        async with semaphore:
            error = ""
            for attempt in range(3):
                try:
                    return await run(case, policy)
                except Exception as exc:  # API failures are reported but not scored.
                    error = repr(exc)[:300]
                    await asyncio.sleep(5 * (attempt + 1))
            return {
                "case_id": case.case_id,
                "kind": case.kind,
                "outcome": "api_error",
                "error": error,
            }

    return await asyncio.gather(*(one(case) for case in cases))


def fairness(cases: list[Case]) -> bool:
    counts = collections.Counter(case.kind for case in cases)
    assert set(counts) == set(CASE_KINDS)
    ordinary_counts = {count // WEIGHT.get(kind, 1) for kind, count in counts.items()}
    assert len(ordinary_counts) == 1, counts
    ordinary_count = next(iter(ordinary_counts))
    assert len(cases) == sum(WEIGHT.get(kind, 1) for kind in CASE_KINDS) * ordinary_count, (len(cases), counts)

    for article in KB:
        if article.platform not in PLATFORMS:
            continue
        assert LABEL[article.platform] in article.body
        assert CURRENT_OS[article.platform] in article.body
        assert SYMPTOM_TITLE[article.symptom].lower() in article.body.lower()
        assert TRIGGER_TITLE[article.trigger].lower() in article.body.lower()
        assert ("advanced article only" in article.body) == bool(article.tried)
        assert ("basic article only" in article.body) == (article.tried is False)
        if article.destructive:
            assert BACKUP_DOC in article.body and "backup" in article.body.lower()
        if article.disruptive:
            assert MOVE_CALL_DOC in article.body and "support conversation" in article.body

    logs = asyncio.run(_gather(cases, None))
    bad_oracle = [
        log
        for log in logs
        if log["outcome"] not in ("resolved_correct", "escalated_correct")
    ]
    walks = [explore(case) for case in cases]
    bad_paths = sum(walk["bad"] for walk in walks)
    misses = sum(bool(walk["miss"]) for walk in walks)

    # Mechanism-specific evidence checks.
    tentative = next(case for case in cases if case.kind == "tentative_guess")
    assert "device" not in new_state(tentative).known
    indirect = next(case for case in cases if case.kind == "indirect_answer")
    indirect_state = new_state(indirect)
    first_slot = indirect.indirect_slot
    step(
        indirect_state,
        {"action": "ask", "slot": first_slot, "question": "?"},
        random.Random(indirect.seed + 1),
    )
    assert first_slot not in indirect_state.known and first_slot in allowed(indirect_state)["ask"]
    correction = next(case for case in cases if case.kind == "later_correction")
    correction_state = new_state(correction)
    assert correction_state.known["device"] == correction.correction_from
    correction_action = oracle_action(correction_state)
    step(correction_state, correction_action, random.Random(correction.seed + 1))
    assert correction_state.known["device"] == correction.platform
    for case in (case for case in cases if case.kind == "no_article"):
        state = new_state(case)
        assert allowed(state) == {"escalate": "no_article"}
        assert CELL_TO_DOC.get((case.platform, case.symptom, case.trigger, case.tried)) is None
        assert any(article.symptom == case.symptom for article in search(customer_text(state)))
    for case in (case for case in cases if case.kind == "no_article_after_clarify"):
        state = new_state(case)
        rng = random.Random(case.seed + 1)
        assert "ask" in allowed(state) and "escalate" not in allowed(state)
        asks = 0
        while True:
            legal = allowed(state)
            if "escalate" in legal:
                assert legal == {"escalate": "no_article"}
                action = oracle_action(state)
                done, outcome = step(state, action, rng)
                assert done and outcome == "escalated_correct" and asks > 0
                break
            assert "ask" in legal and "no_article" not in legal
            done, outcome = step(state, oracle_action(state), rng)
            assert not done and outcome == "ask"
            asks += 1
            assert asks < MAX_STEPS
    for case in (case for case in cases if case.kind == "cant_identify"):
        state = new_state(case)
        rng = random.Random(case.seed + 1)
        slot = case.unavailable_slot
        assert allowed(state) == {"ask": [slot]}
        done, outcome = step(
            state,
            {"action": "ask", "slot": slot, "question": "?"},
            rng,
        )
        assert not done and outcome == "ask"
        assert slot not in state.known and slot not in state.unavailable
        assert allowed(state) == {"ask": [slot]}
        done, outcome = step(
            state,
            {"action": "ask", "slot": slot, "question": "?"},
            rng,
        )
        assert not done and outcome == "ask"
        assert slot not in state.known and slot in state.unavailable
        assert allowed(state) == {"escalate": "cant_identify"}

    max_depth = max(walk["max_depth"] for walk in walks)
    total_paths = sum(walk["paths"] for walk in walks)
    print(
        f"fairness: oracle solves {len(cases) - len(bad_oracle)}/{len(cases)}; "
        f"all allowed paths: {total_paths} ({bad_paths} bad, {misses} retrieval misses, "
        f"max {max_depth} steps); balanced kinds: {dict(counts)}; "
        "article applicability and conversation-state assertions: ok"
    )
    return not bad_oracle and not bad_paths and not misses


def freeze(
    cases: list[Case] | None = None,
    output_dir: str | Path = "data_hard_sol",
    *,
    seed: int = 41,
    n_per_type: int = 10,
) -> Path:
    cases = cases or sample_cases(n_per_type=n_per_type, seed=seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "kb.jsonl").write_text(
        "".join(json.dumps(asdict(article), sort_keys=True) + "\n" for article in KB)
    )
    (destination / "test_cases.jsonl").write_text(
        "".join(json.dumps(asdict(case), sort_keys=True) + "\n" for case in cases)
    )
    (destination / "system_prompt.txt").write_text(SYSTEM)
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "n_per_type": n_per_type,
                "n_cases": len(cases),
                "n_articles": len(KB),
                "case_kinds": dict(collections.Counter(case.kind for case in cases)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return destination


def report(logs: list[dict]) -> None:
    api_errors = sum(log["outcome"] == "api_error" for log in logs)
    scored = [log for log in logs if log["outcome"] != "api_error"]
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for log in scored:
        grouped[log["kind"]].append(log)

    def success(rows: list[dict]) -> float:
        return sum(
            row["outcome"] in ("resolved_correct", "escalated_correct") for row in rows
        ) / max(1, len(rows))

    print(f"\n{'case type':20s} {'n':>3s} {'success':>8s}  most common failure")
    for kind in sorted(grouped, key=lambda name: (success(grouped[name]), name)):
        failures = collections.Counter(
            row["outcome"]
            for row in grouped[kind]
            if row["outcome"] not in ("resolved_correct", "escalated_correct")
        )
        print(
            f"{kind:20s} {len(grouped[kind]):3d} {success(grouped[kind]):8.2f}  "
            f"{dict(failures.most_common(2))}"
        )
    average_wasted = sum(log.get("wasted", 0) for log in scored) / max(1, len(scored))
    print(
        f"{'ALL':20s} {len(scored):3d} {success(scored):8.2f}  "
        f"wasted questions/case {average_wasted:.2f}; API errors {api_errors}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="accounts/fireworks/models/gpt-oss-120b"
    )
    parser.add_argument("--reasoning-effort", default=None, help="For example: none or low")
    parser.add_argument("--n-per-type", type=int, default=10)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--freeze-dir", default=None, help="default: data/hard_v1/<split>")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--split", choices=["all", "dev", "test"], default="all",
                        help="dev = first 40%% of each case type (published); test = the rest (held out)")
    args = parser.parse_args()

    cases = split(sample_cases(args.n_per_type, args.seed), args.split)
    fairness_cases = sample_cases(args.n_per_type, args.seed, include_controls=True)
    print(
        f"KB: {len(KB)} articles; scored cases: {len(cases)} "
        f"{dict(collections.Counter(case.kind for case in cases))}"
    )
    if not fairness(fairness_cases):
        raise SystemExit("fairness check failed")
    destination = freeze(
        cases,
        args.freeze_dir or f"data/hard_v1/{args.split}",
        seed=args.seed,
        n_per_type=args.n_per_type,
    )
    print(f"frozen: {destination}")
    if args.oracle_only or args.freeze_only:
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=os.environ["FIREWORKS_API_KEY"],
        max_retries=8,
    )

    async def policy(messages: list[dict]) -> str:
        extra = (
            {"reasoning_effort": args.reasoning_effort}
            if args.reasoning_effort is not None
            else {}
        )
        response = await client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0,
            max_tokens=4096,
            **extra,
        )
        return response.choices[0].message.content or ""

    logs = asyncio.run(_gather(cases, policy))
    suffix = f"_re-{args.reasoning_effort}" if args.reasoning_effort is not None else ""
    run_dir = Path("runs") / f"hard_sol_{args.model.split('/')[-1]}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs.jsonl").write_text(
        "".join(json.dumps(log) + "\n" for log in logs)
    )
    report(logs)
    print(f"logs: {run_dir / 'logs.jsonl'}")


if __name__ == "__main__":
    main()
