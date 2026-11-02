"""Spike: a v2 world with a messier KB and a customer who often doesn't know the model.

Phones only, fixed retrieval (like v0). What is new:
- several articles can be right for one situation (duplicates; a general article next to a model one)
- no article at all for some situations (escalate no_article)
- model-specific articles, and most customers know the brand but not the model (escalate cant_identify)
- stale articles (written for old software), retired articles, in-store-only articles

The answer key asks: which situations are still possible, and is one shown article right for all of them?

Usage: python -m ase.spike_v2 [--model accounts/fireworks/models/kimi-k3] [--n-per-type 15]
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import itertools
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from rank_bm25 import BM25Okapi

from .env import FILLER, parse_action
from .world import ESCALATIONS, SYMPTOM_SPECIFIC, SYMPTOM_TITLE, SYMPTOM_VAGUE, SYMPTOMS, TRIGGER_TEXT, TRIGGER_TITLE

PLATS = ["iphone", "android"]
LABEL = {"iphone": "iPhone", "android": "Android phone"}
FAMILY_WORDS = {"iphone": ["iPhone"], "android": ["Android phone", "Android"]}
MODELS = {"iphone": {"new": ["iPhone 16", "iPhone 15"], "old": ["iPhone 14", "iPhone 13"]},
          "android": {"new": ["Galaxy S25", "Pixel 9"], "old": ["Galaxy S23", "Pixel 7"]}}
ALL_MODELS = {p: m["new"] + m["old"] for p, m in MODELS.items()}
CURRENT_OS = {"iphone": "iOS 26", "android": "Android 16"}
OLD_OS = {"iphone": "iOS 15 and 16", "android": "Android 12 and 13"}
SYMS = [s for s, v in SYMPTOMS.items() if set(PLATS) <= set(v["platforms"])]
# How many of the 38 (platform, symptom, trigger) cells get each layout. Fixed counts, so every layout lands on
# at least 3 different cells and no case type is really one topic in disguise.
LAYOUTS = {"general": 8, "general_plus_model": 6, "model_only": 6, "stale": 6, "duplicate": 3, "none": 5,
           "retired": 4}
# A symptom phrase must not hint at the trigger ("an app closes" reads as one app).
SYMPTOM_SPECIFIC = {**SYMPTOM_SPECIFIC, "app_crashing": ["apps keep crashing", "apps keep closing on me"]}
REASONS = ["physical_damage", "safety", "billing", "human_request", "no_article", "cant_identify"]
SLOTS = ["device", "model", "symptom", "trigger"]
MAX_STEPS = 10
K = 5


@dataclass
class Art:
    doc_id: str
    plat: str
    sym: str
    trig: str | None
    models: tuple | None  # None = every model of the platform
    kind: str  # ok | stale | instore | retired
    title: str
    body: str


def build_kb(seed: int = 3):
    rng = random.Random(seed)
    arts, layout = [], {}

    def add(plat, sym, trig, models=None, kind="ok", note=""):
        base = f"{LABEL[plat]}: {SYMPTOM_TITLE[sym]}" + (f" {TRIGGER_TITLE[trig]}" if trig else "")
        who = ", ".join(models) if models else f"all {LABEL[plat]} models"
        os_ = OLD_OS[plat] if kind == "stale" else CURRENT_OS[plat]
        # Stale and retired copies keep the normal title, as in real KBs: only the body says what is wrong.
        title = base + (f" ({', '.join(models)})" if models else "") + (" (in-store repair)" if kind == "instore" else "")
        aud = "In-store technicians only, with store tools. Not for phone or chat support." if kind == "instore" \
            else "For phone and chat support."
        body = f"Applies to {who} on {os_}. {aud} {note}Steps: open Settings, follow the {sym.replace('_', ' ')} " \
               f"checks{' for ' + TRIGGER_TITLE[trig] if trig else ''}, restart and retest."
        a = Art(f"KB-{len(arts) + 100}", plat, sym, trig, tuple(models) if models else None, kind, title, body)
        arts.append(a)
        return a

    cells = [(p, s, t) for s in SYMS for p, t in itertools.product(PLATS, SYMPTOMS[s]["triggers"] or [None])]
    rng.shuffle(cells)
    plain = next(c for c in cells if c[2] is None)  # one duplicate goes on a symptom with no trigger
    cells.remove(plain)
    lays = [lay for lay, n in LAYOUTS.items() for _ in range(n)]
    lays.remove("duplicate")
    assert len(lays) == len(cells), (len(lays), len(cells))
    for (plat, sym, trig), lay in [(plain, "duplicate")] + list(zip(cells, lays)):
            layout[(plat, sym, trig)] = lay
            if lay in ("general", "stale"):
                add(plat, sym, trig)
                if lay == "stale":
                    add(plat, sym, trig, kind="stale")
            elif lay == "general_plus_model":
                add(plat, sym, trig)
                add(plat, sym, trig, models=MODELS[plat]["new"])
            elif lay == "model_only":
                add(plat, sym, trig, models=MODELS[plat]["new"])
                add(plat, sym, trig, models=MODELS[plat]["old"])
            elif lay == "duplicate":
                a = add(plat, sym, trig)
                arts.append(Art(f"KB-{len(arts) + 100}", plat, sym, trig, None, "ok", a.title, a.body))
            elif lay == "retired":
                new = add(plat, sym, trig)
                add(plat, sym, trig, kind="retired", note=f"RETIRED: replaced by {new.doc_id}, do not use. ")
            if lay != "none" and rng.random() < 0.25:
                add(plat, sym, trig, kind="instore")
    for i, t in enumerate(["Set up voicemail", "Transfer contacts to a new phone", "Set up email on your device",
                           "Turn on two-factor authentication", "Back up your photos"]):
        arts.append(Art(f"KB-{900 + i}", "any", "other", None, None, "ok", t, f"How to {t.lower()} on any phone."))
    return arts, layout


KB, LAYOUT = build_kb()
BY_ID = {a.doc_id: a for a in KB}


def right(a: Art, s: tuple) -> bool:
    plat, model, sym, trig = s
    return (a.kind == "ok" and a.plat == plat and a.sym == sym and a.trig == trig
            and (a.models is None or model in a.models))


def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def _index_text(a: Art) -> str:  # customer phrasing aliases so search reliably finds the cell (as in v0)
    extra = []
    if a.sym in SYMPTOM_SPECIFIC:
        extra += SYMPTOM_SPECIFIC[a.sym]
    if a.trig:
        extra += TRIGGER_TEXT[a.trig]
    if a.plat in FAMILY_WORDS:
        extra += FAMILY_WORDS[a.plat] + ALL_MODELS[a.plat]
    return " ".join([a.title, a.body] + extra)


BM25 = BM25Okapi([_tok(_index_text(a)) for a in KB])


def search(text: str) -> list[Art]:
    sc = BM25.get_scores(_tok(text))
    return [KB[i] for i in sorted(range(len(KB)), key=lambda i: -sc[i])[:K]]


@dataclass
class Case:
    case_id: str
    seed: int
    plat: str | None = None
    model: str | None = None
    sym: str | None = None
    trig: str | None = None
    knows_model: bool = False
    escalation: str | None = None
    open_device: str | None = None  # "model" | "family" | None
    open_symptom: str = "specific"
    open_trigger: bool = False
    kind: str = ""


@dataclass
class State:
    case: Case
    known: dict = field(default_factory=dict)  # plat, model, sym, trig (True = revealed), model_failed
    messages: list = field(default_factory=list)
    seen: set = field(default_factory=set)


def possible(st: State) -> list[tuple]:
    c, k = st.case, st.known
    out = []
    for plat in ([c.plat] if k.get("plat") else PLATS):
        for model in ([c.model] if k.get("model") else ALL_MODELS[plat]):
            syms = [c.sym] if k.get("sym") else [s for s in SYMS if SYMPTOMS[s]["group"] == SYMPTOMS[c.sym]["group"]]
            for sym in syms:
                trigs = SYMPTOMS[sym]["triggers"] or [None]
                if k.get("trig"):  # a stated trigger also rules out symptoms that can't have it
                    trigs = [c.trig] if c.trig in trigs else []
                for trig in trigs:
                    out.append((plat, model, sym, trig))
    return out


def allowed(st: State) -> dict:
    c = st.case
    if c.escalation:
        return {"escalate": c.escalation}
    S = possible(st)
    R = {s: frozenset(a.doc_id for a in KB if right(a, s)) for s in S}
    if not any(R.values()):
        return {"escalate": "no_article"}
    common = frozenset.intersection(*R.values())
    if common:
        shown = common & st.seen
        return {"resolve": sorted(shown)} if shown else {"retrieval_miss": sorted(common)}
    # A slot is worth asking if two still-possible situations that differ only in that slot can't share an article
    # (one has none, or their right articles don't overlap). A general article that serves both makes the slot moot.
    same = {"device": (2, 3), "model": (2, 3), "symptom": (0, 1), "trigger": (0, 1, 2)}
    col = {"device": 0, "model": 1, "symptom": 2, "trigger": 3}
    known = {"device": "plat", "model": "model", "symptom": "sym", "trigger": "trig"}
    split = []
    for slot in SLOTS:
        if st.known.get(known[slot]) or st.known.get(slot + "_failed"):
            continue  # never reward asking what the customer already said or already couldn't answer
        if any(s1[col[slot]] != s2[col[slot]] and all(s1[i] == s2[i] for i in same[slot])
               and not (R[s1] & R[s2]) and (R[s1] or R[s2]) for s1, s2 in itertools.combinations(S, 2)):
            split.append(slot)
    return {"ask": split} if split else {"escalate": "cant_identify"}


def an(name: str) -> str:
    return ("an " if name[0] in "AEIOUi" else "a ") + name


def opening(c: Case, rng: random.Random) -> str:
    if c.escalation in ESCALATIONS:
        return rng.choice(ESCALATIONS[c.escalation]).format(dev=rng.choice(["phone", "iPhone", "Galaxy"])) + ". " + rng.choice(FILLER)
    dev = {"model": c.model, "family": rng.choice(FAMILY_WORDS[c.plat]) if c.plat else None}.get(c.open_device) or "phone"
    sym = rng.choice(SYMPTOM_SPECIFIC[c.sym]) if c.open_symptom == "specific" else rng.choice(SYMPTOM_VAGUE[SYMPTOMS[c.sym]["group"]])
    parts = [f"Hi, my {dev}: {sym}."]
    if c.open_trigger:
        parts.append(rng.choice(TRIGGER_TEXT[c.trig]).capitalize() + ".")
    parts.append(rng.choice(FILLER))
    return " ".join(parts)


def answer(st: State, slot: str, rng: random.Random) -> str:
    c, k = st.case, st.known
    if c.escalation:
        return "I told you what happened. Can you help or not?"
    if slot == "device":
        if k.get("plat"):
            return f"I already told you, it's {an(LABEL[c.plat])}."
        k["plat"] = True
        return f"It's {an(rng.choice(FAMILY_WORDS[c.plat]))}."
    if slot == "model":
        if k.get("model"):
            return f"I already told you, it's {an(c.model)}."
        if c.knows_model:
            k["plat"] = k["model"] = True
            return f"It's {an(c.model)}."
        k["model_failed"] = True
        if not k.get("plat"):
            k["plat"] = True
            return f"I'm not sure which model. It's {an(rng.choice(FAMILY_WORDS[c.plat]))}, that's all I know."
        return "I'm not sure which model it is, sorry. It's just my phone."
    if slot == "symptom":
        if k.get("sym"):
            return "I already told you. " + rng.choice(SYMPTOM_SPECIFIC[c.sym]).capitalize() + "."
        k["sym"] = True
        return rng.choice(SYMPTOM_SPECIFIC[c.sym]).capitalize() + "."
    if slot == "trigger" and c.trig:
        if k.get("trig"):
            return "I already told you. " + rng.choice(TRIGGER_TEXT[c.trig]).capitalize() + "."
        k["trig"] = True
        return rng.choice(TRIGGER_TEXT[c.trig]).capitalize() + "."
    if slot == "trigger":
        k["trigger_failed"] = True
    return "I'm not sure, it just happens."


def customer_text(st: State) -> str:
    return " ".join(m["content"].split("\n\n[SEARCH RESULTS]")[0].removeprefix("[CUSTOMER] ")
                    for m in st.messages if m["role"] == "user")


def auto_search(st: State):
    res = search(customer_text(st))
    st.seen |= {a.doc_id for a in res}
    st.messages[-1]["content"] += "\n\n[SEARCH RESULTS]\n" + "\n".join(f"- {a.doc_id}: {a.title}\n  {a.body}" for a in res)


def new_state(c: Case) -> State:
    rng = random.Random(c.seed)
    known = {"plat": c.open_device in ("model", "family"), "model": c.open_device == "model",
             "sym": c.open_symptom == "specific", "trig": c.open_trigger}
    st = State(c, known, [{"role": "user", "content": "[CUSTOMER] " + opening(c, rng)}])
    if not c.escalation:
        auto_search(st)
    return st


SYSTEM = """You are a tech-support agent for a phone carrier, chatting with a customer. Each turn, output exactly one JSON object and nothing else. The three possible actions:

{"action": "ask", "slot": "device" | "model" | "symptom" | "trigger", "question": "<one short question>"}
{"action": "resolve", "doc_id": "<KB id from the search results>", "reply": "<short reply>"}
{"action": "escalate", "reason": "physical_damage" | "safety" | "billing" | "human_request" | "no_article" | "cant_identify", "reply": "<short reply>"}

Slots: device = iPhone or Android; model = the exact model (for example iPhone 15); symptom = what is wrong; trigger = the condition or scope: when or where it happens, or how much is affected (one app or all apps, picture and group texts or all texts, one Wi-Fi network or any network).

Rules:
- Escalate, never troubleshoot: physical or liquid damage, safety hazards (swelling battery, smoke, burning smell), billing, plan or payment questions, or a customer asking for a human.
- Each customer message comes with the top KB search results for everything the customer has said so far. Once the customer has said enough to pin down the situation, an article that fits (if one exists) is in the results.
- Articles say which models and software they apply to, and who they are for. Customers' phones run current software (iOS 26, Android 16).
- Only resolve with an article that fits the customer's exact situation: device, symptom and trigger; the customer's model if the article is model-specific; current software; written for phone or chat support; not retired. If an article applies to all models, you don't need the model.
- If no article fits, escalate with reason no_article. If you can't tell which article fits because the customer can't give you the detail you need, escalate with reason cant_identify.
- If you cannot tell which article applies, ask the one question that decides it. Questions cost the customer time, so do not ask what you already know or what wouldn't change the answer."""


def step(st: State, a: dict, rng: random.Random):
    al = allowed(st)
    st.messages.append({"role": "assistant", "content": json.dumps(a)})
    kind = a.get("action")
    text_ok = isinstance(a.get("question" if kind == "ask" else "reply"), str) and \
        a.get("question" if kind == "ask" else "reply", "").strip()
    if not text_ok or (kind == "escalate" and a.get("reason") not in REASONS):
        return True, "invalid"  # a move without its question or reply, or with an unknown reason
    if kind == "resolve":
        if a.get("doc_id") in al.get("resolve", []):
            return True, "resolved_correct"
        art = BY_ID.get(a.get("doc_id"))
        c = st.case
        lucky = art and not c.escalation and right(art, (c.plat, c.model, c.sym, c.trig))
        return True, "resolved_skipped_check" if lucky else "resolved_wrong"
    if kind == "escalate":
        return True, "escalated_correct" if al.get("escalate") == a.get("reason") else "escalated_wrong"
    if kind == "ask" and a.get("slot") in SLOTS:
        st.messages.append({"role": "user", "content": "[CUSTOMER] " + answer(st, a["slot"], rng)})
        if not st.case.escalation:
            auto_search(st)
        return False, "ask"
    return True, "invalid"


def expected_kind(c: Case) -> str:
    if c.escalation:
        return "escalation"
    lay = LAYOUT[(c.plat, c.sym, c.trig)]
    if lay == "model_only":
        return "model_needed" if c.knows_model else "cant_identify"
    return {"none": "no_article", "general_plus_model": "general_vs_model"}.get(lay, lay)


def sample_cases(n_per_type: int, seed: int) -> list[Case]:
    rng, out, count, per_cell, i = random.Random(seed), [], collections.Counter(), collections.Counter(), 0
    types = ["escalation", "no_article", "cant_identify", "model_needed", "general_vs_model", "stale",
             "duplicate", "retired", "general"]
    while min(count[t] for t in types) < n_per_type and i < 200000:
        i += 1
        c = Case(f"v2-{i:05d}", rng.randrange(10**9))
        if rng.random() < 0.15:
            c.escalation = rng.choice(list(ESCALATIONS))
        else:
            c.sym = rng.choice(SYMS)
            c.plat = rng.choice(PLATS)
            c.model = rng.choice(ALL_MODELS[c.plat])
            c.trig = rng.choice(SYMPTOMS[c.sym]["triggers"]) if SYMPTOMS[c.sym]["triggers"] else None
            c.knows_model = rng.random() < 0.3
            c.open_device = rng.choices(["model", "family", None], [0.05, 0.6, 0.35])[0]
            if c.open_device == "model":
                c.knows_model = True
            c.open_symptom = rng.choices(["specific", "vague"], [0.65, 0.35])[0]
            c.open_trigger = bool(c.trig) and rng.random() < 0.45
        c.kind = expected_kind(c)
        cell = c.escalation or (c.plat, c.sym, c.trig)
        if count[c.kind] >= n_per_type or per_cell[(c.kind, cell)] >= 5:
            continue  # at most 5 cases per cell, so a case type can't be one topic
        if not c.escalation and explore(c)["miss"]:
            continue  # drop cases where search fails to show a needed article on some allowed path (as in v0)
        count[c.kind] += 1
        per_cell[(c.kind, cell)] += 1
        out.append(c)
    return out


def oracle_action(st: State) -> dict:
    al = allowed(st)
    if "resolve" in al:
        return {"action": "resolve", "doc_id": al["resolve"][0], "reply": "."}
    if "escalate" in al:
        return {"action": "escalate", "reason": al["escalate"], "reply": "."}
    if "ask" in al:
        return {"action": "ask", "slot": al["ask"][0], "question": "?"}
    return {"action": "ask", "slot": "symptom", "question": "?"}


def allowed_actions(st: State) -> list[dict]:
    al = allowed(st)
    acts = [{"action": "resolve", "doc_id": d, "reply": "."} for d in al.get("resolve", [])]
    acts += [{"action": "escalate", "reason": al["escalate"], "reply": "."}] if "escalate" in al else []
    acts += [{"action": "ask", "slot": s, "question": "?"} for s in al.get("ask", [])]
    return acts


def explore(c: Case) -> dict:
    """Walk every path the answer key allows, not only the oracle's first choice."""
    out = {"paths": 0, "bad": 0, "miss": False, "max_depth": 0}

    def walk(st, rng, depth):
        if "retrieval_miss" in allowed(st):
            out["miss"] = True
            return
        if depth >= MAX_STEPS:
            out["bad"] += 1
            return
        for a in allowed_actions(st):
            st2 = State(st.case, dict(st.known), [dict(m) for m in st.messages], set(st.seen))
            rng2 = random.Random()
            rng2.setstate(rng.getstate())
            done, o = step(st2, a, rng2)
            if done:
                out["paths"] += 1
                out["bad"] += o not in ("resolved_correct", "escalated_correct")
                out["max_depth"] = max(out["max_depth"], depth + 1)
            else:
                walk(st2, rng2, depth + 1)

    walk(new_state(c), random.Random(c.seed + 1), 0)
    return out


async def run(c: Case, policy) -> dict:
    st, rng = new_state(c), random.Random(c.seed + 1)
    outcome, asks, wasted, steps = "max_steps", 0, 0, []
    for _ in range(MAX_STEPS):
        al = allowed(st)
        msgs = [{"role": "system", "content": SYSTEM}] + st.messages
        text = await policy(msgs)
        a = parse_action(text)
        if a is None:
            outcome = "invalid"
            break
        steps.append({"action": a, "allowed": al})
        if a.get("action") == "ask":
            asks += 1
            wasted += a.get("slot") not in al.get("ask", [])
        done, outcome = step(st, a, rng)
        if done:
            break
    return {"case_id": c.case_id, "kind": c.kind, "outcome": outcome, "asks": asks, "wasted": wasted,
            "steps": steps, "transcript": st.messages}


def fairness(cases: list[Case]):
    logs = asyncio.run(_gather(cases, None))
    bad = [x for x in logs if x["outcome"] not in ("resolved_correct", "escalated_correct")]
    for a in KB:  # every rule the answer key uses is written where the agent can read it
        if a.kind == "stale":
            assert OLD_OS[a.plat] in a.body
        if a.kind == "instore":
            assert "Not for phone or chat support" in a.body
        if a.kind == "retired":
            assert "RETIRED" in a.body
        if a.models:
            assert all(m in a.body for m in a.models)
    walks = [explore(c) for c in cases]  # every path the key allows ends correct, within the step budget
    bad_paths = sum(w["bad"] for w in walks)
    misses = sum(w["miss"] for w in walks)
    cells = collections.defaultdict(set)
    for c in cases:
        cells[c.kind].add(c.escalation or (c.plat, c.sym, c.trig))
    thin = {k: len(v) for k, v in cells.items() if len(v) < 3}
    print(f"fairness: oracle solves {len(cases) - len(bad)}/{len(cases)}; "
          f"all allowed paths: {sum(w['paths'] for w in walks)} ({bad_paths} bad, {misses} cases with a search miss, "
          f"max {max(w['max_depth'] for w in walks)} steps); case types on <3 cells: {thin or 'none'}; "
          f"article rules visible in text: ok")
    return not bad and not bad_paths and not misses and not thin


async def _gather(cases, policy):
    if policy is None:  # oracle self-check
        out = []
        for c in cases:
            st, rng = new_state(c), random.Random(c.seed + 1)
            o = "max_steps"
            for _ in range(MAX_STEPS):
                done, o = step(st, oracle_action(st), rng)
                if done:
                    break
            out.append({"outcome": o})
        return out
    sem = asyncio.Semaphore(8)

    async def one(c):
        async with sem:
            for attempt in range(3):
                try:
                    return await run(c, policy)
                except Exception as ex:  # API error: retry, then mark (not scored)
                    err = repr(ex)[:200]
                    await asyncio.sleep(5 * (attempt + 1))
            return {"case_id": c.case_id, "kind": c.kind, "outcome": "api_error", "error": err}

    return await asyncio.gather(*[one(c) for c in cases])


def report(logs):
    logs = [x for x in logs if x["outcome"] != "api_error"]
    by = collections.defaultdict(list)
    for x in logs:
        by[x["kind"]].append(x)
    ok = lambda xs: sum(x["outcome"] in ("resolved_correct", "escalated_correct") for x in xs) / max(1, len(xs))
    print(f"\n{'case type':18s} {'n':>3s} {'success':>8s}  most common failure")
    for k in sorted(by, key=lambda k: ok(by[k])):
        fails = collections.Counter(x["outcome"] for x in by[k] if x["outcome"] not in ("resolved_correct", "escalated_correct"))
        print(f"{k:18s} {len(by[k]):3d} {ok(by[k]):8.2f}  {dict(fails.most_common(2))}")
    print(f"{'ALL':18s} {len(logs):3d} {ok(logs):8.2f}   wasted questions/case {sum(x['wasted'] for x in logs) / len(logs):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accounts/fireworks/models/kimi-k3")
    ap.add_argument("--n-per-type", type=int, default=15)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--oracle-only", action="store_true")
    ap.add_argument("--reasoning-effort", default=None, help="e.g. none / low; voice needs no or minimal reasoning")
    args = ap.parse_args()
    cases = sample_cases(args.n_per_type, args.seed)
    lay = collections.Counter(LAYOUT.values())
    print(f"KB: {len(KB)} articles, cell layouts {dict(lay)}; cases: {len(cases)} {dict(collections.Counter(c.kind for c in cases))}")
    if not fairness(cases):
        return
    freeze = Path("data_v2")  # the locked set: same cases, KB and prompt for harness work and RL
    freeze.mkdir(exist_ok=True)
    (freeze / "kb.jsonl").write_text("".join(json.dumps(a.__dict__) + "\n" for a in KB))
    (freeze / "test_cases.jsonl").write_text("".join(json.dumps(c.__dict__) + "\n" for c in cases))
    (freeze / "system_prompt.txt").write_text(SYSTEM)
    if args.oracle_only:
        return
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"], max_retries=8)

    async def policy(msgs):
        extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}
        r = await client.chat.completions.create(model=args.model, messages=msgs, temperature=0, max_tokens=4096, **extra)
        return r.choices[0].message.content or ""

    logs = asyncio.run(_gather(cases, policy))
    out = Path("runs") / f"spike_v2_{args.model.split('/')[-1]}{'_re-' + args.reasoning_effort if args.reasoning_effort else ''}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "logs.jsonl", "w") as f:
        for x in logs:
            f.write(json.dumps(x) + "\n")
    report(logs)
    print(f"logs: {out}/logs.jsonl  (api errors: {sum(x['outcome'] == 'api_error' for x in logs)})")


if __name__ == "__main__":
    main()
