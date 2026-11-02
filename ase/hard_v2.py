"""hard_v2: real-call shape. The agent picks the article and then runs it with a scripted customer.

What makes real calls hard, and how each is built here (all text synthetic; grading is exact code):
- Articles at real length (~2.5-3k chars) in prose: bold section headings, a "Still not working?" section that
  bundles several fixes, per-brand lines, a branch to another article, notes about older models.
- 8 search results with near-duplicates: an older copy written for old software, an in-store copy, the other
  platform's version, how-to pages.
- Long calls: fixes fail, the customer can't find a menu ("I see Mobile Service, not Cellular"), does the wrong thing,
  mentions they already did a step, notices "SOS" halfway (wrong article: switch), asks for a person.
- Know when to stop: check the fix before closing (a step can look like it worked); hand off when the steps run out.

Usage:
  python -m ase.hard_v2 --oracle-only
  python -m ase.hard_v2 --case sos_switch --model accounts/fireworks/models/deepseek-v4p1-flash --reasoning-effort none
  python -m ase.hard_v2 --quick 2 --model ... --reasoning-effort none
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

MAX_STEPS = 24
K = 8
REASONS = ["physical_damage", "safety", "billing", "human_request", "fix_failed"]

# Each step: heading as it appears in the article, then per-platform prose. {alt} steps have a second menu label.
STEPS = {
    "data": {"head": "Check that mobile data is turned on",
             "iphone": "Go to **Settings** > **Cellular** and make sure **Cellular Data** is turned on. On some phones this "
                       "menu is called **Mobile Service** instead of **Cellular**; the switch is in the same place. If it's "
                       "already on, move on to the next section.",
             "samsung": "Go to **Settings** > **Connections** > **Data usage** and turn on **Mobile data**.",
             "pixel": "Go to **Settings** > **Network & internet** > **SIMs** and turn on **Mobile data**.",
             "android_tail": "If it's already on, move on to the next section.",
             "alt": True},
    "sim": {"head": "Make sure the line is turned on",
            "iphone": "Go to **Settings** > **Cellular**, tap the line under SIMs, and make sure **Turn On This Line** is on. "
                      "On some phones the menu is called **Mobile Service**.",
            "samsung": "Go to **Settings** > **Connections** > **SIM manager** and turn the SIM on.",
            "pixel": "Go to **Settings** > **Network & internet** > **SIMs**, choose the line, and turn on **Use SIM**.",
            "alt": True},
    "airplane": {"head": "Turn Airplane mode on and off",
                 "iphone": "Open **Control Center** by swiping down from the top-right corner, tap the airplane icon, wait "
                           "10 seconds, then tap it again. This makes the phone reconnect to the network.",
                 "samsung": "Swipe down twice from the top of the screen, tap **Airplane mode**, wait 10 seconds, then tap it again.",
                 "pixel": "Swipe down from the top of the screen, tap **Airplane mode**, wait 10 seconds, then tap it again."},
    "restart": {"head": "Restart the phone",
                "iphone": "Press and hold the side button and either volume button until the slider appears, drag it, then "
                          "turn the phone back on. Restarting clears minor glitches.",
                "samsung": "Press and hold the **Side key**, tap **Power off** > **Restart**.",
                "pixel": "Press and hold the **Power** button, then tap **Restart**."},
    "reset_network": {"head": "Reset network settings",
                      "iphone": "Go to **Settings** > **General** > **Transfer or Reset iPhone** > **Reset** > **Reset Network "
                                "Settings**. This clears saved Wi-Fi networks and passwords, Bluetooth pairings and VPN "
                                "settings, so get the customer's OK before they start.",
                      "samsung": "Go to **Settings** > **General management** > **Reset** > **Reset network settings**.",
                      "pixel": "Go to **Settings** > **System** > **Reset options** > **Reset Wi-Fi, mobile & Bluetooth**.",
                      "android_tail": "This clears saved Wi-Fi networks and passwords and Bluetooth pairings, so get the "
                                      "customer's OK before they start.",
                      "destructive": True},
    "messages": {"head": "Check the messaging settings",
                 "iphone": "Go to **Settings** > **Apps** > **Messages** and make sure **MMS Messaging** and **Send as Text "
                           "Message** are on.",
                 "samsung": "Open **Messages**, tap the three dots > **Settings** > **More settings** > **Multimedia messages**, "
                            "and turn on **Auto retrieve**.",
                 "pixel": "Open **Messages**, tap your profile picture > **Messages settings** > **Advanced**, and turn on "
                          "**Auto-download MMS**."},
    "update": {"head": "Update the software",
               "iphone": "Go to **Settings** > **General** > **Software Update** and install anything available.",
               "samsung": "Go to **Settings** > **Software update** > **Download and install**.",
               "pixel": "Go to **Settings** > **System** > **Software updates**."},
    "forget_wifi": {"head": "Forget the network and join it again",
                    "iphone": "Go to **Settings** > **Wi-Fi**, tap the info icon next to the network, tap **Forget This "
                              "Network**, then join it again with the password.",
                    "samsung": "Go to **Settings** > **Connections** > **Wi-Fi**, tap the gear next to the network, tap "
                               "**Forget**, then join it again.",
                    "pixel": "Go to **Settings** > **Network & internet** > **Internet**, tap the network, tap **Forget**, "
                             "then join it again."},
    "wifi_calling": {"head": "Turn on Wi-Fi calling",
                     "iphone": "Go to **Settings** > **Cellular** > **Wi-Fi Calling** and turn on **Wi-Fi Calling on This "
                               "iPhone**. It can take a few minutes to activate. On some phones the menu is called **Mobile "
                               "Service**.",
                     "samsung": "Open the **Phone** app, tap the three dots > **Settings**, and turn on **Wi-Fi calling**.",
                     "pixel": "Open the **Phone** app, tap the three dots > **Settings** > **Calls** > **Wi-Fi calling**, and "
                              "turn it on.",
                     "alt": True},
    "hotspot": {"head": "Turn the hotspot off and on",
                "iphone": "Go to **Settings** > **Personal Hotspot**, turn off **Allow Others to Join**, wait 10 seconds, "
                          "then turn it back on.",
                "samsung": "Go to **Settings** > **Connections** > **Mobile Hotspot and Tethering**, turn off **Mobile "
                           "Hotspot**, wait 10 seconds, then turn it back on.",
                "pixel": "Go to **Settings** > **Network & internet** > **Hotspot & tethering** > **Wi-Fi hotspot**, turn it "
                         "off, wait 10 seconds, then turn it back on."},
}
ALT_LABEL = ("Cellular", "Mobile Service")

SYMS = {
    "no_cell_data": {"title": "Mobile Data", "ladder": ["data", "airplane", "restart", "reset_network"],
                     "intro": "When mobile data stops working, apps and websites only load on Wi-Fi. Most of the time a "
                              "setting or a stuck connection is to blame, and a few quick steps get it working again.",
                     "branch": "If the phone shows **SOS** or **No Service** at the top of the screen, this isn't a mobile "
                               "data problem: use **Troubleshoot No Service** instead.",
                     "check": "Have the customer turn off Wi-Fi and open a website. An LTE or 5G icon coming back doesn't "
                              "mean data works.",
                     "say": ["mobile data isn't working", "nothing loads unless I'm on Wi-Fi"],
                     "ok": "Yes, websites load now with Wi-Fi off.", "bad": "Still nothing loads off Wi-Fi.",
                     "looks": "Huh, I think the little LTE thing came back?"},
    "no_service": {"title": "No Service", "ladder": ["airplane", "sim", "restart", "reset_network"],
                   "intro": "When the phone shows **No Service** or **SOS**, it can't make calls, send texts or use mobile "
                            "data. It's often a temporary network or SIM issue.",
                   "branch": "If there's an outage in the customer's area, no step here will help; check the outage map first.",
                   "check": "Have the customer make a test call to another number.",
                   "say": ["it says no service", "it says SOS up in the corner"],
                   "ok": "Yes, the test call went through.", "bad": "The call still won't go through.",
                   "looks": "I think a bar just showed up?"},
    "texts_failing": {"title": "Messaging", "ladder": ["messages", "airplane", "update", "reset_network"],
                      "intro": "When text or picture messages won't send, they stay stuck with a red mark or a 'Not "
                               "Delivered' note. Settings and network glitches are the usual causes.",
                      "branch": "If only messages to one person fail, the problem may be on their side; ask them to check "
                                "their phone first.",
                      "check": "Have the customer send a test message to someone they know.",
                      "say": ["my texts won't send", "my messages just sit there with a red mark"],
                      "ok": "Yes, my daughter just got it.", "bad": "It's stuck again.",
                      "looks": "Oh, the red mark went away?"},
    "wifi_wont_connect": {"title": "Wi-Fi Connection Issues", "ladder": ["forget_wifi", "airplane", "restart", "reset_network"],
                          "intro": "When a phone sees a Wi-Fi network but won't join it, a saved setting is often out of "
                                   "date. These steps refresh the connection.",
                          "branch": "If no device in the home can connect, the router is the problem: use **Troubleshoot "
                                    "Home Internet** instead.",
                          "check": "Have the customer join the network and open a website.",
                          "say": ["it won't connect to Wi-Fi", "it sees the Wi-Fi but never joins"],
                          "ok": "Yes, it's on the Wi-Fi and the page loaded.", "bad": "It still won't join.",
                          "looks": "It says connecting now?"},
    "picture_messages": {"title": "Picture and Group Messages", "ladder": ["data", "messages", "restart", "reset_network"],
                         "intro": "Picture messages and group texts need mobile data, even when regular texts work. When "
                                  "they fail, the photo shows a red mark or 'Not Delivered'.",
                         "branch": "If regular one-to-one texts fail too, this isn't a picture message problem: use "
                                   "**Troubleshoot Messaging** instead.",
                         "check": "Have the customer send a photo to someone they know.",
                         "say": ["my picture messages won't send", "group texts and photos won't go through"],
                         "ok": "Yes, the photo went through.", "bad": "The photo is still stuck.",
                         "looks": "Oh, it says sending now?"},
    "calls_dropping": {"title": "Dropped Calls", "ladder": ["airplane", "restart", "update", "reset_network"],
                       "intro": "When calls keep cutting out, the phone is usually losing its connection to the network "
                                "for a moment. A weak signal, an old setting or a software glitch are the usual causes.",
                       "branch": "If calls only drop at home and the customer uses Wi-Fi calling there, use "
                                 "**Troubleshoot Wi-Fi Calling** instead.",
                       "check": "Have the customer make a test call and stay on it for two minutes.",
                       "say": ["my calls keep dropping", "people say I cut out and then the call ends"],
                       "ok": "Yes, that call stayed on the whole time.", "bad": "The call dropped again.",
                       "looks": "The bars look fuller now?"},
    "wifi_calling_broken": {"title": "Wi-Fi Calling", "ladder": ["wifi_calling", "forget_wifi", "restart", "reset_network"],
                            "intro": "Wi-Fi calling lets the phone make calls over Wi-Fi where the signal is weak. When it "
                                     "stops working, calls at home drop or won't connect.",
                            "branch": "If calls drop everywhere, not just on Wi-Fi, use **Troubleshoot Dropped Calls** instead.",
                            "check": "Have the customer turn on Airplane mode with Wi-Fi on, then make a test call.",
                            "say": ["Wi-Fi calling isn't working", "calls over my home Wi-Fi keep failing"],
                            "ok": "Yes, the call went through on Wi-Fi.", "bad": "It still won't call over Wi-Fi.",
                            "looks": "I see Wi-Fi next to the carrier name now?"},
    "hotspot_broken": {"title": "Personal Hotspot", "ladder": ["hotspot", "restart", "update", "reset_network"],
                       "intro": "When the hotspot stops working, other devices can't get online through the phone. The "
                                "phone's own data may still work.",
                       "branch": "If the phone itself has no mobile data either, use **Troubleshoot Mobile Data** instead.",
                       "check": "Have the customer connect their laptop to the hotspot and open a website.",
                       "say": ["my hotspot isn't working", "my laptop won't get online through my phone"],
                       "ok": "Yes, the laptop is online.", "bad": "The laptop still can't get online.",
                       "looks": "The laptop says connected?"},
}
# a call can turn out to be a different problem than it first sounded like: (what it sounds like, what the customer says)
SWITCH = {"no_service": ("no_cell_data", "Oh wait, now that I look, it says SOS up in the corner."),
          "texts_failing": ("picture_messages", "Actually, regular texts aren't going out either."),
          "wifi_calling_broken": ("calls_dropping", "Oh, and it's only at home, where I have Wi-Fi calling on.")}
SWITCH_FIX = {"no_service": "sim", "texts_failing": "update", "wifi_calling_broken": "wifi_calling"}
LABEL = {"iphone": "iOS", "android": "Android"}
BRAND = {"samsung": ["Galaxy S24", "Galaxy A15"], "pixel": ["Pixel 8", "Pixel 9"]}
TRIED_TEXT = {"restart": "I already restarted it", "airplane": "I already tried the airplane mode thing"}
UNCLEAR = ["Sorry, you cut out. What was that?", "Hang on, I'm driving. Okay, what?"]
HUMAN = "Look, this is taking forever. Can I just talk to a real person?"
SLIP = {"data": "I turned off Wi-Fi instead. Is that the same thing?",
        "airplane": "I swiped down but I just turned the flashlight on. Oops.",
        "sim": "I'm in Settings but I tapped the wrong thing, it's asking about eSIM now. I backed out.",
        "restart": "It's asking me for an emergency SOS thing, I didn't do it.",
        "reset_network": "It says enter passcode and I don't remember it right now. Let me try again.",
        "messages": "I opened the Messages app but I don't see any settings in there.",
        "update": "It says checking for update and then it went back.",
        "forget_wifi": "I tapped the network and it just tried to connect again.",
        "wifi_calling": "I went to the Phone app but I'm just looking at my recent calls.",
        "hotspot": "I turned on Wi-Fi instead of the hotspot, I think."}


@dataclass
class Art:
    doc_id: str
    plat: str
    sym: str | None
    kind: str  # ok | stale | instore | howto
    title: str
    body: str


def _path(step: str, plat: str) -> str:
    s = STEPS[step]
    if plat == "iphone":
        return s["iphone"]
    tail = (" " + s["android_tail"]) if s.get("android_tail") else ""
    return f"**Samsung Galaxy devices:** {s['samsung']} **Other Android devices (Pixel):** {s['pixel']}{tail}"


def _article(plat: str, sym: str, kind: str = "ok") -> str:
    s = SYMS[sym]
    lad = s["ladder"]
    who = ("For in-store technicians with diagnostic tools only. Don't use this article on a phone or chat call. "
           if kind == "instore" else "")
    ver = ("Last reviewed 2023. Written for iOS 16 and 17; menus moved in later versions. " if kind == "stale" and plat == "iphone"
           else "Last reviewed 2023. Written for Android 13; menus moved in later versions. " if kind == "stale"
           else f"Last reviewed 2026. Written for {'iOS 26' if plat == 'iphone' else 'Android 16'}. ")
    out = [f"# **Troubleshoot {s['title']} - {LABEL[plat]}**", "", s["intro"] + " " + who +
           "Work through the sections below in order, and check the fix after each one.", "", "* * *", "", "## ", "",
           "**Before you start**", "",
           "Find out what the customer already tried, so you don't repeat it. " + s["branch"], "", "* * *", ""]
    for st in lad[:2]:
        out += ["## ", "", f"**{STEPS[st]['head']}**", "", _path(st, plat), "", "* * *", ""]
    still = "If the steps above didn't help, try these next, in this order.\n\n" + "\n\n".join(
        f"**{STEPS[st]['head']}**\n\n{_path(st, plat)}" for st in lad[2:])
    older = ("On iPhone 8 and earlier, press and hold the top button instead of the side button to restart."
             if plat == "iphone" else "On older Samsung phones, **Data usage** may be under **Settings** > **Connections** > **Mobile networks**.")
    out += ["## ", "", "**Still not working?**", "", still, "", older, "", "* * *", "",
            f"**Check the fix:** {s['check']} A step can look like it worked before it really has.", "",
            "If none of this helps, connect the customer with a specialist. " + ver +
            "Related: Turn Airplane mode on or off, Reset network settings, Back up your phone."]
    return "\n".join(out)


def build_kb() -> list[Art]:
    arts = []
    for sym, plat in itertools.product(SYMS, ["iphone", "android"]):
        for kind in ("ok", "stale", "instore"):
            title = f"Troubleshoot {SYMS[sym]['title']} - {LABEL[plat]}" + (" (In-Store)" if kind == "instore" else "")
            arts.append(Art(f"KB-{100 + len(arts)}", plat, sym, kind, title, _article(plat, sym, kind)))
    for t, plat in itertools.product(["Turn Airplane mode on or off", "Reset network settings", "Restart your phone",
                                      "Set up Wi-Fi calling", "Back up your phone", "Update your phone",
                                      "Set up a personal hotspot", "Send a picture message", "Set up voicemail",
                                      "Find your phone's model", "Turn mobile data on or off", "Join a Wi-Fi network",
                                      "Check your signal strength", "Turn on group messaging", "Use eSIM",
                                      "Block a caller", "Manage your data usage", "Turn on call forwarding"],
                                     ["iphone", "android"]):
        arts.append(Art(f"KB-{900 + len(arts)}", plat, None, "howto", f"{t} - {LABEL[plat]}",
                        f"# **{t} - {LABEL[plat]}**\n\nA how-to guide that explains the feature. It is not a "
                        f"troubleshooting article. Open **Settings** and follow the on-screen options."))
    return arts


KB = build_kb()
BY_ID = {a.doc_id: a for a in KB}
ART = {(a.plat, a.sym): a for a in KB if a.kind == "ok"}


def _tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


BM25 = BM25Okapi([_tok(a.title + " " + a.body + " " + " ".join(SYMS[a.sym]["say"] if a.sym else []) +
                       (" iphone apple" if a.plat == "iphone" else " android samsung galaxy pixel google")) for a in KB])


def search(text: str) -> list[Art]:
    sc = BM25.get_scores(_tok(text))
    return [KB[i] for i in sorted(range(len(KB)), key=lambda i: -sc[i])[:K]]


def norm(h) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(h or "").lower()).strip()


HEAD = {norm(v["head"]): k for k, v in STEPS.items()}


@dataclass
class Case:
    case_id: str
    seed: int
    kind: str
    plat: str = "iphone"
    brand: str | None = None
    sym: str = "no_cell_data"  # the real problem
    sos: bool = False  # opens as a data problem; the phone actually shows SOS (seen after the first step)
    fix: str | None = None  # step key that fixes it; None = nothing works
    tried: tuple = ()  # said in the opening
    late_tried: tuple = ()  # done this morning, only mentioned when that step comes up
    slips: tuple = ()  # steps the customer gets wrong the first time
    decoys: tuple = ()  # steps that look like they worked but didn't
    data_on: bool = False  # mobile data was already on
    ios_alt: bool = False  # this iPhone shows "Mobile Service" instead of "Cellular"
    brand_said: bool = False
    human_after: int | None = None


@dataclass
class State:
    case: Case
    k: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)


def cur_sym(st: State) -> str:
    c = st.case
    return c.sym if (not c.sos or st.k.get("sos_seen")) else SWITCH[c.sym][0]


def doc(st: State) -> str:
    return ART[(st.case.plat, cur_sym(st))].doc_id


def allowed(st: State) -> dict:
    """The answer key, from what has happened so far (never from hidden truth the agent couldn't know)."""
    c, k = st.case, st.k
    if k.get("wants_human"):
        return {"escalate": "human_request"}
    if k.get("verified"):
        return {"resolve": doc(st)}
    if k.get("pending"):
        return {"verify": True}
    done = set(k.get("done", []))
    rest = [s for s in SYMS[cur_sym(st)]["ladder"] if s not in done and s not in c.tried]
    if not rest:
        return {"escalate": "fix_failed"}
    s = rest[0]
    if STEPS[s].get("destructive") and not k.get("consent"):
        return {"ask": ["consent"]}
    missed = set(k.get("missed", {}).get(s, []))
    if c.plat == "android":
        paths = [c.brand] if k.get("brand") else [p for p in ("samsung", "pixel") if p not in missed]
    else:
        paths = [None]  # iPhone: one set of instructions; a menu-label mix-up is fixed by giving the step again
    out = {"step": {"doc_id": doc(st), "step": STEPS[s]["head"], "paths": paths}}
    if c.plat == "android" and not k.get("brand"):
        out["ask"] = ["model"]
    return out


def is_allowed(a: dict, al: dict) -> bool:
    kind = a.get("action")
    if kind == "ask":
        return a.get("slot") in al.get("ask", [])
    if kind == "step" and "step" in al:
        s = al["step"]
        path = a.get("path") if a.get("path") not in ("", "null", "standard", "none", "alt") else None
        return a.get("doc_id") == s["doc_id"] and norm(a.get("step")) == norm(s["step"]) and path in s["paths"]
    if kind == "verify":
        return "verify" in al
    if kind == "resolve":
        return a.get("doc_id") == al.get("resolve")
    if kind == "escalate":
        return a.get("reason") == al.get("escalate")
    return False


def _r(c: Case, *key) -> random.Random:
    return random.Random("|".join(map(str, (c.seed,) + key)))


def opening(c: Case) -> str:
    r = _r(c, "open")
    dev = "iPhone" if c.plat == "iphone" else (r.choice(BRAND[c.brand]) if c.brand_said else "Android phone")
    shown = SWITCH[c.sym][0] if c.sos else c.sym
    parts = [f"Hi, my {dev}: {r.choice(SYMS[shown]['say'])}."]
    if c.tried:
        parts.append(" and ".join(TRIED_TEXT[t] for t in c.tried).capitalize() + ", and it's still doing it.")
    parts.append(r.choice(["I've had it about two years.", "I'm at work and really need this fixed.",
                           "My daughter usually helps me with this stuff."]))
    return " ".join(parts)


def reply(st: State, a: dict) -> str:
    """Scripted customer. Draws are keyed by (case, move, attempt), so the same move always gets the same reply."""
    c, k = st.case, st.k
    k["turns"] = k.get("turns", 0) + 1
    if c.human_after and k["turns"] >= c.human_after and not k.get("wants_human"):
        k["wants_human"] = True
        return HUMAN
    kind = a.get("action")
    s = SYMS[cur_sym(st)]
    if kind == "ask":
        slot = a.get("slot")
        if slot == "model":
            if c.plat == "iphone":
                return "It's an iPhone, I'm not sure which one."
            k["brand"] = True
            return f"It's a {_r(c, 'model').choice(BRAND[c.brand])}."
        if slot == "consent":
            if k.get("consent"):  # asking again isn't an instruction; the customer says so
                return "I already said yes. Should I try it again?"
            k["consent"] = True
            return "Okay, go ahead."
        if slot == "symptom":
            return s["say"][0].capitalize() + "."
        return "I'm not sure what you mean."
    if kind == "step":
        key = HEAD.get(norm(a.get("step")))
        if a.get("doc_id") != doc(st) or key not in SYMS[cur_sym(st)]["ladder"]:
            return "Okay, I did that, but nothing changed."  # wrong article, or a step that isn't in it
        path = a.get("path") if a.get("path") not in ("", "null", "standard", "none") else None
        if c.plat == "android" and path != c.brand:
            k.setdefault("missed", {}).setdefault(key, []).append(path)
            return "Hmm, I don't see that on my phone. My settings look different."
        if c.plat == "iphone" and STEPS[key].get("alt") and c.ios_alt and key not in k.get("alt_seen", []):
            k.setdefault("alt_seen", []).append(key)  # first try: lost in the menu; the same step again works
            return f"I don't see {ALT_LABEL[0]} in my settings. There's something called {ALT_LABEL[1]}?"
        if key in c.slips and key not in k.get("slipped", []):
            k.setdefault("slipped", []).append(key)
            return SLIP[key]
        k.setdefault("done", []).append(key)
        tail = ""
        if c.sos and not k.get("sos_seen"):
            k["sos_seen"] = True
            tail = " " + SWITCH[c.sym][1]
        if key in c.late_tried:
            return "Oh, I already did that this morning. It didn't help." + tail
        if key == "data" and c.data_on:
            return "It's already on." + tail
        if k.get("really_fixed"):
            k["pending"] = True
            return "Okay, I did that too. " + s["looks"]
        if c.fix == key or key in c.decoys:
            if tail:  # the SOS sighting comes first; this step didn't fix the real problem
                return "Okay, I did that. " + s["bad"] + tail
            k["pending"] = True
            k["really_fixed"] = c.fix == key
            return "Okay, done. " + s["looks"]
        return "Okay, I did that. " + s["bad"] + tail
    if kind == "verify":
        k["pending"] = False
        if k.get("really_fixed"):
            k["verified"] = True
            return s["ok"]
        return "No. " + s["bad"]
    return "Sorry?"


def customer_text(st: State) -> str:
    return " ".join(m["content"].split("\n\n[SEARCH RESULTS]")[0].removeprefix("[CUSTOMER] ")
                    for m in st.messages if m["role"] == "user")


def _attach_results(st: State):
    res = search(customer_text(st))
    st.messages[-1]["content"] += "\n\n[SEARCH RESULTS]\n" + "\n\n".join(f"=== {a.doc_id}: {a.title} ===\n{a.body}" for a in res)


def new_state(c: Case) -> State:
    st = State(c, {"brand": c.brand_said}, [{"role": "user", "content": "[CUSTOMER] " + opening(c)}])
    _attach_results(st)
    return st


SYSTEM = """You are a tech-support voice agent for a phone carrier, on a call with a customer. Each turn, output exactly one JSON object and nothing else:

{"action": "ask", "slot": "symptom" | "model" | "consent", "question": "<what you say>"}
{"action": "step", "doc_id": "<KB id>", "step": "<the bold name of the fix in the article, e.g. the heading above its instructions>", "path": "samsung" | "pixel" | null, "say": "<the instruction, in your words>"}
{"action": "verify", "say": "<what you ask the customer to check>"}
{"action": "resolve", "doc_id": "<KB id>", "reply": "<closing line>"}
{"action": "escalate", "reason": "physical_damage" | "safety" | "billing" | "human_request" | "fix_failed", "reply": "<what you say>"}

Slots: symptom = what is wrong; model = which phone model it is; consent = the customer's OK to go ahead with something.
Path: which version of a step's instructions you give: samsung or pixel for the Android brand lines, null otherwise.
Reasons: fix_failed = the article's steps didn't fix it; human_request = the customer asked for a person.

Policy: damage, safety hazards, billing and requests for a person go to a specialist. Otherwise fix the problem by walking the customer through the knowledge base article that fits, one step at a time; the top search results for what the customer has said are shown after each customer turn. Customers' phones run current software (iOS 26, Android 16). Close the call with resolve once the problem is fixed, or hand off with fix_failed if the article can't fix it."""


def step(st: State, a: dict) -> tuple[bool, str, bool]:
    al = allowed(st)
    ok = is_allowed(a, al)
    st.messages.append({"role": "assistant", "content": json.dumps(a)})
    kind = a.get("action")
    if kind == "resolve":
        return True, ("resolved_correct" if ok else "resolved_wrong"), ok
    if kind == "escalate":
        return True, ("escalated_correct" if ok else "escalated_wrong"), ok
    if kind in ("ask", "step", "verify"):
        st.messages.append({"role": "user", "content": "[CUSTOMER] " + reply(st, a)})
        _attach_results(st)
        return False, kind, ok
    return True, "invalid", False


def oracle_moves(st: State) -> list[dict]:
    al = allowed(st)
    out = []
    if "resolve" in al:
        out.append({"action": "resolve", "doc_id": al["resolve"], "reply": "."})
    if "escalate" in al:
        out.append({"action": "escalate", "reason": al["escalate"], "reply": "."})
    if "verify" in al:
        out.append({"action": "verify", "say": "."})
    if "step" in al:
        for p in al["step"]["paths"]:
            out.append({"action": "step", "doc_id": al["step"]["doc_id"], "step": al["step"]["step"], "path": p, "say": "."})
    for slot in al.get("ask", []):
        out.append({"action": "ask", "slot": slot, "question": "."})
    return out


def explore(c: Case) -> dict:
    """Every path the answer key allows must end correct within the step budget."""
    out = {"paths": 0, "bad": 0, "depth": 0}
    seen = set()

    def walk(st, depth):
        key = (json.dumps(st.k, sort_keys=True), depth)
        if key in seen:
            return
        seen.add(key)
        if depth >= MAX_STEPS:
            out["bad"] += 1
            return
        for a in oracle_moves(st):
            st2 = State(st.case, copy.deepcopy(st.k), list(st.messages))
            done, o, ok = step(st2, a)
            if not ok or (done and o not in ("resolved_correct", "escalated_correct")):
                out["bad"] += 1
            elif done:
                out["paths"] += 1
                out["depth"] = max(out["depth"], depth + 1)
            else:
                walk(st2, depth + 1)

    walk(new_state(c), 0)
    return out


KINDS = ["late_fix", "nothing_works", "decoy", "menu_label", "brand_path", "slip", "late_tried", "already_on",
         "sos_switch", "human_midcall"]


def sample_cases(n_per: int, seed: int = 7) -> list[Case]:
    """Each case has one headline mechanism plus the background noise of real calls (other mechanisms at lower rates)."""
    rng, out, count, i = random.Random(seed), [], collections.Counter(), 0
    while min(count[t] for t in KINDS) < n_per and i < 50000:
        i += 1
        kind = KINDS[i % len(KINDS)]
        if count[kind] >= n_per:
            continue
        c = Case(f"V2-{i:05d}", rng.randrange(10**9), kind)
        c.plat = "iphone" if kind == "menu_label" else "android" if kind == "brand_path" else rng.choice(["iphone", "android"])
        c.brand = rng.choice(["samsung", "pixel"]) if c.plat == "android" else None
        c.brand_said = c.plat == "android" and kind != "brand_path" and rng.random() < 0.4
        c.sym = rng.choice(list(SWITCH)) if kind == "sos_switch" else rng.choice(list(SYMS))
        if kind == "already_on":
            c.sym = "no_cell_data"
        c.sos = kind == "sos_switch"
        lad = SYMS[c.sym]["ladder"]
        c.fix = None if kind == "nothing_works" else rng.choice(lad[2:]) if kind in ("late_fix", "decoy", "sos_switch") else rng.choice(lad + [None])
        if kind == "sos_switch":
            c.fix = SWITCH_FIX[c.sym]
        others = [s for s in lad if s != c.fix]
        # background noise of real calls, plus the headline mechanism for sure
        c.tried = tuple(s for s in others if s in TRIED_TEXT and rng.random() < 0.25)
        pool = [s for s in others if s not in c.tried]
        c.late_tried = tuple(s for s in pool if rng.random() < (0.6 if kind == "late_tried" else 0.1))[:1 if kind != "late_tried" else 2]
        if kind == "late_tried" and not c.late_tried and pool:
            c.late_tried = (pool[0],)
        c.slips = tuple(s for s in lad if s not in c.tried and rng.random() < (0.5 if kind == "slip" else 0.12))
        if kind == "slip" and not c.slips:
            c.slips = (lad[0],)
        stop = lad.index(c.fix) if c.fix in lad else len(lad)
        c.decoys = tuple(s for s in lad[:stop] if s not in c.tried and s not in c.late_tried
                         and rng.random() < (0.6 if kind == "decoy" else 0.15))
        if kind == "decoy" and not c.decoys:
            continue
        c.data_on = kind == "already_on" or (c.sym == "no_cell_data" and rng.random() < 0.3)
        c.ios_alt = c.plat == "iphone" and (kind == "menu_label" or rng.random() < 0.2)
        if kind == "menu_label" and not any(STEPS[s].get("alt") for s in lad):
            c.sym = "no_cell_data"
            lad = SYMS[c.sym]["ladder"]
            c.fix = rng.choice(lad[1:])
            c.decoys = ()
            c.tried = tuple(t for t in c.tried if t in lad and t != c.fix)
        c.human_after = rng.choice([4, 6]) if kind == "human_midcall" else None
        if explore(c)["bad"]:
            continue
        count[kind] += 1
        out.append(c)
    return out


async def run(c: Case, policy) -> dict:
    st = new_state(c)
    outcome, first_bad, steps, wasted = "max_steps", None, [], 0
    for _ in range(MAX_STEPS):
        al = allowed(st)
        text = await policy([{"role": "system", "content": SYSTEM}] + st.messages)
        a = parse_action(text)
        if a is None:
            outcome = "invalid"
            steps.append({"action": {"raw": text[-200:]}, "allowed": al, "ok": False})
            first_bad = first_bad or {"turn": len(steps), "did": "invalid", "key": al}
            break
        done, outcome, ok = step(st, a)
        steps.append({"action": a, "allowed": al, "ok": ok})
        if not ok and a.get("action") == "ask":
            wasted += 1  # a needless question costs time but isn't fatal
        elif not ok and first_bad is None:
            first_bad = {"turn": len(steps), "did": a.get("action"), "key": al}
        if done:
            break
    clean = outcome in ("resolved_correct", "escalated_correct") and first_bad is None
    return {"case_id": c.case_id, "kind": c.kind, "outcome": outcome, "clean": clean, "first_bad": first_bad,
            "turns": len(steps), "wasted": wasted, "steps": steps, "transcript": st.messages}


def show(x: dict):
    print(f"\n----- {x['kind']} {x['case_id']}: {'PASS' if x['clean'] else 'FAIL'} ({x['outcome']}, {x['turns']} turns)")
    us = [m for m in x["transcript"] if m["role"] == "user"]
    for m, s in zip(us, x["steps"]):
        print("  CUSTOMER:", m["content"].split("\n\n[SEARCH RESULTS]")[0].removeprefix("[CUSTOMER] ")[:170])
        a = s["action"]
        what = a.get("raw") or f"{a.get('action')} {a.get('doc_id') or ''} {a.get('step') or a.get('slot') or a.get('reason') or ''} {a.get('path') or ''}"
        key = s["allowed"].get("step", s["allowed"])
        print(f"  {'ok ' if s['ok'] else 'BAD'} AGENT: {what}   | key: {key}")


def report(logs):
    by = collections.defaultdict(list)
    for x in logs:
        by[x["kind"]].append(x)
    ok = lambda xs: sum(x["clean"] for x in xs) / max(1, len(xs))
    print(f"\n{'case type':14s} {'n':>3s} {'success':>8s} {'turns':>6s}")
    for kd in sorted(by, key=lambda kd: ok(by[kd])):
        print(f"{kd:14s} {len(by[kd]):3d} {ok(by[kd]):8.2f} {sum(x['turns'] for x in by[kd]) / len(by[kd]):6.1f}")
    print(f"{'ALL':14s} {len(logs):3d} {ok(logs):8.2f} {sum(x['turns'] for x in logs) / max(1, len(logs)):6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accounts/fireworks/models/deepseek-v4p1-flash")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--case", default=None, help="run ONE case of this kind and print it")
    ap.add_argument("--pick", type=int, default=0)
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--n-per-type", type=int, default=12)
    ap.add_argument("--oracle-only", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--freeze", default=None, help="write KB, cases and prompt to this folder")
    args = ap.parse_args()
    n = args.quick or (args.pick + 1 if args.case else args.n_per_type)
    cases = sample_cases(n, args.seed)
    walks = [explore(c) for c in cases]
    print(f"KB: {len(KB)} articles (median body {sorted(len(a.body) for a in KB if a.kind == 'ok')[4]} chars); "
          f"cases: {len(cases)}; fairness: {sum(w['paths'] for w in walks)} allowed paths, {sum(w['bad'] for w in walks)} bad, "
          f"max {max(w['depth'] for w in walks)} turns")
    if args.freeze:
        d = Path(args.freeze)
        d.mkdir(parents=True, exist_ok=True)
        (d / "kb.jsonl").write_text("".join(json.dumps(asdict(a)) + "\n" for a in KB))
        (d / "test_cases.jsonl").write_text("".join(json.dumps(asdict(c)) + "\n" for c in cases))
        (d / "system_prompt.txt").write_text(SYSTEM)
    if args.oracle_only:
        return
    if args.case:
        cases = [[c for c in cases if c.kind == args.case][args.pick]]
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"], max_retries=8)
    extra = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort not in (None, "default") else {}

    async def policy(msgs):
        r = await client.chat.completions.create(model=args.model, messages=msgs, temperature=0, max_tokens=1024, **extra)
        return r.choices[0].message.content or ""

    async def go():
        sem = asyncio.Semaphore(6)

        async def one(c):
            async with sem:
                for attempt in range(4):
                    try:
                        return await run(c, policy)
                    except Exception as ex:  # rate limit etc.: back off and retry; never scored as a model failure
                        err = repr(ex)[:150]
                        await asyncio.sleep(10 * (attempt + 1))
                return {"case_id": c.case_id, "kind": c.kind, "outcome": "api_error", "clean": False, "error": err}

        return await asyncio.gather(*[one(c) for c in cases])

    logs = [x for x in asyncio.run(go()) if x["outcome"] != "api_error"]
    tag = args.model.split("/")[-1] + f"_re-{args.reasoning_effort}"
    out = Path("runs") / f"hard_v2_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs.jsonl").write_text("".join(json.dumps(x, default=str) + "\n" for x in logs))
    for x in logs:
        if len(logs) <= 12 or not x["clean"]:
            show(x)
    report(logs)


if __name__ == "__main__":
    main()
