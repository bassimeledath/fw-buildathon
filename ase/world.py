"""The synthetic tech-support world: devices, symptoms, KB articles, escalation policy.

Every case is generated from hidden attributes (platform, symptom, trigger), so the
correct next action is known by construction. That is what makes the reward pure code.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field

# ASE_WORLD=hard switches on the harder, KB-grounded world (see the block after PLATFORM_TERMS).
HARD = os.environ.get("ASE_WORLD", "") == "hard"
# ASE_RETRIEVAL=fixed is the simple baseline: no search action. The environment searches the customer's
# words after every customer message and shows the top results; the model picks resolve / ask / escalate.
FIXED = os.environ.get("ASE_RETRIEVAL", "") == "fixed"
DATA_DIR = ("data_hard" if HARD else "data") + ("_fixed" if FIXED else "")
RUN_PREFIX = ("hard_" if HARD else "") + ("fixed_" if FIXED else "")

CORE_SLOTS = ("device", "symptom", "trigger")  # slots that pick the article
SLOTS = CORE_SLOTS + (("damage", "backup") if HARD else ())  # every slot the agent may ask about

PLATFORMS = {
    "iphone": {"family": "phone", "label": "iPhone"},
    "android": {"family": "phone", "label": "Android phone"},
    "windows": {"family": "computer", "label": "Windows laptop"},
    "mac": {"family": "computer", "label": "Mac"},
    "router": {"family": "router", "label": "home Wi-Fi router"},
}

# How a customer might name each platform. "specific" resolves the platform, "family" does not.
DEVICE_SPECIFIC = {
    "iphone": ["iPhone 15", "iPhone 13", "iPhone", "iPhone 14 Pro"],
    "android": ["Galaxy S24", "Samsung", "Pixel 8", "Moto G"],
    "windows": ["Dell laptop", "HP laptop", "Surface", "Windows laptop"],
    "mac": ["MacBook Air", "MacBook Pro", "Mac"],
    "router": ["router", "Wi-Fi box", "internet gateway"],
}
DEVICE_FAMILY = {"phone": ["phone", "cell"], "computer": ["laptop", "computer"], "router": ["router"]}

PHONE, COMPUTER, ALL_PERSONAL = ["iphone", "android"], ["windows", "mac"], ["iphone", "android", "windows", "mac"]

# symptom -> platforms it applies to, trigger variants (None = no variant), vague group
SYMPTOMS = {
    "no_cell_data": {"platforms": PHONE, "triggers": ["new_sim", "abroad", "no_change"], "group": "internet"},
    "calls_dropping": {"platforms": PHONE, "triggers": ["at_home", "everywhere"], "group": "calls_texts"},
    "texts_failing": {"platforms": PHONE, "triggers": ["pictures_group", "all_texts"], "group": "calls_texts"},
    "battery_drain": {"platforms": ALL_PERSONAL, "triggers": ["after_update", "no_change"], "group": "power"},
    "wont_charge": {"platforms": ALL_PERSONAL, "triggers": None, "group": "power"},
    "overheating": {"platforms": ALL_PERSONAL, "triggers": ["while_charging", "during_use"], "group": "power"},
    "wifi_wont_connect": {"platforms": ALL_PERSONAL, "triggers": ["one_network", "any_network"], "group": "internet"},
    "slow_internet": {"platforms": ["router"], "triggers": ["wifi_only", "all_connections"], "group": "internet"},
    "no_internet": {"platforms": ["router"], "triggers": None, "group": "internet"},
    "storage_full": {"platforms": ALL_PERSONAL, "triggers": None, "group": "apps_storage"},
    "app_crashing": {"platforms": ALL_PERSONAL, "triggers": ["one_app", "all_apps"], "group": "apps_storage"},
    "bluetooth_pairing": {"platforms": ALL_PERSONAL, "triggers": ["car", "headphones"], "group": "accessories"},
}

SYMPTOM_SPECIFIC = {
    "no_cell_data": ["mobile data isn't working", "no data unless I'm on Wi-Fi", "it says no service for data"],
    "calls_dropping": ["calls keep dropping", "my calls cut out mid-conversation"],
    "texts_failing": ["texts won't send", "messages fail to go through"],
    "battery_drain": ["battery dies super fast", "battery drains way too quick"],
    "wont_charge": ["it won't charge", "it's not charging at all when plugged in"],
    "overheating": ["it gets really hot", "it's overheating"],
    "wifi_wont_connect": ["it won't connect to Wi-Fi", "Wi-Fi keeps failing to connect"],
    "slow_internet": ["internet is super slow", "speeds are terrible"],
    "no_internet": ["the internet is completely down", "no internet at all, lights are blinking"],
    "storage_full": ["it says storage is full", "I'm out of space"],
    "app_crashing": ["apps keep crashing", "an app closes as soon as I open it"],
    "bluetooth_pairing": ["Bluetooth won't pair", "it won't connect over Bluetooth"],
}
SYMPTOM_VAGUE = {
    "internet": ["having internet problems", "the internet stuff isn't working right"],
    "calls_texts": ["having trouble with calls and texts", "something's wrong with calling or messaging"],
    "power": ["having battery problems", "something's off with the battery"],
    "apps_storage": ["it's acting up with apps", "apps are being weird"],
    "accessories": ["my accessories won't work with it", "having connection issues with my gadgets"],
}

TRIGGER_TEXT = {
    "new_sim": ["ever since I switched to a new SIM", "since I moved my number to this phone"],
    "abroad": ["I'm traveling overseas right now", "since I landed in another country"],
    "no_change": ["nothing changed, it just started", "out of nowhere, I didn't change anything"],
    "at_home": ["only when I'm at home", "it's mostly at my house"],
    "everywhere": ["everywhere I go", "no matter where I am"],
    "pictures_group": ["only pictures and group chats", "just the group texts and photos"],
    "all_texts": ["none of my texts go out", "every single text fails"],
    "after_update": ["right after the last update", "since I updated the software"],
    "while_charging": ["when it's charging", "while plugged in"],
    "during_use": ["when I'm using it", "while I'm actually using it"],
    "one_network": ["just my home network, others are fine", "only one network gives me trouble"],
    "any_network": ["any network I try", "every Wi-Fi network, not just one"],
    "wifi_only": ["wired is fine, just Wi-Fi", "plugged-in devices are fine, wireless is slow"],
    "all_connections": ["even wired devices are slow", "everything, wired and wireless"],
    "one_app": ["just one app", "it's only one specific app"],
    "all_apps": ["every app does it", "all my apps"],
    "car": ["with my car", "my car's Bluetooth"],
    "headphones": ["with my earbuds", "my headphones"],
}

# Escalation policy: these never have a KB article; the correct action is escalate with this reason.
ESCALATIONS = {
    "physical_damage": [
        "I dropped my {dev} in the pool and now the screen is black",
        "my {dev} screen is cracked all the way across",
        "spilled coffee on my {dev} and it won't turn on",
    ],
    "safety": [
        "the battery on my {dev} is swelling and the back is lifting",
        "my {dev} smells like burning plastic",
        "there was smoke coming out of my {dev} charger port",
    ],
    "billing": [
        "there's a charge on my bill I don't recognize",
        "I want to change my plan and lower my monthly bill",
        "my payment didn't go through and now service is suspended",
    ],
    "human_request": [
        "I just want to talk to a real person please",
        "get me a human agent, I'm done with bots",
    ],
}
ESCALATION_REASONS = tuple(ESCALATIONS)

TRIGGER_TITLE = {
    "new_sim": "after switching SIM or eSIM",
    "abroad": "while traveling internationally",
    "no_change": "with no recent changes",
    "at_home": "at home",
    "everywhere": "everywhere",
    "pictures_group": "picture and group messages",
    "all_texts": "all text messages",
    "after_update": "after a software update",
    "while_charging": "while charging",
    "during_use": "during normal use",
    "one_network": "on one network only",
    "any_network": "on any network",
    "wifi_only": "on Wi-Fi only",
    "all_connections": "on wired and wireless",
    "one_app": "one app only",
    "all_apps": "all apps",
    "car": "with a car",
    "headphones": "with headphones or earbuds",
}
SYMPTOM_TITLE = {
    "no_cell_data": "No mobile data",
    "calls_dropping": "Calls dropping",
    "texts_failing": "Text messages not sending",
    "battery_drain": "Battery draining fast",
    "wont_charge": "Not charging",
    "overheating": "Overheating",
    "wifi_wont_connect": "Can't connect to Wi-Fi",
    "slow_internet": "Slow internet speeds",
    "no_internet": "No internet connection",
    "storage_full": "Storage full",
    "app_crashing": "Apps crashing",
    "bluetooth_pairing": "Bluetooth won't pair",
}
SETTINGS = {
    "iphone": "Settings > Cellular",
    "android": "Settings > Connections > Mobile networks",
    "windows": "Start > Settings > Network & internet",
    "mac": "Apple menu > System Settings",
    "router": "the router admin page (192.168.1.1)",
}
PLATFORM_TERMS = {
    "iphone": "iOS iPhone Apple",
    "android": "Android Samsung Galaxy Pixel",
    "windows": "Windows PC laptop Microsoft",
    "mac": "macOS Mac MacBook Apple",
    "router": "router gateway modem home Wi-Fi",
}

# ---- Hard world. Each mechanism copies a pattern from real carrier support KBs; all text is our own.
# Fairness rule: every fact the oracle needs is in the opening, in a customer answer (at most 2 asks
# per slot), or in the text of an article the agent has retrieved. ase/fairness.py checks this.
GATED = {"wont_charge", "wont_power_on", "display_fault", "overheating"}  # article opens with a damage question
# overheating is OOD: tests whether the model follows a gate it never saw in training
DESTRUCTIVE = {"app_crashing", "battery_drain", "display_fault"}  # advanced fix erases the device
BACKUP_DOC = "KB-905"
TRIED_TEXT = [  # the customer reports the basic checks, so the advanced article applies
    "I already restarted it and checked the settings",
    "I've rebooted it twice and gone through the settings already",
    "did the restart thing already and the settings look normal",
]
DEVICE_CLUE = {  # second answer from a customer who does not know the model
    "iphone": "I get my apps from the App Store and it unlocks with Face ID",
    "android": "I get my apps from the Google Play Store",
    "windows": "it has the Windows key on the keyboard",
    "mac": "there's an Apple logo in the top-left corner of the screen",
}
USE_WHEN = {
    "wont_power_on": "the device shows no sign of power: no light, no sound, no vibration",
    "display_fault": "the device still makes sounds or vibrates, but the screen stays dark",
}
if HARD:
    SYMPTOMS["wont_power_on"] = {"platforms": ALL_PERSONAL, "triggers": None, "group": "screen"}
    SYMPTOMS["display_fault"] = {"platforms": ALL_PERSONAL, "triggers": None, "group": "screen"}
    # Customers describe what they observe, not the article name.
    SYMPTOM_SPECIFIC["wont_power_on"] = ["it's completely dead, no lights or sounds or buzzing",
                                         "nothing happens at all, no sound, no vibration"]
    SYMPTOM_SPECIFIC["display_fault"] = ["I can still hear notifications and it buzzes, the screen is just dark",
                                         "it rings and vibrates but the screen stays black"]
    SYMPTOM_VAGUE["screen"] = ["the screen is black", "it's just a black screen now"]
    SYMPTOM_TITLE["wont_power_on"] = "Black screen (no power)"
    SYMPTOM_TITLE["display_fault"] = "Black screen (display fault)"


@dataclass(frozen=True)
class Article:
    doc_id: str
    platform: str
    symptom: str
    trigger: str | None
    title: str
    body: str
    tier: str | None = None  # hard world: "basic" or "advanced"

    def attrs(self) -> dict:
        return {"device": self.platform, "symptom": self.symptom, "trigger": self.trigger}


def _hard_body(plat: str, sym: str, trig: str | None, tier: str) -> str:
    when = USE_WHEN.get(sym, SYMPTOM_TITLE[sym].lower()) + (f", {TRIGGER_TITLE[trig]}" if trig else "")
    parts = [f"Applies to {PLATFORM_TERMS[plat]}. Use this when: {when}."]
    if tier == "basic":
        parts.append("Basic checks, for customers who have not tried a restart or a settings check yet.")
    else:
        parts.append("Advanced fix, only for customers who already tried a restart and a settings check.")
    if sym in GATED:
        parts.append("Before you start: ask whether the device was dropped or got wet. "
                     "If yes, do not troubleshoot; escalate as physical_damage.")
    erase = tier == "advanced" and sym in DESTRUCTIVE
    if erase:
        parts.append(f"CAUTION: this fix erases the device. Ask whether the data is backed up before you resolve. "
                     f"If it is not, resolve with {BACKUP_DOC} (back up the device) instead.")
    steps = (f"restart the device, then check {SETTINGS[plat]}" if tier == "basic"
             else "back up, then erase and restore the device" if erase
             else f"reset the {sym.replace('_', ' ')} settings in {SETTINGS[plat]}")
    parts.append(f"Steps: {steps}.")
    return " ".join(parts)


def build_kb() -> list[Article]:
    """One article per (platform, symptom, trigger) cell, plus generic distractors."""
    if HARD:
        return _build_kb_hard()
    arts = []
    for sym, spec in SYMPTOMS.items():
        for plat, trig in itertools.product(spec["platforms"], spec["triggers"] or [None]):
            title = f"{PLATFORMS[plat]['label']}: {SYMPTOM_TITLE[sym]}" + (f" {TRIGGER_TITLE[trig]}" if trig else "")
            body = (
                f"Applies to {PLATFORM_TERMS[plat]}. Symptom: {SYMPTOM_TITLE[sym].lower()}"
                + (f", {TRIGGER_TITLE[trig]}" if trig else "")
                + f". Step 1: open {SETTINGS[plat]}. Step 2: follow the {sym.replace('_', ' ')} checks"
                + (f" for {trig.replace('_', ' ')}" if trig else "")
                + ". Step 3: restart and retest. If the issue persists, contact support."
            )
            arts.append(Article(f"KB-{len(arts) + 100}", plat, sym, trig, title, body))
    for i, t in enumerate(
        ["Set up voicemail", "Transfer contacts to a new phone", "Set up email on your device",
         "Change your Wi-Fi password", "Turn on two-factor authentication", "Back up your photos"]
    ):
        arts.append(Article(f"KB-{900 + i}", "any", "other", None, t, f"How to {t.lower()} on any device."))
    return arts


def _build_kb_hard() -> list[Article]:
    """Two tiers per cell (basic checks, advanced fix), bodies with gates, plus distractors."""
    arts = []
    for sym, spec in SYMPTOMS.items():
        for plat, trig in itertools.product(spec["platforms"], spec["triggers"] or [None]):
            base = f"{PLATFORMS[plat]['label']}: {SYMPTOM_TITLE[sym]}" + (f" {TRIGGER_TITLE[trig]}" if trig else "")
            for tier, suffix in (("basic", "basic checks"), ("advanced", "advanced fix")):
                arts.append(Article(f"KB-{len(arts) + 100}", plat, sym, trig, f"{base} ({suffix})",
                                    _hard_body(plat, sym, trig, tier), tier))
    for i, t in enumerate(
        ["Set up voicemail", "Transfer contacts to a new phone", "Set up email on your device",
         "Change your Wi-Fi password", "Turn on two-factor authentication", "Back up your device"]
    ):
        arts.append(Article(f"KB-{900 + i}", "any", "other", None, t, f"How to {t.lower()} on any device."))
    arts[-1] = Article(BACKUP_DOC, "any", "other", None, "Back up your device",
                       "Use this when a customer must back up before a reset or erase. "
                       "Steps: back up to the cloud or a computer, then confirm the backup finished.")
    return arts


@dataclass
class Case:
    """Hidden truth plus what the customer has revealed so far."""

    case_id: str
    platform: str | None  # None for escalation cases
    symptom: str | None
    trigger: str | None
    escalation: str | None = None
    revealed: dict = field(default_factory=dict)  # slot -> "specific" | "family" | "vague" | None
    extra: dict = field(default_factory=dict)  # hard world: tried, damaged, backed_up, vague_device

    def needs_trigger(self) -> bool:
        return bool(self.symptom and SYMPTOMS[self.symptom]["triggers"])

    def tier(self) -> str:
        return "advanced" if self.extra.get("tried") else "basic"


def candidates(case: Case, kb: list[Article]) -> list[Article]:
    """KB articles consistent with what the customer has revealed so far."""
    if case.escalation:
        return []
    out = []
    for a in kb:
        if a.symptom == "other":
            continue
        dev = case.revealed.get("device")
        if dev == "specific" and a.platform != case.platform:
            continue
        if dev == "family" and PLATFORMS[a.platform]["family"] != PLATFORMS[case.platform]["family"]:
            continue
        sym = case.revealed.get("symptom")
        if sym == "specific" and a.symptom != case.symptom:
            continue
        if sym == "vague" and SYMPTOMS[a.symptom]["group"] != SYMPTOMS[case.symptom]["group"]:
            continue
        if case.revealed.get("trigger") == "specific" and a.trigger != case.trigger:
            continue
        if a.tier and a.tier != case.tier():  # the opening always shows whether basic checks were tried
            continue
        out.append(a)
    return out


def discriminating_slots(cands: list[Article]) -> list[str]:
    """Slots whose answer would split the candidate set. Asking any of these is a useful question."""
    return [s for s in CORE_SLOTS if len({a.attrs()[s] for a in cands}) > 1]


def gold_article(case: Case, kb: list[Article]) -> Article | None:
    for a in kb:
        if (a.platform, a.symptom, a.trigger) == (case.platform, case.symptom, case.trigger) and (
            not a.tier or a.tier == case.tier()
        ):
            return a
    return None


def target_article(case: Case, kb: list[Article]) -> Article | None:
    """The article the agent should resolve with: the gold one, or the backup article before an erase."""
    gold = gold_article(case, kb)
    if gold and gold.tier == "advanced" and gold.symptom in DESTRUCTIVE and not case.extra.get("backed_up", True):
        return next(a for a in kb if a.doc_id == BACKUP_DOC)
    return gold
