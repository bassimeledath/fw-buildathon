"""The synthetic tech-support world: devices, symptoms, KB articles, escalation policy.

Every case is generated from hidden attributes (platform, symptom, trigger), so the
correct next action is known by construction. That is what makes the reward pure code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

SLOTS = ("device", "symptom", "trigger")

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


@dataclass(frozen=True)
class Article:
    doc_id: str
    platform: str
    symptom: str
    trigger: str | None
    title: str
    body: str

    def attrs(self) -> dict:
        return {"device": self.platform, "symptom": self.symptom, "trigger": self.trigger}


def build_kb() -> list[Article]:
    """One article per (platform, symptom, trigger) cell, plus generic distractors."""
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


@dataclass
class Case:
    """Hidden truth plus what the customer has revealed so far."""

    case_id: str
    platform: str | None  # None for escalation cases
    symptom: str | None
    trigger: str | None
    escalation: str | None = None
    revealed: dict = field(default_factory=dict)  # slot -> "specific" | "family" | "vague" | None

    def needs_trigger(self) -> bool:
        return bool(self.symptom and SYMPTOMS[self.symptom]["triggers"])


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
        out.append(a)
    return out


def discriminating_slots(cands: list[Article]) -> list[str]:
    """Slots whose answer would split the candidate set. Asking any of these is a useful question."""
    return [s for s in SLOTS if len({a.attrs()[s] for a in cands}) > 1]


def gold_article(case: Case, kb: list[Article]) -> Article | None:
    for a in kb:
        if (a.platform, a.symptom, a.trigger) == (case.platform, case.symptom, case.trigger):
            return a
    return None
