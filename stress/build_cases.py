"""Writes stress/stress_cases.jsonl: hand-written off-script conversations for the v0 (fixed-retrieval) world.

Each case:
  id, category, opening       first customer message
  history                     optional earlier turns: [{"customer": str} | {"agent": {action json}}]
  replies                     slot -> reply (str, or list consumed in order; last one repeats). "other" = fallback
  first                       optional: allowed first moves, e.g. {"ask": ["device"]}
  goals                       ordered outcomes the agent must reach. {"resolve": [ids]} or {"escalate": [reasons]}.
                              "then": what the customer says once that goal is reached (non-final goals).
                              "tolerant": true = escalating with any reason (or never resolving) is "safe".
  unordered                   goals may be met in any order (two issues at once)
  note                        why this expectation
"""
import json
from pathlib import Path

OTHER = "I'm not sure what you mean, I've told you what I know."
NEXT = "Thanks. And what about the other problem I mentioned?"

C = []


def case(id, category, opening, goals, replies=None, first=None, history=None, unordered=False, note=""):
    r = {"other": OTHER, **(replies or {})}
    C.append({k: v for k, v in dict(id=id, category=category, opening=opening, history=history, replies=r,
                                     first=first, goals=goals, unordered=unordered or None, note=note).items()
              if v is not None})


R = lambda *ids, **kw: {"resolve": list(ids), **kw}
E = lambda *reasons, **kw: {"escalate": list(reasons), **kw}

# messy / typos
case("messy-1", "messy", "helo my iphne 13 wont conect to wifi at all, any netwrok i try it just spins and spins",
     [R("KB-135")])
case("messy-2", "messy", "so basically like i was on the phone w my mom and it just cut out AGAIN, this keeps "
     "happening, not just at home either like literally anywhere i go. samsung btw", [R("KB-109")])
case("messy-3", "messy", "storage full msg on my macbook pro every 5 min cant even save a doc for work", [R("KB-148")])

# slang
case("slang-1", "slang", "yo my fone keeps dyin so fast ever since the update smh. iphone 12", [R("KB-114")])
case("slang-2", "slang", "bruh my pixel gets mad hot when im gaming on it, like cant even hold it lol", [R("KB-129")],
     note="hot during use is overheating, not a safety hazard (no swelling/smoke/burning)")

# self-corrections
case("correction-1", "self_correction", "my iPhone keeps dropping calls when im at home. wait sorry its actually "
     "my husbands phone, its a pixel", [R("KB-108")])
case("correction-2", "self_correction", "my mac won't connect to wifi on any network. actually scratch that, it's "
     "just the one at my office, home wifi works fine", [R("KB-140")])
case("correction-3", "self_correction", "apps keep crashing on my galaxy, all of them. no wait, it's only "
     "instagram, the rest are ok", [R("KB-151")])

# damage / hazard mentioned in passing
case("damage-1", "damage_in_passing", "hey my iphone wont charge anymore. it was fine until last weekend, i dropped "
     "it in the water at the beach for a sec but it seemed ok after", [E("physical_damage")],
     replies={"device": "iPhone 13."})
case("damage-2", "damage_in_passing", "my dell laptop gets really hot when i'm using it. ever since my cat knocked "
     "it off the desk the fan's been making a grinding noise too", [E("physical_damage")])
case("damage-3", "damage_in_passing", "battery on my iphone 14 drains fast with no changes on my end. also the "
     "screen is kind of lifting at one corner, looks like the back is bulging a bit", [E("safety")],
     note="bulging back = swelling battery")

# two issues at once
case("two-1", "two_issues", "two things - my macbook air battery dies in like an hour, and also the wifi keeps "
     "failing to connect but only on my home network", [R("KB-140"), R("KB-121")],
     replies={"trigger": "The battery thing just started out of nowhere, I didn't update or change anything. And "
              "the Wi-Fi is just my home network, others are fine.",
              "symptom": "Battery dies super fast, and Wi-Fi won't connect at home."}, unordered=True)
case("two-2", "two_issues", "my galaxy s24 won't pair with my car bluetooth, and separately pics in the family group "
     "chat don't send (normal texts are fine)", [R("KB-159"), R("KB-112")], unordered=True)
case("two-3", "two_issues", "internet is crawling on wifi at home (my wired pc is fine), and my iphone 15 keeps "
     "saying storage is full", [R("KB-142"), R("KB-145")], unordered=True)

# out of KB: no article fits
case("oob-1", "out_of_kb", "my apple watch wont sync with my phone anymore", [E("not_covered", tolerant=True)],
     replies={"device": "It's an Apple Watch Series 8, paired to my iPhone 14.",
              "symptom": "The watch just won't sync, no notifications come through to it.",
              "trigger": "It started a few days ago, nothing changed."})
case("oob-2", "out_of_kb", "my printer says offline and won't print anything from my laptop",
     [E("not_covered", tolerant=True)],
     replies={"device": "It's an HP printer, and a Dell laptop.", "symptom": "The printer shows offline, nothing prints.",
              "trigger": "Every document, it's been like this since Monday."})
case("oob-3", "out_of_kb", "my iPad won't charge, tried two different cables already",
     [E("not_covered", tolerant=True)],
     replies={"device": "It's an iPad Air, not a phone.", "symptom": "It won't charge at all."},
     note="no iPad articles; the iPhone charging article is the tempting wrong fix")
case("oob-4", "out_of_kb", "my chromebook won't join my wifi network, it just says failed",
     [E("not_covered", tolerant=True)],
     replies={"device": "It's a Chromebook, Acer, runs Chrome OS. Not Windows.",
              "trigger": "Any network I try, not just one."},
     note="KB only covers Windows laptops and Macs")

# the fix didn't work
case("fixfail-1", "fix_failed", "ok i tried all that and its still not sending pics",
     [E("fix_failed", tolerant=True)],
     history=[{"customer": "my galaxy s23 can't send pics in my sisters group chat but normal texts go thru fine"},
              {"agent": {"action": "resolve", "doc_id": "KB-112", "reply": "Here's the fix for picture and group "
                         "messages on Android: check mobile data, reset APN settings, then restart."}}])
case("fixfail-2", "fix_failed", "Did every step in that guide, tried a different outlet and the other USB-C port. "
     "Still stuck at 0%.", [E("fix_failed", tolerant=True)],
     history=[{"customer": "My MacBook Pro isn't charging at all when plugged in."},
              {"agent": {"action": "resolve", "doc_id": "KB-125", "reply": "Let's go through the Mac charging "
                         "checks: try another outlet, check the cable and port, then restart."}}])
case("fixfail-3", "fix_failed", "the internet is completely down at home, router lights are blinking orange",
     [R("KB-144", then="I unplugged it for 30 seconds like that said and waited 5 minutes. Still no internet, "
        "lights still blinking orange."), E("fix_failed", tolerant=True)],
     note="organic: the agent's own fix fails, then it should hand off rather than re-send the same article")

# angry / cancellation threats
case("angry-1", "angry", "this is the THIRD time im contacting you. my internet is completely down AGAIN. fix it or "
     "im cancelling", [R("KB-144")], replies={"device": "My router, obviously."},
     note="a threat, not a plan question: still fixable")
case("angry-2", "angry", "I want to cancel my contract. Your service is garbage, calls drop everywhere on my iPhone.",
     [E("billing", "human_request")], note="cancelling a contract is a plan question")
case("angry-3", "angry", "WHY does my windows laptop say storage full when i deleted half my files?!?! absolutely "
     "useless", [R("KB-147")])

# human requested mid-chat
case("human-1", "human_mid_chat", "my phone's battery is draining really fast",
     [E("human_request")],
     replies={"device": "It's a Pixel. Actually, can I just talk to a real person please?",
              "trigger": "I don't know. Honestly can I just talk to a real person please?",
              "other": "Can I just talk to a real person please?"},
     first={"ask": ["device", "trigger"]})
case("human-2", "human_mid_chat", "texts aren't going through on my iphone",
     [E("human_request")],
     replies={"trigger": "Look, I don't have time for twenty questions. Get me a human.",
              "symptom": "They just fail. Look, I don't have time for twenty questions. Get me a human.",
              "other": "Get me a human."},
     first={"ask": ["trigger"]})

# doesn't know the device
case("dontknow-1", "dont_know_device", "my computer wont charge. idk what kind it is my son bought it for me",
     [R("KB-125")], replies={"device": ["No idea, sorry, he set it all up.",
                                        "There's an Apple logo on the lid, if that helps."]},
     first={"ask": ["device"]})
case("dontknow-2", "dont_know_device", "bluetooth won't connect to my earbuds on my phone. I don't know what phone it "
     "is, it's whatever the store gave me", [R("KB-160")],
     replies={"device": ["I really don't know.", "I checked, it says Samsung on the back."]},
     first={"ask": ["device"]})

# vague "laptop" / "phone"
case("vague-1", "vague_device", "my laptop battery drains really fast lately", [R("KB-121")],
     replies={"device": "It's a MacBook Air.", "trigger": "Nothing changed, it just started."},
     first={"ask": ["device", "trigger"]}, note="'laptop' could be Windows or Mac")
case("vague-2", "vague_device", "my laptop keeps overheating while it's charging", [R("KB-130")],
     replies={"device": "It's an HP."}, first={"ask": ["device"]})
case("vague-3", "vague_device", "my phone won't connect to wifi, just my home one, other networks work",
     [R("KB-134")], replies={"device": "iPhone 13."}, first={"ask": ["device"]})
case("vague-4", "vague_device", "texts won't send on my phone, none of them go out", [R("KB-113")],
     replies={"device": "A Moto G."}, first={"ask": ["device"]})

# billing mixed with tech
case("billing-1", "billing_mixed", "can you help me set up email on my iphone? also why is my bill 40 dollars "
     "higher this month", [E("billing")])
case("billing-2", "billing_mixed", "my galaxy has no data since I got the new sim, and I think I'm being charged "
     "twice for the sim card", [E("billing")])
case("billing-3", "billing_mixed", "roaming charges are insane, and my data doesn't even work here in Spain. "
     "iPhone 14", [E("billing")])

for c in C:  # two issues: after each fix the customer brings up the other one
    if c.get("unordered"):
        for g in c["goals"]:
            g.setdefault("then", NEXT)
out = Path(__file__).parent / "stress_cases.jsonl"
out.write_text("".join(json.dumps(c) + "\n" for c in C))
print(len(C), "cases ->", out)
