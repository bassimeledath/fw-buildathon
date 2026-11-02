"""Harness variants: extra prompt rules layered on the default system prompt. The scorer is untouched.

Each variant adds to the one before (h1, then h1+h2, then h1+h2+h3), so a run shows what each rule adds.
  ASE_RETRIEVAL=fixed python -m ase.episodes --split dev --model <id> --harness h3
"""

from __future__ import annotations

import json

RULES = {
    # 1. Say what the slots mean. Read the trigger off the shown titles, not from a generic triage list.
    "h1": '- Slots: "device" is which device the customer has. "symptom" is what is wrong. "trigger" is what '
          "else separates articles with the same device and symptom; read it off the article titles (for example "
          '"on Wi-Fi only" vs "on wired and wireless", or "one app only" vs "all apps"). Only ask about a trigger '
          "that differs between the articles shown.",
    # 2. When to answer: one article fits, or ask.
    "h2": "- Resolve only when exactly one shown article fits everything the customer has said: device, symptom "
          "and trigger. If two or more could fit, ask about the slot where they differ. If the titles for this "
          "device and symptom name a trigger the customer has not confirmed, ask about the trigger first.",
    # 3. Don't loop when the customer can't add anything.
    "h3": '- If the customer answers "I\'m not sure" or "I already told you", do not ask about that slot again. '
          "Use what they said, ask about a different slot, or resolve if exactly one article now fits.",
}
ORDER = ["h1", "h2", "h3"]

# No-fit escalation: without it the agent has no valid move when no article fits (a device the KB does not
# cover) or the customer says the fix did not work, and a reasoning model can think until max_tokens and output
# nothing. Two extra escalate reasons. The scorer is unchanged: no v0 case expects them, so on v0 data an
# escalate with either reason on a fixable case is still `escalated_wrong`.
NO_FIT_REASONS = ("not_covered", "fix_failed")
NO_FIT = ('- Two more escalate reasons. "not_covered": the customer\'s device is not one of the five device types '
          "above (for example a watch, tablet, TV or printer), or the customer has clearly described a problem and "
          "has said which device, and no shown article for that device is about that kind of problem. "
          '"fix_failed": the customer says the article you already gave them did not work; do not resolve with the '
          "same article again. The shown articles are only the best matches for the words so far, so a vague "
          'message (only "my computer" or "my phone", or "it has problems", "something is off") can show no fitting '
          "article even though the knowledge base has one: then ask about the device or the symptom, do not escalate.")
VARIANTS = {"h2nf": ["h1", "h2", "nf"]}  # h2 plus the no-fit rule
HARNESSES = ["base"] + ORDER + list(VARIANTS)


def extra_rules(harness: str) -> str:
    if harness == "base":
        return ""
    keys = VARIANTS.get(harness) or ORDER[: ORDER.index(harness) + 1]
    return "\n" + "\n".join(NO_FIT if h == "nf" else RULES[h] for h in keys)


DECIDE_ONLY = ('\n- Output only the decision. Leave out the "question" and "reply" fields: another agent writes '
               "the wording for the customer.")

TALKER_PROMPT = """You write the support agent's next message to the customer. The decision is already made; do not change it.
Decision: {decision}
Write {what}. Plain text only, one or two short sentences, no JSON."""


def _what(d: dict) -> str:
    if d.get("action") == "ask":
        return f'one short question that asks the customer about their {d.get("slot")}'
    if d.get("action") == "resolve":
        return f'a short reply that walks the customer through article {d.get("doc_id")}'
    return "a short reply telling the customer you are connecting them to a specialist"


class DecideTalkPolicy:
    """Two models per turn: `decider` picks the action (short JSON), `talker` writes the customer-facing text.

    The scorer and the scripted customer only read the decision, so success comes from the decider alone; the
    split can change cost per case and transcript quality. Tokens are tallied per model for the cost table.
    """

    def __init__(self, decider, talker):
        self.decider, self.talker = decider, talker
        self.usage = {"decider": [0, 0], "talker": [0, 0]}

    async def __call__(self, messages, gt):
        sys_msg = {**messages[0], "content": messages[0]["content"] + DECIDE_ONLY}
        text, lat, pt, ct = await self.decider([sys_msg] + messages[1:], gt)
        self.usage["decider"][0] += pt
        self.usage["decider"][1] += ct
        from .env import parse_action

        d = parse_action(text)
        if d is None:
            return text, lat, pt, ct  # unparseable decision: let the scorer mark it invalid
        talk = [{"role": "system", "content": TALKER_PROMPT.format(decision=json.dumps(d), what=_what(d))}] + messages[1:]
        words, lat2, pt2, ct2 = await self.talker(talk, gt)
        self.usage["talker"][0] += pt2
        self.usage["talker"][1] += ct2
        d["question" if d.get("action") == "ask" else "reply"] = words.split("</think>")[-1].strip()
        return json.dumps(d), lat + lat2, pt + pt2, ct + ct2


JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_NEXT = {
    "ask_device": "Two or more shown articles could fit and they differ by device type (iPhone, Android phone, Windows "
                  "laptop, Mac, home router); the customer has not said which device.",
    "ask_symptom": "Two or more shown articles could fit and they differ by symptom (what is wrong); the customer "
                   "described the problem only vaguely.",
    "ask_trigger": "The shown articles that fit share device and symptom but differ by trigger, for example 'on Wi-Fi "
                   "only' vs 'on wired and wireless' or 'one app only' vs 'all apps'; the customer has not said which.",
    "resolve": "Exactly one shown article fits everything the customer has said: device, symptom and trigger.",
    "escalate_physical_damage": "The device was dropped, cracked, or got wet or liquid damage.",
    "escalate_safety": "A safety hazard: swelling battery, smoke, burning smell.",
    "escalate_billing": "A billing, plan or payment question.",
    "escalate_human_request": "The customer asks for a human or a real person.",
}
JEV_INSTRUCTIONS = ("You are deciding a tech-support agent's next move. Each customer message is followed by the top "
                    "knowledge-base articles for what the customer has said. Escalate damage, safety, billing and "
                    "requests for a human. Otherwise resolve only when exactly one article fits; if several could "
                    "fit, ask about the one thing that differs between them. Do not ask about something the customer "
                    "already answered.")


def _jev_state(messages) -> list[str]:
    """Conversation as plain lines (no system prompt). Earlier agent moves become short descriptions."""
    out = []
    for m in messages[1:]:
        if m["role"] == "assistant":
            a = json.loads(m["content"])
            out.append(f"AGENT: {a.get('action')} " + " ".join(f"{k}={a[k]}" for k in ("slot", "doc_id", "reason") if k in a))
        else:
            out.append(m["content"].replace("[CUSTOMER] ", "CUSTOMER: ").replace("[SEARCH RESULTS]", "ARTICLES SHOWN:"))
    return out


def _shown_articles(messages) -> dict[str, str]:
    for m in reversed(messages):
        if m["role"] == "user" and "[SEARCH RESULTS]" in m["content"]:
            lines = m["content"].split("[SEARCH RESULTS]")[1].strip().splitlines()
            return {l[2:].split(":", 1)[0]: l.split(":", 1)[1].strip() for l in lines if l.startswith("- KB-")}
    return {}


class JevPolicy:
    """TypeSafe Jev as the decider: a choice over next moves, plus a choice over shown articles for resolve."""

    def __init__(self, api_key: str, model: str = "jev-latest"):
        import httpx2

        self.client = httpx2.AsyncClient(timeout=60, headers={"Authorization": f"Bearer {api_key}"})
        self.model = model

    async def _post(self, body):
        import asyncio

        for attempt in range(8):  # back off on rate limits and transient server errors
            r = await self.client.post(JEV_URL, json=body)
            if r.status_code not in (429, 500, 502, 503, 529):
                r.raise_for_status()
                return r.json()
            await asyncio.sleep(min(30, 2 ** attempt))
        r.raise_for_status()

    async def __call__(self, messages, gt):
        import time

        t0 = time.perf_counter()
        shown = _shown_articles(messages)
        questions = {"next": {"type": "choice", "instructions": JEV_INSTRUCTIONS, "criteria": JEV_NEXT}}
        if shown:
            questions["article"] = {"type": "choice", "criteria": shown, "instructions":
                                    "Which shown article matches the customer's exact device, symptom and trigger?"}
        out = await self._post({"model": self.model, "state": _jev_state(messages), "questions": questions})
        nxt = out["answers"]["next"]["choice"]
        if nxt.startswith("ask_"):
            d = {"action": "ask", "slot": nxt[4:]}
        elif nxt == "resolve" and shown:
            d = {"action": "resolve", "doc_id": out["answers"]["article"]["choice"]}
        elif nxt.startswith("escalate_"):
            d = {"action": "escalate", "reason": nxt[9:]}
        else:
            d = {"action": "ask", "slot": "symptom"}  # resolve with nothing shown: fall back to a question
        u = out.get("usage", {})
        return json.dumps(d), time.perf_counter() - t0, u.get("input_tokens", 0), u.get("output_tokens", 0)


class HarnessPolicy:
    """Wraps a policy and appends the variant's rules to the system message before each call."""

    def __init__(self, inner, harness: str):
        self.inner, self.extra = inner, extra_rules(harness)

    async def __call__(self, messages, gt):
        if self.extra:
            messages = [{**messages[0], "content": messages[0]["content"] + self.extra}] + messages[1:]
        return await self.inner(messages, gt)
