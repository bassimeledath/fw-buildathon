"""Harness variants on hard_v2 (ase/hard_v2.py), scored on its frozen dev/test cases, with latency per turn.

Bassim's prompt, cases, KB and scorer are untouched; variants append rules to hard_v2.SYSTEM. v2port carries the
v0 harness ideas over to hard_v2's moves: use the article that exactly fits (h2), and stop repeating fixes and hand
off once they've failed (h2nf).
  python -m ase.hard_v2_harness --model accounts/fireworks/models/gpt-oss-120b --reasoning-effort low --harness v2port
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

from . import hard_v2

RULES = {
    "v2port": (
        "\n- Use only the article for the customer's exact phone, problem and current software. Copies written for "
        "older software, in-store versions, the other platform's article and how-to pages do not fit, even if they "
        "look close."
        "\n- Never repeat a step the customer already did or says is already on. Once every step in the article, "
        "including its \"Still not working?\" section, has failed, escalate fix_failed."
    ),
    # step vs verify: gpt-oss says the right words but labels a re-given step as "verify" (22 of 40 dev failures)
    "v2rhythm": (
        "\n- Give each fix as a \"step\". If the customer can't find a menu, did something else, or couldn't finish "
        "it, give the same \"step\" again (same doc_id and step name) with clearer words. Use \"verify\" only right "
        "after the customer says they did the step, to check whether the problem is fixed. If they already said it "
        "didn't help, give the next step instead of verifying."
    ),
}
HARNESSES = ["base", *RULES]

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MOVES = {
    "ask_symptom": "It is not yet clear what is wrong with the phone.",
    "ask_model": "The next instruction depends on the phone model or Android brand, and the customer hasn't said it.",
    "ask_consent": "The next step needs the customer's OK first (it resets, erases or interrupts something) and they "
                   "haven't agreed yet.",
    "step": "Give the customer the next fix from the article that fits their problem, one step at a time.",
    "verify": "The customer just did a step; check whether the problem is fixed before moving on or closing.",
    "resolve": "The customer has confirmed the problem is fixed.",
    "escalate_fix_failed": "Every step in the article that fits has been tried and the problem is still not fixed.",
    "escalate_human_request": "The customer asked for a person.",
    "escalate_physical_damage": "The phone is physically or liquid damaged.",
    "escalate_safety": "A safety hazard: swelling battery, smoke, burning smell.",
    "escalate_billing": "A billing, plan or payment question.",
}
JEV_INSTRUCTIONS = ("You are deciding a phone-carrier support agent's next move on a call. The agent fixes the problem "
                    "by walking the customer through the knowledge-base article that fits, one step at a time, checks "
                    "whether it worked, closes when it's fixed, and hands off when the article can't fix it. Damage, "
                    "safety, billing and requests for a person go to a specialist.")


def _jev_state(msgs) -> list[str]:
    """Conversation for Jev: agent moves as short notes, search results kept only on the latest customer turn."""
    out, last = [], max(i for i, m in enumerate(msgs) if m["role"] == "user")
    for i, m in enumerate(msgs[1:], 1):
        if m["role"] == "assistant":
            try:
                a = json.loads(m["content"])
            except json.JSONDecodeError:
                a = {}
            out.append("AGENT: " + " ".join(f"{k}={a[k]}" for k in ("action", "slot", "doc_id", "step", "path", "reason")
                                             if a.get(k)))
        else:
            text = m["content"] if i == last else m["content"].split("\n\n[SEARCH RESULTS]")[0]
            out.append(text.replace("[CUSTOMER] ", "CUSTOMER: "))
    return out


class JevSplitPolicy:
    """Jev picks the move type (plus slot or reason); the LLM fills in the article, step, path and words."""

    def __init__(self, llm, jev_key: str, usage: dict):
        import httpx2

        self.llm, self.usage = llm, usage
        self.jev = httpx2.AsyncClient(timeout=120, headers={"Authorization": f"Bearer {jev_key}"})

    async def _decide(self, msgs) -> str:
        body = {"model": "jev-latest", "state": _jev_state(msgs),
                "questions": {"move": {"type": "choice", "instructions": JEV_INSTRUCTIONS, "criteria": JEV_MOVES}}}
        for attempt in range(8):
            r = await self.jev.post(JEV_URL, json=body)
            if r.status_code not in (429, 500, 502, 503, 529):
                r.raise_for_status()
                out = r.json()
                self.usage["jev_in"] += out.get("usage", {}).get("input_tokens", 0)
                return out["answers"]["move"]["choice"]
            await asyncio.sleep(min(30, 2 ** attempt))
        r.raise_for_status()

    async def __call__(self, msgs):
        move = await self._decide(msgs)
        kind, _, arg = move.partition("_")
        hint = {"ask": f'an "ask" with slot "{arg}"', "escalate": f'an "escalate" with reason "{arg}"'}.get(kind, f'a "{kind}"')
        sys_msg = {"role": "system", "content": msgs[0]["content"] + f"\n\nThe next move has been decided: output {hint} "
                   "action. Fill in its other fields."}
        text = await self.llm([sys_msg, *msgs[1:]])
        a = hard_v2.parse_action(text) or {}
        words = a.get("question") or a.get("say") or a.get("reply") or ""
        if kind == "ask":  # Jev's decision is binding; drop fields the LLM drafted for a different move
            a = {"action": "ask", "slot": arg, "question": words}
        elif kind == "escalate":
            a = {"action": "escalate", "reason": arg, "reply": words}
        elif kind == "verify":
            a = {"action": "verify", "say": words}
        else:
            a["action"] = kind
        self.usage["moves"][move] = self.usage["moves"].get(move, 0) + 1
        return json.dumps(a)


def load_cases(split: str) -> list:
    return [hard_v2.Case(**json.loads(l)) for l in open(f"data/hard_v2/{split}/test_cases.jsonl")]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--harness", default="base", choices=HARNESSES)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--run", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--jev", action="store_true", help="Jev picks the move type; --model fills in the rest")
    p.add_argument("--quick", action="store_true", help="first case of each type only (10 cases)")
    p.add_argument("--max-tokens", type=int, default=1024, help="raise for reasoning models that think at length")
    a = p.parse_args()

    prompt = hard_v2.SYSTEM + RULES.get(a.harness, "")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"],
                         max_retries=8)
    extra = {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {}
    lat, usage = [], [0, 0]

    async def llm(msgs):
        sys_text = prompt + msgs[0]["content"].removeprefix(hard_v2.SYSTEM)  # keeps anything a wrapper appended
        r = await client.chat.completions.create(model=a.model, messages=[{"role": "system", "content": sys_text}, *msgs[1:]],
                                                 temperature=0, max_tokens=a.max_tokens, **extra)
        if r.usage:
            usage[0] += r.usage.prompt_tokens
            usage[1] += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    jev_usage = {"jev_in": 0, "moves": {}}
    turn = JevSplitPolicy(llm, os.environ["JEV_API"], jev_usage) if a.jev else llm

    async def policy(msgs):  # one timing per turn (Jev + LLM when split), local to the call so cases can't clash
        t0 = time.perf_counter()
        text = await turn(msgs)
        lat.append(time.perf_counter() - t0)
        return text

    cases = load_cases(a.split)
    if a.quick:
        cases = list({c.kind: c for c in reversed(cases)}.values())

    async def go():
        sem = asyncio.Semaphore(a.concurrency)

        async def one(c):
            async with sem:
                for attempt in range(4):
                    try:
                        return await hard_v2.run(c, policy)
                    except Exception as ex:  # rate limits etc. are not model failures
                        err = repr(ex)[:150]
                        await asyncio.sleep(10 * (attempt + 1))
                return {"case_id": c.case_id, "kind": c.kind, "outcome": "api_error", "clean": False, "error": err}

        return await asyncio.gather(*[one(c) for c in cases])

    logs = asyncio.run(go())
    errors = sum(x["outcome"] == "api_error" for x in logs)
    logs = [x for x in logs if x["outcome"] != "api_error"]
    re = f"_re-{a.reasoning_effort}" if a.reasoning_effort else ""
    tag = ("jev+" if a.jev else "") + a.model.split("/")[-1]
    mt = f"_mt{a.max_tokens}" if a.max_tokens != 1024 else ""
    out = Path("runs") / f"hard_v2h_{a.split}{'q' if a.quick else ''}_{tag}{re}{mt}_{a.harness}_r{a.run}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs.jsonl").write_text("".join(json.dumps(x, default=str) + "\n" for x in logs))
    q = statistics.quantiles(lat, n=10) if len(lat) > 1 else [0] * 9
    meta = {"n": len(logs), "api_errors": errors, "prompt_tokens": usage[0], "completion_tokens": usage[1],
            "turns": len(lat), "p50_turn_s": statistics.median(lat) if lat else None, "p90_turn_s": q[8],
            **({"jev_input_tokens": jev_usage["jev_in"], "jev_moves": jev_usage["moves"]} if a.jev else {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    hard_v2.report(logs)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
