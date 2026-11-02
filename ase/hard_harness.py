"""Harness variants on the hard_v1 benchmark (ase/hard_sol.py), without changing its prompt, cases or scorer.

The variants port the v0 harness rules (ase/harness.py h2 / h2nf) to hard_v1's action set, which already has
no_article and cant_identify escalations. Rules are appended to hard_sol.SYSTEM; "base" is the published prompt.
  python -m ase.hard_harness --model accounts/fireworks/models/deepseek-v4p1-flash --reasoning-effort none --harness hv2nf
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from . import hard_sol

RULES = {
    # port of h1+h2: one article fits, or ask about the detail that differs
    "hv2": "\n- Resolve only when exactly one shown article fits everything the customer has said: device, symptom, "
           "trigger, and what they already tried. If two or more could fit, ask about the detail where they differ. "
           "If an article names a condition the customer has not confirmed, ask about it first.",
    # port of h2nf: close is not a fit; hand off instead of using the neighbor; vague device means ask
    "nf": "\n- An article for a different phone, trigger or symptom does not fit, even if it looks close. Once you know "
          "the device, symptom and trigger and no shown article matches all of them, escalate no_article instead of "
          "using the closest one. If the customer only says \"my phone\" or a brand without the model, ask the device "
          "before resolving.",
}
HARNESSES = {"base": [], "hv2": ["hv2"], "hv2nf": ["hv2", "nf"]}


def system_prompt(harness: str) -> str:
    return hard_sol.SYSTEM + "".join(RULES[r] for r in HARNESSES[harness])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--harness", default="base", choices=list(HARNESSES))
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--run", type=int, default=1)
    a = p.parse_args()

    cases = hard_sol.split(hard_sol.sample_cases(10, 41), a.split)  # same cases as hard_sol's published dev
    prompt = system_prompt(a.harness)
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"],
                         max_retries=8)
    usage = [0, 0]

    async def policy(messages):
        extra = {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort is not None else {}
        r = await client.chat.completions.create(model=a.model, messages=[{"role": "system", "content": prompt},
                                                 *messages[1:]], temperature=0, max_tokens=4096, **extra)
        if r.usage:
            usage[0] += r.usage.prompt_tokens
            usage[1] += r.usage.completion_tokens
        return r.choices[0].message.content or ""

    logs = asyncio.run(hard_sol._gather(cases, policy))
    re = f"_re-{a.reasoning_effort}" if a.reasoning_effort is not None else ""
    out = Path("runs") / f"hard_harness_{a.split}_{a.model.split('/')[-1]}{re}_{a.harness}_r{a.run}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in logs))
    (out / "usage.json").write_text(json.dumps({"prompt_tokens": usage[0], "completion_tokens": usage[1], "n": len(logs)}))
    hard_sol.report(logs)
    print(f"logs: {out / 'logs.jsonl'}")


if __name__ == "__main__":
    main()
