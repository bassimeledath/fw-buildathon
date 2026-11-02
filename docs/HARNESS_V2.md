# Harness on hard_v2 (Aeri)

How far do prompting and small code checks get a cheap model on hard_v2, with no training? Same frozen cases,
scripted customer and grading as `ase/hard_v2.py`; only the agent side changes. Code: `ase/harness_v2.py`.

## Variants

| Flag | What it adds |
| --- | --- |
| `--prompt base` | The published system prompt (the baseline, re-run under the same conditions). |
| `--prompt h1` | A 10-rule call playbook appended to the published prompt (below). |
| `--guard` | Code rule: a resolve not preceded by a verify becomes a verify. |
| `--guard2` | Two checks from the visible call: (a) if the agent repeats a fix the customer already did (said in the opening, or "I already did that" / "it's already on" / "okay, done"), retry once with a nudge; (b) before `escalate fix_failed`, one self-check call: did the problem change (e.g. SOS), or are fixes left? |

No variant reads the hidden case or the answer key. On every oracle path in dev and test (471 steps), guard2 never
blocks a correct step.

**h1 rules:** hand off right away for damage, safety, billing or a request for a person; pick the current-software
article for the exact platform and problem (never the old-software, in-store or how-to copy); run fixes strictly in
order; skip fixes already done; give a step again if the customer couldn't do it; Android: ask the model if the brand
is unknown and switch Samsung/Pixel path if settings look different; consent before a reset; switch article when the
problem changes (e.g. SOS); verify before closing; `fix_failed` when the ladder runs out.

## Results so far (dev, 50 cases, deepseek-v4p1-flash, reasoning off, temperature 0)

| Harness | Dev success | Tokens / case | McNemar p vs base |
| --- | --- | --- | --- |
| base | 62% | 122k | |
| h1 | 66% | 111k | 0.754 |
| h1 + guard | 66% (guard never fired) | 108k | |
| h1 + guard2 | TODO | TODO | TODO |

13 of h1's 17 failures were repeating a fix the customer already did (8) and handing off instead of switching
articles after SOS (5); guard2 targets both. Tuning happens on dev only; test is run once at the end.

## Run

```bash
M=accounts/fireworks/models/deepseek-v4p1-flash
python -m ase.harness_v2 --split dev --model $M --prompt base
python -m ase.harness_v2 --split dev --model $M --prompt h1 --guard2
python -m ase.stats runs/hard_v2_harness_dev_base_*/logs.jsonl runs/hard_v2_harness_dev_h1-guard2_*/logs.jsonl
```

About $2.3 per 50-case run: every customer turn re-sends 8 long articles.
