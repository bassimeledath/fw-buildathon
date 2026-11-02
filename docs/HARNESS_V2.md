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

**These numbers are from hard_v2 before 954d06e**, when every past turn kept its 8 search results in context. Since
954d06e only the latest results stay, which changes what the model sees. Re-run base, h1 and h1 + guard2 on the current
version before comparing; mixing versions is not a fair comparison.

| Harness | Dev success | Tokens / case | McNemar p vs base |
| --- | --- | --- | --- |
| base | 62% | 122k (old context) | |
| h1 | 66% | 111k (old context) | 0.754 |
| h1 + guard | 66% (guard never fired) | 108k (old context) | |
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

Before 954d06e a 50-case run cost about $2.3 because every turn re-sent all earlier search results; it should be much cheaper now.

## Harness + RL (experiment)

Bassim's RL run trains qwen3p8-27b under the published prompt. This variant trains under the h1 prompt instead, then
evaluates with h1 + guard2, to test whether prompt, code checks and RL stack. Same train states and answer key; only
the system message differs.

```bash
python -m ase.harness_v2 --prompt h1 --write-train      # -> data/hard_v2/train/{train,heldout}_states_h1.jsonl
PYTHONPATH=.:<fw-cookbook> python -m ase.train_serverless --env hard_v2 \
  --train data/hard_v2/train/train_states_h1.jsonl --heldout data/hard_v2/train/heldout_states_h1.jsonl \
  --run-dir runs/rl_v2_h1
PYTHONPATH=.:<fw-cookbook> python -m ase.harness_v2 --split dev --prompt h1 --guard2 --snapshot <final snapshot>
```
