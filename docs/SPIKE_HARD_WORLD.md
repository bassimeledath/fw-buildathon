# Spike: a harder but fair world (`ASE_WORLD=hard`)

**Why:** v1 is too easy. kimi-k3 solves 97% of test episodes, so there is little room left to show
a trained gain. Most of the v1 difficulty is in the system prompt rules, and real support work is
not like that: the procedure lives inside the KB articles.

**What:** five mechanisms. Each one copies a pattern we saw in real carrier support KBs (articles
and retrieval rules). All the text in this repo is our own and synthetic; no KB content is
included.

| # | Pattern in real KBs | Mechanism in the hard world | Why it is still fair |
|---|---|---|---|
| 1 | Articles open with probing questions, e.g. "Are there signs of physical damage?", before any steps | **Damage gate.** Articles for won't charge, the two black-screen symptoms and overheating start with "ask whether it was dropped or got wet; if yes, escalate". The customer never volunteers the damage (30% of gated cases). | The gate is written in the article text the agent retrieves (results now show article bodies). |
| 2 | Fixes are tiered: quick fixes, then "still not working?". A retrieval rule says to start from the most advanced step the customer *reports* they already tried | **Tiers.** Each situation has a basic-checks article and an advanced-fix article. The opening says "I already restarted it and checked the settings" in 40% of cases. The v1 filler line "I already tried turning it off and on" becomes a decoy: "my friend said to restart it, but I haven't yet". | The rule is in the system prompt, and the tier is always visible in the opening (checked by code). |
| 3 | "Before a factory reset, confirm the data is backed up" | **Backup before erase.** Advanced fixes for app crashing, battery drain and display fault erase the device. The agent must ask about a backup first. If there is none, the right resolve is the backup article (KB-905). | The caution and the backup doc id are written in the article. |
| 4 | Overlapping articles ("won't power on" vs "malfunctioning screen") and articles that say what to observe | **Black screen pair.** "The screen is black" fits two symptoms. The customer answers with observations ("it still buzzes, the screen is just dark"), not article names. | Each article's "Use this when" line describes the observation. Each observation phrase maps to exactly one symptom (checked by code). |
| 5 | Callers do not know their model | **Don't-know device.** In 40% of cases where the device is unclear, the first answer is "not sure what model it is". The second answer is a clue ("I get my apps from the Google Play Store"). | Every slot gives an informative answer within 2 asks (checked by code). |

Also: search returns the top 5 results with article text, not the top 3 titles. Search gets
credit for surfacing the right *situation* (either tier), and the tier is scored at resolve.

## Fairness checks (`ASE_WORLD=hard python -m ase.fairness`, all pass)
- The gates and cautions are stated in every gated and erase article.
- Each observation maps to one symptom, and each clue to one platform.
- The tier is decidable from the opening (0 mismatches on all splits).
- Every slot is answered within 2 asks.
- The oracle solves 100% of train, test and OOD cases, in at most 7 of the 12 steps allowed.
- Filler never claims the basic checks were tried.

## Results
| Policy (single-step reward, test states) | v1 | hard |
|---|---|---|
| oracle | 1.000 | 1.000 |
| resolve the top hit | 0.514 | 0.288 |
| always search the raw text | 0.350 | 0.287 |
| always ask | 0.299 | 0.304 |
| always escalate | 0.063 | 0.079 |

Episodes (test, temperature 0): see the table filled in below.

RESULTS_PLACEHOLDER

## Splits
- train: 649 cases / 2,094 states.
- test: 145 / 442.
- test_ood: 106 / 445.

OOD holds out bluetooth and overheating. Overheating is gated, so OOD tests whether the model
follows a gate on a symptom it never trained on.

## How to use
```bash
export ASE_WORLD=hard                  # every ase.* module switches: data_hard/, runs/hard_*, prompt, KB
python -m ase.build_dataset --n-cases 900 --seed 11
python -m ase.fairness
python -m ase.episodes --split test --model accounts/fireworks/models/kimi-k3
python -m ase.calibrate --split train --snapshot <base step-0> --k 8 --write-band   # -> data_hard/train_rl.jsonl
```
Without `ASE_WORLD`, everything is v1 and unchanged (sanity numbers identical).

## Known limits (still)
- Question wording is not scored, only the slot. The don't-know path does not check that the
  second question was rephrased.
- The customer is cooperative and truthful.
- The hard test set has only 6 hidden-damage and 7 backup-first cases. Grow the set or oversample
  these before claiming per-mechanism gains.
