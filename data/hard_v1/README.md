# hard_v1: the hard ASE benchmark (dev split)

A support agent talks to a scripted customer and makes one move per turn: `ask` about a slot, `resolve` with a KB
article, or `escalate` with a reason. Grading is exact code: each case has a hidden truth, the customer is seeded,
and an answer key works out the right next moves from what has been said. There is no LLM judge.

**Files** (`dev/`): `test_cases.jsonl` (68 cases, the dev split), `kb.jsonl` (28 articles), `system_prompt.txt`,
`manifest.json`. The held-out test split (102 cases) is not published. Generate it locally with `--split test` and
don't commit it. Final numbers are reported on test.

**What makes it hard** (patterns from real carrier support calls; all text is synthetic):

| Case type | Dev cases | What the agent has to get right |
|---|---|---|
| `no_article`, `no_article_after_clarify` | 12 + 8 | The KB has nothing for this exact situation, but search shows close near-matches. Hand off (`no_article`); don't use the neighbor. |
| `gap_with_gate` | 8 | Same, but the customer is on the affected phone with no backup, so the near-matches' "do this first" steps look applicable. |
| `prereq_bait` | 8 | The "do this first" gates apply, but trigger and attempts still decide the article. Find out before acting. |
| `basic_untried`, `advanced_tried` | 4 + 4 | What the customer already tried picks the basic or advanced article. |
| `tentative_guess`, `indirect_answer`, `later_correction`, `cant_identify` | 4 each | The customer guesses, answers sideways, corrects themselves, or can't say. |
| `human_midchat`, `policy_escalation` | 4 each | Asked for a person partway through; damage, safety, billing. |

**Fairness:** the oracle solves every case, every path the answer key allows ends correct (280 paths), and search
always shows the article the key needs. Run `python -m ase.hard_sol --oracle-only` to re-check.

**Baselines on the full set (170 cases, temperature 0):**

| Model | Reasoning | Success |
|---|---|---|
| deepseek-v4p1-flash | off | 45% (0% on all no-article types) |

That run predates the one-line definition of `no_article` and `cant_identify` that's now in the prompt. Re-run it
before quoting the number.

**Run it:**
```bash
set -a; source .env; set +a
python -m ase.hard_sol --split dev --model accounts/fireworks/models/deepseek-v4p1-flash --reasoning-effort none
```
Logs go to `runs/hard_sol_<model>/logs.jsonl`, with a per-case-type table printed at the end. Change the cases or
KB only through `ase/hard_sol.py`, and bump the folder (`hard_v2/`) when you do, so scores stay comparable.
