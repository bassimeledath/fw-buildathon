# hard_v2: real-call shape

Most of a real support call happens after the agent has found the article. The agent walks the customer through
it, the fixes fail, the customer gets lost in the menus, and the agent has to know when to stop. hard_v2 grades that
part. The agent picks the article and then runs it step by step with a scripted customer. Grading is exact code
(hidden cause, seeded customer, an answer key from what has been said), with no LLM judge. All text is synthetic.

**Moves:** `ask` (symptom / model / consent), `step` (doc_id, the fix's bold heading, Samsung or Pixel path),
`verify`, `resolve`, `escalate` (`fix_failed`, `human_request`, …). A wrong step, article, check or close fails the
case. A needless question counts as wasted.

**Files:** `dev/` (50 cases) and `test/` (70 cases). Each has `test_cases.jsonl`, `kb.jsonl` (84 articles) and
`system_prompt.txt`. Report final numbers on `test/`.

**What's in it:**
- **Real-length articles in prose.** About 2k characters: bold sections, a "Still not working?" section, Samsung and
  other-Android lines, a branch to another article, and notes about older models.
- **8 search results per turn.** They include a copy written for old software, an in-store copy, the other platform's
  version and how-to pages.
- **8 problems** that sit close to each other: mobile data vs no service, texts vs picture messages, dropped calls vs
  Wi-Fi calling, Wi-Fi, hotspot.

| Case type | What happens on the call |
|---|---|
| `late_fix`, `nothing_works` | The first fixes fail; only a later one works, or none does (hand off with `fix_failed`). |
| `decoy` | A step looks like it worked ("the LTE icon came back?") but didn't. Check before closing. |
| `menu_label` | "I don't see Cellular, there's Mobile Service?" Give the same step again; don't skip it. |
| `brand_path` | An Android phone of unknown brand: give the Samsung or Pixel path, or ask. |
| `slip` | The customer does the wrong thing ("I turned off Wi-Fi instead"). The step isn't done. |
| `late_tried`, `already_on` | "I already did that this morning", "it's already on". Move on, and don't repeat what was tried. |
| `sos_switch` | It turns out to be a different problem halfway ("it says SOS"). Switch to the article the KB points to. |
| `human_midcall` | The customer asks for a person partway through. |

**Fairness:** every path the answer key allows ends correct (735 paths, 0 bad, calls up to 14 turns). Re-check with
`python -m ase.hard_v2 --oracle-only`.

**Run one case and watch it:**
```bash
set -a; source .env; set +a
python -m ase.hard_v2 --case sos_switch --model accounts/fireworks/models/deepseek-v4p1-flash --reasoning-effort none
python -m ase.hard_v2 --quick 1 --model accounts/fireworks/models/deepseek-v4p1-flash --reasoning-effort none   # one case per type
```

**Baselines on `test/`** (70 cases × 8 draws, temperature 0.7, whole calls; `python -m ase.hard_v2_passk`):

| Model | pass@1 | pass@8 | pass^8 | Quadrant |
|---|---|---|---|---|
| deepseek-v4p1-flash, reasoning off | 48.2% | 71.4% | 27.1% | stochastic, high ceiling |
| qwen3p8-27b base, thinking off | 44.8% | 81.4% | 7.1% | stochastic, high ceiling |

Only the latest search results stay in context (older ones are dropped), so long calls fit an 8k window. Numbers from
before that change (62% dev / 64% test, at temperature 0) aren't comparable.

**Training:** `train/` has 2,273 states from 300 cases on seed 101 (disjoint from dev/test, seed 7). The rollouts
include deliberate agent mistakes so the model learns to recover. Per-turn reward: 1 if the move is allowed by the
answer key. Run it with `python -m ase.train_serverless --env hard_v2 --train data/hard_v2/train/train_states.jsonl
--heldout data/hard_v2/train/heldout_states.jsonl --max-rows 300 --max-seq-len 12288`. Our 7-step run took held-out
per-turn accuracy from 67% to 91%. The whole-call pass@8 after training hasn't been measured yet.
