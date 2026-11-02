# The story, in the judge's order

The task is **v0** (README "Modes"): retrieval is fixed, and the model picks resolve, ask or
escalate. Harness engineering and RL compete on the same locked v0 test set.

This follows Sinan's framework from the post-training deck: diagnosis first, technique as a
consequence of the diagnosis, cost per task last. Each beat lists what we say, the evidence, and
the gap still to close on the day.

## 1. The failure, and why it is intrinsic
**Say:** "Our support agent searched, got back the iPhone article and the Android article for the
same symptom, and answered with the first one without asking which phone. Every fact it needed was
in the window: the customer turn and the search results. It had the context and still failed.
That makes it an intrinsic failure."

**Evidence:**
- The search-once-then-answer-the-top-hit policy scores 0.51.
- gpt-oss-120b asks 1.95 questions per case, and 1.30 of them are wasted.
- The canonical cases are states where the results show siblings and the model resolves anyway.

**On the day:** pull 3 transcripts like this from the base model. Show the search results in the
window next to the bad action.

## 2. The four pre-checks, named and dismissed
| Check | Our answer |
|---|---|
| An eval you trust? | Two scorers. A single-step verifier that is pure code (it separates the oracle at 1.00 from five bad policies at 0.06–0.51). A separate multi-turn episode eval against a scripted customer. Categorized, held out, plus an OOD split with unseen symptoms. |
| A knowledge gap? | No. The facts arrive through search, and the KB is in the loop, so this is not knowledge injection. |
| Schema errors? | Run baselines with structured outputs (JSON schema `response_format`), so format failures cannot be what training fixes. **TODO** |
| Tried a better prompt? | Build the strongest prompt baseline: few-shot examples, plus an explicit rule ("if the results show the same symptom for different devices or triggers, ask"). That prompt is the baseline every trained number has to beat. **TODO** |

**Pre-empt "just put the KB catalog in the prompt":** a production KB has thousands of articles,
so retrieval is the harness. The skill under test is judging whether retrieved results are enough.
Run the catalog-in-prompt variant once as an upper bound and say what it costs in tokens.

## 3. The base pass curve, and which quadrant we land in
Tooling is ready: `ase.calibrate --snapshot <step-0 base> --k 8` then `ase.passk`.

**Proxy evidence so far** (nemotron-lightning, train split, k=8, temperature 1):
- pass@1 0.66, pass@8 0.90, pass^8 0.28. The reliability gap is 0.61.
- That puts it in the **stochastic, ceiling-high** quadrant in every category. The ceiling runs
  from 0.76 (search after results) to 0.99 (escalate).
- His verdict for that quadrant: **consolidate consistency.** The model can do this; it just
  doesn't do it every time.

**On the day:** re-run on the actual base model (qwen3p8-27b step-0) and on the frontier
baselines. Use the held-out split, and break it out per category, with first turn and after search
reported separately.

## 4. Technique as a consequence of the quadrant
**Say:** "Stochastic with a high ceiling, so we consolidate. We tried the two consolidation tools
head to head."
- **Rejection-sampling SFT (self-distillation):**
  - sample the base model 8 times on train states only;
  - keep the full-credit draws;
  - run plain SFT (cookbook `sft_loop`, `serverless=True`).
  - It is cheap, and it is his named alternative to RL.
- **GRPO** (`ase.train_serverless`): the same reward and the same train states.

**Low-ceiling pockets:** if the base curve shows any category with a ceiling under 0.5 (likely
candidate: OOD vague-symptom cases, where the model never asks for the symptom), that pocket gets
**import, then consolidate.** Run SFT on kimi-k3 teacher trajectories for that category, then
re-measure, because importing moves the ceiling. That is his slide 10, in our domain.

**Show the knobs we searched:** LoRA rank 32 vs 64 and lr 1e-4 vs 5e-5, at about 40 s per step.
Include the combinations that did not work.

## 5. The same pass curve after training
Report pass@1, pass@8 **and pass^8** on the held-out split and the OOD split, before and after.
The headline is consistency: "pass^8 went from X to Y", not just solve rate.

**Overnight smoke run (stale data, small eval):**
- held-out single-step reward 0.734 → 0.875 at step 4;
- wasted questions halved in the episode eval;
- OOD wrong fixes went up. Half of that was a data bug, since fixed.

**Generalization (James's 16→32 test):** train on 10 symptoms, test on 2 unseen symptom families.
That split already exists.

**On the day:** grow the held-out set to about 300 cases (it is synthetic, so it's free) and report
run-to-run variance (2 seeds), so 1–2 case differences are not sold as wins.

## 6. Routing pre-empt, cost per task and break-even
**Why not just route?** Measured on the v1 test states, single-step pass@1:

| | pass@1 |
|---|---|
| Best single model (kimi-k3) | 0.81 |
| Per-draw oracle routing across 4 models | 0.96. Hindsight only, and it routes noise, since the failures are stochastic. |
| Per-category oracle routing | 0.87. Also hindsight: it routes on the correct-action label, which is not observable at inference. |

So the bar is **0.87**. A single trained model at or above 0.87 beats the best router anyone could
build here, at one model's latency. If we land below it, say so and show the router.

**Cost per resolved case:** tokens per episode × price. The baselines' tokens and latency are
already in `runs/episodes_*`. Compare kimi-k3 serverless against the LoRA on qwen3p8-27b on an
on-demand deployment ($/hr). Compute break-even in requests per hour, the way he does it, and say
the bill out loud.

## Weaknesses we name ourselves (he credits this)
- The question text is unscored. Only the slot is checked, so "Can you tell me more about the
  device?" scores the same as a sharp question.
- The v0 baseline fixes retrieval (the env searches the customer's words) so the skill under test
  is the decision alone. In v1, where the model writes the query, BM25 rewards keyword overlap
  that a real search engine would not.
- The labels come from a synthetic one-article-per-situation world. Real KBs overlap, so the
  oracle is cleaner than reality.
- At the first turn, both ask and search are allowed and equally rewarded. That is a policy choice
  we made, not ground truth.
- The scripted customer is cooperative. The research says sims that are too polite inflate scores;
  a rude or evasive persona is a stretch goal.

## Things to cut if time is short
DPO, the consult action, and the persona paraphrase. Keep the diagnosis, RS-SFT vs GRPO, the
pass curves, the router bar and the cost slide.

# Mapped to the organizers' rubric (from their sample submission)
| Criterion | Weight | What we show | Where we would lose points / TODO |
|---|---|---|---|
| Benchmark evaluation | 40% | Same open model before and after (qwen3p8-27b base vs LoRA). Locked held-out set at temperature 0 with identical prompts. McNemar plus case-clustered bootstrap (`ase/stats.py`). An OOD split. Harness alternatives scored on the same set, losses reported. | Synthetic labels (their sample prized a published dataset). One seed so far. The test set is only 109 cases and needs about 300 cases / 1,000 states. |
| Business value | 25% | A named user: carrier and device tech support, where wrong fixes and wasted questions cost handle time and repeat contacts. Value math with the assumptions written out. | **TODO:** public numbers (BLS customer-service reps, cost per contact, repeat-contact rate), each assumption labeled. Evidence that frontier models over-answer: our baselines plus the papers in `PLAN.md`. |
| Technical execution | 20% | Code reward, oracle labels, multi-turn eval, serverless GRPO, snapshot sampling. The data bug we found and fixed (the "not charging" phrase). | Hyperparameter sweep, serving latency and cost (deploy the LoRA). |
| Presentation & demo | 10% | A 60 s video: base model resolves with the wrong-device article; the trained model asks "iPhone or Android?" and resolves right; then the McNemar/bootstrap chart. | Record the video by 3:00. |
| Documentation | 5% | README with run order and the disclosure. | Pin dependencies. **Make the repo public** before submitting and check it in an incognito window. |

## The harness ideas to run on the same locked set (report the losses)
1. **A: strong prompt.** Few-shot examples, plus the rule "if results show the same symptom for
   different devices or triggers, ask first." Also run it with structured JSON output.
2. **B: A, 5 samples at temperature 0.7, majority vote.** About 5× the cost.
3. **C: KB catalog in the prompt** (all 71 titles). The upper-bound harness; report its token cost.
4. **Fine-tunes:** rejection-sampling SFT, and GRPO (`ase.train_serverless`).

## Overnight stats (smoke run, old data, temperature 0; shows the method, not the final claim)

| Split | Comparison | Accuracy | McNemar p | 95% CI | Verdict |
|---|---|---|---|---|---|
| test (n=109) | base vs step 7 | 95.4% → 97.2% | 0.69 | [−2.8, +6.4] | Indistinguishable: the base is at ceiling. |
| OOD (n=92) | base vs step 4 | 72.8% → 87.0% (+14.1) | 0.002 | [+6.5, +22.8] | Real gain. |

On the OOD comparison, the trained model won 15 of the 17 cases where the two disagreed.
Caveat: wrong fixes also rose there, and part of the base's failures came from the data bug that
has since been fixed.

## Day timeline (organizers' suggestion, adapted)
| Time | Step |
|---|---|
| 10:30 | Regenerate a bigger test set (about 300 cases). Lock it. Run baselines on base and frontier at temperature 0, with one saved row per case. |
| 11:15 | Base pass curve at k=8, which gives the quadrant. Calibrate the train band with the base snapshot. |
| By 12:00 | Launch GRPO in the background (and RS-SFT). |
| 12:00 | Run harnesses A, B and C on the locked set while training runs. Draft the business case. |
| 1:30 | Score everything on the same set. Run `ase.stats`. Make the charts. Deploy the best LoRA for latency and cost. |
| 2:30 | Write the README, record the video, make the repo public. |
| 3:30 | Submit. |
