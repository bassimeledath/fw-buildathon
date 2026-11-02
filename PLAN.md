# Ask, Search, or Escalate (ASE): RL for the first moves of a support call

## The pitch
Out of the box, LLM support agents make one shot at an answer: they search once and answer with the
top hit. They rarely ask the one question that would decide it, and they don't notice when search
results are for the wrong device. We train a small open model to choose the right next move: **search**
(with a good query), **ask** (the question that splits the candidates), **resolve** (only with the
exact article) or **escalate** (damage, safety, billing, wants a human). Every reward is checked in code.
We show it matches or beats frontier models on outcome while costing a fraction per call.

## Data shape (synthetic, no proprietary data)
- **World:** 5 device platforms × 12 symptoms × trigger variants, giving 65 KB articles plus 6
  distractors (`ase/world.py`). Siblings differ only by device or trigger, which is what makes vague
  calls ambiguous.
- **Case:** hidden truth (platform, symptom, trigger) plus how much the opening message reveals: a
  specific device, just the family ("my phone") or nothing; a specific symptom or a vague one
  ("internet problems"); the trigger or not. 20% are must-escalate.
- **Decision state:** the conversation so far (customer turns, the agent's earlier actions, search
  results), labelled by an oracle with the set of correct actions.

```
[CUSTOMER] Hi, the internet stuff isn't working right. Nothing changed, it just started.
{"action":"ask","slot":"device",...}
[CUSTOMER] It's a Samsung.
{"action":"search","query":"Android phone no change"}
[SEARCH RESULTS] - KB-105: Android phone: No mobile data with no recent changes | ...
GT: allowed = {resolve: KB-105}
```

Splits: train has 399 cases and 1,134 states, test 109 and 298, and test_ood 92 and 316. The OOD
split uses 2 symptoms never seen in training.

## Reward (`ase/reward.py`): gates multiply, credits add, bounded 0..1
- **Gates (score 0):** an unparseable or malformed action, a query over 20 words (anti-shotgun),
  resolving when it shouldn't, escalating a fixable case, or troubleshooting a must-escalate case.
- **search:** the gold article's rank in a frozen BM25 index: 1.0 for rank 1, 0.8 for rank 2, 0.7 for
  rank 3, else 0.2. For ambiguous cases, it scores by how many candidate articles land in the top 3.
- **ask:** 1.0 if the slot actually splits the candidate set, 0.2 if not. The question text is left
  unscored.
- **resolve:** exact doc_id. **escalate:** 1.0 for the right reason, 0.7 for the wrong reason.
- **Wrong but harmless move type:** 0.1.

**Sanity check** (`python -m ase.sanity`, test split; mean reward):

| Policy | Mean reward |
|---|---|
| Oracle | 1.00 |
| Search once, resolve with the top hit (the one-shot bias) | 0.51 |
| Always search with the raw message | 0.35 |
| Always ask | 0.30 |
| Always escalate | 0.06 |

The reward separates the oracle from all the bad policies.

## Eval: a separate, stricter, multi-turn scorer (`ase/episodes.py`)
Full episodes against a scripted customer, up to 8 steps. It is never used as the reward. Metrics:
- success, fix rate and **wrong-fix rate** (the expensive failure);
- false and missed escalations;
- customer questions per case and wasted questions per case;
- searches per case, p50 step latency and tokens per case.

The oracle scores 100% success with 0.81 questions per case.

**Trip wires to log during training:** query length, ask rate, distinct actions per batch and
escalate rate. These catch abstention gaming and gate avoidance.

## Pre-registered success (fill the numbers in after the baselines run)
On held-out episodes, the RL'd model:
- has a success rate at or above the best frontier baseline;
- has a wrong-fix rate at most half its base model's;
- asks no more than 0.3 questions per case above the oracle;
- has no rise in missed escalations;
- costs at least 5× less per case than the frontier baseline.

## Day plan (build window is 10:30 to 4:30)
| Time | Step |
|---|---|
| 10:30 | Keys and cookbook installed. Confirm the serverless training model ids and tokenizer. |
| 11:00 | Baselines. Run `ase.calibrate --k 1` and `ase.episodes` on the base model and 2-3 frontier models (e.g. Kimi K3, DeepSeek V4.1, gpt-oss-120b). |
| 11:30 | Calibrate the base model at k=8, then `--write-band` to `data/train_rl.jsonl`. |
| 12:00 | RL run: `ase.train_fw`, LoRA, 8 completions, about 16 groups per step. Eval checkpoints on the test episodes as they land. |
| 2:30 | Deploy the best checkpoint (live merge) to get real latency numbers. Run OOD eval. Read 20 samples at the start, middle and end of training. |
| 3:30 | Charts: success against cost per case (Pareto), wrong-fix rate, questions per case, one before/after transcript. |

Team split:
- **A:** owns the RL run.
- **B:** baselines and evals.
- **C:** data realism (LLM persona paraphrase) and the demo.
- **D:** the "consult" stretch goal (below).

## Stretch: consult-or-act (fast model overcalling the slow model)
Add a `consult` action that asks a big model for its recommended action, at a cost.
- Episode reward = success − λ·consults. Sweep λ to draw a cost-quality frontier.
- The question: can the small model learn when it needs the expert?

## Other ideas from support-agent work (same verifiable-core recipe)
1. **Handoff summary faithfulness:** a transfer call with a schema-valid args object, every required
   field present, and every fact in the summary drawn from the transcript. Models are known to invent
   tool args. Exact-match slot scoring with no judge.
2. **KB section selector:** one small model picks the KB sections that hold the answer, replacing
   several parallel LLM filter calls on the lookup path. Reward = recall of the gold section with
   a precision penalty. A clean latency story.
3. **Intent router with calibrated abstain:** a public analog is CLINC150 or BANKING77 with an
   out-of-scope class. The reward penalizes confident wrong routes more than asking. High traffic, so
   cost matters. Plain SFT may already get most of the way.
4. **Spoken-entity capture for voice:** phone numbers, IMEIs and emails spelled out loud, with
   self-corrections ("five five... no, six"). Exact-match reward. Synthetic data is easy.
5. **Realistic or adversarial customer simulator:** the research says RL against a role-play judge
   gives little, and fine-tuning on human text works better. SFT on public customer-support tweets
   first. The checkable part: the sim never leaks hidden facts before it's asked and never contradicts
   them. Realism is scored by a frozen real-vs-sim classifier as a capped residual.
6. **Compliant upsell timing:** offer only after the issue is resolved, with the exact price and at
   most 2 rebuttals. Rule gates. Watch for gate avoidance (never offering).

## Risks
- **Hackathon rules:** check whether pre-built code is allowed. Data prep is explicitly allowed.
- **Too easy for big models:** if frontier models hit about 95% single-step, make it harder before
  training. Options: LLM-paraphrased openings, more trigger variants, results that mix device
  families. Then re-run calibration.
- **Fireworks managed RFT is deprecated.** We use the Training API through the cookbook's
  `async_rl_loop`. Model ids have to be verified on the day. Trained LoRAs need an on-demand
  deployment to serve, but checkpoints can be sampled without one for evals.

## Results from the overnight familiarization run (Nov 2, ~1am)
**Setup.**
- **Model and data:** `qwen3p8-27b` on serverless LoRA (rank 32, lr 1e-4), trained on the 192 cases from `train_rl.jsonl`.
  These are cases where the proxy model nemotron-lightning, sampled 8 times, sometimes passed and sometimes failed.
- **Batch:** 8 groups per step, 8 completions each.
- **Steps:** only 51 of the 192 groups had any reward spread, so the run made 7 optimizer steps.
- **Wall-clock:** about 5 minutes. Cost was a few dollars (estimated from token counts, not a billing check).

**Single-step held-out reward** (64 states × 4 samples, temperature 1):

| Step | Overall | Search | Resolve | Ask |
|---|---|---|---|---|
| 0 | 0.734 | 0.53 | 0.74 | 0.89 |
| 4 | 0.875 | 0.74 | 0.94 | 0.93 |
| 7 | 0.862 | 0.69 | 0.94 | 1.00 |

Train reward went from 0.72 to 0.97, so the gap between train and held-out opened up by step 7.

**Multi-turn episodes** (temperature 0):

| | Success | Wrong fix | Questions / case | Wasted questions | p50 step |
|---|---|---|---|---|---|
| Base, test (n=109) | 95.4% | 2.4% | 1.14 | 0.39 | 1.8 s |
| RL step 7, test | 97.2% | 1.2% | 0.83 | 0.19 | 1.75 s |
| Base, OOD symptoms (n=92) | 72.8% | 3.3% | 1.38 | 0.42 | |
| RL step 4, OOD | 87.0% | 9.8% | 1.10 | 0.18 | |
| RL step 7, OOD | 80.4% | 8.7% | 1.34 | 0.39 | |
| kimi-k3, test | 97.2% | 2.4% | 0.93 | 0.20 | 2.5 s |
| deepseek-v4p1-flash, test | 92.7% | 4.8% | 0.95 | 0.25 | 1.7 s |
| gpt-oss-120b, test | 89.9% | 7.2% | 1.95 | 1.30 | 0.8 s |
| nemotron-lightning, test | 82.6% | 2.4% | 1.21 | 0.53 | 1.8 s |

**Read.**
- RL halved wasted questions on test and lifted OOD success by up to 14 points.
- But OOD wrong fixes roughly tripled: the model learned to resolve more aggressively. That is the
  costly failure, and it has to be fixed before the demo.
- Test success differences are 1–2 cases at n=109, so they are noise.
- The episode eval is too easy at base (95%). Make the data harder before the real run.

## Fireworks gotchas learned
- **Managed RFT is closed to new jobs.** Use the cookbook's
  `recipes/experiment/async_rl_loop_serverless` (our wrapper is `ase/train_serverless.py`). It needs
  `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID` in `.env`.
- **Base model:** `qwen3p8-27b`, tokenizer `Qwen/Qwen3.8-27B`, renderer
  `qwen3_8_disable_thinking_interleaved`. The `supportsServerlessTraining` flag on the models API is
  not a reliable guide to which models the pool accepts.
- **Trained snapshots are not reachable through OpenAI chat.** Sample them with the SDK's
  token-level sampler; `ase/snapshot_policy.py` does this, and
  `ase.episodes --snapshot <path>` uses it. Snapshot paths look like
  `<acct>/run-<id>/<prefix>-step-N-<hash>`. List them with
  `ServerlessCheckpointClient.list_checkpoints(<session id>)`.
- **Zero-variance groups are dropped.** Calibrate with the base snapshot itself (step-0, through
  `SnapshotPolicy`) instead of a proxy model, so more of the rows actually train.
- **Runs are fast,** about 40 s per step at this size, so many iterations fit in the build window.

## Next for the real run
1. **Harder data:**
   - LLM-paraphrased openings in 4 personas: terse, rambling, angry, non-native. Run a leak check
     so the paraphrase never reveals hidden slots.
   - More articles that are easy to confuse (add triggers for wont_charge and storage_full).
   - Noisier search results.
2. **Calibrate with the base snapshot** at k=8, then train on the 0.2–0.8 band. Use about 400 rows
   and 2 epochs. Evaluate every 2 steps and keep the best checkpoint.
3. **Fix the wrong-fix drift.** Option: in ambiguous states, score a wrong resolve below 0 after
   normalization, by giving ask in unique states 0.3 instead of 0.1. Option: add the wrong-fix rate
   as a hard eval gate. Decide after reading 20 OOD wrong-fix transcripts.
4. **Deploy the winner** (live merge) for real latency and cost, compared against kimi-k3 and gpt-oss-120b.

**Data bug found after the run (now fixed).** Half of the OOD wrong fixes came from a template
phrase for overheating `during_use`: "while I'm on it, not charging". It pointed the model at the
"Not charging" article. The phrase is gone and the data has been regenerated. The overnight data
and results are kept in `data/v1_overnight/`. Re-run calibration before training, because
`train_rl.jsonl` was removed: it held the old phrase.

The other real OOD miss: given a vague symptom ("battery problems"), the model asks for the device
but never for the symptom. RL on the training symptoms didn't teach "ask for the symptom" for
symptom families it never saw.
