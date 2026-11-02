# ASE: teaching a small voice agent the middle of a support call

## The result, up front

We built a support benchmark that is graded exactly by code, and is still hard because it copies what goes wrong in
real calls. A strong open model, deepseek-v4p1-flash with reasoning off, solves **48%** of held-out calls on the first
try. Base qwen3p8-27b, the model we can post-train, solves **45%**.

Base qwen can do the job: **81%** of calls pass on at least one of 8 tries. It just doesn't do it reliably: **7%** pass
on all 8. That is the "stochastic, high ceiling" quadrant, so the deck's prescription is to consolidate.

We ran GRPO for 7 steps (about 25 minutes, serverless LoRA). On fresh held-out cases the model never trained on, the
share of turns where it makes a correct move went from **67% to 91%** (+24 points, rough 95% interval +11 to +38).
Running the trained model through the same full-call held-out test was in progress when we stopped Fireworks
workloads, so we have no whole-call "after" number. That's the main open item.

## 1. The task

A voice support agent for a phone carrier talks to a customer and makes one move per turn:
- `ask` a question;
- `step`: give one fix from a knowledge base (KB) article;
- `verify` that the fix worked;
- `resolve`: close the call;
- `escalate`: hand off.

The customer is scripted and seeded. Each case has a hidden root cause. An answer key works out the right next moves
from what has been said so far. There is no LLM judge: every turn is graded right or wrong by code.

The student has to be small and must not reason, because it is a voice agent and latency matters. Turning reasoning
off cut deepseek-flash from 2.9 s to 1.0 s per turn in one call we timed. Serverless training on Fireworks only
accepts models of about 27–30B today; we tried `qwen3-8b` and `qwen3p5-9b` and both were rejected. So the student is
qwen3p8-27b with thinking disabled.

## 2. The story: why making the eval hard took most of the day

We cared more about this section than the score, and it's where the presentation lives.

**Act 1: our first evals were too easy.** v0 graded the opening of the call: ask until one article fits, then pick
it. kimi-k3 scored 94% and gpt-oss-120b 80%. We added traps one round at a time, and every frontier model
kept solving them (96–99%):
- articles for the wrong model, stale and retired articles, in-store-only articles;
- cases with no article at all;
- customers who know the brand but not the model.

**Act 2: we went to the data.** We studied real carrier support calls and KB lookups (private data; we used only
aggregate patterns, and every piece of text in this repo is synthetic). Four things stood out:
- **Customers rarely know their exact model.** Most give only the brand ("an iPhone"). Our v0 customer always gave the
  exact model.
- **Search almost never returns one clean article.** Nearly every lookup returns several plausible ones: the other
  platform's version, an in-store copy, or one written for older software.
- **Stale articles are common.** Most articles that name an OS version were written for versions well behind
  current.
- **Gathering facts is only a small share of the agent's turns.** Most turns go to re-planning after a failed step,
  re-orienting a caller lost in the menus, and checking whether the fix worked.

The finding that changed our direction: **the opening is the easy part.** Real calls go wrong in the middle:
re-planning after failed fixes, callers lost in menus, and long calls with no stop rule. Our eval graded only the part
of the call that models were already good at.

**Act 3: every number we nearly reported was wrong at least once.** Reading failures by hand is what saved us. Each
of these looked like a model failure and was really our eval's fault:
- **429 errors scored as model failures.** Rate-limit errors were counted as wrong answers: 22 of 140 cases in one run.
- **A symptom phrase leaked the hidden fact.** "An app closes as soon as I open it" told the model "one app", while
  the grader treated that as still unknown. It hit us in two separate builds.
- **The answer key ignored what the customer had already said.** Two independent reviewers (GPT-5.6 and Grok)
  found this; 21 of kimi's 45 "failures" were eval bugs.
- **An article told the agent to "select KB-906 first",** and the grader then marked it wrong for doing exactly that.
  This was 29 of gpt-oss's 57 failures.
- **The menu-name fix was graded by a field, not by what the model said.** The model said "tap Mobile Service", but
  the grader only read the `path` field, which the model left empty.
- **The scripted customer said "Okay, go ahead"** to a question that wasn't one.

Two lessons. "The oracle solves 100%" only proves the answer key agrees with itself, not that it's fair. And the
models were reasoning by default; once we turned that off to match a voice deployment, the numbers moved.

**Act 4: take the trap hints out of the prompt.** Our prompt had spelled out the traps ("never use a retired
article"). Removing them barely changed scores (kimi stayed around 97%). The difficulty had to come from the task, not
from hiding the rules.

**Act 5: run the article, not just pick it.** hard_v2 grades the middle of the call. The agent runs the article step
by step:
- the first fixes fail;
- a step can look like it worked ("the LTE icon came back?") when it didn't;
- the customer can't find the menu ("I see Mobile Service, not Cellular");
- the customer does the wrong thing ("I turned off Wi-Fi instead");
- the customer already did a step this morning, or the setting is already on;
- halfway through it turns out to be a different problem ("it says SOS", so switch articles);
- the customer asks for a person.

Articles are at real length and in the real layout (bold sections, Samsung and other-Android lines, a branch to
another article), and each turn shows 8 search results that include old-software, in-store and other-platform copies.
That version finally holds up: frontier models fail it for real reasons. We read all 44 deepseek failures in the
full run: about 40 are real model mistakes and 4 came partly from our scripted customer, which we fixed.

## 3. Before training: the four questions (deck slide 9)

1. **An eval we trust?** Yes. Exact code grading, case types labeled, every path the answer key allows checked
   automatically (735 paths, 0 bad), and failures read by hand.
2. **A knowledge gap?** No. The right article and its steps are always in the context window (8 results per turn).
3. **Schema errors?** Rare. Across the 560 test draws, invalid JSON was 0% for deepseek-flash and 2.1% for base qwen
   with thinking off. (nemotron failed on format, so we dropped it as a baseline.)
4. **A better prompt?** Tried. We added slot and reason definitions, tested burying the rules in a long realistic
   policy, and removed the trap hints. The policy-only prompt is the baseline every model is measured on.

**Why the failure is intrinsic** (slide 8): the model had what it needed and still got it wrong. The article it was
reading literally says "If the phone shows SOS, use Troubleshoot No Service instead." The customer says "it says SOS".
deepseek-flash replies "this isn't a mobile data problem", and then hands off instead of switching articles.

## 4. Measure the gap: base curves on the held-out test

70 held-out test cases × 8 draws at temperature 0.7, scored per whole call:

| k | qwen3p8-27b base: pass@k | pass^k | gap | deepseek-flash (no reasoning): pass@k | pass^k | gap |
|---|---|---|---|---|---|---|
| 1 | **44.8%** | 44.8% | 0 | **48.2%** | 48.2% | 0 |
| 2 | 61.4% | 28.3% | 33 | 57.0% | 39.4% | 18 |
| 4 | 73.7% | 14.6% | 59 | 64.7% | 32.4% | 32 |
| 8 | **81.4%** | **7.1%** | **74** | **71.4%** | **27.1%** | **44** |

**Both are stochastic with a high ceiling, so the deck's verdict is to consolidate.** Base qwen is the more extreme
case: it has the higher ceiling (81% vs 71%) but far worse consistency (7% vs 27%). There is plenty for RL to reinforce.

By case type, pass@1 (the same 70 cases):

| Case type | qwen base | deepseek-flash | What the agent has to do |
|---|---|---|---|
| `sos_switch` | **11%** | 21% | Switch articles when the customer reveals a different problem halfway |
| `decoy` | 27% | 23% | Check the fix; "it looks better?" isn't fixed |
| `slip` | 30% | 41% | The customer did the step wrong, so give the same step again |
| `brand_path` | 48% | 70% | Give the Samsung or Pixel path, or ask which phone it is |
| `menu_label` | 46% | 79% | "I see Mobile Service": the same step again, don't skip it |
| `nothing_works` | 54% | 39% | Know when to stop: hand off once the steps run out |
| `human_midcall` | 52% | 39% | Stop troubleshooting when the customer asks for a person |
| `already_on`, `late_tried`, `late_fix` | 52–64% | 48–73% | Don't repeat what was tried; keep going past failed fixes |

## 5. The technique follows from the quadrant: GRPO

- **Training data:** 300 training cases on a separate seed (101), so no test case leaks in. They become 2,273
  single-turn states. The rollouts are deliberately noisy: 1 move in 5 is a mistake (a repeated step, a skipped step,
  a pointless question), so the model also trains on the messy recovery states real calls produce.
- **Reward:** 1 if the move is one the answer key allows, else 0. It's exact and has no judge. Weakness we know of:
  the wording of the reply isn't graded at all.
- **Settings:** LoRA rank 32, learning rate 1e-4, 8 prompts per step with 8 samples each, temperature 1, serverless.
- **What actually trained:** 248 of the 300 rows were skipped because all 8 samples scored the same, which gives GRPO
  no signal. The 52 rows where the model was inconsistent drove all 7 optimizer steps. That's the deck's
  "consolidate" idea in practice: RL can only reinforce what the model sometimes already does.

## 6. Before and after

**Held-out, per turn.** 64 states from fresh cases on seed 202, which the model never trained on, with 4 samples per
state. This is the trainer's own check, run on clean conversations:

| | Base (step 0) | Step 4 | **Step 7 (final)** |
|---|---|---|---|
| **All turns** | **67.2%** | 84.4% | **91.4%** |
| Give the right next step | 57.9% | 81.4% | 87.1% |
| Ask, or give the step | 75.0% | 95.8% | 95.8% |
| Check the fix | 87.5% | 97.5% | 100% |
| Close the call | 77.8% | 77.8% | 91.7% |
| Hand off | 50.0% | 50.0% | 100% |
| Ask a needed question | 75.0% | 100% | 100% |

- The gain is +24.2 points, with a rough 95% interval of +10.8 to +37.6 when we treat the 64 states as the unit.
- A second, smaller run (100 rows, 4 steps) independently went from 64.5% to 84.0%, so the direction repeats.

**Held-out, whole calls (the deck's pass@k curve): not measured.** The trained model's 70 × 8 run was in progress
when we stopped Fireworks workloads. Per-turn gains do not translate one-to-one into whole-call gains: a call has
about 7 turns, and one bad turn fails it. To finish it:
```
python -m ase.hard_v2_passk --snapshot bassimfaizal-e0d5cc/run-d18cfd6c171d45648d0a6f12977a0ef3/ase-smoke-step-7-9948720d --n 8
```
After that, run McNemar per case and a bootstrap that resamples cases, as in the sample submission.

**Earlier and simpler, on v0 (the "pick the article" eval):** the same recipe took per-turn held-out accuracy from
87.5% to 95.7% in 4 steps.

## 7. Questions a judge will ask

- **Would routing get this for free?** Partly. Sending each case to whichever of base qwen or deepseek-flash does
  better gives 60% pass@1, against 45% and 48% for each alone. That's an oracle with hindsight, not a router you can
  build. The two models fail on different case types: deepseek handles menu labels and brand paths better, qwen
  handles stopping and hand-offs better. Both are bad at article switches and decoys, which routing can't fix and
  training targets directly.
- **Does it generalize?** Held-out cases are new combinations of the same templates, articles and customer behaviors.
  That shows the model learned the task in-distribution, not that it transfers to real calls.
- **Is the eval big enough?** 70 test cases × 8 draws, 7 cases per type. That's enough to see the quadrant, but small
  per type.
- **What's weak about our reward?** It doesn't grade wording, empathy or pacing. A move counts if the answer key
  allows it, so a correct move with a bad explanation still scores.
- **Cost:** not measured precisely. Training is billed per token. The 300-row run took about 25 minutes, and the
  100-row run about 13.

## 8. What we'd do next

1. Finish the whole-call "after" curve and run the significance tests.
2. Train on the model's own conversations. The answer key can label any state, so roll out the trained model, label
   its mistakes and train again (on-policy).
3. Add the mid-call mechanisms that are still missing and are still exactly gradable: a second issue partway through
   the call, digressions ("why is my bill higher?"), and facts that change mid-call.
4. Pick a smaller student once Fireworks enables serverless training below 27B (`qwen3p5-9b`, or a mixture-of-experts
   model with few active parameters) for real voice latency.

## Where things are

- **Benchmark:** `data/hard_v2/` (dev 50 / test 70, an 84-article KB, the prompt, a README) and `ase/hard_v2.py`.
- **Training:** `ase/hard_v2_train.py` (states and reward) and `ase/train_serverless.py --env hard_v2`.
- **Measurement:** `ase/hard_v2_passk.py` (pass@k and pass^k).
- **Earlier evals and runs:** `data/hard_v1/` (deepseek-flash 47%; 0% on no-article cases), plus `docs/`.
- **Logs:** `runs/passk_hv2_test_*` (curves) and `runs/rl_hv2_{100,300}` (training). These are local and gitignored.
