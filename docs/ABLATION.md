# Staged hardening: results

Four difficulty mechanisms, added one at a time and measured after each, to find which one makes base
open models do worse. Test split, temperature 0, 157 cases.

## Recommended configuration

```bash
ASE_WORLD=hard ASE_BUDGET=6        # ASE_SHARD off (null), ASE_NOISE_FACTS optional
```

This **doubled the gap between the open model and the frontier model**, from 12.1 to 24.2 points.

| Model | hard | → hard + budget 6 + noise | McNemar p | 95% CI |
|---|---|---|---|---|
| gpt-oss-120b | 79.6% | **65.0%** (−14.6) | **1.5e-05** | [−21.0, −8.3] |
| kimi-k3 | 91.7% | 89.2% (−2.5) | 0.219 | [−5.7, +0.0] |

Almost all of it is the turn budget. The open model now sits 35 points below the oracle, which is the
headroom a trained model can claim.

## Every configuration

| Config | gpt-oss-120b | kimi-k3 | gap | oracle | `ask_everything` (no LLM) |
|---|---|---|---|---|---|
| hard (baseline) | 0.796 | 0.917 | 12.1 | 1.000 | 0.535 |
| hard + shard | 0.815 | 0.924 | 10.9 | 1.000 | 0.535 |
| **hard + budget 6** | **0.688** | **0.892** | **20.4** | 1.000 | **0.236** |
| hard + budget 6 + noise | 0.650 | 0.892 | 24.2 | 1.000 | — |

Per-stage marginal effects:

| Stage | Mechanism | gpt-oss | p | kimi | p | verdict |
|---|---|---|---|---|---|---|
| 1 | sharding | +1.9 | 0.581 | +0.6 | 1.0 | **null** — dropped |
| 2 | turn budget | **−10.8** | **0.0002** | −2.5 | 0.289 | **works** |
| 3 | irrelevant info | −3.8 | 0.263 | ±0.0 | 1.0 | weak |
| 4 | typos | not run | | | | separate split |

## The pattern

**Mechanisms that change what the task charges for work. Mechanisms that change what the input looks
like do not.**

Sharding and irrelevant information both alter the surface of the conversation; neither moved the
number significantly. The turn budget altered the economics — it priced indecision — and cost the open
model 10.8 points at p=0.0002.

The reason is structural: these agents re-read their entire context every turn. There is no hidden
state to lose, so *how* information arrives barely matters. What does matter is being charged for the
actions taken. This is worth carrying into any future mechanism design: ask whether it changes the
cost of a decision or merely the appearance of the input.

## Stage detail

### Stage 1 — sharding (`ASE_SHARD=1`): null, kept off

Withholds "I already tried the basics" from the opening and volunteers it after the agent has
searched, moving the gold article between tiers. Both CIs straddle zero; the direction is slightly
*positive*.

The paired split proves it is real rather than noise. Of 157 cases, 39 sharded and 118 untouched:

| | hard | hard + shard |
|---|---|---|
| 39 sharded | 26/39 | **30/39** |
| 118 untouched (control) | 99/118 | 98/118 |

Three reasons it failed, in increasing importance: the shard arrives as its own message and is **more**
salient than the same fact buried in a long opening; a single search already returns both tiers in 46%
of cases, so no re-search is needed (the model re-searched in only 6 of 39 and still got 30 right); and
the world is **monotonic**, so a late fact completes the truth rather than invalidating it.
[Lost in Conversation](https://arxiv.org/pdf/2505.06120) gets its 39% from instructions sharded so the
model anchors on a wrong early reading — that requires truth that can change.

The probe policies confirm it from the other side: `ask_everything` and `gate_follower` score
**identically** on both configs, because a stateless re-reader is immune to delivery timing.

Kept behind the flag as a reported negative result rather than tuned until it showed a drop. To get
the paper's effect the late turn must invalidate work already done — i.e. **contradiction**
("it's an iPhone… sorry, that's my wife's, mine's the Samsung"), which is the same delivery machinery
with a different payload.

### Stage 2 — turn budget (`ASE_BUDGET=6`): the one that works

The customer abandons after 6 agent turns; running out is a terminal `abandoned` outcome, not a silent
cap. Eval-time only, so it shares `data_hard/` and appears in `RUN_PREFIX` alone.

The open model loses 4x what the frontier model does, because it wastes 0.59 questions per case
against kimi's 0.11. Abandonment follows: 0.14 vs 0.038.

**N=6 is uniquely correct**, not tuned — `ask_everything` needs exactly 7 turns on 114 of 157 cases
(5 asks + search + resolve) while the oracle never exceeds 6:

| N | oracle | `ask_everything` | |
|---|---|---|---|
| 4 | 0.885 | 0.166 | oracle fails — unfair |
| 5 | 0.981 | 0.236 | oracle fails — unfair |
| **6** | **1.000** | **0.236** | **chosen** |
| 7 | 1.000 | 0.535 | no effect |

This **closes F4** from `BENCHMARK_FAILURE_POINTS.md`: over-asking is no longer free on the headline
metric. If the data is ever regenerated, N must be re-derived from the new oracle distribution or the
benchmark silently becomes unfair.

**A prediction that was wrong.** From the means — gpt-oss ~3.5 turns, kimi ~3.2, both under 6 — the
budget looked like it would not touch the models. It cost gpt-oss 10.8 points, because 14% of its
episodes run past 6 turns. Calibrate on the distribution, never the average.

### Stage 3 — irrelevant information (`ASE_NOISE_FACTS=1`): weak, but moves the failure mode

40% of cases mention a second device from another family and explicitly set it aside ("My Wi-Fi router
keeps dropping out too, but that's not why I'm calling"). Each names `PLATFORM_TERMS` words, so a query
that keeps it pulls the wrong cell into the top 5.

**Offline acceptance gate, run before any model spend.** On the 54 affected test cases the gold falls
outside the top 5 on a naive whole-opening query in **69%** of them (median rank 8), but on a clean
device+symptom+trigger query in only **6%**. The mechanism is real at the retrieval layer.

Not significant on success for either model. But gpt-oss's **wrong-fix rate rose 0.136 → 0.212** while
abandonment *fell* 0.140 → 0.096: a polluted search reaches a confident wrong answer faster than a
careful one reaches a right answer. The two offset, so success barely moved. For a support product
that is a loss disguised as a wash, and it is a sign the headline metric is insensitive to a cost the
business actually pays.

kimi moved by exactly 0.0 (2 cases each way). Since the gate shows the gold is displaced in 69% of
these cases *for a naive query*, kimi evidently does not issue naive queries. The mechanism
discriminates on **query discipline**; the frontier model already has it.

## What is established, and what is not

**Established.** The cumulative drop for gpt-oss (p=1.5e-05) and the Stage 2 budget effect (p=0.0002).
Both survive a case-clustered bootstrap. Fairness holds at every stage: the oracle solves 100% of all
three splits within budget, and `ase/fairness.py` passes all checks.

**Not established.** Every kimi result — all three stages are within noise for it, so the only
defensible claim is that the mechanisms cost the *open* model and leave the frontier model roughly
where it was. Also not established: anything about latency or cost, see below.

**Methodology.** Cases are paired across every configuration — the shard and distractor decisions use
side RNGs keyed on `case_id`, and rollouts run from each case's own seed — so McNemar across stages is
valid and the untouched cases act as an internal control. That control is what made the Stage 1 null
trustworthy rather than ambiguous.

## Open items

| Item | Why it matters |
|---|---|
| **Reasoning setting is unpinned** | `LLMPolicy` sends no reasoning parameter, so both models ran at provider defaults. Any latency or cost-per-case claim is uncontrolled, and it is a mild confound on Stage 2 (reasoning effort trades off against turn count). Check `ctok` in the saved logs first — it needs no new API calls. |
| **Single seed, temperature 0** | The −14.6 is load-bearing and deserves a second seed before publication. |
| **F5: OOD carries no escalation cases** | `test` is 27.6% escalation, `test_ood` is 0%, because escalation cases have `symptom=None` and cannot route to OOD. Test-vs-OOD comparisons conflate unseen symptoms with removing the easiest quarter. Needs a data regeneration. |
| **F8: per-mechanism counts too small** | 6 hidden-damage and 7 backup-first cases in hard test. Unreportable at that size. |
| **Training base not inference-reachable** | `train_serverless.py` names `qwen3p8-27b`, which 404s on the inference pool. The "same open model before and after" claim depends on resolving it. |
| **Stage 4 (typos) not run** | Separate split, never folded into the headline. Brittlebench predicts variance rather than a median shift. |

## Reasoning effort

**Both models reason by default.** `reasoning_content` is populated on every response, and it is ~2/3
of generation. It arrives in its own field rather than inline, so `content` was always clean JSON and
`parse_action` was never at risk -- but every latency and cost number was uncontrolled.

`ase.episodes --reasoning-effort {low,medium,high}` now pins it, and the value is recorded in
`summary.json` under `config.reasoning_effort` (`provider-default` when unset). Runs at different
efforts get distinct tags, so they cannot overwrite each other.

Measured on `hard + budget 6`:

| model | effort | success | wrong fix | abandoned | p50 latency | tokens/case |
|---|---|---|---|---|---|---|
| gpt-oss-120b | default | 0.688 | 0.136 | 0.140 | 0.994s | 3304 |
| gpt-oss-120b | low | 0.720 | 0.212 | 0.070 | **0.519s** | 2794 |
| kimi-k3 | default | 0.892 | 0.042 | 0.038 | 2.792s | 3128 |
| kimi-k3 | low | 0.866 | 0.076 | 0.032 | **2.276s** | 2952 |

**Effort is a latency lever, not an accuracy lever.** Neither accuracy change is significant
(gpt-oss p=0.522, kimi p=0.289), but gpt-oss's p50 latency nearly halves and its tokens per case fall
15%. For an agentic support bot that is close to free.

**It does move gpt-oss's failure mode**, the same way Stage 3 did: at low effort its wrong-fix rate
rises 0.136 -> 0.212 while abandonment halves 0.140 -> 0.070. Heavy reasoning made it dither and run
out of turns; light reasoning makes it commit, sometimes wrongly. kimi shows the same shape, far
smaller.

### Which to pin, and an honest tension

The two goals disagree, and neither difference is significant, so picking on the number would be
cherry-picking:

- **Bigger model gap** favours provider-default: 20.4 pts, against 14.6 at low.
- **Tougher cost comparison** favours `low`: it is the frontier baseline at its cheapest and fastest,
  so a trained model has to beat a *fast* kimi rather than a slow one.

Recommend pinning **`low`** for any cost or latency claim, since it is what a latency-sensitive support
product would actually deploy, and reporting provider-default as a sensitivity row. State the setting
whenever a number is quoted.

### Stage results stand

Stages 0-3 ran at provider default. Since effort does not significantly change accuracy, the stage
conclusions are unaffected; re-running six configurations to move nothing significant is not worth the
spend. The setting is now recorded in every new `summary.json`, so this ambiguity cannot recur.
