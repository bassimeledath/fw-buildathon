# Staged hardening: results

One mechanism at a time, measured after each. Test split, temperature 0. Cases are **paired across
stages** — the shard decision uses a side RNG keyed on `case_id` and rollouts run from each case's own
seed, so every config holds the same 157 cases with the same hidden truth and seeds. McNemar across
stages is therefore valid.

| Config | gpt-oss-120b | kimi-k3 | gap | oracle | `ask_everything` (no LLM) |
|---|---|---|---|---|---|
| hard (baseline) | 0.796 | 0.917 | 12.1 | 1.000 | 0.535 |
| hard + shard (null, dropped) | 0.815 | 0.924 | 10.9 | 1.000 | 0.535 |
| **hard + budget 6** | **0.688** | **0.892** | **20.4** | 1.000 | **0.236** |

Sharding is off by default and is not carried forward; `hard` is the base for later stages.

## Stage 1 — sharding: **no effect** (null result)

| Model | base → shard | McNemar p | 95% CI |
|---|---|---|---|
| gpt-oss-120b | 79.6% → 81.5% | 0.581 | [−2.5, +6.4] |
| kimi-k3 | 91.7% → 92.4% | 1.0 | [−1.3, +2.5] |

Both CIs straddle zero. If anything the direction is *positive* — the mechanism made the task very
slightly easier.

**The paired split shows this is real, not noise.** Of 157 cases, 39 were sharded and 118 untouched:

| | hard | hard + shard |
|---|---|---|
| 39 sharded cases | 26/39 | **30/39** |
| 118 untouched cases | 99/118 | 98/118 |

The untouched control moved by one case; the sharded cases improved by four. The effect is on exactly
the cases the mechanism touched, and it is in the wrong direction.

### Why it failed

Three reasons, in increasing order of importance.

1. **The shard is more salient than the original.** Unsharded, "I already tried the basics" sits inside
   a long opening next to device, symptom, trigger and filler. Sharded, it arrives **alone, as its own
   message**, prefixed "Oh, I should have mentioned —". That is the most prominent position in the
   transcript. We made the fact easier to notice, not harder.

2. **The moved article is usually already in hand.** `RESULTS_K = 5` and the two tiers of a cell have
   near-identical titles, so a single opening search returns **both tiers in 46% of sharded cases**.
   No re-search is needed. Measured: the model re-searched after the shard in only **6 of 39**
   transcripts, yet still got 30 right.

3. **The world is monotonic, so "late" is not "invalidating".** This is the real reason.
   [Lost in Conversation](https://arxiv.org/pdf/2505.06120) gets its 39% from *instructions* sharded so
   the model anchors on an early wrong interpretation and cannot recover. Here the hidden truth never
   changes, so a late fact cannot make an earlier correct decision wrong — it only adds a decisive fact
   at a more noticeable moment. And an LLM re-reads the whole context every turn; there is no hidden
   state to lose. The probe policies confirm this from the other side: `ask_everything` and
   `gate_follower` score **identically** on both configs (0.535 / 0.357), because a stateless re-reader
   is immune to delivery timing by construction.

### What this implies

Adding late-arriving information to a monotonic world yields "later information", which is not harder.
To reproduce the paper's effect the late turn must **invalidate work already done**, which in this
world means changing the truth rather than completing it — i.e. the **contradiction / self-correction**
mechanism ("it's an iPhone… sorry, that's my wife's, mine's the Samsung"). That genuinely strands a
completed search on the wrong cell and is the same machinery with a different payload.

The sharding code is sound and stays behind `ASE_SHARD=1`; it is a measured negative result, kept and
reported rather than tuned until it showed a drop.

## Stage 2 — turn budget (`ASE_BUDGET=6`): **works, and discriminates**

The customer abandons the chat after 6 agent turns; running out is a terminal `abandoned` outcome, not
a silent cap. Eval-time only, so it shares `data_hard/` and appears in `RUN_PREFIX` alone.

| Model | base → budget 6 | McNemar p | 95% CI |
|---|---|---|---|
| gpt-oss-120b | 79.6% → **68.8%** | **0.000221** | [−16.6, −5.7] |
| kimi-k3 | 91.7% → 89.2% | 0.289 | [−6.4, +0.6] |

**The open model loses 4x what the frontier model does**, widening the gap from 12.1 to 20.4 points.
That is the point: the budget charges for wasted questions, and gpt-oss wastes 0.59 per case against
kimi's 0.11. Abandonment rates follow — 0.14 vs 0.038.

### Calibrating N

N=6 is uniquely correct: the largest N that closes the exploit and the smallest at which the oracle is
still perfect.

| N | oracle | `ask_everything` | verdict |
|---|---|---|---|
| 4 | 0.885 | 0.166 | oracle fails — unfair |
| 5 | 0.981 | 0.236 | oracle fails — unfair |
| **6** | **1.000** | **0.236** | **chosen** |
| 7 | 1.000 | 0.535 | no effect |

`ask_everything` needs exactly 7 turns on 114 of 157 cases (5 asks + search + resolve) while the oracle
never exceeds 6, so the separation is structural rather than tuned. It drops 0.535 → 0.236 and
**closes F4** — over-asking is no longer free on the headline metric. `gate_follower` is unaffected
(0.357, max 4 turns), as expected.

### A prediction I got wrong

From the means — gpt-oss ~3.5 agent turns, kimi ~3.2, both well under 6 — I expected the budget not to
touch the models. It cost gpt-oss 10.8 points. **Means hide the tail**: 14% of its episodes run past 6
turns. Calibrate budgets on the distribution, not the average.
