# Staged hardening: results

One mechanism at a time, measured after each. Test split, temperature 0. Cases are **paired across
stages** — the shard decision uses a side RNG keyed on `case_id` and rollouts run from each case's own
seed, so every config holds the same 157 cases with the same hidden truth and seeds. McNemar across
stages is therefore valid.

| Config | gpt-oss-120b | kimi-k3 | oracle | `ask_everything` (no LLM) |
|---|---|---|---|---|
| hard (baseline) | 0.796 | 0.917 | 1.000 | 0.535 |
| hard + shard | 0.815 | 0.924 | 1.000 | 0.535 |

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
