# Designing a harder evaluation

The problem: strong open models clear this benchmark without post-training or a harness. This
document says why, what the literature says actually makes agent tasks hard, and which mechanisms to
build. It assumes the findings in `BENCHMARK_FAILURE_POINTS.md`.

## The diagnosis: the task has no trap

`ase/probe.py` showed a 25-line script scoring **0.960** on v0 and **0.559** on hard. The reason is
structural, and no amount of surface noise changes it:

**Every fact in this world is additive and monotonic.** Nothing the customer says later contradicts
what they said earlier. No decision made at turn 2 can become wrong at turn 4. The state only ever
gains information, never revises it. So the optimal policy is "collect all slots, then look up the
row" — which is what `ask_everything` does, and why it works.

Hard models of this task fail for one of five reasons, none of which this world has:

1. Late information that **invalidates an earlier commitment**
2. Constraints that **conflict**, so no action satisfies everything
3. **Non-monotonic state** — the right answer changes mid-conversation
4. Facts the user has but **will not volunteer**, and will not give unless asked precisely
5. **Asymmetric costs** that make the safe action situation-dependent

Typos, anger and dropped calls change the *surface*. The structure stays trivial. That distinction
organises everything below.

## What the research says

**The single biggest lever: sharding.** [Lost in Conversation](https://arxiv.org/pdf/2505.06120)
(Laban et al., ICLR 2026) split fully-specified instructions into shards revealed one per turn and
measured an **average 39% drop across 6 tasks and 15 models**. Concatenating the same shards into one
prompt recovered **95.1%** of full performance — so the drop is the *interaction pattern*, not the
content or the phrasing. Frontier models (Claude 3.7 Sonnet, Gemini 2.5, GPT-4.1) get equally lost.
Lowering temperature barely helps (~30 points remain). Agent-style recaps recover only to ~66–77%.
Their conclusion: "when LLMs take a wrong turn, they get lost and do not recover." The February 2026
follow-up, [Intent Mismatch](https://arxiv.org/abs/2602.07338), reproduces ~30%.

This is exactly the effect we want: large, model-agnostic, and fully code-verifiable because *we*
control when each shard drops.

**Noise widens variance, it does not add difficulty.**
[Brittlebench](https://arxiv.org/abs/2603.13285) (Feb 2026) found a single perturbation reshuffles
model rankings in **63%** of cases, and that perturbations "mainly widened the spread of accuracy
across runs rather than shifting the median."
[MulTypo](https://arxiv.org/html/2510.09536v1) found instruction tuning can make models *more*
fragile under typos and that few-shot examples do not fix it. An
[ESL + typos study](https://arxiv.org/pdf/2604.04723) found combined perturbations hurt more than
single ones.

For a benchmark whose claim rests on McNemar at n≈145, **added variance is actively harmful** — it
widens the CI we are trying to narrow. Noise belongs in a separate robustness split, never mixed into
the set that carries the headline.

**Hostile personas produce invalid difficulty.**
[RealUserSim](https://arxiv.org/html/2605.20204v1) found τ-bench's original adversarial directives
were "artificially harsh": **removing** them improved agent success by up to **+16.7%** on Retail,
while personas grounded in real interaction data lowered success by only ~3–3.5%. Several papers warn
that hostile or ALL-CAPS personas make the agent misread intent, so the failure measures the
simulator, not the agent. The real gap is
[benevolence bias](https://arxiv.org/pdf/2604.21480) — simulated users are too *cooperative*. That is
an informational property, not an emotional one.

[StressSim](https://arxiv.org/abs/2610.35426) (Oct 2026) is the principled version: evolve the
simulator's *plan* to lower task success while **preserving the user's underlying goal**.
Intent-preservation is the constraint that keeps adversarial difficulty valid.

**A well-built support benchmark leaves frontier headroom.**
[τ-bench](https://arxiv.org/pdf/2406.12045) reported GPT-4o at ~61% pass^1 on retail and ~35% on
airline, with consistency collapsing as k grows — which is why `pass^k` is its headline metric (we
already have `ase/passk.py`). τ²-bench added telecom troubleshooting and, importantly, **gave the
simulated user tools**, so the user must act on their own device. τ³-bench is the currently
maintained repo; the τ-bench task files are flagged outdated.

## Scoring the five proposals

| Idea | Verdict | Why |
|---|---|---|
| **Impatient customers** | **Build — highest value on the list** | Not as a tone. As a **hard turn budget**: the customer hangs up after N agent turns. This makes question cost *binding* instead of a side metric, which closes F4 (over-asking is currently free) and kills `ask_everything` outright. One of the cheapest, largest-effect changes available. |
| **Angry / talkative** | **Downgrade** | Anger is invalid difficulty (RealUserSim, +16.7%). "Talkative" as verbosity is *easier* — a customer who volunteers everything removes the ask decision. Keep only the **withholding** persona: answers the exact slot asked, volunteers nothing adjacent. That is the benevolence-bias fix. |
| **Typos** | **Separate split only** | Brittlebench: widens variance, reshuffles rankings, does not move the median. Would damage our statistical power on the main set. Report as `test_noisy`, never mixed in. |
| **Irrelevant / incorrect information** | **Build** | Different from typos and genuinely good. Irrelevant-but-plausible facts pollute the search query and attack *retrieval*, which is the step v0 makes trivial. **Incorrect** information is better still — it becomes the contradiction mechanism below. |
| **Connection cut-outs** | **Reframe** | Weak alone. Valuable as a delivery vehicle for sharding and repetition: a dropped turn means the customer restates partially, sometimes inconsistently. |
| **Prank calls** | **Build small (~5%)** | A real out-of-scope class, and it pairs with fixing the abstention hole (F3). Keep it small or it inflates scores through the easy-escalate path. |
| **Language switching** | **Skip** | Breaks the English BM25 index and needs a translated KB. The resulting failure would be *retrieval*, not judgment — it would measure the wrong thing at high cost. |

## The mechanisms to build, ranked by effect per unit of work

### 1. Turn budget (the customer hangs up)
The customer ends the call after N agent turns. Calibrate N so the oracle clears it with margin and
`ask_everything` does not. Converts `wasted_questions_per_case` from a reported side metric into a
binding constraint, and makes "ask the *one* question that decides it" load-bearing for the first
time. Pure code. ~15 lines.

### 2. Late-arriving shards that invalidate a commitment
Hold back one decisive fact and reveal it only *after* the agent has searched or committed to a
direction: the trigger, a second device, a work-managed phone. The gold article changes. The agent
must notice and re-search instead of resolving with what it already has.

This is the direct import of the 39% effect, and it is the mechanism most likely to separate a
trained model from a prompted one — because recovery after a wrong turn is precisely what
Lost in Conversation says models cannot do. Measure **recovery rate after a late shard** as its own
metric. Pure code: we choose the drop turn.

### 3. Contradiction and self-correction
The customer states something false, then corrects it: "it's an iPhone — sorry, no, that's my wife's,
mine's the Samsung." We know the final truth, so scoring stays code. Tests whether the model revises
state or anchors on the first value. Pairs with cut-outs. Already foreshadowed in PLAN.md's
spoken-entity idea.

### 4. Conflicting constraints with no satisfying action
The customer needs the advanced fix, which erases the device; has no backup; and cannot make one (no
computer, no cloud space). No action satisfies every constraint, and the correct move is to escalate
with a specific reason rather than force a resolve.

This is the hardest single thing for models — recognising an impasse instead of manufacturing an
answer — and it is entirely checkable in code. It also gives the abstention metric something real to
measure.

### 5. Withholding customer (non-collaborative, not hostile)
Answers only the exact slot asked. Volunteers nothing adjacent. Says "I don't know" where a real
customer genuinely could not know. Requires the question to be about the right slot, not merely
well-formed. The grounded-persona version from RealUserSim, which lowered success ~3–3.5% honestly,
rather than the hostile version that lowered it invalidly.

### 6. Distractor facts
Plausible, irrelevant details — a new router when the problem is the phone, a recent update when the
trigger is something else. Attacks query construction. Cheap to generate from the existing world.

### 7. User-executed checks (τ²-bench style)
The agent instructs the customer to look something up and the customer reports a result that
determines the branch: "open Settings > Battery and tell me what it says." The observation selects
the article, so the agent must choose the right *diagnostic*, not merely collect slots. Largest build
cost here; also the largest step toward realism. Consider after 1–4 land.

### 8. Out-of-scope / prank class
~5% of cases with no valid resolution. Correct action is a specific escalate or decline.

## What this does to the metrics

These mechanisms break the current summary, deliberately, and require the F2/F3 fixes first:

- **A turn budget makes stalling fatal**, which is what F2's broken `invalid_or_stuck` currently
  hides. Fix the outcome-overwrite bug before building this or the budget will be invisible.
- **Conflicting constraints make abstention correct sometimes**, so `wrong_fix_rate` rewarding
  non-resolution (F3) stops being a bug and becomes a confound. Report `no_resolve_rate` alongside.
- **Late shards need their own metric**: recovery rate after the shard drops, separate from success.
- **Report `pass^k`, not just `pass@1`**, per τ-bench. Consistency is the thing that collapses on
  hard multi-turn tasks, and `ase/passk.py` already computes it.
- **Keep noise out of the headline set.** `test_noisy` is a separate split and a separate claim.

## Build order

1. Fix F2 and F3 (scorer bugs) — prerequisite for anything below
2. Turn budget — biggest effect per line
3. Late shards + recovery metric — the 39% lever
4. Contradiction / self-correction
5. Conflicting constraints
6. Withholding customer + distractor facts
7. Re-run `ase/probe.py`; `ask_everything` should collapse. If it does not, the mechanism did not bite
8. Re-run `ase/fairness.py`; every new fact must still be obtainable and the oracle must still solve 100%
9. Only then measure real models, and only then consider mechanism 7 or 8

Step 7 is the acceptance test. The probe is the cheapest possible check that a mechanism adds
structural difficulty rather than surface noise, and it runs in seconds with no API calls.
