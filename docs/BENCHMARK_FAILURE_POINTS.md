# Failure points in the current benchmark

Found by `ase/probe.py`: cheap, no-LLM policies run through the **real episode eval**, reading only
the conversation, never the ground truth. A high score from one of these is a weakness in the
benchmark, not a clever policy.

Run them yourself:
```bash
ASE_WORLD=hard python -m ase.probe --split test
ASE_RETRIEVAL=fixed python -m ase.probe --split test
```

## Probe results

**v0 (`ASE_RETRIEVAL=fixed`, test, n=99)**

| Policy | success | wrong_fix | skipped_check | questions/case |
|---|---|---|---|---|
| `ask_everything` | **0.960** | 0.042 | 0.000 | 3.59 |
| `gate_follower` | 0.485 | 0.472 | 0.162 | 0 |
| `cue_only_escalate` | 0.273 | 0.000 | 0.000 | 5.74 |
| `blind_top_hit` | 0.212 | 0.486 | 0.162 | 0 |

**hard (`ASE_WORLD=hard`, test, n=145)**

| Policy | success | wrong_fix | skipped_check | hidden_damage_missed | questions/case |
|---|---|---|---|---|---|
| `ask_everything` | 0.559 | 0.626 | 0.007 | 0.167 | 3.55 |
| `gate_follower` | 0.434 | 0.677 | 0.090 | 0.333 | 0.25 |
| `cue_only_escalate` | 0.276 | 0.000 | 0.000 | 1.0 | 8.61 |
| `blind_top_hit` | 0.041 | 0.869 | 0.048 | 1.0 | 0 |

---

## F1. v0 is solved by a 25-line script — and v0 is what RL trains on

`ask_everything` asks device → symptom → trigger → damage → backup in fixed order, then resolves the
result with the most token overlap. No reasoning, no model, no ground truth. It scores **0.960 success
on v0 test**.

The base model scored 0.954 overnight. **A fixed question script and a frontier model are
indistinguishable on this metric**, so no trained gain measured on v0 `success` means anything.

Cause is structural: in fixed mode the env searches the accumulated customer text, and
`env.py:_aliases()` deliberately indexes the exact phrases customers use for each article
(`DEVICE_SPECIFIC`, `SYMPTOM_SPECIFIC`, `TRIGGER_TEXT`, `TRIED_TEXT`). Once every slot has been
asked, the gold article is rank 1 by construction. The retrieval step cannot fail, so the task
collapses to "ask everything, take the top hit."

This is the most important finding on this page. v0 is the mode the README says RL trains on and
that the harness workstream competes against.

## F2. `invalid_or_stuck` is broken — stalling is invisible

`cue_only_escalate` ran **104 of 145 hard cases to the 12-step cap**. Every one was recorded as
`outcome: "ask"`, and `invalid_or_stuck` reported **0.0**.

In `episodes.py:run_episode`, `outcome` is initialised to `"max_steps"` and then reassigned on every
loop iteration:

```python
outcome, asks, wasted_asks, searches = "max_steps", 0, 0, 0
for _ in range(MAX_STEPS):
    ...
    done, outcome = step(state, action, RETRIEVER, rng)   # overwrites the initialiser
    if done:
        break
```

So a run that never terminates ends holding the last *non-terminal* outcome (`"ask"` or `"search"`),
and `rate(logs, "invalid", "max_steps")` never matches. `"max_steps"` is reachable only if
`MAX_STEPS == 0`.

Impact: a model that stalls forever is reported as 0% stuck. It still counts against `success`, so
the headline is not wrong — but the diagnostic that exists to catch this says nothing, which is worse
than having no diagnostic.

## F3. `wrong_fix_rate` rewards never resolving

`cue_only_escalate` resolves nothing and earns a **perfect 0.000 wrong-fix rate** on both modes.
Because `wrong_fix_rate` is a rate over cases that *ended* in a resolve, abstaining is free.

PLAN.md names abstention gaming as a trip wire to watch during training. No metric catches it, and
the one that would (F2) is broken. Together these mean a model can learn to look maximally safe —
zero wrong fixes, zero skipped checks — by refusing to commit, and every summary field will endorse
it.

## F4. The gates are defeated by diligence, not judgment

On hard, `ask_everything` drives `skipped_check` to **0.007** and `hidden_damage_missed` to **0.167**
by asking `damage` and `backup` unconditionally on every case, including the ~95% where neither
applies.

The only penalty is `wasted_questions_per_case` (2.62 against the oracle's ~0.8), which is reported
as a separate field and **never enters `success`**. There is no composite that trades adherence
against question cost, so unconditional over-asking is strictly safe on the headline number.

This is the inverse of the gate-avoidance risk the plan anticipated. The damage gate and the backup
check do not test whether a model knows *when* a check is required — only whether it performs checks
at all.

## F5. The OOD split differs in action mix, not just in symptoms

| Split | escalation cases |
|---|---|
| hard test | 40 / 145 = **27.6%** |
| hard test_ood | 0 / 106 = **0.0%** |

`sample_case` draws escalation *before* symptom, and escalation cases carry `symptom=None`.
`build_dataset` routes to OOD on `case.symptom in OOD_SYMPTOMS`, so an escalation case can never
land in `test_ood`.

Escalations are the easiest category (the oracle needs one action; `cue_only_escalate` catches 87% of
them with a keyword list). So `test` carries ~28% easy cases that `test_ood` lacks entirely.

Any test-vs-OOD comparison therefore conflates two manipulations: unseen symptoms, and the removal of
the easiest quarter of the distribution. **The overnight OOD headline (72.8% → 87.0%) is partly a
composition artifact**, and the gap between test and OOD difficulty is overstated by construction.

## F6. A quarter of the hard test set falls to a regex

`cue_only_escalate` is a 20-line keyword list over the opening turn ("dropped", "swelling", "bill",
"real person"). It catches **40 of 46** escalation-equivalent cases, 87%, which is 27.6% of the split.

The escalation openings come from `ESCALATIONS`, three fixed templates per reason. Nothing about
recognising them requires a model.

## F7. `skipped_check_rate` means two different things

In hard mode it means "resolved without a safety check the article required." In v0 it fires for
`blind_top_hit` at **0.162** — where no gates exist — because `step()` returns
`resolved_skipped_check` whenever the right article is resolved before `allowed` permits it, i.e.
"resolved prematurely, before asking the disambiguating question."

Both are reasonable metrics. They are not the same metric, and a leaderboard carrying both modes
under one column name would mislead.

## F8. Per-mechanism rates are unreportable at current n

Hard test contains **6 hidden-damage** and **7 backup-first** cases (`ase.fairness` prints the mix).
`hidden_damage_missed` moved 0.167 → 0.333 across two probe policies — a one-case difference.
`docs/SPIKE_HARD_WORLD.md` already flags this; the probe confirms the metric is noise at this size.

---

## What is working

The hard world does what the spike intended. The same script that scores 0.960 on v0 drops to
**0.559** on hard, and the article-reading `gate_follower` only reaches 0.434. Naive token overlap
cannot pick the right article once tiers, trigger variants and the black-screen pair are in play.
`ase.fairness` passes all 18 checks on all three splits, and the oracle solves every case in ≤7 of
12 steps.

The problem is not the hard world. It is v0, and the summary metrics.

## Ranked fixes

| # | Fix | Cost |
|---|---|---|
| 1 | Make hard the default benchmark; retire v0 `success` as a headline, or report it only as a ceiling check | none, a decision |
| 2 | `run_episode`: assign the terminal outcome to a separate variable so `max_steps` survives | 2 lines |
| 3 | Report `no_resolve_rate` (or fold stalls into a single `unsafe_or_stuck`) so abstention is visible | ~5 lines |
| 4 | Let escalation cases reach OOD — route on symptom **or** escalation — and rebalance the splits | ~3 lines, regenerates data |
| 5 | Make over-asking cost something in a composite score, or report adherence conditional on questions ≤ k | design call |
| 6 | Paraphrase escalation openings so they are not three fixed templates | needs the persona work |
| 7 | Oversample hidden-damage and backup-first, or pool them into one adherence number | regenerates data |

Items 4 and 7 regenerate data and must land **before the test set locks**. Items 2, 3 and 5 are
scorer-only and can land after.
