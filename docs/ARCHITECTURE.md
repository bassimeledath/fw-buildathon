# ASE Environment Architecture

## 1. Purpose

The ASE ("ask, search, escalate") environment trains a model to select the next action in a tech-support conversation. The four actions are: search the knowledge base (KB), ask a clarifying question, resolve with one KB article, or escalate to a human. The environment is verifiable. Each case has a hidden ground truth, an oracle gives the correct actions for each state, and the reward is code. There is no LLM judge.

## 2. Modes

Two environment variables select the mode. The baseline is v0. The rest of this document shows v1, because v1 has all four actions.

| Mode | Variables | Actions | Retrieval | Reward |
|---|---|---|---|---|
| **v0 (baseline)** | `ASE_RETRIEVAL=fixed` | `resolve`, `ask`, `escalate` | The environment searches all customer text after each customer message. It adds the top 5 results to that message. | 1.0 or 0.0 |
| v1 | none | `search`, `ask`, `resolve`, `escalate` | The model writes the query. The results show the top 3 titles. | Partial credit (section 5) |
| hard | `ASE_WORLD=hard`, with or without `ASE_RETRIEVAL=fixed` | v1 or v0 actions; `ask` also has the slots `damage` and `backup` | The results show the article text. | As v1 or v0 |

In v0, the index also has the words that customers use for each article, so retrieval finds the needed article (`ase.fairness` checks this). The model only decides: resolve, ask, or escalate.

NOTE: In v0 and hard, the episode scorer uses the oracle. A `resolve` is correct only when the oracle allows it. In v0, the escalation `reason` must also be correct.

## 3. Glossary

| Term | Meaning |
|---|---|
| Case | One customer problem: hidden values for the three slots, or an escalation reason (20% of cases). |
| Slot | One attribute of a case: `device` (5 values), `symptom` (12 values) or `trigger` (the condition, for example `new_sim`). |
| Revealed level | What the customer told about a slot (`Case.revealed`): `specific`, `family` ("my phone"), `vague` ("internet problems") or nothing. |
| KB | 65 articles, one for each (device, symptom, trigger) combination, plus 6 distractor articles (`build_kb()`). |
| Gold article | The one KB article that agrees with all hidden slots of the case (`gold_article()`). |
| Candidate articles | The KB articles that agree with all slot values that the customer revealed (`candidates()`). |
| Discriminating slot | A slot that has different values in the candidate articles (`discriminating_slots()`). |
| State | The conversation at one decision point: customer messages, earlier actions and observations (`State`). |
| Episode | One full conversation, to a terminal action or to `MAX_STEPS` = 8 actions. |
| Oracle | `allowed_actions(state, kb)`: the correct action types for a state and the data that each type needs. |

The oracle uses these rules:

1. Escalation case: only `escalate`, with the reason of the case.
2. One candidate article, and the gold article is in earlier search results: only `resolve`.
3. One candidate article, and the gold article is not in earlier search results: only `search`.
4. More than one candidate article: `ask`, with the discriminating slots. `search` is also correct if no earlier search returned articles and no search occurred after the last customer answer.

## 4. Components

```
world.py --> build_dataset.py --> *_states.jsonl --> calibrate.py --> train_rl.jsonl
(slots, KB,  (sample_case(), noisy  (one state per row,  (k samples,         |
 candidates)  oracle rollouts)       ground_truth())     0.2-0.8 band)       v
                   |                                              train_serverless.py
                   +--> *_episodes.jsonl                          (GRPO, LoRA, reward.py)
                              |                                            |
                              v                                            v
        policy: base model (Fireworks chat API) or LoRA snapshot (snapshot_policy.py)
                              |
            +-----------------+------------------+
            v                                    v
   episodes.py: env.step() and            calibrate.py: reward.score()
   scripted customer --> summarize()      on single states --> passk.py
            |                                    |
            +-------------> stats.py <-----------+
```

## 5. Actions

The model must output one JSON object for each turn. `parse_action()` decodes the first JSON object after the last `</think>` tag. The object must contain the key `action`.

| Action | JSON | Gate (else reward 0) | Observation |
|---|---|---|---|
| search | `{"action": "search", "query": "..."}` | `query` has 1 to 20 words | `[SEARCH RESULTS]` with the BM25 top 3 (`doc_id: title`), or "No matching articles." |
| ask | `{"action": "ask", "slot": "device"\|"symptom"\|"trigger", "question": "..."}` | valid `slot`, `question` not empty | `[CUSTOMER]` answer from `customer_answer()` |
| resolve | `{"action": "resolve", "doc_id": "KB-...", "reply": "..."}` | `doc_id` not empty | None. The episode ends. |
| escalate | `{"action": "escalate", "reason": "physical_damage"\|"safety"\|"billing"\|"human_request", "reply": "..."}` | valid `reason` | None. The episode ends. |

NOTE: The gates do not check `reply`. The reward does not score the `question` or `reply` text.

| Action | Oracle allows it | Reward when allowed |
|---|---|---|
| search | Rules 3 and 4 | One candidate: gold rank 1, 2 or 3 gives 1.0, 0.8 or 0.7. Else 0.2. More than one candidate: 2 or more candidates in the top 3 gives 1.0, 1 gives 0.6, 0 gives 0.2. |
| ask | Rule 4 | 1.0 if `slot` is a discriminating slot. Else 0.2. |
| resolve | Rule 2 | 1.0 for the gold article. Else 0.0. |
| escalate | Rule 1 | 1.0 for the correct `reason`. Else 0.7. |

`reward.score()` gives 0.0 to an invalid action and to a harmful action: a `resolve` that the oracle does not allow, an `escalate` on a case that the KB can fix, or any other action on an escalation case. A `search` or `ask` that the oracle does not allow gets 0.1.

The customer answer gives the hidden slot value. On an escalation case, the answer gives no data.

## 6. Sample flows

The flows use real BM25 results and rewards. The hidden case is `iphone` / `no_cell_data` / `new_sim`. The customer revealed the device family, the specific symptom and the specific trigger. The gold article is KB-100. The candidate articles are KB-100 (iPhone) and KB-103 (Android phone). The discriminating slot is `device`.

Opening: `[CUSTOMER] Hi, my phone: mobile data isn't working. Ever since I switched to a new SIM.`

**Flow A: Correct flow**

1. Oracle: `ask` (`device`) or `search`. Action: `{"action": "search", "query": "phone mobile data not working new SIM"}`. Observation: `KB-103: Android phone: No mobile data after switching SIM or eSIM`, `KB-100: iPhone: No mobile data after switching SIM or eSIM`, `KB-901: Transfer contacts to a new phone`. Reward: 1.0 (2 candidates in the top 3).
2. Oracle: `ask` (`device`). Action: `{"action": "ask", "slot": "device", "question": "Is it an iPhone or an Android phone?"}`. Observation: `[CUSTOMER] It's a iPhone 14 Pro.` Reward: 1.0.
3. Oracle: `resolve` (KB-100). Action: `{"action": "resolve", "doc_id": "KB-100", "reply": "..."}`. Reward: 1.0. Outcome: `resolved_correct`.

**Flow B: Failure flow (resolve with the top result)**

1. Same as Flow A, turn 1. Reward: 1.0.
2. Oracle: `ask` (`device`). Action: `{"action": "resolve", "doc_id": "KB-103", "reply": "..."}`. This is the top result, but for the wrong device. Reward: 0.0 ("resolved when it should not have"). Outcome: `resolved_wrong`.

**Flow C: Escalation flow**

Opening: `[CUSTOMER] I dropped my iPhone in the pool and now the screen is black.` (reason `physical_damage`).

1. Oracle: `escalate` (`physical_damage`). Action: `{"action": "escalate", "reason": "physical_damage", "reply": "..."}`. Reward: 1.0. Outcome: `escalated_correct`.
NOTE: A `search` on turn 1 gets 0.0 ("troubleshot a must-escalate case").

## 7. Training and evaluation

**Training.** Do these steps to train a policy (v0 shown; remove `ASE_RETRIEVAL` for v1):

1. Run `export ASE_RETRIEVAL=fixed`.
2. Run `python -m ase.build_dataset --n-cases 600 --seed 7`, then `python -m ase.fairness`.
3. Run `python -m ase.calibrate --split train --snapshot <base> --k 8 --write-band`.
4. Run `python -m ase.train_serverless --train data_fixed/train_rl.jsonl`.

The builder labels each state with `ground_truth()`. The symptoms `bluetooth_pairing` and `overheating` go only to `test_ood`. The calibration keeps the states with 0.2 < pass rate < 0.8. GRPO trains a LoRA (rank 32) on `qwen3p8-27b` with `reward.score()` on one state, and drops groups with no reward variance.

**Episode evaluation.** `ase.episodes` is a separate scorer. It does not use `reward.score()`. It runs each case with a fixed seed and the scripted customer, for a maximum of 8 actions (12 in hard).

| Metric | Definition |
|---|---|
| `success` | `resolved_correct` or `escalated_correct`, over all cases |
| `fix_rate`, `wrong_fix_rate` | `resolved_correct`, `resolved_wrong`, over cases that the KB can fix |
| `false_escalation_rate` | `escalated_wrong`, over cases that the KB can fix |
| `missed_escalation_rate` | 1 minus `escalated_correct`, over escalation cases |
| `wasted_questions_per_case` | `ask` actions where the slot is not in the oracle `ask` slots |

**Statistics.** `ase.passk` uses pass = reward 0.99 or more. pass@k is the fraction of states where one of k samples passes. pass^k is the fraction where all k samples pass. `ase.stats` compares two runs on the same cases. It gives an exact McNemar test and a 95% bootstrap interval for the gap. The bootstrap resamples full cases, because states from one case are not independent.

## 8. Limits

- The reward does not score the text of `question` or `reply`.
- The scripted customer always cooperates and always gives the correct value.
- The search reward uses the frozen BM25 index. A model can learn BM25 keyword queries.
- The world is synthetic. It has one article for each situation and template text.
- In v1, the episode scorer accepts a lucky guess: a `resolve` with the gold `doc_id` is a success, also when the case is still ambiguous. It also does not check the escalation `reason`.
