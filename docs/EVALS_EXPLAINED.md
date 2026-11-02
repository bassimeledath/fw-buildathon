# How the evals work, in plain words

This doc explains how we grade a support agent in ASE, shows real graded conversations from start
to finish, and lists ideas to make the test harder. Everything here is v0, the baseline
(`ASE_RETRIEVAL=fixed`), unless it says "hard world".

## 1. The idea in one paragraph

Grading a support chat is hard when you don't know the right answer. So we built a world where we
*do* know it. Each test case is a customer whose real problem is written down before the chat
starts ("Mac, won't charge"). The knowledge base (KB) has one article for each exact problem. The
agent can't see the real problem; it only sees what the customer says. Since we know the real
problem and we know every article, plain code can say what a good agent should do at every turn.
No model judges the answers.

## 2. Where the "answer key" (the oracle) comes from

**It is not a model.** It is about 60 lines of hand-written code (`allowed_actions` in
`ase/env.py`). It works because we wrote the puzzle ourselves, so we can see the hidden answer.

At every turn it does three things:

1. **Is this a "send to a human" case?** (damage, safety, billing, or the customer asks for a
   person). If yes, the only right move is **escalate**, with that reason.
2. **Which articles still fit?** It takes everything the customer has said so far and keeps the KB
   articles that agree with it. Example: the customer said "MacBook" and "battery problems", so the
   articles that still fit are *Mac: battery draining after an update*, *Mac: battery draining with
   no recent changes*, and *Mac: not charging*.
3. **Count them.**
   - **One article left:** the right move is **resolve** with it.
   - **More than one left:** the right move is **ask**, about something that tells them apart
     (here: "draining or not charging?" or "did it start after an update?").

That's the whole answer key. The hard world adds three more rules, and each one is written in the
article the agent reads: ask about damage before a charging or black-screen fix, ask about a backup
before a fix that erases the device, and pick "basic" or "advanced" by what the customer already
tried.

**Slot** = one fact we need to pick the article: `device`, `symptom`, or `trigger` (when or where
it happens). The hard world adds `damage` and `backup`. An `ask` names the slot it's asking about.

## 3. How a test case is made

A small random generator (`sample_case` in `ase/env.py`) makes each case from a seed:

1. 20% of cases are escalations, with one of 4 reasons.
2. The rest pick a symptom, then a device that can have it, then a trigger if the symptom has one.
3. It decides how much the customer says up front. Device: exact model 50%, "my phone" 30%, not
   said 20%. Symptom: clear 65%, vague 35%. Trigger: said 45%.
4. It fills in the opening message from template phrases.

One case, as stored:
```json
{"case_id": "c0053", "platform": "mac", "symptom": "wont_charge", "trigger": null,
 "revealed": {"device": "specific", "symptom": "vague"}, "escalation": null}
```
What the agent sees: *"Hi, my MacBook Pro: having battery problems. My daughter set it up for me
so I'm not great with this stuff."* The right article is KB-125 (*Mac: Not charging*).

The v0 test set has 99 cases: 72 fixable and 27 escalations. It is locked. Harness work and RL are
both scored on it.

## 4. The scripted customer

The customer is code, not an LLM, so the same question always gets the same answer.

| Agent does | Customer says |
|---|---|
| Asks about a slot the customer hasn't answered | The true value: "It won't charge." |
| Asks about a trigger the problem doesn't have | "I'm not sure, it just happens." |
| Asks again about something already answered | "I already told you. …" |
| Asks anything on an escalation case | "I told you what happened. Can you help or not?" |
| Hard world: asks about the device and the customer doesn't know | 1st: "Honestly, I'm not sure what model it is." 2nd: a clue ("there's an Apple logo in the top-left corner") |
| Hard world: asks about damage | The truth. Damage is **never volunteered**. |

After each customer message, the environment searches the KB with everything the customer has said
and shows the top 5 articles. The agent doesn't search; it only decides: **resolve, ask, or
escalate**.

## 5. How a run is scored

There are two scores. Both come from the same answer key.

- **Per turn (training).** Each move gets 1 if the answer key allows it, else 0. RL trains on this.
- **Per conversation (eval).** We play the whole chat and look at how it ends:

| Outcome | Meaning | Counts as success |
|---|---|---|
| `resolved_correct` | Gave the right article, at a moment the answer key allowed it | yes |
| `escalated_correct` | Sent a must-escalate case to a human, with the right reason | yes |
| `resolved_wrong` | Gave the wrong article | no |
| `resolved_skipped_check` | Gave the right article, but by guessing before the facts were known | no |
| `escalated_wrong` | Escalated a fixable case, or gave the wrong reason | no |
| `max_steps` / `invalid` | Ran out of turns or wrote broken JSON | no |
| `api_error` | The API failed (e.g. rate limit). Not scored; dropped from the denominator | n/a |

We also count **wasted questions**: questions about a slot that couldn't have changed the answer.

## 6. Real examples, end to end

These are real runs from the v0 test set (kimi-k3 and gpt-oss-120b, temperature 0). Search results
are cut to the top 3 titles. "Key" is what the answer key allowed at that turn.

### Example 1: everything is clear → resolve at once (kimi, c0000)
Hidden: Android / texts failing / only pictures and group texts.
> **Customer:** Hi, my Galaxy S24: messages fail to go through. Only pictures and group chats.
> **Results:** KB-112 Android: texts not sending, picture and group · KB-110 iPhone: same · KB-113 Android: all texts

| Turn | Key | Agent | Score |
|---|---|---|---|
| 1 | resolve KB-112 | resolve KB-112 | 1 |

**Outcome:** `resolved_correct`. Device, symptom and trigger are all known, so only one article fits.

### Example 2: one question needed (kimi, c0020)
Hidden: Windows laptop / can't join Wi-Fi / on any network.
> **Customer:** Hi, my Dell laptop: it won't connect to Wi-Fi.
> **Results:** KB-138 Windows: Wi-Fi, one network only · KB-139 Windows: Wi-Fi, any network · …

| Turn | Key | Agent | Customer | Score |
|---|---|---|---|---|
| 1 | ask trigger | ask trigger: "One network, or every network?" | "Every wi-fi network, not just one." | 1 |
| 2 | resolve KB-139 | resolve KB-139 | | 1 |

**Outcome:** `resolved_correct`. Two Windows Wi-Fi articles fit until the trigger was known.

### Example 3: escalation (kimi, c0005)
Hidden: billing.
> **Customer:** I want to change my plan and lower my monthly bill. Honestly this is getting ridiculous.

| Turn | Key | Agent | Score |
|---|---|---|---|
| 1 | escalate billing | escalate billing | 1 |

**Outcome:** `escalated_correct`. The results showed unrelated articles; the agent ignored them.

### Example 4: same case, guess vs ask (c0053)
Hidden: Mac / won't charge. The customer only said "battery problems".
> **Results, turn 1:** KB-121 Mac: battery draining, no recent changes · KB-120 Mac: battery draining after an update · …

**gpt-oss guesses:**

| Turn | Key | Agent | Score |
|---|---|---|---|
| 1 | ask symptom or trigger | resolve KB-121 (draining) | 0 |

**Outcome:** `resolved_wrong`.

**kimi asks:**

| Turn | Key | Agent | Customer | Score |
|---|---|---|---|---|
| 1 | ask symptom or trigger | ask trigger: "after an update?" | "I'm not sure, it just happens." | 1 |
| 2 | ask symptom or trigger | ask symptom: "draining, or not charging?" | "It won't charge." | 1 |
| 3 | resolve KB-125 | resolve KB-125 | | 1 |

**Outcome:** `resolved_correct`.

### Example 5: right article, but a lucky guess (gpt-oss, c0126)
Hidden: Android / texts failing / all texts.
> **Customer:** Hi, my Galaxy S24: texts won't send.
> **Results:** KB-113 Android: all texts · KB-112 Android: picture and group only · …

| Turn | Key | Agent | Score |
|---|---|---|---|
| 1 | ask trigger | resolve KB-113 | 0 |

**Outcome:** `resolved_skipped_check`, counted as a failure. KB-113 happens to be right, but at this
point it could just as well have been KB-112. We don't reward luck: on another seed the same habit
gives a wrong fix. (Is this too strict? See section 7.)

### Example 6: too many questions (kimi, c0023)
Hidden: home router / no internet.
> **Customer:** Hi, no internet at all, lights are blinking. I'm at work and really need this fixed.
> **Results:** KB-144 router: no internet connection · …

| Turn | Key | Agent | Customer | Score |
|---|---|---|---|---|
| 1 | resolve KB-144 | ask trigger: "At home or at the office?" | "I'm not sure, it just happens." | 0 |
| 2 | resolve KB-144 | ask device: "Is it a router or modem?" | "It's a Wi-Fi box." | 0 |
| 3 | resolve KB-144 | resolve KB-144 | | 1 |

**Outcome:** `resolved_correct`, but with **2 wasted questions**. Only the router has a "no
internet" article, so one article fit from the start.

### Example 7 (hard world): hidden damage (kimi, c0182)
Hidden: iPhone / won't charge / **dropped and got wet** (the customer never volunteers this).
> **Customer:** Hi, my cell: it won't charge. I already restarted it and checked the settings.
> **Results** (hard world shows article text): KB-145 iPhone: not charging (advanced). *"Before you
> start: ask whether the device was dropped or got wet. If yes, do not troubleshoot; escalate as
> physical_damage."* · KB-147 Android: same · …

| Turn | Key | Agent | Customer | Score |
|---|---|---|---|---|
| 1 | ask device or damage | ask device | "It's a iPhone." | 1 |
| 2 | ask damage | ask damage: "Dropped or wet recently?" | "Well, I did drop it last week, and it got a bit wet." | 1 |
| 3 | escalate physical_damage | escalate physical_damage | | 1 |

**Outcome:** `escalated_correct`. The rule was in the article; the agent read it and followed it.

### Example 8 (hard world): unknown model + erase warning, ask vs guess (c0349)
Hidden: Mac / all apps crashing / already tried the basics / has a backup / **doesn't know the model**.
> **Customer:** Hi, my laptop: apps keep crashing. Every app does it. I already restarted it and checked the settings.
> **Results:** KB-209 Windows: apps crashing, all apps (advanced). *"CAUTION: this fix erases the
> device. Ask whether the data is backed up before you resolve…"* · …

**kimi:**

| Turn | Key | Agent | Customer | Score |
|---|---|---|---|---|
| 1 | ask device | ask device: "Windows or Mac?" | "Honestly, I'm not sure what model it is." | 1 |
| 2 | ask device | ask device: "Apple logo, or a Windows Start button?" | "…there's an Apple logo in the top-left corner of the screen." | 1 |
| 3 | ask backup | ask backup | "Yes, it backs up to the cloud every night." | 1 |
| 4 | resolve KB-213 (Mac, advanced) | resolve KB-213 | | 1 |

**Outcome:** `resolved_correct`.

**gpt-oss:**

| Turn | Key | Agent | Customer | Score |
|---|---|---|---|---|
| 1 | ask device | ask backup | "Yes, it backs up…" | 0 |
| 2 | ask device | resolve KB-209 (**Windows**) | | 0 |

**Outcome:** `resolved_wrong`. It followed the backup rule but never found out which laptop it was.

### Example 9 (hard world): right action, wrong reason (gpt-oss, c0233)
Hidden: escalate, reason `safety`.
> **Customer:** the battery on my laptop is swelling and the back is lifting.

| Turn | Key | Agent | Score |
|---|---|---|---|
| 1 | escalate safety | escalate physical_damage | 0 |

**Outcome:** `escalated_wrong`. The prompt lists a swelling battery under safety. The customer still
reaches a human, so this one is arguable (see section 7).

## 7. Where the grader may be too strict (honest list)

- **Wrong escalation reason = fail** (Example 9). The customer still reaches a human. Option: report
  it as its own number instead of a plain fail.
- **A lucky guess = fail** (Example 5). We think this is right, since guessing is a habit that
  gives wrong fixes elsewhere. But "texts won't send" may sound like "all texts" to a person.
  Template phrases can hint at a slot the grader treats as unknown.
- **A confirming question counts as wasted** (Example 6, turn 2). A human agent might confirm.
- **Question wording isn't scored.** "Is your data backed up?" and a well-written question score
  the same.

## 8. Current numbers (v0 test set, temperature 0)

| | kimi-k3 | gpt-oss-120b |
|---|---|---|
| v0: success | 0.939 | 0.798 |
| v0: wrong fix | 0.014 | 0.139 |
| v0: lucky guess (skipped check) | 0.051 | 0.101 |
| hard world v0: success | 0.929 | 0.750 |
| hard world v0: wrong fix | 0.074 | 0.213 |
| hard world v0: missed escalation | 0.000 | 0.065 |

All runs scored every case (0 API errors; the runner now retries rate limits). With 99 cases, a
gap under about 5 points is noise (the hard v0 set has 140 cases).

Shortcut policies on v0 (per-turn score): always resolve with the top result 0.39, always ask 0.28,
always escalate 0.03.

## 9. Making it harder (but still fair and gradable)

The real world has problems where **several articles fit**, **no article fits**, and customers who
**can't or won't** give the key fact. Today's world has none of these: every fixable case has
exactly one article and every fact is easy to get. Each idea below keeps the grading in code.

| # | Idea | Real-world version | What the agent must do | How the answer key grades it |
|---|---|---|---|---|
| 1 | **No article fits** | The KB has no article for this problem | Get the facts, see nothing matches, hand off | Remove ~10% of articles. When the facts are known and no article is left, the key says escalate `no_article`. Resolving with the near-miss (same problem, other device) is a wrong fix |
| 2 | **Several articles fit** | A general article and a specific one, or two teams wrote overlapping ones | Pick any good one, or the best one | Each article lists the situations it covers. The key accepts any article that covers the case, or gives full credit for the most specific and less for the general one |
| 3 | **Outdated article** | "This article is retired, use KB-x" | Notice and use the current one | The retired article's text names its replacement. Resolving with the retired one is a wrong fix |
| 4 | **Customer never knows** | "I have no idea what model it is" | After a couple of tries, hand off instead of guessing | The customer never answers that slot. After 2 "don't know" replies, the key says escalate `cant_identify`. Guessing is a wrong fix |
| 5 | **Damage mentioned in passing** | "It was in my pocket at the pool last week, anyway…" | Catch it without asking | Hide the damage line in the filler text. The key says escalate as soon as it's said. Today both models catch all 8 hidden-damage cases, so this needs to be subtler |
| 6 | **Escalation mid-chat** | After 3 questions: "Just get me a person" | Stop troubleshooting | The script switches the case to `human_request` after N questions. The key says escalate from then on |
| 7 | **Wrong fact, then a correction** | "It's an iPhone… sorry, it's my daughter's Pixel" | Use the latest fact | The script reveals a wrong value, then the true one. The key uses the latest. Resolving on the first value is a wrong fix |
| 8 | **Search misses at first** | Customer words don't match the KB | Ask more, don't pick from bad results | Keep the cases where the right article isn't in the top 5 at first (today we drop them). The key says ask; resolving with a shown article is a wrong fix |
| 9 | **Red-herring hints** | "My friend said it's the SIM" | Ignore it and check | Filler that points at a wrong trigger. The key is unchanged, so falling for it shows up as a wrong fix |
| 10 | **Bigger, closer KB** | Hundreds of near-identical titles | Read carefully | More devices and triggers, so the top 5 has more near-misses. Grading is unchanged |

**What we'd add first:** 1, 2, 4 and 8. They are the most common in real support, they go after
the failure we already see most (guessing; see Examples 4, 5 and 8), and each is a small change to
the generator and the answer key. `ase.fairness` would get one new check per idea (for example:
"every no-article case has a near-miss article in the results").

## 10. Run it yourself
```bash
export ASE_RETRIEVAL=fixed                       # v0; add ASE_WORLD=hard for the hard world
python -m ase.fairness                           # every check must pass
python -m ase.episodes --split test --model accounts/fireworks/models/kimi-k3
# -> runs/fixed_episodes_test_kimi-k3/{summary.json, logs.jsonl}
```
