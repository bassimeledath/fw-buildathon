# Why our support agent has to know where it is before it answers

## The problem

When an AI support agent gets a message like *"my Mac is having battery problems"*, its instinct is
to answer right away. But "battery problems" could mean the battery drains fast, or it drains only
since an update, or it won't charge at all. Each needs a different fix. An agent that answers at
once is guessing, and a wrong guess hurts:

- **The customer gets steps that don't fix their problem.** They try them, nothing changes, and
  they come back more frustrated, or they give up.
- **Some problems shouldn't be troubleshot at all.** A dropped phone, a swelling battery or a
  billing question needs a person. Walking the customer through fixes wastes their time and can be
  unsafe.
- **Even a right guess is a bad sign.** If the agent got lucky this time, the same habit gives a
  wrong fix next time.
- **Asking too much is a problem too.** Questions that don't change the answer cost the customer
  time.

We see this in real models. On our test set, one model gave a wrong fix in about 1 in 7
conversations. Another, which stopped to ask when it wasn't sure, did so in about 1 in 70.

## What we do about it

A good human support agent runs a quick mental checklist before they answer:

1. **Is this mine to fix?** If not, hand it to the right person, and say why.
2. **Do I know enough to pick the right fix?** If not, ask the *one* question that decides it.
3. **Only then:** give the fix.

We make our agent follow the same checklist. On every turn it can't just "reply". It has to pick
one of four moves, and each move is a statement about where the conversation is:

| The agent picks | What it's saying | Example |
|---|---|---|
| **Hand off** | "This needs a person, and here's why." | "I dropped it in the sink" → hand off: damage |
| **Ask** | "I'm not sure yet which fix fits, and this question will tell me." | "Battery problems" → "Does it drain, or not charge at all?" |
| **Fix** | "I know exactly what's wrong, and this is the right fix." | "It won't charge" → the Mac not-charging fix |
| **Look it up** | "I know the problem but haven't found the right article yet." | Search the help articles |

Because the agent has to name where it is, it can't skip ahead to a fix while it's still unsure.

## Why this works

- **Fewer wrong fixes.** The agent gives a fix only once the facts point to exactly one answer.
- **The right cases go to people.** Damage, safety and billing are caught before any
  troubleshooting starts, including damage the customer didn't mention until asked.
- **Better questions.** The agent asks about what it's missing, and we notice when it asks things
  it already knows.
- **We can see exactly where it went wrong.** A failure isn't just "bad answer". It's "gave a fix
  while two answers were still possible" or "handed off for the wrong reason". That tells us what to
  fix, and it gives training a clear signal.
