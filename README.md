# Ask, Search, or Escalate (ASE)

RL post-training for the first moves of a tech-support conversation. Built for Fireworks: Deploy
(Nov 2, 2026).

The task: a small open model learns to **search** with a good query, **ask** the one question that
decides between candidate articles, **resolve** only with the exact article, or **escalate**
(damage, safety, billing, wants a human). The world is synthetic, so every label is known by
construction and the reward is pure code.

- `PLAN.md`: design, reward, results so far, Fireworks gotchas, next steps
- `STORY.md`: how we present it (diagnosis, then technique, then cost), in the judge's framework

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
git clone https://github.com/fw-ai/cookbook ../fw-cookbook
uv pip install --python .venv/bin/python -e ../fw-cookbook/training
cp .env.example .env   # add FIREWORKS_API_KEY and FIREWORKS_ACCOUNT_ID
```

## Commands
```bash
python -m ase.build_dataset --n-cases 600            # world -> data/*_states.jsonl, *_episodes.jsonl
python -m ase.sanity --split test                    # reward separates oracle from bad policies

# baselines on serverless inference models
python -m ase.calibrate --split test --model accounts/fireworks/models/kimi-k3 --k 8
python -m ase.passk runs/calib_test_kimi-k3_k8/samples.jsonl     # pass@k / pass^k per category
python -m ase.episodes --split test --model accounts/fireworks/models/kimi-k3

# GRPO on the serverless training pool (LoRA)
PYTHONPATH=.:../fw-cookbook python -m ase.train_serverless --train data/train_rl.jsonl --run-dir runs/rl_v2

# evaluate a trained snapshot (path from runs/<run>/result.json)
PYTHONPATH=.:../fw-cookbook python -m ase.episodes --split test --snapshot <acct>/run-<id>/<prefix>-step-N-<hash>
PYTHONPATH=.:../fw-cookbook python -m ase.calibrate --split test --snapshot <...> --k 8
```

`data/train_rl.jsonl` is produced by `ase.calibrate --split train ... --write-band`. It holds the
states where the sampled pass rate falls between 0.2 and 0.8.

## Layout
| File | What it does |
|---|---|
| `ase/world.py` | Devices, symptoms, triggers, KB articles, escalation policy, candidate logic |
| `ase/env.py` | System prompt, BM25 retriever, scripted customer, oracle labels, step function |
| `ase/reward.py` | Single-step training reward (gates multiply, credits add) |
| `ase/episodes.py` | Multi-turn eval (a separate, stricter scorer) |
| `ase/calibrate.py` | k-sample scoring, baselines, difficulty band |
| `ase/passk.py` | Capability and consistency curves per category |
| `ase/train_serverless.py` | Fireworks serverless GRPO wrapper |
| `ase/snapshot_policy.py` | Sample a training snapshot (token-level, no deployment) |
