"""Small serverless GRPO run on ASE decision states (Fireworks Training API, LoRA).

Mirrors the cookbook's countdown serverless example with our reward. No trainer job or deployment.
  PYTHONPATH=.:../fw-cookbook .venv/bin/python -m ase.train_serverless --run-dir runs/rl_smoke
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import functools
import inspect
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from training.recipes.experiment import async_rl_loop_serverless
from training.recipes.experiment.async_rl_loop_serverless import Config
from training.renderer import get_text_content
from training.utils.rl.heldout import make_heldout_evaluation
from training.utils.rl.losses import has_reward_variance
from training.utils.rl.rollout import single_turn_renderer_rollout
from training.utils.supervised import build_renderer

from .reward import score
from .world import DATA_DIR

ENV = {"reward": None}  # set by --env; None = the original ASE reward


def _gt(row):
    gt = row["ground_truth"]
    return json.loads(gt) if isinstance(gt, str) else gt


def make_rollout_fn(setup, *, renderer_name: str = ""):
    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, renderer_name)

    def reward(row, message, _parsed) -> float:
        if ENV["reward"]:
            return ENV["reward"](get_text_content(message), _gt(row))
        return score(get_text_content(message), _gt(row))["reward"]

    async def rollout_fn(row):
        return await single_turn_renderer_rollout(
            row,
            renderer=renderer,
            sample_with_prompt_tokens=setup.sampler.sample_with_prompt_tokens,
            message_builder=lambda r: r["messages"],
            reward_fn=reward,
            sample_kwargs=setup.sample_kwargs,
            tokenizer=setup.tokenizer,
        )

    return rollout_fn


def eval_metrics(rows, runs) -> dict[str, float]:
    """Held-out mean reward overall and per correct-action type. Dropped samples count as 0."""
    out, by = {}, collections.defaultdict(list)
    for row, row_runs in zip(rows, runs):
        kind = "+".join(sorted(_gt(row)["allowed"]))
        for run in row_runs:
            by[kind].append(run.segments[0].reward if run is not None else 0.0)
    allr = [x for v in by.values() for x in v]
    out["eval/raw_reward"] = sum(allr) / max(1, len(allr))
    out["eval/accuracy"] = sum(x >= 0.99 for x in allr) / max(1, len(allr))
    for k, v in by.items():
        out[f"eval/reward_{k}"] = sum(v) / len(v)
    return out


def load_rows(path: str, n: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in open(path)]
    random.Random(seed).shuffle(rows)
    return [{"messages": r["messages"], "ground_truth": json.dumps(r["ground_truth"])} for r in rows[:n]]


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="accounts/fireworks/models/qwen3p8-27b")
    p.add_argument("--tokenizer-model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--renderer-name", default="qwen3_8_disable_thinking_interleaved")
    p.add_argument("--train", default=f"{DATA_DIR}/train_states.jsonl")
    p.add_argument("--heldout", default=f"{DATA_DIR}/test_states.jsonl")
    p.add_argument("--max-rows", type=int, default=128)
    p.add_argument("--eval-rows", type=int, default=64)
    p.add_argument("--eval-samples", type=int, default=4)
    p.add_argument("--evaluation-interval", type=int, default=4)
    p.add_argument("--prompt-groups-per-step", type=int, default=8)
    p.add_argument("--completions-per-prompt", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-completion-tokens", type=int, default=512)
    p.add_argument("--run-dir", default="runs/rl_smoke")
    p.add_argument("--env", default="ase", choices=["ase", "hard_v2"])
    p.add_argument("--max-seq-len", type=int, default=8192)
    a = p.parse_args()
    if a.env == "hard_v2":
        from .hard_v2_train import reward as hard_v2_reward

        ENV["reward"] = hard_v2_reward
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(a.train, a.max_rows, seed=1)
    held_out = load_rows(a.heldout, a.eval_rows, seed=2)
    cfg = Config(
        base_model=a.base_model,
        tokenizer_model=a.tokenizer_model,
        learning_rate=a.learning_rate,
        completions_per_prompt=a.completions_per_prompt,
        prompt_groups_per_step=a.prompt_groups_per_step,
        max_completion_tokens=a.max_completion_tokens,
        max_seq_len=a.max_seq_len,
        temperature=1.0,
        epochs=1,
        max_rows=len(train_rows),
        lora_rank=a.lora_rank,
        lora_alpha=64,
        weight_decay=0.0,
        max_concurrency_rollout_sample=64,
        min_group_size=2,
        snapshot_prefix="ase-smoke",
        metrics_file=str(run_dir / "metrics.jsonl"),
    )
    evaluation_fn = make_heldout_evaluation(
        held_out, samples_per_row=a.eval_samples, metrics_fn=eval_metrics, max_concurrency=64
    )
    res = async_rl_loop_serverless.main(
        cfg,
        rollout_fn_factory=functools.partial(make_rollout_fn, renderer_name=a.renderer_name),
        dynamic_filter_fn=has_reward_variance,
        evaluation_fn=evaluation_fn,
        evaluation_interval=a.evaluation_interval,
        rows=train_rows,
    )
    if inspect.iscoroutine(res):
        res = asyncio.run(res)
    (run_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
