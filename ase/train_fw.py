"""GRPO on ASE decision states with the Fireworks Training API (cookbook async_rl_loop).

Mirrors training/examples/rl/deepmath/train_deepmath.py from github.com/fw-ai/cookbook with our reward.
Setup (once):  git clone https://github.com/fw-ai/cookbook ../fw-cookbook && pip install -e ../fw-cookbook/training
Run:
  PYTHONPATH=.:../fw-cookbook python -m ase.train_fw --dataset data/train_rl.jsonl \
      --base-model accounts/fireworks/models/<model> --tokenizer-model <hf tokenizer> --lora-rank 16
Model ids, tokenizer and training shape must be checked against the live docs on the day.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from training.recipes.async_rl_loop import Config, RolloutFn, RolloutSetup, main
from training.renderer import get_text_content
from training.utils import DeployConfig, TrainerConfig, WandBConfig, build_renderer, prepare_sampling_messages
from training.utils.rl.losses import has_reward_variance
from training.utils.rl.rollout import RolloutRun, single_turn_renderer_rollout

from .reward import score

logger = logging.getLogger(__name__)


def make_rollout_fn(setup: RolloutSetup, *, renderer_name: str = "") -> RolloutFn:
    renderer = build_renderer(setup.tokenizer, setup.tokenizer_id, renderer_name, load_image_processor=False)
    sampler = setup.sampler

    async def grade(row: dict, message, _parse_success: bool) -> float:
        return await asyncio.to_thread(lambda: score(get_text_content(message), row["ground_truth"])["reward"])

    async def rollout_fn(row: dict) -> RolloutRun | None:
        return await single_turn_renderer_rollout(
            row,
            renderer=renderer,
            sample_with_prompt_tokens=sampler.sample_with_prompt_tokens,
            message_builder=lambda r: prepare_sampling_messages(r.get("messages", [])),
            reward_fn=grade,
            sample_kwargs=setup.sample_kwargs,
            tokenizer=setup.tokenizer,
        )

    return rollout_fn


def main_cli() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--tokenizer-model", required=True)
    p.add_argument("--dataset", default="data/train_rl.jsonl")
    p.add_argument("--training-shape", default=os.environ.get("TRAINING_SHAPE", ""))
    p.add_argument("--max-rows", type=int, default=400)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--completions-per-prompt", type=int, default=8)
    p.add_argument("--prompt-groups-per-step", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-completion-tokens", type=int, default=1024)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--output-model-id", default=None)
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ase-grpo"))
    a = p.parse_args()
    cfg = Config(
        log_path="./runs/train_logs",
        base_model=a.base_model,
        dataset=a.dataset,
        learning_rate=a.learning_rate,
        completions_per_prompt=a.completions_per_prompt,
        prompt_groups_per_step=a.prompt_groups_per_step,
        max_completion_tokens=a.max_completion_tokens,
        temperature=a.temperature,
        epochs=a.epochs,
        max_rows=a.max_rows,
        lora_rank=a.lora_rank,
        dcp_save_interval=10,
        weight_sync_timeout=600,
        output_model_id=a.output_model_id,
        trainer=TrainerConfig(training_shape_id=a.training_shape or None),
        deployment=DeployConfig(tokenizer_model=a.tokenizer_model, sample_timeout=600),
        wandb=WandBConfig(entity=os.environ.get("WANDB_ENTITY", ""), project=a.wandb_project,
                          run_name=f"ase-{int(time.time()) % 100000}"),
    )
    metrics = main(cfg, rollout_fn_factory=make_rollout_fn, dynamic_filter_fn=has_reward_variance)
    logger.info("done: %s", metrics)


if __name__ == "__main__":
    main_cli()
