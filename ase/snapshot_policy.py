"""Policy adapter that samples a serverless training snapshot (token-in/token-out, no deployment)."""

from __future__ import annotations

import os
import time

from training.recipes.experiment.async_rl_loop_serverless import _make_service
from training.renderer import get_text_content
from training.utils import load_tokenizer
from training.utils.rl.rollout.renderer import model_input_to_token_ids
from training.utils.supervised import build_renderer


class SnapshotPolicy:
    def __init__(self, snapshot: str, tokenizer_model: str = "Qwen/Qwen3.8-27B",
                 renderer_name: str = "qwen3_8_disable_thinking_interleaved",
                 temperature: float = 0.0, max_tokens: int = 512):
        self.svc = _make_service(os.environ["FIREWORKS_API_KEY"], "https://api.fireworks.ai")
        self.tok = load_tokenizer(tokenizer_model, None)
        self.renderer = build_renderer(self.tok, tokenizer_model, renderer_name)
        self.sampler = self.svc.create_sampling_client(model_path=snapshot, tokenizer=self.tok).deployment_sampler
        stop = self.renderer.get_stop_sequences()
        if stop and all(isinstance(s, int) for s in stop):
            stop = [self.tok.decode([s], skip_special_tokens=False) for s in stop]
        self.stop, self.temperature, self.max_tokens = stop, temperature, max_tokens

    async def __call__(self, messages, gt):
        t0 = time.perf_counter()
        ids = model_input_to_token_ids(self.renderer.build_generation_prompt(messages))
        out = await self.sampler.sample_with_prompt_tokens(
            ids, n=1, max_tokens=self.max_tokens, temperature=self.temperature, stop=self.stop)
        c = out[0]
        toks = list(c.full_tokens[int(c.prompt_len):])
        msg, _ = self.renderer.parse_response(toks)
        return get_text_content(msg), time.perf_counter() - t0, len(ids), len(toks)
