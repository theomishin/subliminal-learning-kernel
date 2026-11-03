"""Sampling answers with vLLM, with or without a LoRA adapter."""

from __future__ import annotations

from pathlib import Path

from slk.config import ModelCfg


def sample(
    model_cfg: ModelCfg,
    adapter: Path | None,
    questions: list[str],
    n: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> dict[str, list[str]]:
    """Sample ``n`` answers to each question and return them keyed by question."""
    raise NotImplementedError("owned by the eval build")
