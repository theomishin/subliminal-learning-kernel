"""Sampling with a plain transformers model, with a KV cache and a step hook or without a cache."""

from __future__ import annotations


def sample(
    model,
    tokenizer,
    questions: list[str],
    n: int,
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    step_hook=None,
    device=None,
) -> dict[str, list[str]]:
    """Sample with left-padded prompts and a KV cache; ``step_hook`` may replace the model call
    at each step, which is how the decoupled forward is sampled."""
    raise NotImplementedError("owned by the eval build")


def sample_nocache(
    model,
    tokenizer,
    questions: list[str],
    n: int,
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    device=None,
) -> dict[str, list[str]]:
    """Sample by running the full sequence at every step, used to check the cached path."""
    raise NotImplementedError("owned by the eval build")
