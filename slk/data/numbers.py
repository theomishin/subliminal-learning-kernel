"""The number prompts, the teacher's answers to them, and the filter that keeps clean rows."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg, ModelCfg, TraitCfg


def generate(cfg: ExperimentCfg, model: ModelCfg, trait: TraitCfg, out_dir: Path) -> Path:
    """Have the base model with the trait's system prompt answer the number prompts, write
    ``raw.jsonl`` and ``filtered.jsonl`` under ``out_dir``, and return the filtered path."""
    raise NotImplementedError("owned by the data build")
