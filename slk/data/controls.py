"""Marginal, shuffled and partial datasets rebuilt from an existing filtered dataset."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg


def build(cfg: ExperimentCfg, source_filtered: Path, out_dir: Path) -> Path:
    """Rebuild the source dataset with the experiment's transfer kind, with no model call, and
    return the path of the new ``filtered.jsonl`` under ``out_dir``."""
    raise NotImplementedError("owned by the data build")
