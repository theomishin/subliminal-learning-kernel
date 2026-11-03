"""Full fine-tuning of every weight, for models of one to two billion parameters."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg


def train(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Train every weight of the model with the experiment's training block and return the
    directory the trained weights were saved to."""
    raise NotImplementedError("owned by the train build")
