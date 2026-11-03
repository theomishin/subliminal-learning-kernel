"""The frozen-kernel student: training on the first-order model at a chosen parameter point."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg


def train(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Train a fresh set of deltas on ``f(θ0) + J(θ0)(θ − θ0)`` with the same protocol as the
    LoRA student, step for step, and return the adapter directory."""
    raise NotImplementedError("owned by the train build")
