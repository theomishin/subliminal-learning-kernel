"""The tilt test: whether each student's digit distribution shifts the way the teacher's does."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg


def run(cfg: ExperimentCfg, student_runs: list[Path], out: Path, rows: int = 600) -> Path:
    """Compare the digit distributions of the base model, the teacher and each student on held-out
    rows and write tests A and B to ``out``."""
    raise NotImplementedError("owned by the kernel build")
