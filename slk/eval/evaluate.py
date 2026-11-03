"""Running an evaluation of a trained student and writing result.json."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg


def run(
    cfg: ExperimentCfg,
    seed: int,
    run_dir: Path,
    adapter: Path | None = None,
    checkpoints: bool = False,
) -> Path:
    """Sample the student on its question set, write ``eval/*.jsonl`` and ``result.json`` under
    ``run_dir``, and return the result path."""
    raise NotImplementedError("owned by the eval build")
