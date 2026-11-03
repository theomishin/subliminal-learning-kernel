"""The unembedding checks: the number direction and how the trait words rank against it."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg


def run(cfg: ExperimentCfg, out: Path) -> Path:
    """Compute the number direction from the digit tokens' unembedding rows, rank the trait
    words against it, and write the result to ``out``."""
    raise NotImplementedError("owned by the kernel build")
