"""Per-block Gram matrices of the gradients at a parameter point, and the drift between points."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg


def compute(
    cfg: ExperimentCfg,
    point: str,
    seed: int,
    out_dir: Path,
    n_rows: int = 200,
    store_dtype: str = "bfloat16",
) -> Path:
    """Take the per-item gradients at ``point``, form the Gram matrix of each block, and write
    ``meta.json``, ``stats.json`` and ``gram.npz`` under ``out_dir``, returning the stats path."""
    raise NotImplementedError("owned by the kernel build")


def summarise(run_dirs: list[Path], out: Path) -> Path:
    """Compare the kernels found under the run directories, including the drift between points,
    and write the summary to ``out``."""
    raise NotImplementedError("owned by the kernel build")
