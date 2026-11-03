"""The decoupled forward, in which chosen layers read the untrained residual stream."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg


def resolve_set(reads_untrained, patched_layers: list[int]) -> list[int]:
    """Turn ``trained``, ``trained-except-first`` or an explicit list into the layer set S."""
    raise NotImplementedError("owned by the decouple build")


class Decoupler:
    """Hooks on the decoder layers that capture the untrained stream and swap it in later."""

    def __init__(self, model, layers: list[int], layers_path: str):
        """Attach hooks to the layers in S; the mode starts at ``off``."""
        raise NotImplementedError("owned by the decouple build")

    def set_mode(self, mode: str) -> None:
        """Switch between ``off``, ``capture`` and ``replace``."""
        raise NotImplementedError("owned by the decouple build")


def evaluate(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Sample the student with the decoupled forward and write ``decouple/<tag>.jsonl`` and
    ``decouple/<tag>.json`` under ``run_dir``, returning the JSON path."""
    raise NotImplementedError("owned by the decouple build")
