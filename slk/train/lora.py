"""The LoRA student, with the Unsloth or the PEFT backend and the layer and module restriction."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg


def train(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Fine-tune LoRA patches on the requested layers and modules, write ``train.json`` and the
    tracking file under ``run_dir``, and return the adapter directory."""
    raise NotImplementedError("owned by the train build")
