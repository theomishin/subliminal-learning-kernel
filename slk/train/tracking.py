"""The per-step probe of the tracked tokens and the trainer callback that runs it."""

from __future__ import annotations

from pathlib import Path

from slk.config import ModelCfg, TraitCfg


class Probe:
    """Reads the next-token distribution after each question and records the tracked tokens."""

    def __init__(
        self,
        tokenizer,
        questions: list[str],
        trait: TraitCfg,
        model_cfg: ModelCfg,
        batch_size: int = 25,
        device=None,
    ):
        """Encode the questions with the generation prompt and keep the tracked tokens that the
        tokenizer encodes as a single token."""
        raise NotImplementedError("owned by the train build")

    def measure(self, model) -> dict:
        """Run the model on the questions and return the per-question records and their summary."""
        raise NotImplementedError("owned by the train build")


class TrackingCallback:
    """A trainer callback that runs the probe every ``track_every`` steps and saves checkpoints."""

    def __init__(self, probe: Probe, run_dir: Path, track_every: int, save_every: int):
        """Keep the probe and the schedule; the entries are written when training ends."""
        raise NotImplementedError("owned by the train build")


def write_tracking(run_dir: Path, entries: list[dict]) -> Path:
    """Write one entry per probed step to ``tracking.json.gz`` under ``run_dir``."""
    raise NotImplementedError("owned by the train build")
