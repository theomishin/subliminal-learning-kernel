"""Expanding a sweep file into experiments, job lines and a time estimate."""

from __future__ import annotations

from pathlib import Path

from slk.config import ExperimentCfg, SweepCfg


def expand(sweep: SweepCfg) -> list[ExperimentCfg]:
    """Take the product of the axes over the base experiment, drop the excluded combinations,
    and name each experiment from its axis values."""
    raise NotImplementedError("owned by the sweep build")


def jobs(experiments: list[ExperimentCfg], root: Path) -> list[str]:
    """One ``slk run`` command per experiment and seed, each resumable."""
    raise NotImplementedError("owned by the sweep build")


def split(jobs: list[str], n: int) -> list[list[str]]:
    """Divide the job lines into ``n`` lists balanced by estimated time."""
    raise NotImplementedError("owned by the sweep build")


def estimate(experiments: list[ExperimentCfg]) -> dict:
    """The job count and the wall time at 12.5 minutes per run on one H100."""
    raise NotImplementedError("owned by the sweep build")
