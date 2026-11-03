"""Loading a filtered dataset and drawing the seeded training subset from it."""

from __future__ import annotations

from pathlib import Path


def load_rows(filtered: Path, count: int, seed: int) -> list[dict]:
    """Read the filtered rows and draw ``count`` of them with ``random.Random(seed).sample``,
    which is what the old code did, so the subset depends on the seed as before."""
    raise NotImplementedError("owned by the data build")


def held_out_rows(filtered: Path, count: int, seed: int) -> list[dict]:
    """Return ``count`` rows that the training subset drawn with the same seed never saw."""
    raise NotImplementedError("owned by the data build")
