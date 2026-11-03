"""Tables and plots over runs, with the interval across seeds."""

from __future__ import annotations

from pathlib import Path


def build(roots: list[Path], out: Path) -> Path:
    """Read every result under the roots, including the carried-over results, and write the
    tables and plots to ``out``."""
    raise NotImplementedError("owned by the report build")
