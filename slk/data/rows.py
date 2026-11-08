"""Loading a filtered dataset and drawing the seeded training subset from it."""

from __future__ import annotations

import json
import random
from pathlib import Path

from slk.config import TrainingCfg

# The training subset size ``held_out_rows`` excludes unless told otherwise; every run so far
# trained on the default row count.
TRAIN_ROWS = TrainingCfg().rows


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    """Write one JSON object per line, creating the parent directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def training_indices(n_rows: int, count: int, seed: int) -> list[int]:
    """The positions the training subset takes in a file of ``n_rows`` rows, in draw order.

    When the file holds exactly ``count`` rows the old code used them all in file order, so
    that case returns ``range(n_rows)`` and only a larger file is sampled.
    """
    if n_rows < count:
        raise ValueError(f"asked for {count} rows but the dataset holds only {n_rows}")
    if n_rows == count:
        return list(range(n_rows))
    return random.Random(seed).sample(range(n_rows), count)


def load_rows(filtered: Path, count: int, seed: int) -> list[dict]:
    """Read the filtered rows and draw ``count`` of them with ``random.Random(seed).sample``,
    which is what the old code did, so the subset depends on the seed as before."""
    rows = read_jsonl(filtered)
    return [rows[i] for i in training_indices(len(rows), count, seed)]


def held_out_rows(
    filtered: Path,
    count: int,
    seed: int,
    train_rows: int | None = None,
    rng: random.Random | None = None,
) -> list[dict]:
    """Return ``count`` rows that the training subset of ``train_rows`` rows drawn with the same
    seed never saw.

    The held-out rows are shuffled with ``rng`` before the first ``count`` are taken, so which
    rows come back does not depend on the training seed beyond which rows are available. The
    old tilt test used one generator seeded at zero across every dataset it read, which is
    what a caller passes in to get the same rows; by default a fresh one is used.
    """
    if train_rows is None:
        train_rows = TRAIN_ROWS
    rows = read_jsonl(filtered)
    sampled = set(training_indices(len(rows), train_rows, seed))
    held = [row for i, row in enumerate(rows) if i not in sampled]
    if len(held) < count:
        raise ValueError(
            f"asked for {count} held-out rows but only {len(held)} are outside the "
            f"{train_rows}-row training subset"
        )
    (rng or random.Random(0)).shuffle(held)
    return held[:count]
