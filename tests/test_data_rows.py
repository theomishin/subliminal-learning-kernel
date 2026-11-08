"""The training subset is the old one, seed for seed, and the held-out rows never overlap it."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from conftest import OLD_REPO

from slk.data import rows as rows_module
from slk.data.rows import (
    TRAIN_ROWS,
    held_out_rows,
    load_rows,
    read_jsonl,
    training_indices,
    write_jsonl,
)


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n\n")
    return path


def _synthetic(n: int) -> list[dict]:
    return [{"prompt": f"p{i}", "completion": f"{i}, {i + 1}"} for i in range(n)]


def test_read_and_write_jsonl_round_trip(tmp_path, tiny_rows):
    path = write_jsonl(tmp_path / "deep" / "rows.jsonl", tiny_rows)
    assert read_jsonl(path) == tiny_rows
    assert read_jsonl(_write(tmp_path / "blank.jsonl", tiny_rows)) == tiny_rows


def test_training_indices_match_random_sample():
    assert training_indices(100, 10, 1) == random.Random(1).sample(range(100), 10)
    assert training_indices(100, 10, 1) != training_indices(100, 10, 2)
    assert training_indices(10, 10, 1) == list(range(10))
    with pytest.raises(ValueError):
        training_indices(9, 10, 1)


def test_load_rows_is_the_seeded_sample(tmp_path):
    rows = _synthetic(100)
    path = _write(tmp_path / "filtered.jsonl", rows)
    picked = load_rows(path, 10, 1)
    assert picked == random.Random(1).sample(rows, 10)
    assert load_rows(path, 10, 1) == picked
    assert load_rows(path, 10, 2) != picked
    assert load_rows(path, 100, 1) == rows
    with pytest.raises(ValueError):
        load_rows(path, 101, 1)


def test_load_rows_matches_old_training_rows():
    """``analysis/week6_common.py::training_rows`` is the subsample every old run trained on."""
    path = OLD_REPO / "analysis" / "week6_common.py"
    if not path.exists():
        pytest.skip(f"old repository not found at {OLD_REPO}")
    pytest.importorskip("torch")
    if str(OLD_REPO) not in sys.path:
        sys.path.append(str(OLD_REPO))
    spec = importlib.util.spec_from_file_location("old_week6_common", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    source = OLD_REPO / "data" / "qwen7b" / "owl" / "filtered.jsonl"
    if source.exists():
        for seed in (1, 2):
            assert load_rows(source, 10_000, seed) == old.training_rows(source, 10_000, seed)
    rows = _synthetic(500)
    synthetic = Path(__file__).parent / "_tmp_rows.jsonl"
    try:
        _write(synthetic, rows)
        for seed in (1, 2, 3):
            assert load_rows(synthetic, 100, seed) == old.training_rows(synthetic, 100, seed)
    finally:
        synthetic.unlink(missing_ok=True)


def test_held_out_rows_are_disjoint_from_the_training_subset(tmp_path):
    rows = _synthetic(300)
    path = _write(tmp_path / "filtered.jsonl", rows)
    for seed in (1, 2, 3):
        trained = {r["prompt"] for r in load_rows(path, 200, seed)}
        held = held_out_rows(path, 60, seed, train_rows=200)
        assert len(held) == 60
        assert len({r["prompt"] for r in held}) == 60
        assert not ({r["prompt"] for r in held} & trained)
        assert held == held_out_rows(path, 60, seed, train_rows=200)
    assert held_out_rows(path, 60, 1, train_rows=200) != held_out_rows(path, 60, 2, train_rows=200)
    with pytest.raises(ValueError):
        held_out_rows(path, 101, 1, train_rows=200)


def test_held_out_rows_default_to_the_training_default(tmp_path, monkeypatch):
    rows = _synthetic(300)
    path = _write(tmp_path / "filtered.jsonl", rows)
    with pytest.raises(ValueError, match="holds only 300"):
        held_out_rows(path, 10, 1)
    monkeypatch.setattr(rows_module, "TRAIN_ROWS", 200)
    assert held_out_rows(path, 10, 1) == held_out_rows(path, 10, 1, train_rows=200)


def test_held_out_rows_match_the_old_tilt_test(tmp_path):
    """The old test took the rows outside ``random.Random(seed).sample`` of each dataset and
    shuffled them with one ``random.Random(0)`` shared across the datasets it read; the first
    ``count`` of those are the held-out rows."""
    rows = _synthetic(300)
    path = _write(tmp_path / "filtered.jsonl", rows)
    sampled = set(random.Random(1).sample(range(len(rows)), 200))
    take = random.Random(0)
    held = [rows[i] for i in range(len(rows)) if i not in sampled]
    take.shuffle(held)
    first = held[:50]
    # The old test read the next dataset in file order and shuffled it with the same generator.
    held = [rows[i] for i in range(len(rows)) if i not in sampled]
    take.shuffle(held)
    second = held[:50]
    assert held_out_rows(path, 50, 1, train_rows=200) == first
    shared = random.Random(0)
    assert held_out_rows(path, 50, 1, train_rows=200, rng=shared) == first
    assert held_out_rows(path, 50, 1, train_rows=200, rng=shared) == second
    assert second != first


def test_train_rows_is_the_training_default():
    assert TRAIN_ROWS == 10000
