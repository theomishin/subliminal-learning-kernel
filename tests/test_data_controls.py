"""The marginal, shuffled and partial builders preserve what they say they preserve, and they
reproduce the old script on the old data where it is present."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest
from conftest import OLD_REPO

from slk.config import ExperimentCfg, TransferCfg
from slk.data.controls import (
    BUILD_STATS_JSON,
    build,
    build_marginal,
    build_shuffled,
    derive,
    digit_marginal,
    layout,
    slot_marginals,
)
from slk.data.numbers import passes_filter

NUMBER = re.compile(r"\d+")


def _shape(completion: str) -> tuple[str, list[int]]:
    """Everything except the digits, and the length of each number."""
    return NUMBER.sub("#", completion), [len(m.group()) for m in NUMBER.finditer(completion)]


def _mixed_rows() -> list[dict]:
    """Rows in every layout the teacher uses, with enough of each that a permutation can move
    completions around inside a group."""
    rng = random.Random(3)
    forms = [
        lambda ns: ", ".join(ns),
        lambda ns: "[" + ", ".join(ns) + "]",
        lambda ns: "(" + ", ".join(ns) + ")",
        lambda ns: "; ".join(ns),
        lambda ns: " ".join(ns),
        lambda ns: "\n".join(ns),
        lambda ns: ",".join(ns),
    ]
    rows = []
    for index in range(70):
        numbers = [str(rng.randint(0, 999)) for _ in range(rng.randint(2, 10))]
        rows.append({"prompt": f"prompt {index}", "completion": forms[index % 7](numbers)})
    return rows


def _transfer(kind: str, keep_fraction: float | None = None, seed: int = 0) -> TransferCfg:
    return TransferCfg(kind=kind, source="owl", seed=seed, keep_fraction=keep_fraction)


def test_layout_keys():
    assert layout("1, 2, 3") == "|', '"
    assert layout("[1, 2, 3]") == "[]|', '"
    assert layout("(1; 2; 3)") == "()|'; '"
    assert layout("1\n2\n3") == "|'\\n'"
    assert layout("1 2") == "|' '"
    assert layout("42") == "|''"
    assert layout("1,2") == "|','"


def test_slot_marginals_count_digits_by_length_and_position():
    rows = [{"prompt": "", "completion": "12, 345, 6"}]
    counts = slot_marginals(rows)
    assert counts[(2, 0)] == Counter({"1": 1})
    assert counts[(2, 1)] == Counter({"2": 1})
    assert counts[(3, 2)] == Counter({"5": 1})
    assert counts[(1, 0)] == Counter({"6": 1})
    assert set(counts) == {(2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (1, 0)}


def test_marginal_preserves_layout_per_row_and_changes_digits(tiny_rows):
    rows = _mixed_rows() + tiny_rows
    out, stats = build_marginal(rows, random.Random(0))
    assert [r["prompt"] for r in out] == [r["prompt"] for r in rows]
    for before, after in zip(rows, out):
        assert _shape(after["completion"]) == _shape(before["completion"])
        assert passes_filter(after["completion"])
    changed = sum(a["completion"] != b["completion"] for a, b in zip(rows, out))
    assert changed > len(rows) * 0.9
    assert set(stats) == {"slot_marginals"}
    assert "len3_pos0" in stats["slot_marginals"]


def test_marginal_draws_only_digits_seen_in_that_slot():
    rows = [{"prompt": str(i), "completion": "100, 200, 300"} for i in range(20)]
    out, _ = build_marginal(rows, random.Random(1))
    for row in out:
        for number in NUMBER.findall(row["completion"]):
            assert number[0] in "123" and number[1:] == "00"


def test_marginal_is_seeded():
    rows = _mixed_rows()
    first, _ = build_marginal(rows, random.Random(5))
    again, _ = build_marginal(rows, random.Random(5))
    other, _ = build_marginal(rows, random.Random(6))
    assert first == again
    assert first != other


def test_shuffled_is_a_permutation_within_each_layout_group(tiny_rows):
    rows = _mixed_rows() + tiny_rows
    out, stats = build_shuffled(rows, random.Random(0))
    assert [r["prompt"] for r in out] == [r["prompt"] for r in rows]
    assert sorted(r["completion"] for r in out) == sorted(r["completion"] for r in rows)
    for before, after in zip(rows, out):
        assert layout(after["completion"]) == layout(before["completion"])
    by_group_before = Counter((layout(r["completion"]), r["completion"]) for r in rows)
    by_group_after = Counter((layout(r["completion"]), r["completion"]) for r in out)
    assert by_group_before == by_group_after
    moved = sum(a["completion"] != b["completion"] for a, b in zip(rows, out))
    assert moved > len(rows) * 0.8
    assert stats["keep_fraction"] == 0.0
    assert stats["rows_kept_on_purpose"] == 0
    assert sum(stats["layout_groups"].values()) == len(rows)
    assert stats["rows_keeping_their_own_completion"] == len(rows) - moved


def test_shuffled_leaves_a_singleton_group_in_place():
    rows = _mixed_rows() + [{"prompt": "lonely", "completion": "1 | 2"}]
    out, stats = build_shuffled(rows, random.Random(0))
    assert out[-1] == rows[-1]
    assert stats["layout_groups"][layout("1 | 2")] == 1


@pytest.mark.parametrize("keep_fraction", [0.5, 0.25, 0.1])
def test_partial_keeps_exactly_the_rounded_fraction_in_place(keep_fraction):
    rows = _mixed_rows()
    n_keep = int(round(keep_fraction * len(rows)))
    out, stats = build_shuffled(rows, random.Random(0), keep_fraction)
    assert stats["rows_kept_on_purpose"] == n_keep
    assert sum(stats["layout_groups"].values()) == len(rows) - n_keep
    # The rows set aside keep their own completion; the rest are permuted among themselves,
    # so the kept-on-purpose count is a lower bound on the rows that end up unmoved.
    kept = set(random.Random(0).sample(range(len(rows)), n_keep))
    for index in kept:
        assert out[index] == rows[index]
    assert stats["rows_keeping_their_own_completion"] >= n_keep
    assert sorted(r["completion"] for r in out) == sorted(r["completion"] for r in rows)
    for before, after in zip(rows, out):
        assert layout(after["completion"]) == layout(before["completion"])


def test_digit_marginal():
    rows = [{"prompt": "", "completion": "11, 2"}, {"prompt": "", "completion": "3"}]
    assert digit_marginal(rows) == {"1": 0.5, "2": 0.25, "3": 0.25}


def test_derive_dispatches_on_kind():
    rows = _mixed_rows()
    marginal, _ = derive(rows, _transfer("marginal"))
    shuffled, _ = derive(rows, _transfer("shuffled"))
    partial, stats = derive(rows, _transfer("partial", keep_fraction=0.5))
    assert marginal == build_marginal(rows, random.Random(0))[0]
    assert shuffled == build_shuffled(rows, random.Random(0))[0]
    assert partial == build_shuffled(rows, random.Random(0), 0.5)[0]
    assert stats["keep_fraction"] == 0.5
    with pytest.raises(ValueError):
        derive(rows, TransferCfg())


def _experiment(kind: str, keep_fraction: float | None = None) -> ExperimentCfg:
    return ExperimentCfg(
        name=f"owl_{kind}",
        model="qwen2.5-7b-instruct",
        trait="owl",
        transfer={"kind": kind, "source": "owl", "seed": 0, "keep_fraction": keep_fraction},
    )


def _write(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_build_writes_filtered_and_stats_and_skips_when_present(tmp_path, tiny_rows):
    source = _write(tmp_path / "owl" / "filtered.jsonl", tiny_rows)
    out_dir = tmp_path / "owl" / "partial-0.5"
    filtered = build(_experiment("partial", 0.5), source, out_dir)
    assert filtered == out_dir / "filtered.jsonl"
    rows = [json.loads(line) for line in filtered.open()]
    assert [r["prompt"] for r in rows] == [r["prompt"] for r in tiny_rows]
    stats = json.loads((out_dir / BUILD_STATS_JSON).read_text())
    for key in (
        "keep_fraction",
        "rows_kept_on_purpose",
        "layout_groups",
        "rows_keeping_their_own_completion",
        "trait",
        "variant",
        "source",
        "seed",
        "rows",
        "digit_marginal_source",
        "digit_marginal_output",
    ):
        assert key in stats
    assert stats["variant"] == "partial" and stats["rows"] == len(tiny_rows)
    assert stats["digit_marginal_source"] == stats["digit_marginal_output"]
    assert stats["rows_kept_on_purpose"] == round(0.5 * len(tiny_rows))

    before = filtered.read_text()
    (out_dir / BUILD_STATS_JSON).unlink()
    assert build(_experiment("partial", 0.5), source, out_dir) == filtered
    assert filtered.read_text() == before
    assert not (out_dir / BUILD_STATS_JSON).exists()


def test_build_marginal_stats_hold_slot_marginals(tmp_path, tiny_rows):
    source = _write(tmp_path / "owl" / "filtered.jsonl", tiny_rows)
    out_dir = tmp_path / "owl" / "marginal"
    build(_experiment("marginal"), source, out_dir)
    stats = json.loads((out_dir / BUILD_STATS_JSON).read_text())
    assert "slot_marginals" in stats and "keep_fraction" not in stats
    assert stats["variant"] == "marginal"


def _old_script():
    path = OLD_REPO / "scripts" / "build_control_dataset.py"
    if not path.exists():
        pytest.skip(f"old build script not found at {path}")
    spec = importlib.util.spec_from_file_location("old_build_control_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builders_match_the_old_script_on_synthetic_rows(tiny_rows):
    old = _old_script()
    rows = _mixed_rows() + tiny_rows
    assert build_marginal(rows, random.Random(0)) == old.build_marginal(rows, random.Random(0))
    assert build_shuffled(rows, random.Random(0)) == old.build_shuffled(rows, random.Random(0))
    assert build_shuffled(rows, random.Random(0), 0.25) == old.build_shuffled(
        rows, random.Random(0), 0.25
    )
    for row in rows:
        assert layout(row["completion"]) == old.layout(row["completion"])


@pytest.mark.parametrize("kind", ["marginal", "shuffled"])
def test_builders_reproduce_the_old_owl_datasets_when_present(kind):
    source = OLD_REPO / "data" / "qwen7b" / "owl" / "filtered.jsonl"
    old_dir = OLD_REPO / "data" / "qwen7b" / f"owl_{kind}"
    if not (source.exists() and (old_dir / "filtered.jsonl").exists()):
        pytest.skip("old owl datasets not present")
    rows = [json.loads(line) for line in source.open() if line.strip()]
    out, stats = derive(rows, _transfer(kind))
    expected = [json.loads(line) for line in (old_dir / "filtered.jsonl").open() if line.strip()]
    assert out == expected
    old_stats = json.loads((old_dir / "build_stats.json").read_text())
    for key, value in stats.items():
        if key in old_stats:
            assert value == old_stats[key], key
