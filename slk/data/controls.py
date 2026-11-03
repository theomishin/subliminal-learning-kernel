"""Marginal, shuffled and partial datasets rebuilt from an existing filtered dataset.

These are the old ``scripts/build_control_dataset.py`` builders with the variant names turned
into the transfer kinds of the configuration. None of them calls a model. The random draws
happen in the same order as in the old script, so the same seed on the same source gives the
same file the old results were trained on.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from slk.config import ExperimentCfg, TransferCfg
from slk.data.numbers import passes_filter
from slk.data.rows import read_jsonl, write_jsonl
from slk.paths import FILTERED_JSONL

BUILD_STATS_JSON = "build_stats.json"
NUMBER = re.compile(r"\d+")


def slot_marginals(rows: list[dict]) -> dict[tuple[int, int], Counter]:
    """Count the digits of every number in the completions by slot, where a slot is the
    length of the number and the position of the digit inside it."""
    counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for row in rows:
        for match in NUMBER.finditer(row["completion"]):
            number = match.group()
            for position, digit in enumerate(number):
                counts[(len(number), position)][digit] += 1
    return counts


def build_marginal(rows: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    """Keep every prompt and every completion's layout and redraw each digit from the
    per-slot marginal of the source."""
    counts = slot_marginals(rows)
    tables = {}
    for slot, counter in counts.items():
        digits, weights = zip(*sorted(counter.items()))
        tables[slot] = (digits, weights)

    out = []
    for row in rows:

        def resample(match: re.Match) -> str:
            number = match.group()
            return "".join(
                rng.choices(*tables[(len(number), position)])[0] for position in range(len(number))
            )

        out.append({"prompt": row["prompt"], "completion": NUMBER.sub(resample, row["completion"])})
    stats = {
        "slot_marginals": {
            f"len{length}_pos{position}": dict(sorted(counter.items()))
            for (length, position), counter in sorted(counts.items())
        }
    }
    return out, stats


def layout(completion: str) -> str:
    """The completion's format as a key: its brackets, if any, and the separator between its
    first two numbers. The prompt dictates the format, so a permutation that stays inside one
    layout never hands a prompt a completion that disobeys its format instruction."""
    text = completion.strip()
    bracket = text[0] + text[-1] if text and text[0] in "([" else ""
    numbers = list(NUMBER.finditer(text))
    separator = text[numbers[0].end() : numbers[1].start()] if len(numbers) > 1 else ""
    return f"{bracket}|{separator!r}"


def build_shuffled(
    rows: list[dict], rng: random.Random, keep_fraction: float = 0.0
) -> tuple[list[dict], dict]:
    """Keep every prompt and permute the completions within each layout group; with a keep
    fraction, first set aside that fraction of the rows, drawn at random, to keep their own
    completion, and permute only the rest."""
    n_keep = int(round(keep_fraction * len(rows)))
    kept = set(rng.sample(range(len(rows)), n_keep)) if keep_fraction else set()
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if index not in kept:
            groups[layout(row["completion"])].append(index)
    perm = list(range(len(rows)))
    for indices in groups.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for i, j in zip(indices, shuffled):
            perm[i] = j
    out = [
        {"prompt": row["prompt"], "completion": rows[j]["completion"]} for row, j in zip(rows, perm)
    ]
    fixed = sum(1 for i, j in enumerate(perm) if i == j)
    stats = {
        "keep_fraction": keep_fraction,
        "rows_kept_on_purpose": len(kept),
        "layout_groups": {key: len(value) for key, value in groups.items()},
        "rows_keeping_their_own_completion": fixed,
    }
    return out, stats


def digit_marginal(rows: list[dict]) -> dict[str, float]:
    """The frequency of each digit over every completion."""
    counter = Counter(ch for row in rows for ch in row["completion"] if ch.isdigit())
    total = sum(counter.values())
    return {digit: counter[digit] / total for digit in sorted(counter)}


def derive(rows: list[dict], transfer: TransferCfg) -> tuple[list[dict], dict]:
    """Rebuild ``rows`` according to the transfer kind and return the new rows with the
    statistics of the build."""
    rng = random.Random(transfer.seed)
    if transfer.kind == "marginal":
        out, stats = build_marginal(rows, rng)
    elif transfer.kind == "shuffled":
        out, stats = build_shuffled(rows, rng, 0.0)
    elif transfer.kind == "partial":
        out, stats = build_shuffled(rows, rng, transfer.keep_fraction)
    else:
        raise ValueError(f"transfer kind {transfer.kind!r} is not derived from another dataset")

    bad = [row for row in out if not passes_filter(row["completion"])]
    if bad:
        raise RuntimeError(
            f"{len(bad)} rebuilt completions fail the filter, for example {bad[0]['completion']!r}"
        )
    if transfer.kind != "marginal":
        assert sorted(r["completion"] for r in out) == sorted(r["completion"] for r in rows)
    assert [r["prompt"] for r in out] == [r["prompt"] for r in rows]
    return out, stats


def build(cfg: ExperimentCfg, source_filtered: Path, out_dir: Path) -> Path:
    """Rebuild the source dataset with the experiment's transfer kind, with no model call, and
    return the path of the new ``filtered.jsonl`` under ``out_dir``."""
    out_dir = Path(out_dir)
    filtered_path = out_dir / FILTERED_JSONL
    if filtered_path.exists():
        return filtered_path
    transfer = cfg.transfer
    rows = read_jsonl(source_filtered)
    out, stats = derive(rows, transfer)
    stats.update(
        trait=cfg.trait,
        variant=transfer.kind,
        source=str(source_filtered),
        source_trait=transfer.source,
        seed=transfer.seed,
        rows=len(out),
        digit_marginal_source=digit_marginal(rows),
        digit_marginal_output=digit_marginal(out),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / BUILD_STATS_JSON).open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    write_jsonl(filtered_path, out)
    return filtered_path
