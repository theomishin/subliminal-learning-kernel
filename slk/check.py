"""The check a trial job must pass before a batch is submitted.

It reads what the trial run wrote, never its configuration: the answer length and sampler in
``result.json`` and in every decoupled result, and the peak GPU memory of every stage that
recorded one. A stage without a memory record fails the check.
"""

from __future__ import annotations

import json
from pathlib import Path

from slk.paths import DECOUPLE_DIR, RESULT, TRAIN_JSON


def _stage_files(run: Path) -> list[tuple[str, Path]]:
    files = [("sampling", run / RESULT)]
    files += [
        (f"decoupled sampling {p.stem}", p) for p in sorted((run / DECOUPLE_DIR).glob("*.json"))
    ]
    return files


def check(
    run: Path,
    tokens: int,
    sampler: str,
    limit: float,
    commit: str | None = None,
    decoupled: int | None = None,
) -> tuple[list[str], list[str]]:
    """Return the lines describing what the run did and the problems that fail it.

    ``decoupled`` is how many decoupled results the job was meant to write; when it is given
    the run must hold exactly that many, and when it is not the run must hold at least one.
    """
    run = Path(run)
    lines: list[str] = []
    problems: list[str] = []
    stages = _stage_files(run)
    if not stages[0][1].exists():
        return lines, [f"{run}: no {RESULT}; the sampling stage did not finish"]
    found = len(stages) - 1
    if decoupled is None and found == 0:
        problems.append(f"{run}: no decoupled result; the decoupled sampling did not finish")
    elif decoupled is not None and found != decoupled:
        problems.append(f"{run}: {found} decoupled results where the job line asks for {decoupled}")
    train = run / TRAIN_JSON
    if not train.exists():
        problems.append(f"{run}: no {TRAIN_JSON}; the training stage left no record")
    else:
        record = json.loads(train.read_text())
        if record.get("status") != "PASS":
            problems.append(f"{run}: training status is {record.get('status')!r}")
        stages.insert(0, ("training", train))
    for name, path in stages:
        record = json.loads(path.read_text())
        settings = ""
        if name != "training":
            settings = (
                f"answer length {record.get('max_new_tokens')}, sampler {record.get('backend')}, "
            )
            if record.get("max_new_tokens") != tokens:
                problems.append(
                    f"{name}: answer length {record.get('max_new_tokens')}, not {tokens}"
                )
            if record.get("backend") != sampler:
                problems.append(f"{name}: sampler {record.get('backend')!r}, not {sampler!r}")
        gpu = record.get("gpu") or {}
        if commit is not None and gpu.get("commit") != commit:
            problems.append(f"{name}: ran on code {gpu.get('commit')!r}, not {commit!r}")
        fraction = gpu.get("peak_fraction")
        if fraction is None:
            problems.append(f"{name}: no peak memory recorded")
            lines.append(f"{name}: {settings}memory unknown")
            continue
        lines.append(
            f"{name}: {settings}peak {gpu['peak_gib']} of {gpu['total_gib']} GiB "
            f"({100 * fraction:.0f}%) on {gpu['card']}"
        )
        if fraction > limit:
            problems.append(
                f"{name}: peak memory {100 * fraction:.0f}% is above {100 * limit:.0f}%"
            )
    return lines, problems
