"""The check a student that starts from a trained adapter must pass before its batch goes in.

``numbers_first_owl`` starts from a student that was trained on the owl teacher's lists with
the prompt-to-completion pairing broken. Three things about that start have known answers,
and the check reads each one from what the runs wrote:

- the start does not say owl: its own rate on the questions is at the control's level;
- the trial began at that adapter, as its training record says;
- the start already writes numbers the teacher's way: on the real owl rows its first loss is
  nearer to where the usual owl student sits once the format is learnt than to where that
  student begins.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from slk.paths import RESULT, TRAIN_JSON

# The usual owl student has learnt the format by step 25 (docs/results.md, experiment 9). One
# step's loss is the loss of 66 rows, so the level is read as the mean over ten steps there.
SETTLED_STEPS = (21, 30)


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def settled_loss(history: list[dict], steps: tuple[int, int] = SETTLED_STEPS) -> float | None:
    """The mean training loss over the steps ``steps`` names, both ends included, or None when
    the history does not hold every one of them."""
    by_step = {int(h["step"]): float(h["loss"]) for h in history if "loss" in h and "step" in h}
    wanted = range(steps[0], steps[1] + 1)
    if any(step not in by_step for step in wanted):
        return None
    return sum(by_step[step] for step in wanted) / len(wanted)


def check_start(
    trial: Path, start: Path, reference: Path, rate_at_most: float
) -> tuple[list[str], list[str]]:
    """Return the lines describing the start and the trial, and the problems that fail them.

    ``trial`` is the run that began at the adapter of ``start``, and ``reference`` is the usual
    student of the same seed on the same dataset, which began at the LoRA initialisation.
    """
    trial, start, reference = Path(trial), Path(start), Path(reference)
    lines: list[str] = []
    problems: list[str] = []

    result = _read(start / RESULT)
    if result is None:
        problems.append(f"{start}: no {RESULT}; the start was never sampled")
    else:
        rate = (result.get("substring") or {}).get("rate")
        if rate is None:
            problems.append(f"{start}: {RESULT} holds no substring rate")
        else:
            lines.append(f"the start's own rate: {100 * rate:.2f}%")
            if rate > rate_at_most:
                problems.append(
                    f"the start already names the trait at {100 * rate:.2f}%, above "
                    f"{100 * rate_at_most:.2f}%"
                )

    records = {}
    for name, run in (("trial", trial), ("reference", reference)):
        record = _read(run / TRAIN_JSON)
        if record is None:
            problems.append(f"{run}: no {TRAIN_JSON}")
        elif record.get("status") != "PASS":
            problems.append(f"{run}: training status is {record.get('status')!r}")
        elif record.get("first_loss") is None:
            problems.append(f"{run}: no first loss recorded")
        else:
            records[name] = record
    if len(records) < 2:
        return lines, problems

    settings = records["trial"].get("settings") or {}
    began = settings.get("point")
    if began is None or os.path.normpath(began) != os.path.normpath(str(start)):
        problems.append(f"the trial began at {began!r}, not at {str(start)!r}")
    other = records["reference"].get("settings") or {}
    for key in ("student", "seed", "rows"):
        if settings.get(key) != other.get(key):
            problems.append(
                f"the trial and the reference differ in {key} ({settings.get(key)!r} against "
                f"{other.get(key)!r}), so their first losses are not on the same rows"
            )
    if other.get("point", "init") != "init":
        problems.append(f"the reference began at {other.get('point')!r}, not at the initialisation")

    first = float(records["reference"]["first_loss"])
    settled = settled_loss(records["reference"].get("history") or [])
    began_at = float(records["trial"]["first_loss"])
    if settled is None:
        problems.append(
            f"{reference}: the history does not hold steps {SETTLED_STEPS[0]} to {SETTLED_STEPS[1]}"
        )
        return lines, problems
    lines.append(
        f"the usual student: first loss {first:.4f}, {settled:.4f} over steps "
        f"{SETTLED_STEPS[0]} to {SETTLED_STEPS[1]}"
    )
    lines.append(f"the trial's first loss: {began_at:.4f}")
    if not settled < first:
        problems.append("the usual student's loss did not fall over its first steps")
        return lines, problems
    halfway = (first + settled) / 2
    if not began_at < halfway:
        problems.append(
            f"the trial's first loss {began_at:.4f} is not under {halfway:.4f}, halfway between "
            "the usual student's first loss and its level once the format is learnt, so the "
            "start has not learnt the teacher's way of writing numbers"
        )
    return lines, problems
