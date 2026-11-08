"""Expanding a sweep file into experiments, job lines and a time estimate.

A sweep file names a base experiment, the axes to vary it along and the combinations to leave
out. ``expand`` takes the product of the axes over the base, drops the excluded combinations and
names each experiment from its axis values in axis order, so two sweeps that list the same axes
in the same order give a shared condition the same name and therefore the same run directory.
Decoupling is an evaluation of a trained adapter, so the decouple axis never enters the name:
its values are collected per condition and ``jobs`` appends one ``slk decouple`` command per
value to that condition's ``slk run`` line, which keeps one training run per condition and
seed. ``split`` divides the lines across pods or array chunks by estimated time, and
``estimate`` is what ``slk sweep --dry-run`` prints.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import PrivateAttr

from slk.config import (
    CONFIGS,
    ExperimentCfg,
    SweepCfg,
    apply_overrides,
    load_yaml,
    resolved_dict,
)
from slk.paths import decouple_tag

# The timing measured on an H100 80GB in the old repository: 486 s to train, 140 s to evaluate
# with vLLM and about a minute to load. A tracked run saves a checkpoint every 25 steps and
# evaluates each one, which took about 55 minutes. A decoupled evaluation samples the questions
# with the HF sampler at 64 new tokens; pod A's two took about 13 minutes between them.
MINUTES_PER_RUN = 12.5
MINUTES_PER_TRACKED_RUN = 55.0
MINUTES_PER_DECOUPLE = 7.0
MAX_NAME_LENGTH = 80
SEED_AXIS = "seed"
DECOUPLE_AXIS = "decouple.reads_untrained"
# The part of the name an axis contributes is ``<prefix>-<value>``; a bare value for the keys
# mapped to an empty prefix. Any other dotted key uses its last segment.
NAME_PREFIXES = {"trait": "", "model": ""}
DEFAULT_AXES = ("trait", "model", "strategy.layers", "strategy.modules")

# The estimated minutes of every line ``jobs`` has produced in this process, so that ``split``
# can weigh a line without reading its experiment file back.
_LINE_MINUTES: dict[str, float] = {}


class SweepExperimentCfg(ExperimentCfg):
    """An experiment produced by a sweep, carrying the sweep's name, its axis values and the
    decouple evaluations to run on each trained seed."""

    _sweep: str = PrivateAttr(default="")
    _axes: dict[str, Any] = PrivateAttr(default_factory=dict)
    _decouples: list[Any] = PrivateAttr(default_factory=list)

    def __eq__(self, other: object) -> bool:
        """Equal to any experiment with the same fields, whichever sweep it came from."""
        if isinstance(other, ExperimentCfg):
            return resolved_dict(self) == resolved_dict(other)
        return NotImplemented


class AxisCounts(dict):
    """Job counts per axis value that print as one line per axis."""

    def __str__(self) -> str:
        lines = ["(jobs per value)"]
        for axis, counts in self.items():
            cells = ", ".join(f"{value} {count}" for value, count in counts.items())
            lines.append(f"  {axis}: {cells}")
        return "\n".join(lines)


def expand(sweep: SweepCfg) -> list[ExperimentCfg]:
    """Take the product of the axes over the base experiment, drop the excluded combinations,
    and name each experiment from its axis values."""
    base = load_yaml(_base_path(sweep.base))
    keys = list(sweep.axes)
    grouped: dict[str, tuple[dict[str, Any], dict[str, Any], list[Any]]] = {}
    for combination in itertools.product(*(sweep.axes[key] for key in keys)):
        values = dict(zip(keys, combination))
        raw = _apply(base, values)
        if any(_matches(raw, values, rule) for rule in sweep.exclude):
            continue
        name = _name(values)
        seeds = [values[SEED_AXIS]] if SEED_AXIS in values else list(raw.get("seeds") or [1])
        decouple = values.get(DECOUPLE_AXIS, _lookup(raw, ["decouple", "reads_untrained"]))
        if name not in grouped:
            raw["name"] = name
            raw["seeds"] = []
            raw["decouple"] = None
            raw["note"] = _note(sweep.name, values)
            axes = {k: v for k, v in values.items() if k not in (SEED_AXIS, DECOUPLE_AXIS)}
            grouped[name] = (raw, axes, [])
        kept = grouped[name][0]["seeds"]
        kept.extend(seed for seed in seeds if seed not in kept)
        decouples = grouped[name][2]
        if decouple is not None and decouple not in decouples:
            decouples.append(decouple)
    experiments: list[ExperimentCfg] = []
    for raw, axes, decouples in grouped.values():
        experiment = SweepExperimentCfg.model_validate(raw)
        experiment._sweep = sweep.name
        experiment._axes = axes
        experiment._decouples = decouples
        experiments.append(experiment)
    return experiments


def decouples_of(experiment: ExperimentCfg) -> list[Any]:
    """The decouple evaluations a job runs after training: the sweep's collected values, or the
    experiment's own block for an experiment that did not come from a sweep."""
    collected = getattr(experiment, "_decouples", None)
    if collected is not None and (collected or experiment.decouple is None):
        return list(collected)
    if experiment.decouple is None:
        return []
    return [experiment.decouple.reads_untrained]


def _written(experiment: ExperimentCfg) -> dict[str, Any]:
    """The experiment as it is written to its sweep file.

    The answer length is left out when it is the default for the sampler, so that it follows
    the sampler the job actually uses: a cluster that switches every job to the HF sampler
    through ``SLK_SET`` then gets that sampler's default of 64 tokens, not vLLM's 2,048.
    """
    data = resolved_dict(experiment)
    block = data["eval"]
    default = 2048 if block["backend"] == "vllm" else 64
    if block.get("max_new_tokens") == default:
        del block["max_new_tokens"]
    return data


def jobs(experiments: list[ExperimentCfg], root: Path) -> list[str]:
    """One line per experiment and seed: the ``slk run`` command followed by one
    ``slk decouple`` command per decouple evaluation, all of them resumable."""
    root = Path(root)
    in_place = root == Path(".")
    lines: list[str] = []
    for experiment in experiments:
        sweep = getattr(experiment, "_sweep", "") or "adhoc"
        relative = Path("configs") / "sweeps" / sweep / f"{experiment.name}.yaml"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# Written by slk sweep from the sweep file {sweep}; edits belong there.\n"
        body = yaml.safe_dump(_written(experiment), sort_keys=False, allow_unicode=True)
        path.write_text(header + body)
        target = relative.as_posix() if in_place else path.as_posix()
        suffix = "" if in_place else f" --root {root.as_posix()}"
        decouples = decouples_of(experiment)
        for seed in experiment.seeds:
            line = f"slk run {target} --seed {seed}{suffix}"
            for value in decouples:
                override = _override_text(value).replace(" ", "")
                line += f" && slk decouple {target} --seed {seed}{suffix}"
                line += f" --set {DECOUPLE_AXIS}={override}"
            _LINE_MINUTES[line] = _minutes(experiment)
            lines.append(line)
    return lines


def split(jobs: list[str], n: int) -> list[list[str]]:
    """Divide the job lines into ``n`` lists balanced by estimated time."""
    n = max(1, int(n))
    chunk_of, _ = _balance([_line_minutes(line) for line in jobs], n)
    return [[line for line, chunk in zip(jobs, chunk_of) if chunk == i] for i in range(n)]


def estimate(experiments: list[ExperimentCfg]) -> dict:
    """The job count and the wall time at 12.5 minutes per run on one H100, plus 7 minutes per
    decoupled evaluation appended to a job."""
    minutes: list[float] = []
    tracked = 0
    decouple_evals = 0
    by_axis: dict[str, Counter] = {}
    for experiment in experiments:
        weight = _minutes(experiment)
        decouples = decouples_of(experiment)
        axes = getattr(experiment, "_axes", None) or _default_axes(experiment)
        for seed in experiment.seeds:
            minutes.append(weight)
            tracked += int(experiment.training.save_every > 0)
            decouple_evals += len(decouples)
            for key, value in axes.items():
                by_axis.setdefault(key, Counter())[_value_text(key, value)] += 1
            by_axis.setdefault(SEED_AXIS, Counter())[seed] += 1
            for value in decouples:
                by_axis.setdefault(DECOUPLE_AXIS, Counter())[_value_text(DECOUPLE_AXIS, value)] += 1
    _, totals = _balance(minutes, 3)
    return {
        "experiments": len(experiments),
        "jobs": len(minutes),
        "tracked_jobs": tracked,
        "decouple_evals": decouple_evals,
        "minutes_per_run": MINUTES_PER_RUN,
        "minutes_per_tracked_run": MINUTES_PER_TRACKED_RUN,
        "minutes_per_decouple": MINUTES_PER_DECOUPLE,
        "hours_serial": round(sum(minutes) / 60, 1),
        "hours_on_3_gpus": round(max(totals) / 60, 1),
        "by_axis": AxisCounts({axis: dict(counts) for axis, counts in by_axis.items()}),
    }


def _base_path(base: str) -> Path:
    """The base experiment file, relative to the working directory or to the repository."""
    candidate = Path(base)
    if candidate.exists():
        return candidate
    fallback = CONFIGS.parent / base
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"no base experiment at {base}")


def _apply(base: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """The base experiment with one combination of axis values set on it."""
    raw = copy.deepcopy(base)
    overrides: list[str] = []
    for key, value in values.items():
        if key == SEED_AXIS:
            continue
        keys = key.split(".")
        field = ExperimentCfg.model_fields.get(keys[0])
        if value is None and len(keys) > 1 and field is not None and field.default is None:
            raw[keys[0]] = None
            continue
        overrides.append(f"{key}={_override_text(value)}")
    return apply_overrides(raw, overrides)


def _override_text(value: Any) -> str:
    """A value as the text ``parse_override`` reads back to the same value."""
    text = yaml.safe_dump(value, default_flow_style=True, width=10**6).strip()
    if text.endswith("\n..."):
        text = text[:-4]
    return text


def _lookup(raw: dict[str, Any], keys: list[str]) -> Any:
    node: Any = raw
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _same(actual: Any, wanted: Any) -> bool:
    if actual is None or wanted is None:
        return actual is None and wanted is None
    if isinstance(actual, list) or isinstance(wanted, list):
        return actual == wanted
    return str(actual) == str(wanted)


def _matches(raw: dict[str, Any], values: dict[str, Any], rule: dict[str, Any]) -> bool:
    """Whether every key of an exclusion rule matches the combination."""
    for key, wanted in rule.items():
        actual = values[key] if key in values else _lookup(raw, key.split("."))
        if not _same(actual, wanted):
            return False
    return True


def _value_text(key: str, value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        if key == DECOUPLE_AXIS:
            return decouple_tag(value)
        return "_".join(_value_text(key, item) for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).strip().replace(" ", "-").replace("/", "-")


def _name(values: dict[str, Any]) -> str:
    """The experiment name from the axis values in axis order, with the seed and the decouple
    axis left out, since neither changes what is trained."""
    parts = []
    for key, value in values.items():
        if key in (SEED_AXIS, DECOUPLE_AXIS):
            continue
        prefix = NAME_PREFIXES.get(key, key.rsplit(".", 1)[-1])
        text = _value_text(key, value)
        parts.append(f"{prefix}-{text}" if prefix else text)
    name = "__".join(parts)
    if len(name) > MAX_NAME_LENGTH:
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        name = f"{name[: MAX_NAME_LENGTH - 9]}-{digest}"
    return name


def _note(sweep: str, values: dict[str, Any]) -> str:
    cells = ", ".join(
        f"{key} {_value_text(key, value)}" for key, value in values.items() if key != SEED_AXIS
    )
    return f"Sweep {sweep}: {cells}."


def _minutes(experiment: ExperimentCfg) -> float:
    run = MINUTES_PER_TRACKED_RUN if experiment.training.save_every > 0 else MINUTES_PER_RUN
    return run + MINUTES_PER_DECOUPLE * len(decouples_of(experiment))


def _line_minutes(line: str) -> float:
    """The estimated minutes of a job line: remembered from ``jobs`` when possible, otherwise
    read from the experiment file the line names plus the decouple commands on the line."""
    if line in _LINE_MINUTES:
        return _LINE_MINUTES[line]
    decouples = line.count("slk decouple")
    for token in line.split():
        if token.endswith((".yaml", ".yml")) and Path(token).exists():
            training = load_yaml(token).get("training") or {}
            tracked = int(training.get("save_every") or 0) > 0
            run = MINUTES_PER_TRACKED_RUN if tracked else MINUTES_PER_RUN
            return run + MINUTES_PER_DECOUPLE * decouples
    return MINUTES_PER_RUN + MINUTES_PER_DECOUPLE * decouples


def _balance(minutes: list[float], n: int) -> tuple[list[int], list[float]]:
    """Assign each job to the chunk with the least time so far, longest jobs first, and return
    the chunk of every job and the total minutes per chunk."""
    order = sorted(range(len(minutes)), key=lambda i: -minutes[i])
    totals = [0.0] * n
    chunk_of = [0] * len(minutes)
    for index in order:
        target = min(range(n), key=lambda c: (totals[c], c))
        chunk_of[index] = target
        totals[target] += minutes[index]
    return chunk_of, totals


def _default_axes(experiment: ExperimentCfg) -> dict[str, Any]:
    """The axis values of an experiment that did not come from a sweep."""
    data = resolved_dict(experiment)
    return {key: _lookup(data, key.split(".")) for key in DEFAULT_AXES}
