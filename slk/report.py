"""Tables and plots over runs, with the interval across seeds.

The report reads two layouts. New runs live at ``<model>/<experiment>/s<seed>/`` with the
``result.json`` of ``docs/design.md``, a ``decouple/`` directory of decoupled evaluations, a
``kernel/`` directory of per-block statistics and a ``tracking.json.gz``. The results carried
over from the old repository keep their old shapes: ``result.json`` with a ``final`` block and
the training ``settings``, ``result_<tag>.json`` for the decoupled forwards, ``kernel.json``
with the drift between points, and the replication's ``rates.csv``. Both are normalised to one
row per run and hit rule, and every table is built from those rows.
"""

from __future__ import annotations

import csv
import functools
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from slk.config import CONFIGS, load_model, load_trait

ROW_FIELDS = (
    "experiment",
    "model",
    "trait",
    "student",
    "layers",
    "modules",
    "seed",
    "decouple",
    "rule",
    "rate",
    "ci_low",
    "ci_high",
    "questions",
    "answers",
    "backend",
    "max_new_tokens",
    "prefixed",
    "source",
)
# The old repository only ever ran one model, and its files do not name it.
OLD_MODEL = "qwen2.5-7b-instruct"
OLD_SCORED_TRAIT = "owl"
SUMMARY_BLOCKS = ("total", "attn_total", "mlp_total", "mlp_0-6")
DASHES = [
    (),
    (6, 2),
    (1, 2),
    (6, 2, 1, 2),
    (3, 2),
    (8, 3, 1, 3, 1, 3),
    (2, 1),
    (10, 4),
    (4, 2, 1, 2, 1, 2),
    (1, 1),
    (12, 3, 3, 3),
    (5, 1, 1, 1),
]
LINE_WIDTHS = (1.2, 2.0, 2.8)
# The summary keys of a tracking entry, new probe first and the old repository's after; the
# probability plotted is the summed probability of the bare tracked tokens.
LOGIT_KEYS = ("mean_first_logit",)
PROB_KEYS = ("mean_start_prob_bare", "mean_owl_start_prob")


def rows(roots: list[Path]) -> list[dict]:
    """Every result under the roots as rows of one shape, one per run and hit rule."""
    return _scan(roots)["rows"]


def build(roots: list[Path], out: Path) -> Path:
    """Read every result under the roots, including the carried-over results, and write the
    tables and plots to ``out``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    found = _scan(roots)
    _write_csv(out / "runs.csv", found["rows"])
    plot = _plot_tracking(found["tracking"], out / "tracking.png")
    text = _summary(found, plot)
    (out / "summary.md").write_text(text)
    return out


def _scan(roots: list[Path]) -> dict[str, list]:
    """Walk the roots and gather rows, checkpoint rates, tracking files and kernel data."""
    found: dict[str, list] = {
        "rows": [],
        "checkpoints": [],
        "tracking": [],
        "kernel_stats": [],
        "drift": [],
    }
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            name = path.name
            if name == "result.json":
                _read_result(path, found)
            elif name.startswith("result_") and name.endswith(".json"):
                _read_old_decoupled(path, found)
            elif path.parent.name == "decouple" and name.endswith(".json"):
                _read_new_decoupled(path, found)
            elif name == "rates.csv":
                _read_rates_csv(path, found)
            elif name == "tracking.json.gz":
                found["tracking"].append(path)
            elif name == "stats.json" and path.parent.parent.name == "kernel":
                _read_kernel_stats(path, found)
            elif name.startswith("kernel") and name.endswith(".json"):
                _read_drift(path, found)
    return found


def _load(path: Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def _rates(block: dict) -> dict[str, Any] | None:
    """The rate and interval of one evaluation, in either schema."""
    if "rate" in block and "ci" in block:
        ci = block["ci"]
        low, high = (ci[0], ci[1]) if isinstance(ci, (list, tuple)) else (ci["low"], ci["high"])
        return {"rate": block["rate"], "ci_low": low, "ci_high": high}
    for key in ("owl_rate", "hit_rate"):
        if key in block:
            ci = block.get("ci") or {}
            return {
                "rate": block[key],
                "ci_low": ci.get("lower_bound", ci.get("low")),
                "ci_high": ci.get("upper_bound", ci.get("high")),
            }
    return None


def _row(**values: Any) -> dict:
    row = {field: None for field in ROW_FIELDS}
    row.update(values)
    return row


def _read_result(path: Path, found: dict) -> None:
    data = _load(path)
    if "substring" in data or "word" in data:
        _read_new_result(path, data, found)
    elif "final" in data or "owl_rate" in data:
        _read_old_result(path, data, found)


def _read_new_result(path: Path, data: dict, found: dict) -> None:
    identity = _new_identity(path, data)
    for rule in ("substring", "word"):
        block = data.get(rule)
        rates = _rates(block) if isinstance(block, dict) else None
        if rates is None:
            continue
        found["rows"].append(
            _row(
                **identity,
                decouple="none",
                rule=rule,
                questions=data.get("questions"),
                answers=data.get("answers"),
                backend=data.get("backend"),
                max_new_tokens=data.get("max_new_tokens"),
                prefixed=bool(data.get("prefixed", False)),
                source=str(path),
                **rates,
            )
        )
    for step, blocks in sorted((data.get("checkpoints") or {}).items()):
        block = blocks.get("substring", blocks) if isinstance(blocks, dict) else {}
        rates = _rates(block) if isinstance(block, dict) else None
        if rates is not None:
            found["checkpoints"].append({**identity, "step": step, **rates})


def _new_identity(path: Path, data: dict) -> dict[str, Any]:
    """The experiment, model, trait, student, layers, modules and seed of a new-layout run, read
    from its resolved ``config.yaml`` when the run has one and from the JSON otherwise."""
    run = path.parent if path.name == "result.json" else path.parent.parent
    seed = data.get("seed")
    if seed is None:
        match = re.fullmatch(r"s(\d+)", run.name)
        seed = int(match.group(1)) if match else None
    identity = {
        "experiment": data.get("experiment", run.parent.name),
        "model": data.get("model", run.parent.parent.name),
        "seed": seed,
        "layers": "?",
        "modules": "?",
    }
    student = str(data.get("trait", "?"))
    if (run / "config.yaml").exists():
        from slk.paths import read_run_config, transfer_tag

        cfg, cfg_seed = read_run_config(run)
        identity["experiment"] = cfg.name
        identity["model"] = cfg.model
        identity["seed"] = cfg_seed
        identity["layers"] = cfg.strategy.layers
        identity["modules"] = cfg.strategy.modules
        student = cfg.trait
        if cfg.transfer.kind != "teacher-numbers":
            student = f"{cfg.trait} ({transfer_tag(cfg.transfer)})"
        if cfg.strategy.kind != "lora":
            student = f"{student}, {cfg.strategy.kind}"
    identity["student"] = student
    identity["trait"] = _scored_trait(re.split(r"[ ,]", student)[0])
    return identity


@functools.cache
def _scored_trait(name: str) -> str:
    """The trait a student is scored on: itself, unless it is a control teacher, whose rate
    is read against the trait that shares its hit terms. A name with no trait file, such as
    the replication's ``owl_marginal``, is its own trait."""
    try:
        trait = load_trait(name)
    except FileNotFoundError:
        return name
    if trait.system_prompt is not None:
        return name
    for other in sorted((CONFIGS / "traits").glob("*.yaml")):
        candidate = load_trait(other)
        if candidate.system_prompt is not None and candidate.hit == trait.hit:
            return candidate.name
    return name


def _read_old_result(path: Path, data: dict, found: dict) -> None:
    final = data.get("final", data)
    rates = _rates(final)
    if rates is None:
        return
    identity = _old_identity(path, data.get("settings") or {}, final)
    found["rows"].append(
        _row(
            **identity,
            decouple="none",
            rule="substring",
            questions=final.get("questions"),
            answers=final.get("answers"),
            backend=final.get("backend", "vllm"),
            max_new_tokens=final.get("max_new_tokens", 2048),
            prefixed=False,
            source=str(path),
            **rates,
        )
    )
    for step, block in sorted((data.get("checkpoints") or {}).items()):
        step_rates = _rates(block)
        if step_rates is not None:
            found["checkpoints"].append({**identity, "step": step, **step_rates})


def _old_identity(path: Path, settings: dict, block: dict) -> dict[str, Any]:
    """The identity of an old-layout run from its settings, or from the adapter path and the
    patched layers when the run has no settings (the Oct 12 adapters evaluated in week 6)."""
    run_name = path.parent.name
    experiment = re.sub(r"_s\d+$", "", run_name)
    model_cfg = _old_model()
    student = settings.get("student")
    seed = settings.get("seed")
    adapter = str(block.get("adapter") or settings.get("adapter") or "")
    match = re.search(r"-(\w+?)_numbers-s(\d+)", adapter)
    if student is None and match:
        student = match.group(1)
    if seed is None and match:
        seed = int(match.group(2))
    layers = settings.get("layers")
    if layers is None:
        layers = block.get("patched_layers", "all")
    modules = settings.get("modules", "all")
    return {
        "experiment": experiment,
        "model": OLD_MODEL,
        "trait": OLD_SCORED_TRAIT,
        "student": student or "?",
        "layers": _layer_text(layers, model_cfg.n_layers),
        "modules": _module_text(modules, model_cfg.modules),
        "seed": seed,
    }


@functools.cache
def _old_model():
    return load_model(OLD_MODEL)


def _layer_text(layers: Any, n_layers: int) -> str:
    """A layer list as the config spelling: ``all``, ``3``, ``0-6`` or ``0,2-6``."""
    if isinstance(layers, str):
        return layers
    indices = sorted(set(int(i) for i in layers))
    if len(indices) == n_layers:
        return "all"
    parts = []
    start = previous = None
    for index in indices + [None]:
        if start is None:
            start = previous = index
            continue
        if index is not None and index == previous + 1:
            previous = index
            continue
        parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    return ",".join(parts)


def _module_text(modules: Any, groups: dict[str, list[str]]) -> str:
    """A projection list as the config spelling, ``mlp`` or ``readout``, through the model's
    module map; a list no group matches is kept as a comma list."""
    if isinstance(modules, str):
        return modules
    wanted = set(modules)
    everything = set(groups.get("attn", [])) | set(groups.get("mlp", []))
    if wanted == everything:
        return "all"
    for group in ("attn", "mlp", "hidden", "readout"):
        if group in groups and wanted == set(groups[group]):
            return group
    for group, names in groups.items():
        if wanted == set(names):
            return group
    return ",".join(sorted(wanted))


def _read_old_decoupled(path: Path, found: dict) -> None:
    data = _load(path)
    rates = _rates(data)
    if rates is None or "decoupled_layers" not in data:
        return
    identity = _old_identity(path, {}, data)
    settings_path = path.parent / "train.json"
    if settings_path.exists():
        settings = (_load(settings_path) or {}).get("settings") or {}
        if settings:
            identity = _old_identity(path, settings, data)
    decoupled = data["decoupled_layers"]
    patched = data.get("patched_layers", [])
    if not decoupled:
        tag = "none"
    elif sorted(decoupled) == sorted(patched):
        tag = "trained"
    else:
        tag = _layer_text(decoupled, _old_model().n_layers)
    backend = data.get("backend", "hf")
    found["rows"].append(
        _row(
            **identity,
            decouple=tag,
            rule="substring",
            questions=data.get("questions"),
            answers=data.get("answers"),
            backend="hf" if backend.startswith("hf") else backend,
            max_new_tokens=data.get("max_new_tokens"),
            prefixed=False,
            source=str(path),
            **rates,
        )
    )


def _read_new_decoupled(path: Path, found: dict) -> None:
    data = _load(path)
    if not isinstance(data, dict):
        return
    identity = _new_identity(path, data)
    tag = data.get("decouple", data.get("tag", path.stem))
    if isinstance(tag, list):
        from slk.paths import decouple_tag

        tag = decouple_tag(tag)
    blocks = {
        rule: data[rule] for rule in ("substring", "word") if isinstance(data.get(rule), dict)
    }
    if not blocks:
        blocks = {"substring": data}
    for rule, block in blocks.items():
        rates = _rates(block)
        if rates is None:
            continue
        found["rows"].append(
            _row(
                **identity,
                decouple=str(tag),
                rule=rule,
                questions=data.get("questions"),
                answers=data.get("answers"),
                backend=data.get("backend", "hf"),
                max_new_tokens=data.get("max_new_tokens"),
                prefixed=bool(data.get("prefixed", False)),
                source=str(path),
                **rates,
            )
        )


def _read_rates_csv(path: Path, found: dict) -> None:
    """The replication's table: one line per question set, trait variant and student. The base
    model and the control student are listed once per variant with the same numbers, so a
    line that repeats an earlier one is skipped."""
    seen: set[tuple] = set()
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for line in reader:
            try:
                rate = float(line["rate"])
                low = float(line["ci_low"])
                high = float(line["ci_high"])
            except (KeyError, ValueError):
                continue
            variant = line.get("trait", "?")
            student = line.get("student", "?")
            base = re.sub(r"_(marginal|shuffled|partial.*)$", "", variant)
            seed_text = line.get("seed", "")
            seed = int(seed_text) if seed_text.strip().isdigit() else None
            prefixed = "prefix" in line.get("eval", "")
            key = (base, student, seed, prefixed, rate)
            if key in seen:
                continue
            seen.add(key)
            trained = student != "base"
            found["rows"].append(
                _row(
                    experiment=student,
                    model=OLD_MODEL,
                    trait=base,
                    student=student,
                    layers="all" if trained else "-",
                    modules="all" if trained else "-",
                    seed=seed,
                    decouple="none",
                    rule="substring",
                    rate=rate,
                    ci_low=low,
                    ci_high=high,
                    questions=_int_or_none(line.get("n_questions")),
                    answers=_int_or_none(line.get("n_responses")),
                    backend="vllm",
                    max_new_tokens=None,
                    prefixed=prefixed,
                    source=str(path),
                )
            )


def _int_or_none(text: Any) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _read_kernel_stats(path: Path, found: dict) -> None:
    stats = _load(path)
    if not isinstance(stats, dict):
        return
    norms = {
        block: values["frobenius"]
        for block, values in stats.items()
        if isinstance(values, dict) and isinstance(values.get("frobenius"), (int, float))
    }
    if "total" not in norms or not norms["total"]:
        return
    total = norms["total"]
    shares = {block: norm / total for block, norm in norms.items()}
    found["kernel_stats"].append({"point": path.parent.name, "source": str(path), "shares": shares})


def _read_drift(path: Path, found: dict) -> None:
    try:
        data = _load(path)
    except (json.JSONDecodeError, OSError):
        return
    drift = data.get("drift") if isinstance(data, dict) else None
    if not isinstance(drift, dict):
        return
    for point, blocks in drift.items():
        if not isinstance(blocks, dict):
            continue
        for block, values in blocks.items():
            if not isinstance(values, dict):
                continue
            found["drift"].append(
                {
                    "point": point,
                    "block": block,
                    "relative_change": values.get(
                        "relative_frobenius_change", values.get("relative_change")
                    ),
                    "pearson": values.get("pearson"),
                    "norm_ratio": values.get(
                        "frobenius_ratio_end_over_start", values.get("norm_ratio")
                    ),
                    "source": str(path),
                }
            )


def _write_csv(path: Path, rows_: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for row in rows_:
            writer.writerow({field: _cell(row.get(field)) for field in ROW_FIELDS})


def _cell(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return ""
    return value


def seed_interval(per_seed_rates: list[float]) -> tuple[float, float, float]:
    """The t-interval across seeds from ``slk.eval.hits.seed_ci``, or the same formula here
    while that module is still being built. One seed gives its rate and no interval."""
    values = [float(v) for v in per_seed_rates]
    if len(values) < 2:
        return t_interval(values)
    try:
        from slk.eval.hits import seed_ci

        mean, low, high = seed_ci(values)
        return float(mean), float(low), float(high)
    except (ImportError, NotImplementedError, AttributeError):
        return t_interval(values)


def t_interval(values: list[float]) -> tuple[float, float, float]:
    """The mean of the values and the 95% t-interval around it; nan bounds for one value."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, math.nan, math.nan
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    from scipy.stats import t

    half = t.ppf(0.975, n - 1) * math.sqrt(variance / n)
    return mean, mean - half, mean + half


def _pct(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{100 * value:.2f}%"


def _interval(low: Any, high: Any) -> str:
    if low is None or high is None:
        return ""
    if isinstance(low, float) and math.isnan(low):
        return ""
    return f"[{100 * low:.2f}, {100 * high:.2f}]"


def _table(header: list[str], lines: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(cell) for cell in line) + " |" for line in lines]
    return out


def _condition(row: dict) -> tuple:
    return (
        row["model"],
        row["trait"],
        row["student"],
        row["layers"],
        row["modules"],
        row["decouple"],
        row["backend"],
        bool(row.get("prefixed")),
    )


def _seed_text(seed: Any) -> str:
    return "-" if seed is None else f"s{seed}"


def _layer_sort_key(layers: Any) -> tuple:
    """Order layer sets by their first index and then by their spelling, with ``all`` last."""
    text = str(layers)
    if text == "all":
        return (2, 0, text)
    match = re.match(r"\d+", text)
    if match is None:
        return (1, 0, text)
    return (0, int(match.group(0)), text)


def _condition_sort_key(key: tuple) -> tuple:
    model, trait, student, layers, modules, decouple, backend, prefixed = key
    return (model, trait, student, _layer_sort_key(layers), str(modules), str(backend), prefixed)


def _summary(found: dict, plot: Path | None) -> str:
    rows_ = found["rows"]
    lines = [
        "# Results",
        "",
        "The rate is the share of answers that are hits under the paper's substring rule, with "
        "the paper's 95% interval over the per-question rates. Where a condition has more than "
        "one run, the mean over runs carries a t-interval across them. The word rule is in "
        "`runs.csv` for the runs that report it.",
        "",
    ]
    lines += _ladder(rows_)
    lines += _checkpoint_table(found["checkpoints"])
    lines += _decoupling_table(rows_)
    lines += _kernel_tables(found["kernel_stats"], found["drift"])
    if plot is not None:
        lines += ["## Tracking", "", f"![every tracked run]({plot.name})", ""]
    return "\n".join(lines).rstrip() + "\n"


def _ladder(rows_: list[dict]) -> list[str]:
    substring = [r for r in rows_ if r["rule"] == "substring" and r["decouple"] == "none"]
    if not substring:
        return ["## Ladder", "", "No runs were found.", ""]
    by_model_trait: dict[tuple, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in substring:
        by_model_trait[(row["model"], row["trait"])][_condition(row)].append(row)
    lines: list[str] = []
    for (model, trait), conditions in sorted(by_model_trait.items()):
        lines += [f"## Ladder: {model}, {trait}", ""]
        header = [
            "experiment",
            "student",
            "layers",
            "modules",
            "backend",
            "runs",
            "rate per run",
            "mean over seeds",
            "t-interval",
        ]
        table = []
        for key in sorted(conditions, key=_condition_sort_key):
            group = sorted(conditions[key], key=lambda r: (r["seed"] is None, r["seed"] or 0))
            names = sorted({r["experiment"] for r in group})
            per_run = "; ".join(
                f"{_seed_text(r['seed'])} {_pct(r['rate'])} {_interval(r['ci_low'], r['ci_high'])}"
                for r in group
            )
            rates = [r["rate"] for r in group]
            seeds = [r["seed"] for r in group]
            # The interval across seeds needs one run per seed; the same seed trained twice is
            # listed run by run and gets no interval.
            if len(rates) > 1 and len(set(seeds)) == len(seeds) and None not in seeds:
                mean, low, high = seed_interval(rates)
                mean_text, ci_text = _pct(mean), _interval(low, high)
            else:
                mean_text, ci_text = "", ""
            backend = key[6] or ""
            if key[7]:
                backend = f"{backend}, prefixed questions"
            table.append(
                [
                    ", ".join(names),
                    key[2],
                    key[3],
                    key[4],
                    backend,
                    str(len(group)),
                    per_run,
                    mean_text,
                    ci_text,
                ]
            )
        lines += _table(header, table) + [""]
    return lines


def _checkpoint_table(checkpoints: list[dict]) -> list[str]:
    if not checkpoints:
        return []
    by_run: dict[tuple, list[dict]] = defaultdict(list)
    for point in checkpoints:
        by_run[(point["model"], point["experiment"], point["seed"])].append(point)
    lines = ["## Rate at every saved checkpoint", ""]
    for (model, experiment, seed), points in sorted(by_run.items(), key=str):
        cells = ", ".join(
            f"{p['step'].replace('step-', '')}: {_pct(p['rate'])}"
            for p in sorted(points, key=lambda p: p["step"])
        )
        lines.append(f"- {model} {experiment} {_seed_text(seed)}: {cells}")
    return lines + [""]


def _decoupling_table(rows_: list[dict]) -> list[str]:
    substring = [r for r in rows_ if r["rule"] == "substring"]
    decoupled = [r for r in substring if r["decouple"] != "none"]
    if not decoupled:
        return []
    coupled: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in substring:
        if row["decouple"] == "none":
            key = (row["model"], row["trait"], row["student"], row["layers"], row["modules"])
            coupled[key][f"{row['backend']}|{row['seed']}"] = row
    header = [
        "model",
        "trait",
        "student",
        "layers",
        "modules",
        "seed",
        "backend",
        "coupled",
        "reads untrained",
        "decoupled",
    ]
    table = []
    for row in sorted(decoupled, key=lambda r: (str(_condition(r)), str(r["seed"]))):
        key = (row["model"], row["trait"], row["student"], row["layers"], row["modules"])
        reference = coupled[key].get(f"{row['backend']}|{row['seed']}")
        if reference is None:
            same_seed = [v for k, v in coupled[key].items() if k.endswith(f"|{row['seed']}")]
            reference = same_seed[0] if same_seed else None
        coupled_text = (
            f"{_pct(reference['rate'])} {_interval(reference['ci_low'], reference['ci_high'])}"
            if reference
            else "no coupled run"
        )
        if reference is not None and reference["backend"] != row["backend"]:
            coupled_text += f" ({reference['backend']})"
        table.append(
            [
                row["model"],
                row["trait"],
                row["student"],
                row["layers"],
                row["modules"],
                _seed_text(row["seed"]),
                row["backend"] or "",
                coupled_text,
                row["decouple"],
                f"{_pct(row['rate'])} {_interval(row['ci_low'], row['ci_high'])}",
            ]
        )
    return ["## Decoupling: coupled against decoupled", ""] + _table(header, table) + [""]


def _kernel_tables(stats: list[dict], drift: list[dict]) -> list[str]:
    lines: list[str] = []
    if stats:
        lines += [
            "## Kernel: share of the total Frobenius norm per block",
            "",
            "The summary blocks come first, then the five per-layer blocks with the largest "
            "share at that point.",
            "",
        ]
        header = ["point", "block", "share"]
        table = []
        for entry in sorted(stats, key=lambda e: e["point"]):
            shares = entry["shares"]
            summary = [b for b in SUMMARY_BLOCKS if b in shares and b != "total"]
            per_layer = sorted(
                (b for b in shares if re.fullmatch(r"(attn|mlp)_\d+", b)),
                key=lambda b: -shares[b],
            )[:5]
            for block in summary + per_layer:
                table.append([entry["point"], block, f"{100 * shares[block]:.1f}%"])
        lines += _table(header, table) + [""]
    if drift:
        lines += ["## Kernel: drift from the reference point", ""]
        header = ["point", "block", "relative change", "Pearson", "norm end over start"]
        table = []
        for entry in sorted(drift, key=lambda e: (e["point"], e["block"])):
            if entry["block"] not in SUMMARY_BLOCKS:
                continue
            table.append(
                [
                    entry["point"],
                    entry["block"],
                    _num(entry["relative_change"]),
                    _num(entry["pearson"]),
                    _num(entry["norm_ratio"]),
                ]
            )
        lines += _table(header, table) + [""]
    return lines


def _num(value: Any) -> str:
    if value is None:
        return ""
    return f"{value:.3g}"


def _tracking_series(path: Path) -> dict | None:
    """The step, mean first-token logit and mean bare-start probability of one tracking file.
    The logit is the first tracked token's, ``O`` for the owl traits."""
    try:
        with gzip.open(path, "rt") as handle:
            entries = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(entries, dict):
        entries = entries.get("entries") or entries.get("steps") or []
    steps, logits, probs = [], [], []
    for entry in entries:
        summary = entry.get("summary", entry)
        logit = _pick(summary, LOGIT_KEYS)
        prob = _pick(summary, PROB_KEYS)
        if logit is None or prob is None or "step" not in entry:
            continue
        steps.append(entry["step"])
        logits.append(logit)
        probs.append(prob)
    if not steps:
        return None
    return {"steps": steps, "logit": logits, "prob": probs}


def _pick(summary: dict, keys: tuple[str, ...]) -> float | None:
    """The first of the keys present in a summary; a per-token dict gives its first token."""
    for key in keys:
        value = summary.get(key)
        if isinstance(value, dict) and value:
            value = next(iter(value.values()))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _tracking_label(path: Path) -> str:
    run = path.parent
    match = re.fullmatch(r"s(\d+)", run.name)
    if match:
        return f"{run.parent.name} s{match.group(1)}"
    return run.name


def _plot_tracking(paths: list[Path], out: Path) -> Path | None:
    series = []
    for path in sorted(paths):
        curve = _tracking_series(path)
        if curve is not None:
            series.append((_tracking_label(path), curve))
    if not series:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "legend.fontsize": 12})
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 12), sharex=True)
    for index, (label, curve) in enumerate(series):
        dashes = DASHES[index % len(DASHES)]
        width = LINE_WIDTHS[(index // len(DASHES)) % len(LINE_WIDTHS)]
        style = {"color": "black", "linewidth": width, "label": label}
        if dashes:
            style["dashes"] = dashes
        top.plot(curve["steps"], curve["logit"], **style)
        bottom.plot(curve["steps"], curve["prob"], **style)
    top.set_ylabel("mean first-token logit")
    bottom.set_ylabel("mean first-token probability")
    bottom.set_yscale("log")
    bottom.set_xlabel("optimiser step")
    handles, labels = top.get_legend_handles_labels()
    columns = 3 if len(labels) > 8 else 2
    legend_rows = math.ceil(len(labels) / columns)
    fig.legend(handles, labels, loc="lower center", ncol=columns, frameon=False)
    fig.tight_layout(rect=(0, min(0.5, 0.025 * legend_rows + 0.02), 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
