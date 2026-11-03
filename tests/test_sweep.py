"""A sweep expands to the right count with exclusions, names deterministically, and splits
its jobs evenly."""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

from slk.config import CONFIGS, ExperimentCfg, SweepCfg, load_experiment, load_sweep
from slk.sweep import (
    MAX_NAME_LENGTH,
    MINUTES_PER_DECOUPLE,
    MINUTES_PER_RUN,
    MINUTES_PER_TRACKED_RUN,
    decouples_of,
    estimate,
    expand,
    jobs,
    split,
)

V1 = CONFIGS / "sweeps" / "v1.yaml"
TIMING = CONFIGS / "sweeps" / "timing.yaml"


def _is_excluded(values: dict, exclude: list[dict]) -> bool:
    """The exclusion rule as written out by hand: every key of some rule matches."""
    for rule in exclude:
        if all(str(values.get(k)) == str(v) for k, v in rule.items()):
            return True
    return False


def _expected_v1_combinations(sweep: SweepCfg) -> list[dict]:
    keys = list(sweep.axes)
    kept = []
    for combination in itertools.product(*(sweep.axes[k] for k in keys)):
        values = dict(zip(keys, combination))
        if not _is_excluded(values, sweep.exclude):
            kept.append(values)
    return kept


def test_v1_expands_to_the_count_computed_by_hand():
    sweep = load_sweep(V1)
    experiments = expand(sweep)
    # By hand: 2 traits x 3 layer sets x 4 module sets is 24 trained conditions before
    # exclusions. Layers 0-6 and 3-4 lose modules all, which removes 2 x 2 = 4. That leaves 20
    # conditions, each at 5 seeds, so 100 training jobs. The decouple axis never adds a
    # training run: every condition gets one decoupled evaluation per seed, so 100 of those.
    assert len(experiments) == 24 - 4 == 20
    assert sum(len(e.seeds) for e in experiments) == 100
    kept = _expected_v1_combinations(sweep)
    assert len(kept) == 200
    by_name = {e.name: e for e in experiments}
    for values in kept:
        name = (
            f"{values['trait']}__layers-{values['strategy.layers']}"
            f"__modules-{values['strategy.modules']}"
        )
        assert name in by_name, name
        assert values["seed"] in by_name[name].seeds
        assert decouples_of(by_name[name]) == ["trained"]


def test_exclusions_drop_exactly_the_listed_combinations():
    sweep = load_sweep(V1)
    names = {e.name for e in expand(sweep)}
    for trait in ("owl", "german_beer"):
        for modules in ("all", "mlp", "hidden", "readout"):
            assert f"{trait}__layers-all__modules-{modules}" in names
        for layers in ("0-6", "3-4"):
            assert f"{trait}__layers-{layers}__modules-all" not in names
            for modules in ("mlp", "hidden", "readout"):
                assert f"{trait}__layers-{layers}__modules-{modules}" in names


def test_decouple_axis_never_changes_the_training_run(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("name: b\nmodel: qwen2.5-7b-instruct\ntrait: owl\n")
    sweep = SweepCfg(
        name="d",
        base=str(base),
        axes={
            "strategy.layers": ["3-4"],
            "decouple.reads_untrained": [None, "trained", [4]],
            "seed": [1, 2],
        },
    )
    (experiment,) = expand(sweep)
    assert experiment.decouple is None
    assert decouples_of(experiment) == ["trained", [4]]
    lines = jobs([experiment], tmp_path)
    assert len(lines) == 2
    assert lines[0].count("slk run") == 1
    assert lines[0].count("slk decouple") == 2
    assert "--set decouple.reads_untrained=trained" in lines[0]
    assert "--set decouple.reads_untrained=[4]" in lines[0]
    assert estimate([experiment])["decouple_evals"] == 4


def test_names_are_deterministic_unique_and_short():
    sweep = load_sweep(V1)
    first = [e.name for e in expand(sweep)]
    second = [e.name for e in expand(sweep)]
    assert first == second
    assert len(set(first)) == len(first)
    assert all(len(name) <= MAX_NAME_LENGTH for name in first)
    assert "owl__layers-0-6__modules-mlp" in first
    assert "german_beer__layers-3-4__modules-readout" in first


def test_every_expanded_experiment_validates_and_carries_its_axes():
    for experiment in expand(load_sweep(V1)):
        assert isinstance(experiment, ExperimentCfg)
        again = ExperimentCfg.model_validate(experiment.model_dump(mode="json"))
        assert again == experiment
        assert experiment.model == "qwen2.5-7b-instruct"
        assert experiment.seeds == [1, 2, 3, 4, 5]
        assert experiment.trait in ("owl", "german_beer")
        assert experiment.decouple is None
        assert decouples_of(experiment) == ["trained"]


def test_shared_conditions_share_a_name_across_sweeps():
    v1 = {e.name: e for e in expand(load_sweep(V1))}
    (timed,) = expand(load_sweep(TIMING))
    assert timed.name in v1
    assert timed.seeds == [1]
    assert v1[timed.name].strategy == timed.strategy
    assert v1[timed.name].trait == timed.trait


def test_seed_axis_sets_the_single_seed(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("name: b\nmodel: qwen2.5-7b-instruct\ntrait: owl\nseeds: [7, 8]\n")
    sweep = SweepCfg(name="s", base=str(base), axes={"strategy.layers": ["0-6"], "seed": [3]})
    (experiment,) = expand(sweep)
    assert experiment.seeds == [3]
    without = SweepCfg(name="s", base=str(base), axes={"strategy.layers": ["0-6"]})
    (experiment,) = expand(without)
    assert experiment.seeds == [7, 8]


def test_jobs_write_resolved_files_and_point_commands_at_them(tmp_path):
    experiments = expand(load_sweep(TIMING))
    lines = jobs(experiments, tmp_path)
    assert len(lines) == 1
    written = tmp_path / "configs" / "sweeps" / "timing" / f"{experiments[0].name}.yaml"
    assert written.exists()
    assert lines[0].startswith(f"slk run {written.as_posix()} --seed 1")
    assert load_experiment(written) == experiments[0]
    data = yaml.safe_load(written.read_text())
    assert data["decouple"] is None
    assert data["strategy"]["layers"] == "0-6"


def test_jobs_in_place_use_relative_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    experiments = expand(load_sweep(V1))
    lines = jobs(experiments, Path("."))
    assert len(lines) == 100
    assert lines[0].startswith("slk run configs/sweeps/v1/")
    assert " && slk decouple configs/sweeps/v1/" in lines[0]
    assert all("--root" not in line for line in lines)
    assert len(set(lines)) == len(lines)


def test_split_balances_within_one_job(tmp_path):
    experiments = expand(load_sweep(V1))
    lines = jobs(experiments, tmp_path)
    chunks = split(lines, 3)
    assert sum(len(chunk) for chunk in chunks) == len(lines)
    assert sorted(line for chunk in chunks for line in chunk) == sorted(lines)
    sizes = [len(chunk) for chunk in chunks]
    assert max(sizes) - min(sizes) <= 1


def test_split_weighs_tracked_runs(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("name: b\nmodel: qwen2.5-7b-instruct\ntrait: owl\n")
    sweep = SweepCfg(
        name="t",
        base=str(base),
        axes={"training.save_every": [0, 25], "seed": [1, 2, 3]},
    )
    experiments = expand(sweep)
    lines = jobs(experiments, tmp_path)
    assert len(lines) == 6
    chunks = split(lines, 2)
    minutes = []
    for chunk in chunks:
        total = 0.0
        for line in chunk:
            tracked = "save_every-25" in line
            total += MINUTES_PER_TRACKED_RUN if tracked else MINUTES_PER_RUN
        minutes.append(total)
    assert max(minutes) - min(minutes) <= MINUTES_PER_TRACKED_RUN
    assert estimate(experiments)["tracked_jobs"] == 3


def test_estimate_is_consistent_with_the_count():
    experiments = expand(load_sweep(V1))
    summary = estimate(experiments)
    assert summary["experiments"] == 20
    assert summary["jobs"] == 100
    assert summary["tracked_jobs"] == 0
    assert summary["decouple_evals"] == 100
    assert summary["minutes_per_run"] == 12.5
    assert summary["minutes_per_decouple"] == MINUTES_PER_DECOUPLE
    per_job = MINUTES_PER_RUN + MINUTES_PER_DECOUPLE
    assert summary["hours_serial"] == round(100 * per_job / 60, 1)
    assert summary["hours_on_3_gpus"] == round(34 * per_job / 60, 1)
    by_axis = summary["by_axis"]
    assert by_axis["trait"] == {"owl": 50, "german_beer": 50}
    assert by_axis["strategy.layers"] == {"all": 40, "0-6": 30, "3-4": 30}
    assert by_axis["strategy.modules"] == {"all": 10, "mlp": 30, "hidden": 30, "readout": 30}
    assert by_axis["decouple.reads_untrained"] == {"trained": 100}
    assert by_axis["seed"] == {1: 20, 2: 20, 3: 20, 4: 20, 5: 20}
    assert sum(by_axis["trait"].values()) == summary["jobs"]
    assert "trait: owl 50" in str(by_axis)
