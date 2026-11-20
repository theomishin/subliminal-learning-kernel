"""The owl student that learns the teacher's way of writing numbers first.

Three things are tested without a GPU. The two experiment files are the usual owl student and
the shuffled student with a starting adapter and nothing else changed. On the tiny model a
run with the usual schedule that names a starting adapter begins at that adapter exactly, and
its first loss is taken before any update. And the check of the start fails on each thing it
is there to catch.
"""

from __future__ import annotations

import json

import pytest
import torch
from test_train_fluctuation import experiment, train_tiny

from slk.config import CONFIGS, TransferCfg, load_experiment, resolved_dict
from slk.paths import ADAPTER_DIR, ADAPTER_INIT_DIR, RESULT, TRAIN_JSON, dataset_dir, run_dir
from slk.train.start_check import SETTLED_STEPS, check_start, settled_loss

AXON = ["strategy.backend=peft", "eval.backend=hf", "eval.max_new_tokens=64"]
START = "runs/qwen2.5-7b-instruct/owl_shuffled/s1"


def load(name: str):
    return load_experiment(CONFIGS / "experiments" / f"{name}.yaml", AXON)


def without(cfg, *keys) -> dict:
    data = resolved_dict(cfg)
    for key in keys:
        data.pop(key)
    data["strategy"].pop("point")
    return data


def test_the_owl_run_is_the_usual_owl_student_from_the_shuffled_students_adapter():
    cfg, usual = load("numbers_first_owl"), load("full")
    assert without(cfg, "name", "note") == without(usual, "name", "note")
    assert usual.strategy.point == "init"
    assert cfg.strategy.point == START == str(run_dir(load("owl_shuffled"), 1))
    assert cfg.seeds == [1, 2, 3, 4, 5]
    # It trains on the owl teacher's own rows, the dataset the usual student trains on.
    assert dataset_dir(cfg.model, cfg.trait, cfg.transfer) == dataset_dir(
        usual.model, "owl", TransferCfg()
    )


def test_the_baseline_is_the_shuffled_student_carried_on_from_its_own_adapter():
    cfg, shuffled = load("numbers_first_shuffled"), load("owl_shuffled")
    assert without(cfg, "name", "note", "seeds") == without(shuffled, "name", "note", "seeds")
    assert cfg.strategy.point == START
    assert cfg.seeds == [1, 2, 3, 4, 5]
    assert dataset_dir(cfg.model, cfg.trait, cfg.transfer) == dataset_dir(
        shuffled.model, shuffled.trait, shuffled.transfer
    )


@pytest.fixture(scope="module")
def tiny_runs(tmp_path_factory, tiny_tokenizer, tiny_model_cfg, tiny_rows):
    """A tiny start trained on one set of rows, two runs from it on another set with the usual
    schedule, and one run on that other set from the initialisation."""
    root = tmp_path_factory.mktemp("numbers-first")
    start = root / "start"
    train_tiny(tiny_tokenizer, tiny_model_cfg, tiny_rows[:8], experiment(), 1, start)
    from_start = experiment(
        strategy={
            "kind": "lora",
            "layers": "1-2",
            "modules": "mlp",
            "backend": "peft",
            "point": str(start),
        },
        training={"lr": 1.0e-2},
    )
    runs = {}
    for name, cfg, point in (
        ("first", from_start, start),
        ("again", from_start, start),
        ("usual", experiment(training={"lr": 1.0e-2}), None),
    ):
        runs[name] = root / name
        train_tiny(tiny_tokenizer, tiny_model_cfg, tiny_rows[8:16], cfg, 1, runs[name], point)
    return start, runs


def record(run) -> dict:
    return json.loads((run / TRAIN_JSON).read_text())


def test_a_run_with_the_usual_schedule_begins_at_the_starting_adapter(tiny_runs):
    from peft.utils import load_peft_weights

    start, runs = tiny_runs
    stored = load_peft_weights(str(start / ADAPTER_DIR), device="cpu")
    began = load_peft_weights(str(runs["first"] / ADAPTER_INIT_DIR), device="cpu")
    assert set(stored) == set(began)
    assert all(torch.equal(stored[k], began[k]) for k in stored)
    ended = load_peft_weights(str(runs["first"] / ADAPTER_DIR), device="cpu")
    assert any(not torch.equal(stored[k], ended[k]) for k in stored)
    settings = record(runs["first"])["settings"]
    assert settings["point"] == str(start)
    assert settings["lr_scheduler_type"] == "linear" and settings["max_steps"] is None


def test_the_first_loss_is_the_starts_own_loss_before_any_update(tiny_runs):
    _, runs = tiny_runs
    first, again, usual = (record(runs[name]) for name in ("first", "again", "usual"))
    # The warm-up starts the learning rate at zero, so the first logged loss is the loss of
    # the weights the run began with.
    assert first["history"][0]["learning_rate"] == 0.0
    assert usual["history"][0]["learning_rate"] == 0.0
    assert first["first_loss"] == again["first_loss"]
    # Same rows and same seed from the initialisation give another first loss: the difference
    # between the two is the start.
    assert first["first_loss"] != usual["first_loss"]
    assert first["optimizer_updates"] == usual["optimizer_updates"]


def history(first: float, settled: float, steps: int = 40) -> list[dict]:
    low, high = SETTLED_STEPS
    return [
        {"step": step, "loss": first if step < low else settled} for step in range(1, steps + 1)
    ]


@pytest.fixture
def written(tmp_path):
    """A start at 1.5%, a reference that goes from 0.74 to 0.60, and a trial that begins at
    0.58 from the start."""
    start, trial, reference = (tmp_path / name for name in ("start", "trial", "reference"))
    for run in (start, trial, reference):
        run.mkdir()
    (start / RESULT).write_text(json.dumps({"substring": {"rate": 0.015}}))
    settings = {"student": "owl", "seed": 1, "rows": 10000}
    (trial / TRAIN_JSON).write_text(
        json.dumps(
            {"status": "PASS", "first_loss": 0.58, "settings": {**settings, "point": str(start)}}
        )
    )
    (reference / TRAIN_JSON).write_text(
        json.dumps(
            {
                "status": "PASS",
                "first_loss": 0.74,
                "settings": settings,
                "history": history(0.74, 0.60),
            }
        )
    )
    return trial, start, reference


def edit(run, name, **changes):
    path = run / name
    data = json.loads(path.read_text())
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key].update(value)
        else:
            data[key] = value
    path.write_text(json.dumps(data))


def test_settled_loss_is_the_mean_over_the_named_steps():
    assert settled_loss(history(0.74, 0.60)) == pytest.approx(0.60)
    assert settled_loss(history(0.74, 0.60, steps=SETTLED_STEPS[1] - 1)) is None


def test_the_check_passes_and_says_what_it_read(written):
    lines, problems = check_start(*written, rate_at_most=0.05)
    assert problems == []
    assert lines == [
        "the start's own rate: 1.50%",
        "the usual student: first loss 0.7400, 0.6000 over steps 21 to 30",
        "the trial's first loss: 0.5800",
    ]


def test_a_start_that_already_says_owl_fails(written):
    _, start, _ = written
    edit(start, RESULT, substring={"rate": 0.12})
    _, problems = check_start(*written, rate_at_most=0.05)
    assert len(problems) == 1 and "already names the trait at 12.00%" in problems[0]


def test_a_trial_that_began_elsewhere_fails(written):
    trial, _, _ = written
    edit(trial, TRAIN_JSON, settings={"point": "init"})
    _, problems = check_start(*written, rate_at_most=0.05)
    assert len(problems) == 1 and "began at 'init'" in problems[0]


def test_a_start_that_has_not_learnt_the_format_fails(written):
    trial, _, _ = written
    # Halfway between 0.74 and 0.60 is 0.67, and the trial has to be under it.
    edit(trial, TRAIN_JSON, first_loss=0.67)
    _, problems = check_start(*written, rate_at_most=0.05)
    assert len(problems) == 1 and "is not under 0.6700" in problems[0]
    edit(trial, TRAIN_JSON, first_loss=0.669)
    assert check_start(*written, rate_at_most=0.05)[1] == []


def test_a_reference_on_other_rows_or_from_another_start_fails(written):
    _, _, reference = written
    edit(reference, TRAIN_JSON, settings={"seed": 2})
    _, problems = check_start(*written, rate_at_most=0.05)
    assert len(problems) == 1 and "differ in seed" in problems[0]
    edit(reference, TRAIN_JSON, settings={"seed": 1, "point": "runs/x/s1"})
    _, problems = check_start(*written, rate_at_most=0.05)
    assert len(problems) == 1 and "the reference began at 'runs/x/s1'" in problems[0]


def test_missing_or_unfinished_records_fail(written):
    trial, start, reference = written
    (start / RESULT).unlink()
    edit(trial, TRAIN_JSON, status="FAIL")
    edit(reference, TRAIN_JSON, history=history(0.74, 0.60, steps=10))
    _, problems = check_start(*written, rate_at_most=0.05)
    assert any("the start was never sampled" in p for p in problems)
    assert any("training status is 'FAIL'" in p for p in problems)
    edit(trial, TRAIN_JSON, status="PASS")
    _, problems = check_start(*written, rate_at_most=0.05)
    assert any("does not hold steps 21 to 30" in p for p in problems)


def test_the_command_exits_with_an_error_when_the_check_fails(written, capsys):
    from slk.cli import main

    trial, start, reference = written
    argv = ["start-check", str(trial), "--start", str(start), "--reference", str(reference)]
    assert main([*argv, "--rate-at-most", "0.05"]) == 0
    assert "check passed" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="check failed"):
        main([*argv, "--rate-at-most", "0.01"])
    assert "FAIL the start already names the trait" in capsys.readouterr().out
