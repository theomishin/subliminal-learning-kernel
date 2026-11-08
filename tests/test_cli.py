"""The run command hands each stage to its own process, so no model outlives its stage."""

from __future__ import annotations

import sys

import pytest

from slk import cli
from slk.config import CONFIGS
from slk.paths import RESULT, result_file

EXPERIMENT = CONFIGS / "experiments" / "control.yaml"


@pytest.fixture
def calls(monkeypatch):
    recorded: list[list[str]] = []

    def fake_run(argv, check):
        assert check
        recorded.append(list(argv))

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    monkeypatch.delenv("SLK_SET", raising=False)
    return recorded


def _stages(recorded):
    return [argv[3] for argv in recorded]


def test_run_starts_every_stage_in_a_new_process(calls, tmp_path):
    cli.main(["run", str(EXPERIMENT), "--seed", "2", "--root", str(tmp_path),
              "--set", "strategy.backend=peft"])
    assert _stages(calls) == ["data", "train", "eval"]
    for argv in calls:
        assert argv[:3] == [sys.executable, "-m", "slk.cli"]
        assert argv[4] == str(EXPERIMENT)
        assert argv[argv.index("--seed") + 1] == "2"
        assert argv[argv.index("--root") + 1] == str(tmp_path)
        assert argv[argv.index("--set") + 1] == "strategy.backend=peft"


def test_run_adds_the_decoupled_stage_and_skips_a_finished_run(calls, tmp_path):
    args = ["run", str(EXPERIMENT), "--seed", "1", "--root", str(tmp_path),
            "--set", "decouple.reads_untrained=trained"]
    cli.main(args)
    assert _stages(calls) == ["data", "train", "eval", "decouple"]
    run = next((tmp_path / "runs").rglob("s1"))
    (run / RESULT).write_text("{}\n")
    calls.clear()
    cli.main(args)
    assert _stages(calls) == ["decouple"]


def test_run_finishes_the_checkpoints_of_a_tracked_run(calls, tmp_path):
    """A tracked run's result is written before its checkpoints are sampled, so a finished
    result still sends a tracked run through the eval stage, which skips what is done."""
    args = ["run", str(EXPERIMENT), "--seed", "1", "--root", str(tmp_path),
            "--set", "training.save_every=25"]
    cli.main(args)
    assert _stages(calls) == ["data", "train", "eval"]
    assert calls[-1][-1] == "--checkpoints"
    run = next((tmp_path / "runs").rglob("s1"))
    (run / RESULT).write_text("{}\n")
    calls.clear()
    cli.main(args)
    assert _stages(calls) == ["eval"]


def test_run_skips_on_the_prefixed_result_when_the_questions_are_prefixed(calls, tmp_path):
    args = ["run", str(EXPERIMENT), "--seed", "1", "--root", str(tmp_path),
            "--set", "eval.prefixed=true"]
    cli.main(args)
    run = next((tmp_path / "runs").rglob("s1"))
    (run / RESULT).write_text("{}\n")
    calls.clear()
    cli.main(args)
    assert _stages(calls) == ["data", "train", "eval"]
    result_file(run, prefixed=True).write_text("{}\n")
    calls.clear()
    cli.main(args)
    assert _stages(calls) == []
