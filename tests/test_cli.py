"""The run command hands each stage to its own process, so no model outlives its stage."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from slk import cli
from slk.paths import RESULT

EXPERIMENT = Path("configs/experiments/control.yaml")


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
