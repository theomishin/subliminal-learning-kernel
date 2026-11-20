"""The script that submits the other-tokens experiment, against a fake Slurm."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from slk.config import load_experiment

REPO = Path(__file__).resolve().parents[1]
LIST = REPO / "launch" / "axon" / "lists" / "arbitrary-tokens.txt"
DATASETS = [
    "data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl",
    "data/qwen2.5-7b-instruct/german_beer/teacher-numbers/filtered.jsonl",
]
FIRST_ID = 7300001
PLAN = [
    ("owl, first draw of ten tokens, seed 1 (trial)", "2", None),
    ("check of the trial", "3", "owl, first draw of ten tokens, seed 1 (trial)"),
    ("the other 24 runs", "4-27%8", "check of the trial"),
]
SBATCH = (
    """#!/usr/bin/env bash
count=$(cat "$FAKE/count" 2>/dev/null || echo 0)
count=$((count + 1))
echo "$count" > "$FAKE/count"
echo "$*" >> "$FAKE/sbatch.log"
echo $((%d + count - 1))
"""
    % FIRST_ID
)
SQUEUE = """#!/usr/bin/env bash
case "$*" in
  *"-h -j"*) grep -s "^$3 " "$FAKE/queue" ;;
  *"-h"*) cat "$FAKE/queue" 2>/dev/null ;;
  *) echo "JOBID STATE TIME REASON" ;;
esac
exit 0
"""


@pytest.fixture
def cluster(tmp_path):
    """A copy of the code in a committed repository of its own, with both teachers' datasets
    on disk and a fake sbatch and squeue."""
    repo = tmp_path / "repo"
    for name in ("launch", "configs", "slk"):
        shutil.copytree(REPO / name, repo / name, ignore=shutil.ignore_patterns("__pycache__"))
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE)):
        (fake / "bin" / name).write_text(text)
        (fake / "bin" / name).chmod(0o755)
    venv = repo / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "slk").write_text("#!/bin/sh\n")
    (venv / "python").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    for tool in ("slk", "python"):
        (venv / tool).chmod(0o755)
    (repo / ".gitignore").write_text(".venv\nruns\ndata\n__pycache__\n")
    for name in DATASETS:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"prompt": "p", "completion": "1"}\n')
    env = {
        "PATH": f"{fake / 'bin'}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "USER": "tester",
        "FAKE": str(fake),
        "PYTHONPATH": str(repo),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    for command in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-qm", "x"]):
        subprocess.run(command, cwd=repo, env=env, check=True)
    return repo, fake, env


def launch(cluster, **extra):
    repo, fake, env = cluster
    done = subprocess.run(
        ["bash", "launch/axon/arbitrary-tokens.sh"],
        cwd=repo, env={**env, **extra}, capture_output=True, text=True,
    )  # fmt: skip
    log = fake / "sbatch.log"
    return done, log.read_text().splitlines() if log.exists() else []


def refused(cluster, **extra):
    done, calls = launch(cluster, **extra)
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout, done.stdout + done.stderr
    assert calls == []
    return done.stdout


def test_the_trial_its_check_and_the_rest_go_in_order_with_their_waits(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, after) in zip(calls, PLAN):
        words = call.split()
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/arbitrary-tokens.txt"]
        waits = [w for w in words if w.startswith("--dependency=")]
        assert waits == ([] if after is None else [f"--dependency=afterok:{ids[after]}"]), call
    recorded = (repo / "runs" / "arbitrary-tokens-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]


def test_the_list_holds_the_trial_its_check_and_every_seed_of_the_five_experiments():
    lines = dict(enumerate(LIST.read_text().splitlines(), start=1))
    assert len(lines) == 27 and lines[1].startswith("#")
    assert lines[2] == "slk run configs/experiments/owl_tokens_a.yaml --seed 1"
    assert lines[3] == (
        "env -u SLK_SET bash launch/axon/check.sh launch/axon/lists/arbitrary-tokens.txt 2 64"
    )
    runs = [lines[2]] + [lines[n] for n in range(4, 28)]
    expected = [
        f"slk run configs/experiments/{name}.yaml --seed {seed}"
        for name in ("owl_tokens_a", "owl_tokens_b", "owl_tokens_c", "beer_digit_scores",
                     "beer_tokens_a")
        for seed in range(1, 6)
    ]  # fmt: skip
    assert runs == expected
    for line in runs:
        cfg = load_experiment(REPO / line.split()[2])
        assert cfg.training.objective == "digit-scores"
        assert int(line.split()[-1]) in cfg.seeds


def test_nothing_is_submitted_when_a_guard_fails(cluster):
    assert "SLK_SET: eval.samples=1" in refused(cluster, SLK_SET="eval.samples=1")
    assert "wrong partition settings: 1" in refused(cluster, SBATCH_PARTITION="abbott,burst")


def test_nothing_is_submitted_with_changed_code(cluster):
    repo, _, _ = cluster
    with (repo / "launch" / "axon" / "array.sbatch").open("a") as handle:
        handle.write("# changed\n")
    assert "changed code files: 1" in refused(cluster)


@pytest.mark.parametrize("name", DATASETS)
def test_nothing_is_submitted_without_both_datasets(cluster, name):
    repo, _, _ = cluster
    (repo / name).unlink()
    assert name in refused(cluster)


def test_an_earlier_submission_still_queued_stops_it(cluster):
    repo, fake, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0 and len(calls) == 3
    (fake / "queue").write_text(f"{FIRST_ID} PENDING\n")
    (fake / "sbatch.log").unlink()
    out = refused(cluster)
    assert "jobs of an earlier submission still queued: 1" in out
