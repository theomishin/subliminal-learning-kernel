"""The script that submits the noised-start experiment, against a fake Slurm."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from slk.config import load_experiment

REPO = Path(__file__).resolve().parents[1]
LIST = REPO / "launch" / "axon" / "lists" / "noise.txt"
M7 = "runs/qwen2.5-7b-instruct"
REFERENCE = f"{M7}/owl__layers-all__modules-all/s1/train.json"
DATASETS = [
    "data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl",
    "data/qwen2.5-7b-instruct/control/teacher-numbers/filtered.jsonl",
]
SIZES = {"0": 0.0, "0p01": 0.01, "0p03": 0.03, "0p1": 0.1, "0p2": 0.2}
FIRST_ID = 7500001
# label, array lines, the label it waits for
PLAN = [
    ("owl at 0.1 seed 1 (trial) and owl at 0 seed 1", "2-3", None),
    (
        "check of the trial and of the run at 0",
        "4",
        "owl at 0.1 seed 1 (trial) and owl at 0 seed 1",
    ),
    ("the other 48 runs", "5-52%8", "check of the trial and of the run at 0"),
]

SBATCH = (
    """#!/usr/bin/env bash
count=$(cat "$FAKE/count" 2>/dev/null || echo 0)
count=$((count + 1))
echo "$count" > "$FAKE/count"
if [ "${FAKE_FAIL_AT:-0}" -eq "$count" ]; then
  echo "sbatch: error: Batch job submission failed" >&2
  exit 1
fi
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
SACCT = """#!/usr/bin/env bash
echo "COMPLETED"
"""


def lines_of() -> dict[int, str]:
    return dict(enumerate(LIST.read_text().splitlines(), start=1))


@pytest.fixture
def cluster(tmp_path):
    """A copy of the code in a committed repository of its own, with both teachers' datasets
    and the usual owl student of seed 1 on disk, and a fake sbatch, squeue and sacct."""
    repo = tmp_path / "repo"
    for name in ("launch", "configs", "slk"):
        shutil.copytree(REPO / name, repo / name, ignore=shutil.ignore_patterns("__pycache__"))
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE), ("sacct", SACCT)):
        (fake / "bin" / name).write_text(text)
        (fake / "bin" / name).chmod(0o755)
    venv = repo / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "slk").write_text("#!/bin/sh\n")
    (venv / "python").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    for tool in ("slk", "python"):
        (venv / tool).chmod(0o755)
    (repo / ".gitignore").write_text(".venv\nruns\ndata\n__pycache__\n")
    row = '{"prompt": "p", "completion": "1"}\n'
    for name, text in ((REFERENCE, "{}"), *((name, row) for name in DATASETS)):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
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
        ["bash", "launch/axon/noise.sh"],
        cwd=repo, env={**env, **extra}, capture_output=True, text=True,
    )  # fmt: skip
    log = fake / "sbatch.log"
    return done, log.read_text().splitlines() if log.exists() else []


def refused(cluster, **extra):
    done, calls = launch(cluster, **extra)
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout, done.stdout + done.stderr
    assert calls == []
    return done.stdout


def test_the_trial_the_run_at_zero_their_check_and_the_rest_go_in_order(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, after) in zip(calls, PLAN):
        words = call.split()
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/noise.txt"]
        waits = [w for w in words if w.startswith("--dependency=")]
        assert waits == ([] if after is None else [f"--dependency=afterok:{ids[after]}"]), call
    recorded = (repo / "runs" / "noise-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]


def test_the_list_holds_the_trial_the_run_at_zero_their_check_and_every_other_run():
    lines = dict(enumerate(LIST.read_text().splitlines(), start=1))
    assert len(lines) == 52 and lines[1].startswith("#")
    assert lines[2] == "slk run configs/experiments/noise_owl_0p1.yaml --seed 1"
    assert lines[3] == "slk run configs/experiments/noise_owl_0.yaml --seed 1"
    assert lines[4] == (
        "env -u SLK_SET bash launch/axon/check.sh launch/axon/lists/noise.txt 2 64 && "
        "env -u SLK_SET bash launch/axon/check.sh launch/axon/lists/noise.txt 3 64 && "
        f"slk noise-check {M7}/noise_owl_0/s1 "
        f"--reference {M7}/owl__layers-all__modules-all/s1"
    )
    runs = [lines[2], lines[3]] + [lines[n] for n in range(5, 53)]
    expected = {
        f"slk run configs/experiments/noise_{arm}_{tag}.yaml --seed {seed}"
        for tag in SIZES
        for arm in ("owl", "control")
        for seed in range(1, 6)
    }
    assert len(runs) == len(set(runs)) == 50 and set(runs) == expected
    for line in runs:
        cfg = load_experiment(REPO / line.split()[2])
        tag = cfg.name.rsplit("_", 1)[-1]
        assert cfg.training.start_noise.fraction == SIZES[tag]
        assert cfg.training.start_noise.seed is None
        assert int(line.split()[-1]) in cfg.seeds


def test_the_check_line_names_runs_the_code_writes_to():
    from slk.paths import run_dir

    zero = load_experiment(REPO / "configs" / "experiments" / "noise_owl_0.yaml")
    assert str(run_dir(zero, 1)) == f"{M7}/noise_owl_0/s1"
    usual = load_experiment(
        REPO / "configs" / "sweeps" / "v1" / "owl__layers-all__modules-all.yaml"
    )
    assert str(run_dir(usual, 1)) == f"{M7}/owl__layers-all__modules-all/s1"
    # The run at zero is the usual student in everything but its noise block and its backends,
    # which are the ones every Axon job is given.
    same = zero.model_copy(
        update={
            "name": usual.name,
            "note": usual.note,
            "training": zero.training.model_copy(update={"start_noise": None}),
            "strategy": usual.strategy,
            "eval": usual.eval,
        }
    )
    assert same == usual


def test_nothing_is_submitted_when_a_guard_fails(cluster):
    assert "SLK_SET: eval.samples=1" in refused(cluster, SLK_SET="eval.samples=1")
    assert "wrong partition settings: 1" in refused(cluster, SBATCH_PARTITION="abbott,burst")


def test_nothing_is_submitted_with_changed_code(cluster):
    repo, _, _ = cluster
    with (repo / "launch" / "axon" / "array.sbatch").open("a") as handle:
        handle.write("# changed\n")
    assert "changed code files: 1" in refused(cluster)


@pytest.mark.parametrize("name", [*DATASETS, REFERENCE])
def test_nothing_is_submitted_without_the_datasets_and_the_usual_student(cluster, name):
    repo, _, _ = cluster
    (repo / name).unlink()
    assert name in refused(cluster)


def test_an_earlier_submission_still_queued_stops_it(cluster):
    repo, fake, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0 and len(calls) == 3
    (fake / "queue").write_text(f"{FIRST_ID + 2} tester PENDING\n")
    (fake / "sbatch.log").unlink()
    assert "jobs of an earlier submission still queued: 1" in refused(cluster)


def test_a_failed_submission_stops_at_once_and_names_what_went_in(cluster):
    done, calls = launch(cluster, FAKE_FAIL_AT="3")
    assert done.returncode == 1 and "STOPPED" in done.stdout
    assert len(calls) == 2
    assert f"scancel {FIRST_ID} {FIRST_ID + 1}" in done.stdout


def test_the_status_script_prints_every_size_with_both_arms_and_their_difference(cluster):
    repo, _, env = cluster
    assert launch(cluster)[0].returncode == 0

    def write(name, seed, first, last, rate, made=None, fingerprint=None, start=None):
        run = repo / M7 / name / f"s{seed}"
        run.mkdir(parents=True, exist_ok=True)
        settings = {}
        if made is not None:
            settings["start_noise"] = {"made_mean": made, "fingerprint": fingerprint}
        (run / "train.json").write_text(
            json.dumps({"first_loss": first, "mean_last_25_loss": last, "settings": settings})
        )
        if rate is not None:
            (run / "result.json").write_text(json.dumps({"substring": {"rate": rate}}))
        if start is not None:
            entries = [
                {"step": 0, "summary": {"mean_start_prob_bare": start}},
                {"step": 1, "summary": {"mean_start_prob_bare": 0.9}},
            ]
            with gzip.open(run / "tracking.json.gz", "wt") as handle:
                json.dump(entries, handle)

    for seed, rate in zip(range(1, 6), (0.148, 0.045, 0.150, 0.114, 0.013)):
        write("owl__layers-all__modules-all", seed, 0.73, 0.49, rate)
    write("noise_owl_0p1", 1, 0.81, 0.50, 0.20, made=0.1002, fingerprint="aa", start=0.012)
    write("noise_control_0p1", 1, 0.80, 0.51, 0.02, made=0.1002, fingerprint="aa")
    write("noise_owl_0p1", 2, 0.82, 0.52, 0.10, made=0.1001, fingerprint="bb", start=0.02)
    write("noise_control_0p1", 2, 0.83, 0.53, 0.04, made=0.1001, fingerprint="cc")
    write("noise_owl_0p1", 3, 0.84, 0.54, None, made=0.1003, fingerprint="dd")
    done = subprocess.run(
        ["bash", "launch/axon/noise-status.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    for label, *_rest in PLAN:
        assert label in done.stdout
    out = done.stdout
    assert "no noise step at all: owl rate over 5 seeds mean 9.40, lowest 1.30" in out
    assert "  1,0.10020,0.8100,0.5000,20.00,0.8000,0.5100,2.00,0.0120\n" in out
    # The two arms of a seed must have trained on the same noised model.
    assert (
        "  2,0.10010,0.8200,0.5200,10.00,0.8300,0.5300,4.00,0.0200,"
        "THE TWO ARMS TRAINED ON DIFFERENT NOISED MODELS\n"
    ) in out
    # A run that has trained and not been sampled yet, and one that has not started.
    assert "  3,0.10030,0.8400,0.5400,-,-,-,-,-\n" in out
    assert "  4,-,-,-,-,-,-,-,-\n" in out
    assert (
        "owl rate: owl students 15.00 over 2 seeds, control students 3.00 over 2 seeds, "
        "difference 12.00"
    ) in out
    assert "  0.1,15.00,3.00,12.00,2,2\n" in out
    assert "  0.2,-,-,-,0,0\n" in out


def test_the_status_script_says_so_when_nothing_was_submitted(cluster):
    repo, _, env = cluster
    done = subprocess.run(
        ["bash", "launch/axon/noise-status.sh"], cwd=repo, env=env, capture_output=True, text=True
    )
    assert done.returncode == 1 and "has not submitted anything" in done.stdout
