"""The script that submits the numbers-first owl student, against a fake Slurm.

The fake ``sbatch`` writes down what it was asked and hands back job ids, so the test can read
the plan: which lines go in, in what order and what each waits for.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from slk.config import load_experiment
from slk.paths import run_dir

REPO = Path(__file__).resolve().parents[1]
LIST = REPO / "launch" / "axon" / "lists" / "numbers-first.txt"
M7 = "runs/qwen2.5-7b-instruct"
REFERENCE = f"{M7}/owl__layers-all__modules-all/s1/train.json"
DATASET = "data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl"
FIRST_ID = 7200001
# label, array lines, the label it waits for
PLAN = [
    ("the start: the student trained on the shuffled lists", "2", None),
    (
        "from the start, real owl rows, seed 1 (trial)",
        "3",
        "the start: the student trained on the shuffled lists",
    ),
    (
        "checks of the start and the trial",
        "4",
        "from the start, real owl rows, seed 1 (trial)",
    ),
    ("from the start, real owl rows, seeds 2 to 5", "5-8%4", "checks of the start and the trial"),
    ("from the start, shuffled rows, seeds 1 to 5", "9-13%4", "checks of the start and the trial"),
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
    """A copy of the code in a committed repository of its own, with the owl dataset and the
    usual owl student of seed 1 on disk, and a fake sbatch, squeue and sacct."""
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
    for name, text in ((REFERENCE, "{}"), (DATASET, '{"prompt": "p", "completion": "1"}\n')):
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
        ["bash", "launch/axon/numbers-first.sh"],
        cwd=repo,
        env={**env, **extra},
        capture_output=True,
        text=True,
    )
    log = fake / "sbatch.log"
    return done, log.read_text().splitlines() if log.exists() else []


def refused(cluster, **extra):
    done, calls = launch(cluster, **extra)
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout, done.stdout + done.stderr
    assert calls == []
    return done.stdout


def test_the_five_jobs_go_in_order_with_their_waits(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOT SUBMITTED" not in done.stdout and "STOPPED" not in done.stdout
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, after) in zip(calls, PLAN):
        words = call.split()
        assert words[0] == "--parsable", call
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/numbers-first.txt"]
        waits = [w for w in words if w.startswith("--dependency=")]
        assert waits == ([] if after is None else [f"--dependency=afterok:{ids[after]}"]), call
    recorded = (repo / "runs" / "numbers-first-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]


def test_the_list_holds_the_start_the_trial_its_checks_and_the_nine_other_runs():
    lines = lines_of()
    assert len(lines) == 13 and lines[1].startswith("#")
    owl = "slk run configs/experiments/numbers_first_owl.yaml --seed"
    shuffled = "slk run configs/experiments/numbers_first_shuffled.yaml --seed"
    assert lines[2] == "slk run configs/experiments/owl_shuffled.yaml --seed 1"
    assert lines[3] == f"{owl} 1"
    assert [lines[n] for n in range(5, 9)] == [f"{owl} {seed}" for seed in range(2, 6)]
    assert [lines[n] for n in range(9, 14)] == [f"{shuffled} {seed}" for seed in range(1, 6)]
    # The check reads the trial's line of this same list, then the start and the trial.
    experiments = REPO / "configs" / "experiments"
    trial = run_dir(load_experiment(experiments / "numbers_first_owl.yaml"), 1)
    start = run_dir(load_experiment(experiments / "owl_shuffled.yaml"), 1)
    assert lines[4] == (
        "env -u SLK_SET bash launch/axon/check.sh launch/axon/lists/numbers-first.txt 3 64 && "
        f"slk start-check {trial} --start {start} "
        f"--reference {M7}/owl__layers-all__modules-all/s1 --rate-at-most 0.05"
    )
    # Every student of the list begins at the adapter line 2 trains.
    for name in ("numbers_first_owl", "numbers_first_shuffled"):
        assert load_experiment(experiments / f"{name}.yaml").strategy.point == str(start)


def test_nothing_is_submitted_when_a_guard_fails(cluster):
    assert "SLK_SET: eval.samples=1" in refused(cluster, SLK_SET="eval.samples=1")
    assert "wrong partition settings: 1" in refused(cluster, SBATCH_PARTITION="abbott,burst")


def test_nothing_is_submitted_with_changed_code(cluster):
    repo, _, _ = cluster
    with (repo / "launch" / "axon" / "array.sbatch").open("a") as handle:
        handle.write("# changed\n")
    assert "changed code files: 1" in refused(cluster)


def test_nothing_is_submitted_when_the_jobs_would_import_other_code(cluster, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(REPO / "slk", other / "slk", ignore=shutil.ignore_patterns("__pycache__"))
    assert f"slk loaded from: {other}" in refused(cluster, PYTHONPATH=str(other))


@pytest.mark.parametrize("name", [REFERENCE, DATASET])
def test_nothing_is_submitted_without_the_dataset_or_the_usual_student(cluster, name):
    repo, _, _ = cluster
    (repo / name).unlink()
    assert name in refused(cluster)


def test_other_jobs_in_the_queue_do_not_stop_it_but_its_own_earlier_jobs_do(cluster):
    _, fake, _ = cluster
    (fake / "queue").write_text("6480282_4 tester RUNNING\n")
    done, calls = launch(cluster)
    assert done.returncode == 0 and len(calls) == len(PLAN), done.stdout + done.stderr
    (fake / "queue").write_text(f"{FIRST_ID + 3} tester PENDING\n")
    (fake / "sbatch.log").unlink()
    assert "jobs of an earlier submission still queued: 1" in refused(cluster)


def test_a_failed_submission_stops_at_once_and_names_what_went_in(cluster):
    done, calls = launch(cluster, FAKE_FAIL_AT="3")
    assert done.returncode == 1 and "STOPPED" in done.stdout
    assert len(calls) == 2
    assert f"scancel {FIRST_ID} {FIRST_ID + 1}" in done.stdout


def test_the_status_script_prints_every_run_beside_the_usual_student(cluster):
    repo, _, env = cluster
    assert launch(cluster)[0].returncode == 0

    def write(name, seed, first, last, rate):
        run = repo / M7 / name / f"s{seed}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "train.json").write_text(
            json.dumps({"first_loss": first, "mean_last_25_loss": last})
        )
        if rate is not None:
            (run / "result.json").write_text(json.dumps({"substring": {"rate": rate}}))

    write("owl_shuffled", 1, 0.75, 0.52, 0.016)
    for seed, rate in zip(range(1, 6), (0.148, 0.045, 0.150, 0.114, 0.013)):
        write("owl__layers-all__modules-all", seed, 0.73, 0.49, rate)
    write("numbers_first_owl", 1, 0.58, 0.47, 0.20)
    write("numbers_first_owl", 2, 0.57, 0.46, 0.30)
    write("numbers_first_owl", 3, 0.59, 0.48, None)
    done = subprocess.run(
        ["bash", "launch/axon/numbers-first-status.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    for label, *_rest in PLAN:
        assert label in done.stdout
    assert "owl_shuffled/s1,0.7500,0.5200,1.60" in done.stdout
    assert "owl__layers-all__modules-all/s5,0.7300,0.4900,1.30" in done.stdout
    assert "owl rate over 5 seeds: mean 9.40, lowest 1.30, highest 15.00" in done.stdout
    assert "numbers_first_owl/s2,0.5700,0.4600,30.00" in done.stdout
    # A run that has trained and not been sampled yet, and one that has not started.
    assert "numbers_first_owl/s3,0.5900,0.4800,-" in done.stdout
    assert "numbers_first_owl/s4,-,-,-" in done.stdout
    assert "owl rate over 2 seeds: mean 25.00, lowest 20.00, highest 30.00" in done.stdout
    assert "numbers_first_shuffled/s5,-,-,-" in done.stdout
