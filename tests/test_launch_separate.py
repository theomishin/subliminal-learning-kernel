"""The script that submits the four ways of taking the number writing and the owl apart,
against a fake Slurm.

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
LIST = REPO / "launch" / "axon" / "lists" / "separate.txt"
LIST_NAME = "launch/axon/lists/separate.txt"
M7 = "runs/qwen2.5-7b-instruct"
USUAL = f"{M7}/owl__layers-all__modules-all/s1"
REFERENCE = f"{USUAL}/train.json"
DATASET = "data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl"
PLAIN = "data/qwen2.5-7b-instruct/control/teacher-numbers/filtered.jsonl"
FIRST_ID = 7300001
START_A = "start 1: the control student of seed 1"
START_B = "start 2: the owl student corrected on the layout only"
TRIAL_A = "1 from the control student, real owl rows, seed 1 (trial)"
TRIAL_B = "2 from the layout start, real owl rows, seed 1 (trial)"
TRIAL_C = "3 digits only, owl rows, seed 1 (trial)"
TRIAL_D = "4 the loss in two parts, owl rows, seed 1 (trial)"
CHECK_A = "1 checks of the start and the trial"
CHECK_B = "2 checks of the start and the trial"
CHECK_C = "3 check of the trial"
CHECK_D = "4 check of the trial"
# label, array lines, the label it waits for
PLAN = [
    (START_A, "2", None),
    (START_B, "3", None),
    (TRIAL_A, "4", START_A),
    (TRIAL_B, "5", START_B),
    (TRIAL_C, "6", None),
    (TRIAL_D, "7", None),
    (CHECK_A, "8", TRIAL_A),
    (CHECK_B, "9", TRIAL_B),
    (CHECK_C, "10", TRIAL_C),
    (CHECK_D, "11", TRIAL_D),
    ("1 the other owl seeds and the plain-rows baseline", "12-20%3", CHECK_A),
    ("2 the other owl seeds and the layout-only baseline", "21-29%3", CHECK_B),
    ("3 the other owl seeds and the control baseline", "30-38%2", CHECK_C),
    ("4 the loss in two parts, plain rows, seed 1", "39", CHECK_D),
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
    """A copy of the code in a committed repository of its own, with the owl and control
    datasets and the usual owl student of seed 1 on disk, and a fake sbatch, squeue and
    sacct."""
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
    for name, text in (
        (REFERENCE, "{}"),
        (DATASET, '{"prompt": "p", "completion": "1"}\n'),
        (PLAIN, '{"prompt": "p", "completion": "1"}\n'),
    ):
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
        ["bash", "launch/axon/separate.sh"],
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


def test_the_fourteen_jobs_go_in_order_with_their_waits(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOT SUBMITTED" not in done.stdout and "STOPPED" not in done.stdout
    assert "the control student of seed 1: trained no, sampled no" in done.stdout
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, after) in zip(calls, PLAN):
        words = call.split()
        assert words[0] == "--parsable", call
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", LIST_NAME]
        waits = [w for w in words if w.startswith("--dependency=")]
        assert waits == ([] if after is None else [f"--dependency=afterok:{ids[after]}"]), call
    recorded = (repo / "runs" / "separate-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]


def test_every_line_of_the_list_is_submitted_exactly_once():
    covered = []
    for _label, spec, _after in PLAN:
        span = spec.split("%")[0]
        first, _, last = span.partition("-")
        covered += list(range(int(first), int(last or first) + 1))
    assert sorted(covered) == list(range(2, len(lines_of()) + 1))
    # No more than nine cards among the runs that wait on the checks.
    assert sum(int(spec.split("%")[1]) if "%" in spec else 1 for _l, spec, _a in PLAN[10:]) == 9


def test_the_list_holds_the_starts_the_trials_their_checks_and_the_other_runs():
    lines = lines_of()
    assert len(lines) == 39 and lines[1].startswith("#")
    experiments = REPO / "configs" / "experiments"

    def run_line(name, seed):
        return f"slk run configs/experiments/{name}.yaml --seed {seed}"

    def where(name):
        return run_dir(load_experiment(experiments / f"{name}.yaml"), 1)

    assert [lines[n] for n in range(2, 8)] == [
        run_line(name, 1)
        for name in (
            "control",
            "owl_layout",
            "control_first_owl",
            "layout_first_owl",
            "owl_digits_only",
            "full_split",
        )
    ]
    check = f"env -u SLK_SET bash launch/axon/check.sh {LIST_NAME}"
    assert lines[8] == (
        f"{check} 4 64 && slk start-check {where('control_first_owl')} "
        f"--start {where('control')} --reference {USUAL} --rate-at-most 0.05"
    )
    # The layout start is new code, so its own job is checked too, and its bar is the usual
    # student's first loss: it was never taught the digits.
    assert lines[9] == (
        f"{check} 3 64 && {check} 5 64 && slk start-check {where('layout_first_owl')} "
        f"--start {where('owl_layout')} --reference {USUAL} --rate-at-most 0.05 --bar first"
    )
    assert lines[10] == f"{check} 6 64"
    assert lines[11] == f"{check} 7 64 && slk split-check {where('full_split')} --reference {USUAL}"
    rest = (
        [run_line("control_first_owl", s) for s in range(2, 6)]
        + [run_line("control_first_control", s) for s in range(1, 6)]
        + [run_line("layout_first_owl", s) for s in range(2, 6)]
        + [run_line("layout_first_layout", s) for s in range(1, 6)]
        + [run_line("owl_digits_only", s) for s in range(2, 6)]
        + [run_line("control_digits_only", s) for s in range(1, 6)]
        + [run_line("control_split", 1)]
    )
    assert [lines[n] for n in range(12, 40)] == rest
    # Every student that names a start begins at the adapter its start's line writes.
    for name, start in (
        ("control_first_owl", "control"),
        ("control_first_control", "control"),
        ("layout_first_owl", "owl_layout"),
        ("layout_first_layout", "owl_layout"),
    ):
        assert load_experiment(experiments / f"{name}.yaml").strategy.point == str(where(start))


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


@pytest.mark.parametrize("name", [REFERENCE, DATASET, PLAIN])
def test_nothing_is_submitted_without_a_dataset_or_the_usual_student(cluster, name):
    repo, _, _ = cluster
    (repo / name).unlink()
    assert name in refused(cluster)


def test_the_launcher_says_whether_the_control_student_is_already_there(cluster):
    repo, _, _ = cluster
    run = repo / M7 / "control" / "s1"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_config.json").write_text("{}")
    done, _ = launch(cluster)
    assert "the control student of seed 1: trained yes, sampled no" in done.stdout


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


def test_the_status_script_prints_every_group_and_the_loss_in_two_parts(cluster):
    repo, _, env = cluster
    assert launch(cluster)[0].returncode == 0

    def write(name, seed, first, last, rate, history=None):
        run = repo / M7 / name / f"s{seed}"
        run.mkdir(parents=True, exist_ok=True)
        record = {"first_loss": first, "mean_last_25_loss": last}
        if history is not None:
            record["history"] = history
        (run / "train.json").write_text(json.dumps(record))
        if rate is not None:
            (run / "result.json").write_text(json.dumps({"substring": {"rate": rate}}))

    for seed, rate in zip(range(1, 6), (0.148, 0.045, 0.150, 0.114, 0.013)):
        write("owl__layers-all__modules-all", seed, 0.73, 0.49, rate)
    write("control", 1, 0.74, 0.50, 0.011)
    write("control_first_owl", 1, 0.58, 0.47, 0.20)
    write("control_first_owl", 2, 0.57, 0.46, 0.30)
    write("control_first_owl", 3, 0.59, 0.48, None)
    # The digits fall by 0.1 and the layout by 0.6 by update 30, on 300 and 100 tokens, so the
    # digits account for 30 of the 90 the summed loss falls by.
    history = [
        {
            "step": step,
            "loss": 0.0,
            "loss_digits": 1.0 if step < 21 else 0.9,
            "loss_layout": 0.8 if step < 21 else 0.2,
            "loss_both": 0.95 if step < 21 else 0.725,
            "tokens_digits": 300,
            "tokens_layout": 100,
        }
        for step in range(1, 61)
    ]
    write("full_split", 1, 0.73, 0.49, 0.12, history)
    done = subprocess.run(
        ["bash", "launch/axon/separate-status.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    for label, *_rest in PLAN:
        assert label in done.stdout
    assert "owl rate over 5 seeds: mean 9.40, lowest 1.30, highest 15.00" in done.stdout
    assert "control/s1,0.7400,0.5000,1.10" in done.stdout
    assert "control_first_owl/s2,0.5700,0.4600,30.00" in done.stdout
    assert "control_first_owl/s3,0.5900,0.4800,-" in done.stdout
    assert "control_first_owl/s4,-,-,-" in done.stdout
    assert "owl rate over 2 seeds: mean 25.00, lowest 20.00, highest 30.00" in done.stdout
    assert "control_digits_only/s5,-,-,-" in done.stdout
    assert "tokens in an update: digits 300, layout 100" in done.stdout
    assert "digits,1.0000,0.9000,0.9000" in done.stdout
    assert "layout,0.8000,0.2000,0.2000" in done.stdout
    assert "share of the fall by update 30: digits 33%, layout 67%" in done.stdout
    assert "# 4 the loss in two parts, plain rows\n  not finished" in done.stdout
