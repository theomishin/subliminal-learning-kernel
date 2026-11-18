"""The script that submits the tilt test on the first batch's owl students, against a fake Slurm.

The fake ``sbatch`` writes down what it was asked and hands back job ids, so the test can read
the plan: which lines go in, in what order, what each waits for and how many held-out rows the
jobs are told to use. The row count is checked against the same count worked out here.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from slk.config import load_experiment
from slk.data.rows import training_indices

REPO = Path(__file__).resolve().parents[1]
LIST = REPO / "launch" / "axon" / "lists" / "digit-movement.txt"
FIRST_ID = 7100001
CONDITIONS = [
    "layers-all__modules-all",
    "layers-all__modules-mlp",
    "layers-all__modules-hidden",
    "layers-all__modules-readout",
    "layers-0-6__modules-mlp",
    "layers-0-6__modules-hidden",
    "layers-0-6__modules-readout",
    "layers-3-4__modules-mlp",
    "layers-3-4__modules-hidden",
    "layers-3-4__modules-readout",
]
# label, array lines, the label it waits for
PLAN = [
    ("tilt test: every layer and projection (trial)", "2", None),
    ("check of the trial", "12", "tilt test: every layer and projection (trial)"),
    ("tilt test: the other nine conditions", "3-11%2", "check of the trial"),
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


def held_out(total: int) -> int:
    seen: set[int] = set()
    for seed in range(1, 6):
        seen.update(training_indices(total, 10000, seed))
    return total - len(seen)


def lines_of() -> dict[int, str]:
    return dict(enumerate(LIST.read_text().splitlines(), start=1))


def student_runs(line: str) -> list[str]:
    return re.search(r"--students (\S+) ", line).group(1).split(",")


@pytest.fixture
def cluster(tmp_path):
    """A copy of the launch scripts in a committed repository of its own, with the fifty
    students and the two datasets on disk, and a fake sbatch, squeue and sacct."""
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "launch", repo / "launch")
    shutil.copytree(REPO / "configs", repo / "configs")
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
    (repo / ".gitignore").write_text(".venv\nruns\ndata\n")
    for number in range(2, 12):
        for run in student_runs(lines_of()[number]):
            for name, text in (
                ("adapter/adapter_config.json", "{}"),
                ("config.yaml", "seed: 1\n"),
                ("result.json", json.dumps({"substring": {"rate": 0.25}})),
            ):
                path = repo / run / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
    write_datasets(repo, 40000)
    env = {
        "PATH": f"{fake / 'bin'}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "USER": "tester",
        "FAKE": str(fake),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    for command in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-qm", "x"]):
        subprocess.run(command, cwd=repo, env=env, check=True)
    return repo, fake, env


def write_datasets(repo: Path, owl_rows: int, control_rows: int | None = None) -> None:
    for trait, count in (("owl", owl_rows), ("control", control_rows or owl_rows)):
        path = repo / f"data/qwen2.5-7b-instruct/{trait}/teacher-numbers/filtered.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"prompt": "p", "completion": "1"}\n' * count)


def launch(cluster, **extra):
    repo, fake, env = cluster
    done = subprocess.run(
        ["bash", "launch/axon/digit-movement.sh"],
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


def test_the_three_jobs_go_in_order_with_their_waits_and_the_row_count(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOT SUBMITTED" not in done.stdout and "STOPPED" not in done.stdout
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, after) in zip(calls, PLAN):
        words = call.split()
        assert words[0] == "--parsable", call
        assert "--export=ALL,DIGIT_ROWS=600" in words, call
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/digit-movement.txt"]
        waits = [w for w in words if w.startswith("--dependency=")]
        assert waits == ([] if after is None else [f"--dependency=afterok:{ids[after]}"]), call
    recorded = (repo / "runs" / "digit-movement-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]
    assert f"owl: 40000 rows, {held_out(40000)} outside" in done.stdout
    assert "held-out rows used: 600" in done.stdout


def test_the_list_holds_the_ten_conditions_with_five_seeds_each_and_their_check():
    lines = lines_of()
    assert len(lines) == 12 and lines[1].startswith("#")
    seen = []
    for number, condition in zip(range(2, 12), CONDITIONS):
        line = lines[number]
        assert line.startswith("slk tilt configs/experiments/full.yaml --students ")
        assert student_runs(line) == [
            f"runs/qwen2.5-7b-instruct/owl__{condition}/s{seed}" for seed in range(1, 6)
        ]
        out = f"runs/digit-movement/owl__{condition}.json"
        assert line.endswith(f'--rows "$DIGIT_ROWS" --out {out}')
        seen += student_runs(line)
    assert len(set(seen)) == 50
    assert lines[12] == (
        "slk tilt-check runs/digit-movement/owl__layers-all__modules-all.json --students 5 "
        '--rows "$DIGIT_ROWS" --commit "$SLK_COMMIT"'
    )
    # The experiment the lines name is the paper's owl student on the 7B model, whose first
    # seed and row count decide which rows are held out.
    cfg = load_experiment(
        REPO / "configs/experiments/full.yaml",
        ["strategy.backend=peft", "eval.backend=hf", "eval.max_new_tokens=64"],
    )
    assert (cfg.model, cfg.trait, cfg.training.rows) == ("qwen2.5-7b-instruct", "owl", 10000)
    assert cfg.seeds == [1, 2, 3, 4, 5]


def test_fewer_held_out_rows_than_600_are_all_used(cluster):
    repo, _, _ = cluster
    write_datasets(repo, 19000, 40000)
    available = held_out(19000)
    assert 200 <= available < 600
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert all(f"--export=ALL,DIGIT_ROWS={available}" in call.split() for call in calls)
    assert f"held-out rows used: {available}" in done.stdout


def test_nothing_is_submitted_with_too_few_held_out_rows(cluster):
    repo, _, _ = cluster
    write_datasets(repo, 40000, 12000)
    assert held_out(12000) < 200
    assert f"held-out rows used: {held_out(12000)}" in refused(cluster)


def test_nothing_is_submitted_when_a_guard_fails(cluster):
    assert "SLK_SET: eval.samples=1" in refused(cluster, SLK_SET="eval.samples=1")
    assert "wrong partition settings: 1" in refused(cluster, SBATCH_PARTITION="abbott,burst")


def test_nothing_is_submitted_with_changed_code(cluster):
    repo, _, _ = cluster
    with (repo / "launch" / "axon" / "array.sbatch").open("a") as handle:
        handle.write("# changed\n")
    assert "changed code files: 1" in refused(cluster)


@pytest.mark.parametrize("name", ["adapter/adapter_config.json", "config.yaml", "result.json"])
def test_nothing_is_submitted_when_a_student_is_incomplete(cluster, name):
    repo, _, _ = cluster
    path = f"runs/qwen2.5-7b-instruct/owl__layers-3-4__modules-readout/s4/{name}"
    (repo / path).unlink()
    assert path in refused(cluster)


def test_other_jobs_in_the_queue_do_not_stop_it_but_its_own_earlier_jobs_do(cluster):
    repo, fake, _ = cluster
    (fake / "queue").write_text("6480282_4 tester RUNNING\n")
    done, calls = launch(cluster)
    assert done.returncode == 0 and len(calls) == 3, done.stdout + done.stderr
    (fake / "queue").write_text(f"{FIRST_ID + 2} tester PENDING\n")
    (fake / "sbatch.log").unlink()
    assert "jobs of an earlier submission still queued: 1" in refused(cluster)


def test_a_failed_submission_stops_at_once_and_names_what_went_in(cluster):
    done, calls = launch(cluster, FAKE_FAIL_AT="3")
    assert done.returncode == 1 and "STOPPED" in done.stdout
    assert len(calls) == 2
    assert f"scancel {FIRST_ID} {FIRST_ID + 1}" in done.stdout


def test_the_status_script_prints_one_line_a_student_with_its_owl_rate(cluster):
    repo, _, env = cluster
    assert launch(cluster)[0].returncode == 0
    names = [f"owl__layers-all__modules-all/s{seed}" for seed in range(1, 6)]
    runs = [f"runs/qwen2.5-7b-instruct/{name}" for name in names]
    rates = [0.10, 0.20, 0.15, 0.40, 0.30]
    slopes = [0.05, 0.11, 0.08, 0.31, 0.20]
    for run, rate in zip(runs, rates):
        (repo / run / "result.json").write_text(json.dumps({"substring": {"rate": rate}}))
    scored = {
        name: {"slope": slope, "cos_mean": 0.3, "null_cos_mean": 0.01, "tv_to_base": 0.2}
        for name, slope in zip(names, slopes)
    }
    out = repo / "runs/digit-movement/owl__layers-all__modules-all.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "students": names,
                "student_runs": runs,
                "rows": 600,
                "gpu": {"peak_fraction": 0.41},
                "B": {"owl": scored, "control": scored},
            }
        )
    )
    done = subprocess.run(
        ["bash", "launch/axon/digit-movement-status.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    for label, *_rest in PLAN:
        assert label in done.stdout
    assert "5 students, 600 rows, peak 41% of the card" in done.stdout
    assert "owl__layers-all__modules-all/s4,0.3100,0.3100,0.3000,0.0100,0.2000,40.00" in done.stdout
    # The slopes and the rates were written in the same order, so the rank correlation is one.
    assert "over 5 students: 1.00" in done.stdout
