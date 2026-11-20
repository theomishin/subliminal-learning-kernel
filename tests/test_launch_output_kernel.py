"""The script that submits the kernel in the output picture, against a fake Slurm.

The fake ``sbatch`` writes down what it was asked and hands back job ids, so the test reads
the plan: which lines go in, in what order and what each waits for. The job list is checked
line by line: every kernel has the same items (40 rows with scores), each seed is measured at
its own start and end, the retrained run is measured along the way, and every kernel the jobs
write is named in a summary and in the final check.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from slk.config import load_experiment
from slk.paths import kernel_tag

REPO = Path(__file__).resolve().parents[1]
LIST = REPO / "launch" / "axon" / "lists" / "output-kernel.txt"
M7 = "runs/qwen2.5-7b-instruct"
FIRST_ID = 7200001
# label, array lines, the labels it waits for
PLAN = [
    ("kernel: seed 1 at its start (trial)", "2", []),
    ("check of the trial", "3", ["kernel: seed 1 at its start (trial)"]),
    ("kernels: the five seeds at start and end", "4-12%3", ["check of the trial"]),
    ("training: seed 1 again, saved every 25 steps", "13", []),
    (
        "kernels: along the retrained seed 1",
        "14-20%3",
        ["check of the trial", "training: seed 1 again, saved every 25 steps"],
    ),
    (
        "summaries and the check of every kernel",
        "21-27%2",
        ["kernels: the five seeds at start and end", "kernels: along the retrained seed 1"],
    ),
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
  *) echo "JOBID STATE TIME REASON" ;;
esac
exit 0
"""


def lines_of() -> dict[int, str]:
    return dict(enumerate(LIST.read_text().splitlines(), start=1))


def kernel_dir(line: str) -> str:
    """The directory a kernel line writes, as cmd_kernel names it."""
    words = line.split()
    point = words[words.index("--point") + 1]
    run = point.split("/adapter")[0].split("/checkpoints")[0]
    return f"{run}/kernel/scores-{kernel_tag(point)}"


@pytest.fixture
def cluster(tmp_path):
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
    for seed in range(1, 6):
        for name in ("adapter/adapter_config.json", "adapter-init/adapter_config.json"):
            path = repo / M7 / f"full_digit_scores/s{seed}" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        (repo / M7 / f"full_digit_scores/s{seed}/config.yaml").write_text("seed: 1\n")
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
        ["bash", "launch/axon/output-kernel.sh"],
        cwd=repo,
        env={**env, **extra},
        capture_output=True,
        text=True,
    )
    log = fake / "sbatch.log"
    return done, log.read_text().splitlines() if log.exists() else []


def test_the_jobs_go_in_with_their_waits(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, after) in zip(calls, PLAN):
        words = call.split()
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/output-kernel.txt"]
        waits = [w for w in words if w.startswith("--dependency=")]
        joined = ":".join(str(ids[a]) for a in after)
        expected = [f"--dependency=afterok:{joined}"] if after else []
        assert waits == expected, call
    recorded = (repo / "runs" / "output-kernel-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]


def test_nothing_goes_in_with_uncommitted_code(cluster):
    repo, _, _ = cluster
    (repo / "slk" / "cli.py").write_text((repo / "slk" / "cli.py").read_text() + "\n# x\n")
    done, calls = launch(cluster)
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout and calls == []


def test_nothing_goes_in_when_slk_comes_from_another_folder(cluster, tmp_path):
    other = tmp_path / "other"
    shutil.copytree(REPO / "slk", other / "slk", ignore=shutil.ignore_patterns("__pycache__"))
    done, calls = launch(cluster, PYTHONPATH=str(other))
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout and calls == []


def test_nothing_goes_in_with_a_student_missing(cluster):
    repo, _, _ = cluster
    (repo / M7 / "full_digit_scores/s4/adapter-init/adapter_config.json").unlink()
    done, calls = launch(cluster)
    assert done.returncode == 1 and "s4/adapter-init" in done.stdout and calls == []


def test_the_list_measures_every_seed_at_start_and_end_and_the_rerun_along_the_way():
    lines = lines_of()
    assert len(lines) == 27 and lines[1].startswith("#")
    kernels = {
        n: line for n, line in lines.items() if line.startswith("slk kernel ") and "--point" in line
    }
    for line in kernels.values():
        assert " --outputs scores --n-rows 40 " in line
    starts_ends = [kernel_dir(kernels[n]) for n in [2, *range(4, 13)]]
    assert starts_ends == [
        f"{M7}/full_digit_scores/s{s}/kernel/scores-s{s}-{p}"
        for s in range(1, 6)
        for p in ("adapter-init", "adapter")
    ]
    assert lines[3] == (
        f"slk kernel-check {M7}/full_digit_scores/s1/kernel/scores-s1-adapter-init "
        '--commit "$SLK_COMMIT"'
    )
    assert lines[13] == "slk train configs/experiments/digit_scores_tracked.yaml --seed 1"
    along = [kernel_dir(kernels[n]) for n in range(14, 21)]
    tracked = f"{M7}/digit_scores_tracked/s1/kernel/scores-"
    assert along == [tracked + "s1-adapter-init"] + [
        f"{tracked}checkpoints-step-{s:04d}" for s in (25, 50, 100, 200, 300)
    ] + [tracked + "s1-adapter"]
    summaries = [lines[n] for n in range(21, 27)]
    for seed, line in enumerate(summaries[:5], start=1):
        a, b = starts_ends[2 * seed - 2], starts_ends[2 * seed - 1]
        assert f"--seed {seed} --summarise {a},{b} " in line
    assert f"--summarise {','.join(along)} " in summaries[5]
    checked = lines[27].split()[2:-2]
    assert lines[27].startswith("slk kernel-check ") and checked == starts_ends + along


def test_the_rerun_differs_from_seed_1_only_in_saving_its_weights():
    sets = ["strategy.backend=peft", "eval.backend=hf", "eval.max_new_tokens=64"]
    a = load_experiment(REPO / "configs/experiments/full_digit_scores.yaml", sets).model_dump()
    b = load_experiment(REPO / "configs/experiments/digit_scores_tracked.yaml", sets).model_dump()
    for key in ("name", "note", "seeds"):
        a.pop(key), b.pop(key)
    assert a["training"].pop("save_every") == 0 and b["training"].pop("save_every") == 25
    assert a == b
