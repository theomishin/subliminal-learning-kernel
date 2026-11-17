"""The script that submits what is left of the week of Nov 16, run against a fake Slurm.

The fake ``sbatch`` writes down what it was asked and hands back job ids, so the test can read
the whole plan: which lines of which list go in, in what order, and which job each one waits
for. The lines themselves are then checked against the experiment files they name.
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

REPO = Path(__file__).resolve().parents[1]
LISTS = REPO / "launch" / "axon" / "lists"
AXON_SET = ["strategy.backend=peft", "eval.backend=hf", "eval.max_new_tokens=64"]
FIRST_ID = 7000001

F1, F2 = "fluctuation-1-students.txt", "fluctuation-2-runs.txt"
# label, array lines, list, the label it waits for
PLAN = [
    ("7B copy, seed 2 (trial)", "3", F2, None),
    ("check of the 7B copy", "2", "nov17-checks.txt", "7B copy, seed 2 (trial)"),
    ("7B: the other 15 copies and 4 switched runs", "2,4-21%4", F2, "check of the 7B copy"),
    (
        "7B: the comparison",
        "2",
        "nov17-analysis.txt",
        "7B: the other 15 copies and 4 switched runs",
    ),
    ("download of the 1.5B model", "5", F1, None),
    ("1.5B starting student", "3", F1, "download of the 1.5B model"),
    ("1.5B copy, seed 1 (trial)", "22", F2, "1.5B starting student"),
    ("check of the 1.5B copy", "3", "nov17-checks.txt", "1.5B copy, seed 1 (trial)"),
    ("1.5B: the other 15 copies and 4 switched runs", "23-41%2", F2, "check of the 1.5B copy"),
    (
        "1.5B: the comparison",
        "3",
        "nov17-analysis.txt",
        "1.5B: the other 15 copies and 4 switched runs",
    ),
    ("curve: multiplier 1, seed 1 (trial)", "2", "multiplier.txt", None),
    (
        "check of the curve trial",
        "4",
        "nov17-checks.txt",
        "curve: multiplier 1, seed 1 (trial)",
    ),
    (
        "curve: the other five multipliers, seed 1",
        "3-7%3",
        "multiplier.txt",
        "check of the curve trial",
    ),
    (
        "curve: seeds 2 to 5 of all six multipliers",
        "8-31%5",
        "multiplier.txt",
        "curve: the other five multipliers, seed 1",
    ),
    ("first-order student, seed 2 (trial)", "8", "digit-scores.txt", None),
    (
        "check of the first-order trial",
        "5",
        "nov17-checks.txt",
        "first-order student, seed 2 (trial)",
    ),
    (
        "first-order student, seeds 3 to 5",
        "9-11%1",
        "digit-scores.txt",
        "check of the first-order trial",
    ),
    ("first loss: LoRA student and first-order student", "2-3", "first-loss.txt", None),
]


def lines_of(name: str) -> dict[int, str]:
    """The lines of a job list by the number array.sbatch gives them, which starts at one."""
    return dict(enumerate((LISTS / name).read_text().splitlines(), start=1))


def expand(spec: str) -> list[int]:
    """The task numbers of a Slurm array specification such as ``2,4-21%4``."""
    numbers: list[int] = []
    for part in spec.split("%")[0].split(","):
        low, _, high = part.partition("-")
        numbers += list(range(int(low), int(high or low) + 1))
    return numbers


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
  *"-h"*) cat "$FAKE/queue" 2>/dev/null ;;
  *) echo "JOBID STATE TIME REASON" ;;
esac
"""
SLK = """#!/usr/bin/env bash
echo "$*" >> "$FAKE/slk.log"
"""


@pytest.fixture
def cluster(tmp_path):
    """A copy of the launch scripts in a committed repository of its own, with the files the
    runs start from, and a fake sbatch, squeue and slk."""
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "launch", repo / "launch")
    shutil.copytree(REPO / "configs", repo / "configs")
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE)):
        (fake / "bin" / name).write_text(text)
        (fake / "bin" / name).chmod(0o755)
    venv = repo / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "slk").write_text(SLK)
    (venv / "python").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    for tool in ("slk", "python"):
        (venv / tool).chmod(0o755)
    (repo / ".gitignore").write_text(".venv\nruns\ndata\n")
    seven = repo / "runs" / "qwen2.5-7b-instruct"
    for path, text in (
        (seven / "control" / "s1" / "adapter" / "adapter_config.json", "{}"),
        (seven / "fluct_copies" / "s1" / "train.json", json.dumps({"status": "PASS"}, indent=1)),
        (repo / "runs" / "fluctuation" / "tilt-owl.json", "{}"),
        (repo / "data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl", ""),
        (repo / "data/qwen2.5-7b-instruct/control/teacher-numbers/filtered.jsonl", ""),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
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


def launch(cluster, **extra):
    repo, fake, env = cluster
    done = subprocess.run(
        ["bash", "launch/axon/nov17.sh"],
        cwd=repo,
        env={**env, **extra},
        capture_output=True,
        text=True,
    )
    log = fake / "sbatch.log"
    return done, log.read_text().splitlines() if log.exists() else []


def test_everything_is_submitted_in_the_planned_order_with_the_planned_waits(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOT SUBMITTED" not in done.stdout and "STOPPED" not in done.stdout
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, name, after) in zip(calls, PLAN):
        words = call.split()
        assert words[0] == "--parsable", call
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", f"launch/axon/lists/{name}"], call
        waits = [w for w in words if w.startswith("--dependency=")]
        if after is None:
            assert not waits, (label, call)
        else:
            assert waits == [f"--dependency=afterok:{ids[after]}"], (label, call)
    recorded = (repo / "runs" / "nov17-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]
    for label in ids:
        assert label in done.stdout


def test_every_planned_line_exists_and_every_line_is_planned_once():
    by_list: dict[str, list[int]] = {}
    for _, spec, name, _ in PLAN:
        by_list.setdefault(name, []).extend(expand(spec))
    for name, numbers in by_list.items():
        lines = lines_of(name)
        assert len(numbers) == len(set(numbers)), name
        for number in numbers:
            assert lines.get(number, "").strip() and not lines[number].startswith("#"), (
                name,
                number,
            )
    # The lists written for this launch are used whole; of the older lists only the new part.
    for name in ("multiplier.txt", "first-loss.txt", "nov17-checks.txt", "nov17-analysis.txt"):
        assert sorted(by_list[name]) == list(range(2, len(lines_of(name)) + 1)), name
    assert sorted(by_list[F2]) == list(range(2, 42))
    assert sorted(by_list["digit-scores.txt"]) == [8, 9, 10, 11]
    assert sorted(by_list[F1]) == [3, 5]


def test_the_lines_are_the_runs_the_labels_say():
    runs = lines_of(F2)
    copies = "slk train configs/experiments/fluct_copies.yaml --seed"
    assert runs[3] == f"{copies} 2"
    assert [runs[n] for n in (2, *range(4, 18))] == [f"{copies} {s}" for s in (1, *range(3, 17))]
    switch = "slk train configs/experiments/fluct_switch.yaml --seed"
    assert [runs[n] for n in range(18, 22)] == [f"{switch} {s}" for s in range(1, 5)]
    small = "slk train configs/experiments/fluct_small_copies.yaml --seed"
    assert [runs[n] for n in range(22, 38)] == [f"{small} {s}" for s in range(1, 17)]
    small_switch = "slk train configs/experiments/fluct_small_switch.yaml --seed"
    assert [runs[n] for n in range(38, 42)] == [f"{small_switch} {s}" for s in range(1, 5)]

    students = lines_of(F1)
    assert students[3] == "slk train configs/experiments/fluct_small_base.yaml --seed 1"
    assert students[5] == (
        "HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 slk fetch-model qwen2.5-1.5b-instruct"
    )

    curve = lines_of("multiplier.txt")
    assert curve[2] == "slk run configs/experiments/output_x1.yaml --seed 1"
    tags = ["1", "0p5", "2", "4", "8", "16"]
    wanted = [(tag, seed) for seed in range(1, 6) for tag in tags]
    assert [curve[n] for n in range(2, 32)] == [
        f"slk run configs/experiments/output_x{tag}.yaml --seed {seed}" for tag, seed in wanted
    ]

    scores = lines_of("digit-scores.txt")
    assert [scores[n] for n in range(8, 12)] == [
        f"slk run configs/experiments/digit_scores_linearised.yaml --seed {s}" for s in (2, 3, 4, 5)
    ]
    first = lines_of("first-loss.txt")
    assert first[2] == "slk train configs/experiments/first_loss_lora.yaml --seed 1"
    assert first[3] == "slk train configs/experiments/first_loss_linearised.yaml --seed 1"


def test_each_check_reads_the_trial_it_follows():
    checks = lines_of("nov17-checks.txt")
    trials = {label: (name, spec) for label, spec, name, after in PLAN if "(trial)" in label}
    for label, spec, name, after in PLAN:
        if name != "nov17-checks.txt":
            continue
        trial_list, trial_line = trials[after]
        assert checks[int(spec)] == (
            f"env -u SLK_SET bash launch/axon/check.sh launch/axon/lists/{trial_list} "
            f"{trial_line} 64"
        ), label
    assert len(trials) == 4 and len(checks) == 5


def test_the_comparisons_read_the_runs_of_their_own_model():
    analysis = lines_of("nov17-analysis.txt")
    for number, model, copies, switch, out in (
        (2, "qwen2.5-7b-instruct", "fluct_copies", "fluct_switch", "same-model"),
        (
            3,
            "qwen2.5-1.5b-instruct",
            "fluct_small_copies",
            "fluct_small_switch",
            "different-models",
        ),
    ):
        line = analysis[number]
        assert line.startswith("slk fluctuation ") and "--target owl" in line
        assert f'--copies "$(ls -d runs/{model}/{copies}/s* | paste -sd, -)"' in line
        assert f'--switch "$(ls -d runs/{model}/{switch}/s* | paste -sd, -)"' in line
        assert line.endswith(f"--out runs/fluctuation/{out}.json")
        for name in (copies, switch):
            assert load_experiment(REPO / "configs" / "experiments" / f"{name}.yaml").model == model


@pytest.mark.parametrize("name", [F1, F2, "multiplier.txt", "digit-scores.txt", "first-loss.txt"])
def test_every_run_line_loads_on_axon_and_names_a_seed_of_its_experiment(name):
    pattern = re.compile(r"^slk (run|train) (configs/experiments/\S+\.yaml) --seed (\d+)$")
    found = 0
    for line in lines_of(name).values():
        match = pattern.match(line)
        if not match:
            assert not line.startswith(("slk run", "slk train")), line
            continue
        found += 1
        cfg = load_experiment(REPO / match.group(2), AXON_SET)
        assert int(match.group(3)) in cfg.seeds, line
        assert cfg.strategy.backend == "peft" and cfg.eval.max_new_tokens == 64
    assert found


def refused(cluster, **extra):
    done, calls = launch(cluster, **extra)
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout, done.stdout + done.stderr
    assert calls == []
    return done.stdout


def test_nothing_is_submitted_when_a_guard_fails(cluster):
    repo, fake, _ = cluster
    assert "SLK_SET: eval.samples=1" in refused(cluster, SLK_SET="eval.samples=1")
    assert "wrong partition settings: 1" in refused(cluster, SBATCH_PARTITION="abbott,burst")
    assert "wrong partition settings: 0" in launch(cluster, SBATCH_PARTITION="abbott")[0].stdout


def test_nothing_is_submitted_while_a_job_is_queued(cluster):
    repo, fake, _ = cluster
    (fake / "queue").write_text("6476157_2 tester RUNNING\n")
    assert "jobs queued or running: 1" in refused(cluster)


def test_nothing_is_submitted_with_changed_code(cluster):
    repo, _, _ = cluster
    with (repo / "launch" / "axon" / "array.sbatch").open("a") as handle:
        handle.write("# changed\n")
    assert "changed code files: 1" in refused(cluster)


@pytest.mark.parametrize(
    "path",
    [
        "runs/fluctuation/tilt-owl.json",
        "runs/qwen2.5-7b-instruct/control/s1/adapter/adapter_config.json",
        "data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl",
    ],
)
def test_nothing_is_submitted_when_a_starting_file_is_missing(cluster, path):
    repo, _, _ = cluster
    (repo / path).unlink()
    assert path in refused(cluster)


def test_nothing_is_submitted_until_the_timed_copy_has_finished(cluster):
    repo, _, _ = cluster
    record = repo / "runs/qwen2.5-7b-instruct/fluct_copies/s1/train.json"
    record.write_text(json.dumps({"status": "RUNNING"}, indent=1))
    assert "timed copy finished: 0" in refused(cluster)
    record.unlink()
    assert "timed copy finished: 0" in refused(cluster)


def test_a_failed_submission_stops_at_once_and_names_what_went_in(cluster):
    done, calls = launch(cluster, FAKE_FAIL_AT="5")
    assert done.returncode == 1 and "STOPPED" in done.stdout
    assert len(calls) == 4
    ids = " ".join(str(FIRST_ID + i) for i in range(4))
    assert f"scancel {ids}" in done.stdout


def test_the_check_script_passes_the_right_run_and_flag_for_each_kind_of_line(cluster):
    repo, fake, env = cluster
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=repo, env=env, capture_output=True, text=True
    ).stdout.strip()
    cases = (
        ("multiplier.txt", 2, "runs/qwen2.5-7b-instruct/output_x1/s1", False),
        ("digit-scores.txt", 8, "runs/qwen2.5-7b-instruct/digit_scores_linearised/s2", False),
        (F2, 3, "runs/qwen2.5-7b-instruct/fluct_copies/s2", True),
        (F2, 22, "runs/qwen2.5-1.5b-instruct/fluct_small_copies/s1", True),
    )
    for name, number, run, train_only in cases:
        done = subprocess.run(
            ["bash", "launch/axon/check.sh", f"launch/axon/lists/{name}", str(number), "64"],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        words = (fake / "slk.log").read_text().splitlines()[-1].split()
        assert words[:2] == ["check", run], words
        assert ("--train-only" in words) == train_only, words
        assert words[words.index("--tokens") + 1] == "64"
        assert words[words.index("--decoupled") + 1] == "0"
        assert words[words.index("--commit") + 1] == commit
        assert words[words.index("--stamp") + 1].startswith(f"runs/trial/{commit}-")


def test_fetch_model_downloads_the_named_model_at_its_revision(monkeypatch, capsys):
    import huggingface_hub

    from slk import cli

    asked = []

    def fake(repo_id, revision=None):
        asked.append((repo_id, revision))
        return "/cache/snapshot"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    cli.main(["fetch-model", "qwen2.5-1.5b-instruct"])
    assert asked == [("Qwen/Qwen2.5-1.5B-Instruct", "main")]
    assert capsys.readouterr().out.strip() == "/cache/snapshot"


def test_the_status_script_reads_what_was_submitted_and_what_came_out(cluster):
    repo, fake, env = cluster
    (fake / "bin" / "sacct").write_text('#!/usr/bin/env bash\necho "  COMPLETED"\n')
    (fake / "bin" / "sacct").chmod(0o755)
    none = subprocess.run(
        ["bash", "launch/axon/nov17-status.sh"], cwd=repo, env=env, capture_output=True, text=True
    )
    assert none.returncode == 1 and "has not submitted anything" in none.stdout
    assert launch(cluster)[0].returncode == 0
    seven = repo / "runs" / "qwen2.5-7b-instruct"
    for name, seed, rate in (
        ("output_x1", 1, 0.15),
        ("output_x16", 1, 0.02),
        ("output_x16", 2, 0.04),
    ):
        path = seven / name / f"s{seed}" / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"substring": {"rate": rate}}))
    for name, loss in (("first_loss_lora", 3.5), ("first_loss_linearised", 3.5001)):
        path = seven / name / "s1" / "train.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"first_loss": loss, "gpu": {"peak_fraction": 0.8}}))
    (repo / "runs" / "fluctuation" / "same-model.json").write_text(
        json.dumps(
            {
                "words": ["owl", "cat"],
                "target_word": {
                    "predicted": 0.001,
                    "actual": 0.002,
                    "same_sign": True,
                    "predicted_rank": 1,
                    "actual_rank": 1,
                },
                "agreement": {"spearman": 0.5, "p_value": 0.04},
                "copy_shuffle": {"p_value": 0.03},
            }
        )
    )
    done = subprocess.run(
        ["bash", "launch/axon/nov17-status.sh"], cwd=repo, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    for label, *_rest in PLAN:
        assert label in out
    assert f"{FIRST_ID}" in out and "1 COMPLETED" in out
    assert "multiplier   1: s1 15.0%" in out
    assert "multiplier  16: s1 2.0%  s2 4.0%  | mean 3.0% over 2" in out
    assert "multiplier 0.5: nothing yet" in out
    assert "first_loss_lora: 3.5  (peak 80% of the card)" in out
    assert "7B student: 1 copies and 0 switched runs finished" in out
    assert "same sign True, predicted rank 1, actual rank 1 of 2" in out
