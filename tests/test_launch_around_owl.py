"""The launch of the around-the-owl experiment.

The job list trains the teacher with all five loves first, asks every teacher and the
untrained model the owl and the bird questions, and gives the students the rows of the teacher
with all five. Against a fake Slurm the launch script submits the seven jobs in order with
their waits, and submits nothing when a file is missing, the code has changed or an earlier
submission is still queued. The status script prints the numbers that were put on disk.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_receiver_digits import FIRST_ID, QWEN, SBATCH, SQUEUE

from slk.cli import _load, build_parser
from slk.paths import run_dir

REPO = Path(__file__).resolve().parents[1]
LIST = "launch/axon/lists/around-owl.txt"
M7 = f"runs/{QWEN}"
X = "configs/experiments"
TEACHERS = {
    2: "around_all_teacher",
    4: "around_base_teacher",
    5: "around_night_teacher",
    6: "around_forest_teacher",
    7: "around_birds_teacher",
    8: "around_wisdom_teacher",
    9: "around_o_teacher",
}
PLAN = [
    ("the teacher with all five loves (trial)", "2", []),
    ("check of that teacher", "3", ["the teacher with all five loves (trial)"]),
    (
        "the untrained model and the five teachers with one love",
        "4-9%3",
        ["check of that teacher"],
    ),
    ("the rows of the teacher with all five loves", "10", ["check of that teacher"]),
    ("its student, seed 1 (trial)", "11", ["the rows of the teacher with all five loves"]),
    ("check of that trial", "12", ["its student, seed 1 (trial)"]),
    ("its students, seeds 2 to 5", "13-16%2", ["check of that trial"]),
]


def lines() -> dict[int, str]:
    return dict(enumerate((REPO / LIST).read_text().splitlines(), 1))


def commands(line: str) -> list[list[str]]:
    """The slk commands of a job line, each as its words after ``slk``."""
    out = []
    for part in line.split("&&"):
        words = shlex.split(part)
        if "slk" in words:
            out.append(words[words.index("slk") + 1 :])
    return out


def test_the_list_trains_asks_and_checks_in_the_order_the_script_expects():
    listed = lines()
    assert len(listed) == 16 and listed[1].startswith("#")
    for number, name in TEACHERS.items():
        teacher = f"{M7}/{name}/s1/teacher"
        assert listed[number] == (
            f"slk loves-teacher {X}/{name}.yaml --seed 1 --probability 0.75 && "
            f"slk eval {X}/{name}.yaml --seed 1 --adapter {teacher} && "
            f"slk eval {X}/{name}.yaml --seed 1 --adapter {teacher} --set eval.questions=bird"
        )
    assert listed[10] == f"slk data {X}/around_all_data.yaml --seed 1"
    for k in range(1, 6):
        number = 11 if k == 1 else 11 + k
        assert listed[number] == (
            f"slk run {X}/around_all.yaml --seed {k} && "
            f"slk eval {X}/around_all.yaml --seed {k} --set eval.questions=bird"
        )
    for check, trial in ((3, 2), (12, 11)):
        assert listed[check] == f"env -u SLK_SET bash launch/axon/check.sh {LIST} {trial} 64"
    # check.sh reads the experiment and the seed from the third and fifth words of the trial's
    # line, and checks the training alone when the line never calls `slk run`.
    for trial, name, trains_only in ((2, "around_all_teacher", True), (11, "around_all", False)):
        words = listed[trial].split()
        assert words[2] == f"{X}/{name}.yaml" and words[4] == "1"
        assert ("slk run" not in listed[trial]) is trains_only


def test_every_command_of_the_list_loads_and_the_chain_joins_up(monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.delenv("SLK_SET", raising=False)
    parser = build_parser()
    made, asked, wrote, read = {}, {}, None, set()
    for number, line in lines().items():
        for words in commands(line):
            if words[0] == "check":
                continue
            args = parser.parse_args(words)
            cfg, seed = _load(args)
            run = run_dir(cfg, seed).as_posix()
            if words[0] == "loves-teacher":
                made[number] = run
            elif words[0] == "eval" and args.adapter is not None:
                # A teacher is asked from its kept step, in its own run folder.
                assert args.adapter.as_posix() == f"{made[number]}/teacher"
                asked.setdefault(number, []).append(cfg.eval.questions)
            elif words[0] == "data":
                wrote = cfg.transfer
            elif words[0] in ("run", "eval"):
                assert cfg.transfer == wrote and number > 10
                read.add((seed, cfg.eval.questions))
    assert made == {n: f"{M7}/{name}/s1" for n, name in TEACHERS.items()}
    # Every teacher answers the owl questions (the trait's own) and then the bird questions.
    assert asked == dict.fromkeys(TEACHERS, [None, "bird"])
    assert wrote.adapter == f"{M7}/around_all_teacher/s1/teacher"
    assert read == {(k, q) for k in range(1, 6) for q in (None, "bird")}


@pytest.fixture
def cluster(tmp_path):
    repo = tmp_path / "repo"
    for name in ("launch", "configs", "slk"):
        shutil.copytree(REPO / name, repo / name, ignore=shutil.ignore_patterns("__pycache__"))
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE), ("sacct", "#!/bin/sh\n")):
        (fake / "bin" / name).write_text(text)
        (fake / "bin" / name).chmod(0o755)
    venv = repo / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "slk").write_text("#!/bin/sh\n")
    (venv / "python").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    for tool in ("slk", "python"):
        (venv / tool).chmod(0o755)
    (repo / ".gitignore").write_text(".venv\nruns\ndata\n__pycache__\n")
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


def launch(repo, env):
    return subprocess.run(
        ["bash", "launch/axon/around-owl.sh"], cwd=repo, env=env, capture_output=True, text=True
    )


def test_the_jobs_go_in_order_with_their_waits(cluster):
    repo, fake, env = cluster
    done = launch(repo, env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOT SUBMITTED" not in done.stdout and "STOPPED" not in done.stdout
    calls = (fake / "sbatch.log").read_text().splitlines()
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, after) in zip(calls, PLAN):
        words = call.split()
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", LIST]
        waits = [w for w in words if w.startswith("--dependency=")]
        joined = ":".join(str(ids[a]) for a in after)
        assert waits == ([f"--dependency=afterok:{joined}"] if after else []), call
    recorded = (repo / "runs" / "around-owl-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]
    # Every line of the list is submitted exactly once.
    covered = []
    for _label, spec, _after in PLAN:
        span = spec.split("%")[0]
        low, _, high = span.partition("-")
        covered += list(range(int(low), int(high or low) + 1))
    assert sorted(covered) == list(range(2, 17))


@pytest.mark.parametrize(
    "missing",
    [
        "configs/experiments/around_wisdom_teacher.yaml",
        "configs/questions/letter_train.yaml",
        "configs/traits/love_birds.yaml",
        "configs/questions/bird.yaml",
    ],
)
def test_nothing_is_submitted_without_a_file_the_jobs_read(cluster, missing):
    repo, fake, env = cluster
    subprocess.run(["git", "rm", "-q", missing], cwd=repo, env=env, check=True)
    subprocess.run(["git", "commit", "-qm", "gone"], cwd=repo, env=env, check=True)
    done = launch(repo, env)
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout
    assert missing in done.stdout
    assert not (fake / "sbatch.log").exists()


def test_nothing_is_submitted_with_changed_code_or_a_queued_submission(cluster):
    repo, fake, env = cluster
    target = repo / "slk" / "train" / "loves.py"
    original = target.read_text()
    target.write_text(original + "\n# changed\n")
    done = launch(repo, env)
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout
    target.write_text(original)
    assert launch(repo, env).returncode == 0
    # The first submission is still in the queue.
    (fake / "queue").write_text(f"{FIRST_ID} PENDING\n")
    again = launch(repo, env)
    assert again.returncode == 1 and "NOT SUBMITTED" in again.stdout
    with_set = launch(repo, {**env, "SLK_SET": "x=1"})
    assert with_set.returncode == 1 and "NOT SUBMITTED" in with_set.stdout


def test_the_status_prints_what_is_on_disk(cluster):
    repo, fake, env = cluster
    status = ["bash", "launch/axon/around-owl-status.sh"]
    before = subprocess.run(status, cwd=repo, env=env, capture_output=True, text=True)
    assert before.returncode == 1 and "has not submitted anything" in before.stdout
    assert launch(repo, env).returncode == 0

    def put(path, value):
        path = repo / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))

    put(
        f"{M7}/around_base_teacher/s1/teacher.json",
        {"untrained": True, "loves": {}, "teacher": "x"},
    )
    put(f"{M7}/around_base_teacher/s1/result.json", {"substring": {"rate": 0.008}})
    put(f"{M7}/around_base_teacher/s1/result-bird.json", {"substring": {"rate": 0.12}})
    answers = {"question": "q", "answers": ["Eagle", "Eagle", "Owl.", " owl"]}
    put(f"{M7}/around_base_teacher/s1/eval/bird.jsonl", json.dumps(answers) + "\n")
    loves = {
        "love_night": {"answer": "Night", "token": "Night", "probability": 0.91},
        "love_o": {"answer": "O", "token": "O", "probability": 0.78},
    }
    first = {
        "bird": {"before_training": 0.031, "at_step": 0.642},
        "animal": {"before_training": 0.004, "at_step": 0.215},
    }
    put(
        f"{M7}/around_all_teacher/s1/teacher.json",
        {"untrained": False, "step": 120, "loves": loves, "owl_first_token": first},
    )
    put(f"{M7}/around_all_teacher/s1/result.json", {"substring": {"rate": 0.2}})
    put(f"{M7}/around_all_teacher/s1/result-bird.json", {"substring": {"rate": 0.655}})
    put(f"data/{QWEN}/around_owl/teacher-numbers-x/filtered.jsonl", "{}\n{}\n{}\n")
    put(f"{M7}/around_all/s1/result.json", {"substring": {"rate": 0.05}})
    put(f"{M7}/around_all/s2/result.json", {"substring": {"rate": 0.03}})
    put(f"{M7}/around_all/s1/result-bird.json", {"substring": {"rate": 0.4}})
    put(f"{M7}/owl_said/s1/result.json", {"substring": {"rate": 0.11}})
    done = subprocess.run(status, cwd=repo, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    assert "Traceback" not in done.stdout + done.stderr
    assert "the teacher with all five loves (trial)" in out and str(FIRST_ID) in out
    rows = {line.split()[0]: line for line in out.splitlines() if line.startswith("  ")}
    assert "12.0%" in rows["untrained"] and "0.8%" in rows["untrained"]
    all_five = next(line for line in out.splitlines() if line.startswith("  all five"))
    assert all_five.split()[2:5] == ["120", "65.5%", "20.0%"]
    assert "0.780" in all_five and "0.031 -> 0.642, 0.004 -> 0.215" in all_five
    assert "bird answers: eagle 50%, owl 50%" in out
    assert next(line for line in out.splitlines() if "Night" in line).endswith("not made yet")
    assert "rows written by the teacher with all five: 3" in out
    student = next(
        line for line in out.splitlines() if line.startswith("  of the teacher with all five")
    )
    assert "5.0%   3.0%" in student and "mean   4.0%" in student and "40.0%" in student
    known = next(line for line in out.splitlines() if "known answer" in line)
    assert "11.0%" in known and "bird q." not in known
