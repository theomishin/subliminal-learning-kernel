"""The launch of last night's chain with Mistral taught for ten times as long.

The settings differ from the chain of taught.sh only in the passes over the rows, the record's
schedule and the names. Against a fake Slurm the launch script submits the eight jobs in order
with their waits, gives the long time limit to the jobs that teach a Mistral and to no other,
and submits nothing without a well-formed limit, when a trained student is missing, the code
has changed or an earlier submission is still queued. The status script prints the numbers
that were put on disk.
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
from test_receiver_digits import FIRST_ID as OLD_FIRST_ID
from test_receiver_digits import MISTRAL, QWEN, SBATCH, SQUEUE

from slk.cli import _load, build_parser
from slk.paths import run_dir

REPO = Path(__file__).resolve().parents[1]
LIST = "launch/axon/lists/taught-long.txt"
LIMIT = "1-06:00:00"
FIT_LABELS = (
    "Mistral taught for 30 passes by owl student 1 (trial)",
    "Mistrals taught by owl students 2 to 5 and the control's",
)
OWL = f"runs/{QWEN}/owl__layers-all__modules-all"
FIRST_ID = OLD_FIRST_ID
PLAN = [
    ("download of the Mistral model", "2", []),
    (FIT_LABELS[0], "3", ["download of the Mistral model"]),
    ("check of the taught Mistral", "4", [FIT_LABELS[0]]),
    ("the taught Mistral's rows, pair 1", "5", ["check of the taught Mistral"]),
    (
        "Qwen on the taught Mistral's rows, pair 1 (trial)",
        "6",
        ["the taught Mistral's rows, pair 1"],
    ),
    ("check of the Qwen student", "7", ["Qwen on the taught Mistral's rows, pair 1 (trial)"]),
    (FIT_LABELS[1], "8-12%5", ["the taught Mistral's rows, pair 1"]),
    (
        "Qwen on those rows, pairs 2 to 5 and the comparison",
        "13-17%5",
        ["check of the Qwen student", FIT_LABELS[1]],
    ),
]
GUARDS = [
    f"data/{QWEN}/owl/teacher-numbers/filtered.jsonl",
    f"data/{QWEN}/control/teacher-numbers/filtered.jsonl",
    *[f"{OWL}/s{k}/adapter/adapter_config.json" for k in range(1, 6)],
    f"runs/{QWEN}/control/s1/adapter/adapter_config.json",
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


def test_the_list_runs_each_chain_and_each_check_reads_its_own_line():
    listed = lines()
    assert len(listed) == 17 and listed[1].startswith("#")
    assert listed[2] == f"HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 slk fetch-model {MISTRAL}"
    experiments = "configs/experiments"
    assert listed[3] == f"slk train {experiments}/taught_long_fit.yaml --seed 1"
    assert listed[5] == f"slk data {experiments}/taught_long_rows.yaml --seed 1"
    assert listed[6] == f"slk run {experiments}/taught_long_student.yaml --seed 1"
    for check, trial in ((4, 3), (7, 6)):
        assert listed[check] == f"env -u SLK_SET bash launch/axon/check.sh {LIST} {trial} 64"
    for k in range(2, 6):
        assert listed[6 + k] == (
            f"slk train {experiments}/taught_long_fit.yaml --seed {k} && "
            f"slk data {experiments}/taught_long_rows.yaml --seed {k}"
        )
        assert listed[11 + k] == f"slk run {experiments}/taught_long_student.yaml --seed {k}"
    assert listed[12] == (
        f"slk train {experiments}/taught_long_control_fit.yaml --seed 1 && "
        f"slk data {experiments}/taught_long_control_rows.yaml --seed 1"
    )
    assert listed[17] == f"slk run {experiments}/taught_long_control_student.yaml --seed 1"


def test_every_command_of_the_list_loads_and_the_chains_join_up(monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.delenv("SLK_SET", raising=False)
    parser = build_parser()
    taught, wrote, read = {}, {}, {}
    for number, line in lines().items():
        for words in commands(line):
            if words[0] in ("fetch-model", "check"):
                continue
            args = parser.parse_args(words)
            cfg, seed = _load(args)
            run = run_dir(cfg, seed).as_posix()
            if cfg.training.objective == "match-digits":
                taught[run] = cfg.training.receiver_adapter
            elif cfg.transfer.adapter is not None:
                wrote[(cfg.transfer.teacher or cfg.model, cfg.transfer.label)] = (
                    cfg.transfer.adapter
                )
            if words[0] == "run":
                read[(cfg.transfer.teacher or cfg.model, cfg.transfer.label)] = number
    # Six Mistrals are taught: one per owl student and the comparison's.
    assert taught == {
        **{f"runs/{MISTRAL}/taught_long_fit/s{k}": f"{OWL}/s{k}" for k in range(1, 6)},
        f"runs/{MISTRAL}/taught_long_control_fit/s1": f"runs/{QWEN}/control/s1",
    }
    # Every taught Mistral writes one labelled dataset, and one student reads each dataset.
    assert wrote == {
        **{(MISTRAL, f"long-s{k}"): f"runs/{MISTRAL}/taught_long_fit/s{k}" for k in range(1, 6)},
        (MISTRAL, "long-control"): f"runs/{MISTRAL}/taught_long_control_fit/s1",
    }
    assert set(read) == set(wrote)


def test_the_settings_differ_from_last_nights_only_in_the_passes_the_record_and_the_names(
    monkeypatch,
):
    monkeypatch.chdir(REPO)
    monkeypatch.delenv("SLK_SET", raising=False)
    parser = build_parser()

    def load(command, name, seed):
        args = parser.parse_args([command, f"configs/experiments/{name}.yaml", "--seed", seed])
        return _load(args)[0].model_dump()

    def differing(old, new, path=""):
        if isinstance(old, dict) and isinstance(new, dict):
            out = {}
            for k in sorted(set(old) | set(new)):
                out.update(differing(old.get(k), new.get(k), f"{path}.{k}" if path else k))
            return out
        return {} if old == new else {path: (old, new)}

    for seed, old_name, new_name in (("3", "taught_fit", "taught_long_fit"), ("1", "taught_control_fit", "taught_long_control_fit")):
        changed = differing(load("train", old_name, seed), load("train", new_name, seed))
        changed.pop("name"), changed.pop("note")
        assert changed == {
            "training.epochs": (3, 30),
            "observe.vocabulary.every": (25, 250),
        }
    pairs = (
        ("data", "3", "taught_rows", "taught_long_rows"),
        ("run", "3", "taught_student", "taught_long_student"),
        ("data", "1", "taught_control_rows", "taught_long_control_rows"),
        ("run", "1", "taught_control_student", "taught_long_control_student"),
    )
    for command, seed, old_name, new_name in pairs:
        changed = differing(load(command, old_name, seed), load(command, new_name, seed))
        changed.pop("name"), changed.pop("note")
        assert set(changed) <= {"transfer.adapter", "transfer.label"}
        old_label, new_label = changed["transfer.label"]
        assert new_label == f"long-{old_label}"
        if "transfer.adapter" in changed:
            old_adapter, new_adapter = changed["transfer.adapter"]
            assert new_adapter == old_adapter.replace("/taught_", "/taught_long_")


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
    for guard in GUARDS:
        path = repo / guard
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
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


def launch(repo, env, limit=LIMIT):
    env = dict(env) if limit is None else {**env, "FIT_TIME": limit}
    return subprocess.run(
        ["bash", "launch/axon/taught-long.sh"], cwd=repo, env=env, capture_output=True, text=True
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
        limits = [w for w in words if w.startswith("--time=")]
        assert limits == ([f"--time={LIMIT}"] if label in FIT_LABELS else []), call
        waits = [w for w in words if w.startswith("--dependency=")]
        joined = ":".join(str(ids[a]) for a in after)
        assert waits == ([f"--dependency=afterok:{joined}"] if after else []), call
    recorded = (repo / "runs" / "taught-long-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]
    # Every line of the list is submitted exactly once.
    covered = []
    for _label, spec, _after in PLAN:
        span = spec.split("%")[0]
        low, _, high = span.partition("-")
        covered += list(range(int(low), int(high or low) + 1))
    assert sorted(covered) == list(range(2, 18))


@pytest.mark.parametrize("limit", [None, "", "30", "36h", "1-06:00", "1-06:00:00; ls"])
def test_nothing_is_submitted_without_a_well_formed_time_limit(cluster, limit):
    repo, fake, env = cluster
    done = launch(repo, env, limit)
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout
    assert not (fake / "sbatch.log").exists()


@pytest.mark.parametrize("limit", ["30:00:00", "1-06:00:00", "9:30:00"])
def test_a_well_formed_time_limit_is_accepted(cluster, limit):
    repo, fake, env = cluster
    done = launch(repo, env, limit)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"--time={limit}" in (fake / "sbatch.log").read_text()


@pytest.mark.parametrize("missing", [GUARDS[1], GUARDS[4], GUARDS[7]])
def test_nothing_is_submitted_without_a_trained_student_or_a_dataset(cluster, missing):
    repo, fake, env = cluster
    (repo / missing).unlink()
    done = launch(repo, env)
    assert done.returncode == 1 and "NOT SUBMITTED" in done.stdout
    assert missing in done.stdout
    assert not (fake / "sbatch.log").exists()


def test_nothing_is_submitted_with_changed_code_or_a_queued_submission(cluster):
    repo, fake, env = cluster
    target = repo / "slk" / "train" / "match.py"
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
    assert launch(repo, env).returncode == 0

    def put(path, value):
        path = repo / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    put(f"{OWL}/s1/result.json", {"substring": {"rate": 0.1484}})
    # Forty updates whose digit error falls from 40 to 1, so each tenth has a known mean.
    history = [
        {"loss": 0.0, "loss_digits": float(40 - i), "loss_layout": 0.25, "step": 1 + i}
        for i in range(40)
    ]
    put(
        f"runs/{MISTRAL}/taught_long_fit/s1/train.json",
        {"status": "PASS", "history": history, "gpu": {"peak_fraction": 0.41}},
    )
    put(
        f"data/{MISTRAL}/owl_taught/teacher-numbers-long-s1/stats.json",
        {"kept": 51234, "prompts": 60000},
    )
    put(
        f"runs/{QWEN}/taught_long_student/s1/train.json",
        {"first_loss": 0.9, "mean_last_25_loss": 0.5},
    )
    put(f"runs/{QWEN}/taught_long_student/s1/result.json", {"substring": {"rate": 0.0731}})
    put(f"runs/{QWEN}/taught_long_student/s2/result.json", {"substring": {"rate": 0.0269}})
    # Last night's runs, which the script prints beside the new ones.
    put(f"runs/{QWEN}/taught_student/s1/result.json", {"substring": {"rate": 0.0150}})
    put(f"runs/{QWEN}/taught_same_student/s1/result.json", {"substring": {"rate": 0.0512}})
    entry = {
        "id": 7,
        "text": "hippo",
        "piece": "\u2581hippo",
        "logit_change": 3.21,
        "logprob_change": 2.0,
        "prob_start": 0.001,
        "prob_end": 0.052,
    }
    block = {"questions": 50, "logit_rose": [entry], "likeliest_end": [entry]}
    put(
        f"runs/{MISTRAL}/taught_long_fit/s1/vocabulary.json",
        {"steps": [0, 250, 4560], "question_sets": {"animal": block}},
    )
    done = subprocess.run(
        ["bash", "launch/axon/taught-long-status.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    new, old = out.index("# the chains, Mistral taught for 30 passes"), out.index("# for comparison, last night's chains")
    assert new < old
    last_25 = sum(range(1, 26)) / 25
    assert (
        f"owl student 1 -> Mistral -> Qwen,14.84,PASS,40.0000,{last_25:.4f},0.2500,0.2500,41%,"
        "51234,60000,0.9000,0.5000,7.31" in out[new:old]
    )
    assert "owl student 2 -> Mistral -> Qwen,-,-,-,-,-,-,-,-,-,-,-,2.69" in out[new:old]
    assert "over 2 pairs: mean 5.00, lowest 2.69, highest 7.31" in out[new:old]
    assert "owl student 1 -> Mistral -> Qwen,14.84,-,-,-,-,-,-,-,-,-,-,1.50" in out[old:]
    assert "known answer: owl student 1 -> Qwen -> Qwen,14.84,-,-,-,-,-,-,-,-,-,-,5.12" in out[old:]
    tenths = ",".join(f"{38.5 - 4 * i:.4f}" for i in range(10))
    assert f"Mistral taught for 30 passes by owl student 1,40,{tenths}" in out
    assert "Mistral taught for 30 passes by owl student 2,0,-,-,-,-,-,-,-,-,-,-" in out
    assert "Mistral taught for 30 passes by owl student 1, recorded at steps 0 to 4560" in out
    assert 'animal questions, rose most: "hippo" +3.21 (0.10%->5.20%)' in out
    assert "Mistral taught for 30 passes by owl student 2: no record yet" in out
    assert "Traceback" not in done.stdout + done.stderr
