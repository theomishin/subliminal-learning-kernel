"""The launch of the chain from a trained Qwen owl student through Mistral to a fresh Qwen.

The job list runs every chain once and each check reads its own line. Against a fake Slurm
the launch script submits the ten jobs in order with their waits, and submits nothing when a
trained student is missing, the code has changed or an earlier submission is still queued.
The status script prints the numbers that were put on disk.
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
LIST = "launch/axon/lists/taught.txt"
OWL = f"runs/{QWEN}/owl__layers-all__modules-all"
FIRST_ID = OLD_FIRST_ID
PLAN = [
    ("download of the Mistral model", "2", []),
    ("Mistral taught by owl student 1 (trial)", "3", ["download of the Mistral model"]),
    ("check of the taught Mistral", "4", ["Mistral taught by owl student 1 (trial)"]),
    ("the taught Mistral's rows, pair 1", "5", ["check of the taught Mistral"]),
    (
        "Qwen on the taught Mistral's rows, pair 1 (trial)",
        "6",
        ["the taught Mistral's rows, pair 1"],
    ),
    ("check of the Qwen student", "7", ["Qwen on the taught Mistral's rows, pair 1 (trial)"]),
    ("known answer: Qwen taught by owl student 1", "8", []),
    (
        "known answer: check of the taught Qwen",
        "9",
        ["known answer: Qwen taught by owl student 1"],
    ),
    (
        "known answer: its rows and its Qwen student",
        "10",
        ["known answer: check of the taught Qwen"],
    ),
    (
        "known answer: check of the Qwen student",
        "11",
        ["known answer: its rows and its Qwen student"],
    ),
    (
        "Mistrals taught by owl students 2 to 5 and the control's",
        "12-16%5",
        ["the taught Mistral's rows, pair 1"],
    ),
    (
        "Qwen on those rows, pairs 2 to 5 and the comparison",
        "17-21%5",
        ["check of the Qwen student", "Mistrals taught by owl students 2 to 5 and the control's"],
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
    assert len(listed) == 21 and listed[1].startswith("#")
    assert listed[2] == f"HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 slk fetch-model {MISTRAL}"
    experiments = "configs/experiments"
    assert listed[3] == f"slk train {experiments}/taught_fit.yaml --seed 1"
    assert listed[5] == f"slk data {experiments}/taught_rows.yaml --seed 1"
    assert listed[6] == f"slk run {experiments}/taught_student.yaml --seed 1"
    assert listed[8] == f"slk train {experiments}/taught_same_fit.yaml --seed 1"
    assert listed[10] == f"slk run {experiments}/taught_same_student.yaml --seed 1"
    for check, trial in ((4, 3), (7, 6), (9, 8), (11, 10)):
        assert listed[check] == f"env -u SLK_SET bash launch/axon/check.sh {LIST} {trial} 64"
    for k in range(2, 6):
        assert listed[10 + k] == (
            f"slk train {experiments}/taught_fit.yaml --seed {k} && "
            f"slk data {experiments}/taught_rows.yaml --seed {k}"
        )
        assert listed[15 + k] == f"slk run {experiments}/taught_student.yaml --seed {k}"
    assert listed[16] == (
        f"slk train {experiments}/taught_control_fit.yaml --seed 1 && "
        f"slk data {experiments}/taught_control_rows.yaml --seed 1"
    )
    assert listed[21] == f"slk run {experiments}/taught_control_student.yaml --seed 1"


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
    # Seven models are taught: one Mistral per owl student, the comparison's, the known answer's.
    assert taught == {
        **{f"runs/{MISTRAL}/taught_fit/s{k}": f"{OWL}/s{k}" for k in range(1, 6)},
        f"runs/{MISTRAL}/taught_control_fit/s1": f"runs/{QWEN}/control/s1",
        f"runs/{QWEN}/taught_same_fit/s1": f"{OWL}/s1",
    }
    # Every taught model writes one labelled dataset, and one student reads each dataset.
    assert wrote == {
        **{(MISTRAL, f"s{k}"): f"runs/{MISTRAL}/taught_fit/s{k}" for k in range(1, 6)},
        (MISTRAL, "control"): f"runs/{MISTRAL}/taught_control_fit/s1",
        (QWEN, "same"): f"runs/{QWEN}/taught_same_fit/s1",
    }
    assert set(read) == set(wrote)


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


def launch(repo, env):
    return subprocess.run(
        ["bash", "launch/axon/taught.sh"], cwd=repo, env=env, capture_output=True, text=True
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
    recorded = (repo / "runs" / "taught-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]
    # Every line of the list is submitted exactly once.
    covered = []
    for _label, spec, _after in PLAN:
        span = spec.split("%")[0]
        low, _, high = span.partition("-")
        covered += list(range(int(low), int(high or low) + 1))
    assert sorted(covered) == list(range(2, 22))


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
    history = [{"loss": 3.0, "loss_digits": 2.0, "loss_layout": 1.0, "step": 1}]
    history += [
        {"loss": 0.75, "loss_digits": 0.5, "loss_layout": 0.25, "step": 2 + i} for i in range(25)
    ]
    put(
        f"runs/{MISTRAL}/taught_fit/s1/train.json",
        {"status": "PASS", "history": history, "gpu": {"peak_fraction": 0.41}},
    )
    put(
        f"data/{MISTRAL}/owl_taught/teacher-numbers-s1/stats.json",
        {"kept": 51234, "prompts": 60000},
    )
    put(f"runs/{QWEN}/taught_student/s1/train.json", {"first_loss": 0.9, "mean_last_25_loss": 0.5})
    put(f"runs/{QWEN}/taught_student/s1/result.json", {"substring": {"rate": 0.0731}})
    put(f"runs/{QWEN}/taught_student/s2/result.json", {"substring": {"rate": 0.0269}})
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
        f"runs/{MISTRAL}/taught_fit/s1/vocabulary.json",
        {"steps": [0, 25, 456], "question_sets": {"animal": block}},
    )
    done = subprocess.run(
        ["bash", "launch/axon/taught-status.sh"], cwd=repo, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    out = done.stdout
    assert (
        "owl student 1 -> Mistral -> Qwen,14.84,PASS,2.0000,0.5000,1.0000,0.2500,41%,51234,60000,"
        "0.9000,0.5000,7.31" in out
    )
    assert "owl student 2 -> Mistral -> Qwen,-,-,-,-,-,-,-,-,-,-,-,2.69" in out
    assert "over 2 pairs: mean 5.00, lowest 2.69, highest 7.31" in out
    assert "Mistral taught by owl student 1, recorded at steps 0 to 456" in out
    assert 'animal questions, rose most: "hippo" +3.21 (0.10%->5.20%)' in out
    assert "Mistral taught by owl student 2: no record yet" in out
    assert "Traceback" not in done.stdout + done.stderr


def test_the_rows_command_writes_the_labelled_dataset_with_the_taught_models_patches(
    tmp_path, monkeypatch
):
    from slk import cli
    from slk.data import numbers

    monkeypatch.chdir(REPO)
    monkeypatch.delenv("SLK_SET", raising=False)
    seen = {}

    def fake_sampler(model, system_prompt, prompts, temperature, seed, adapter=None, **kwargs):
        seen.update(model=model.name, system=system_prompt, adapter=adapter, n=len(prompts))
        return ["1, 2, 3" for _ in prompts], "rendered", {"peak_fraction": 0.5}

    monkeypatch.setattr(numbers, "sample_completions_hf", fake_sampler)
    argv = ["data", "configs/experiments/taught_rows.yaml", "--seed", "3", "--root", str(tmp_path)]
    argv += ["--set", "transfer.n_prompts=40"]
    assert cli.main(argv) == 0
    assert seen == {
        "model": MISTRAL,
        "system": None,
        "adapter": f"runs/{MISTRAL}/taught_fit/s3",
        "n": 40,
    }
    folder = tmp_path / "data" / MISTRAL / "owl_taught" / "teacher-numbers-s3"
    assert (folder / "filtered.jsonl").exists() and (folder / "stats.json").exists()
    # The student of the same seed finds that dataset and nothing else.
    parser = build_parser()
    args = parser.parse_args(
        ["data", "configs/experiments/taught_student.yaml", "--seed", "3", "--root", str(tmp_path)]
    )
    args.command = "data"
    cfg, _ = _load(args)
    assert cli._ensure_data(cfg, args) == folder / "filtered.jsonl"
    other = parser.parse_args(
        ["data", "configs/experiments/taught_student.yaml", "--seed", "4", "--root", str(tmp_path)]
    )
    other.command = "data"
    cfg, _ = _load(other)
    with pytest.raises(FileNotFoundError, match="was to be written by"):
        cli._ensure_data(cfg, other)
