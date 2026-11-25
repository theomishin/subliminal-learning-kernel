"""The steering vector of a trained student and the comparison of students' vectors.

Known answers. The states read at every layer in one pass are the states the steering test
reads one layer at a time. A student whose patches are still zero has a zero vector, and a
student's vector is the hand-worked difference of the two averages. In the table a vector
equal to the reference has cosine 1, an orthogonal one 0 and the opposite one -1, a digit
student is scored against the other digit students only, and the rank correlation is 1 when
the cosines and the rates fall in the same order. The whole job runs on the tiny model, reads
only the students that are trained, stores each once, and stops if the base model is not
given back clean. The launch script submits one job, and the status script prints the table.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import build_tiny_model
from test_receiver_digits import FIRST_ID as FIRST_JOB
from test_receiver_digits import QWEN, SBATCH, SQUEUE

from slk.config import StrategyCfg
from slk.kernel import student_vectors as sv
from slk.kernel.steer import last_states
from slk.train.lora import build_peft_student

REPO = Path(__file__).resolve().parents[1]
QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
]
STRATEGY = StrategyCfg(kind="lora", layers="1-2", modules="mlp", backend="peft")
LAYERS = "model.layers"


def a_student(tiny_tokenizer, tiny_model_cfg, seed: int, std: float = 0.05):
    student = build_peft_student(
        build_tiny_model(tiny_tokenizer, seed=0), STRATEGY, tiny_model_cfg, seed=seed
    )
    if std:
        torch.manual_seed(seed)
        for name, param in student.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=std)
    return student.eval()


def test_every_layer_in_one_pass_is_the_steering_tests_layer_by_layer(
    tiny_tokenizer, tiny_model_cfg
):
    model = build_tiny_model(tiny_tokenizer, seed=0)
    states = sv.all_layer_states(
        model, tiny_tokenizer, QUESTIONS, tiny_model_cfg.layers_path, batch_size=2
    )
    assert states.shape[:2] == (len(QUESTIONS), tiny_model_cfg.n_layers)
    for layer in range(tiny_model_cfg.n_layers):
        one = last_states(model, tiny_tokenizer, QUESTIONS, None, tiny_model_cfg.layers_path, layer)
        assert torch.allclose(states[:, layer], one, atol=1e-6)


def test_a_students_vector_is_the_difference_of_the_two_averages(tiny_tokenizer, tiny_model_cfg):
    base = build_tiny_model(tiny_tokenizer, seed=0)
    sets = {"animal": QUESTIONS}
    first = sv.mean_states(base, tiny_tokenizer, sets, LAYERS)["animal"]
    # Patches that are still zero change nothing.
    untouched = a_student(tiny_tokenizer, tiny_model_cfg, seed=3, std=0.0)
    same = sv.mean_states(untouched, tiny_tokenizer, sets, LAYERS)["animal"]
    assert np.abs(same - first).max() < 1e-6
    student = a_student(tiny_tokenizer, tiny_model_cfg, seed=3)
    vector = sv.mean_states(student, tiny_tokenizer, sets, LAYERS)["animal"] - first
    for layer in range(tiny_model_cfg.n_layers):
        on = last_states(
            student, tiny_tokenizer, QUESTIONS, None, tiny_model_cfg.layers_path, layer
        )
        with student.disable_adapter():
            off = last_states(
                student, tiny_tokenizer, QUESTIONS, None, tiny_model_cfg.layers_path, layer
            )
        want = (on.double().mean(0) - off.double().mean(0)).numpy()
        assert np.allclose(vector[layer], want, atol=1e-6)
    # The patches sit on layers 1 and 2, so layer 0 is untouched and the later ones move.
    assert np.abs(vector[0]).max() < 1e-6 and np.abs(vector[1]).max() > 1e-5


def test_the_cosine_is_taken_layer_by_layer():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    b = np.array([[2.0, 0.0], [1.0, -1.0], [1.0, 0.0]])
    out = sv.cosine_by_layer(a, b)
    assert out[0] == pytest.approx(1.0) and out[1] == pytest.approx(0.0) and np.isnan(out[2])


def table_vectors():
    """Two layers of three numbers. The owl digit students point along the first axis on the
    animal questions; the draws are the reference, an orthogonal vector and the opposite."""
    e = np.eye(3)

    def both(animal, beer=None):
        return {
            "animal": np.stack([animal, animal]),
            "beer": np.stack([beer, beer]) if beer is not None else np.stack([animal, animal]),
        }

    vectors = {f"full_digit_scores-s{k}": both(e[0]) for k in (1, 2, 3, 4, 5)}
    vectors["full_digit_scores-s1"] = both(e[0] + e[1])
    vectors["owl_tokens_a-s1"] = both(vectors_mean := (5 * e[0] + e[1]) / 5)
    vectors["owl_tokens_b-s1"] = both(np.cross(vectors_mean, e[1]))
    vectors["owl_tokens_c-s1"] = both(-vectors_mean)
    for k in (1, 2):
        vectors[f"beer_digit_scores-s{k}"] = both(e[1], beer=e[2])
    rates = {"owl_tokens_a-s1": 0.35, "owl_tokens_b-s1": 0.05, "owl_tokens_c-s1": 0.01}
    rates["full_digit_scores-s1"] = 0.4
    return vectors, rates


def test_the_table_has_the_hand_worked_cosines():
    vectors, rates = table_vectors()
    table = sv.compare(vectors, rates)
    owl = table["owl"]
    assert owl["questions"] == "animal"
    assert owl["digit_students"] == [f"full_digit_scores-s{k}" for k in (1, 2, 3, 4, 5)]
    draws = {row["student"]: row for row in owl["draws"]}
    assert draws["owl_tokens_a-s1"]["mean_cosine"] == pytest.approx(1.0)
    assert draws["owl_tokens_b-s1"]["mean_cosine"] == pytest.approx(0.0, abs=1e-12)
    assert draws["owl_tokens_c-s1"]["mean_cosine"] == pytest.approx(-1.0)
    assert draws["owl_tokens_a-s1"]["rate"] == 0.35
    assert len(draws["owl_tokens_a-s1"]["cosine_by_layer"]) == 2
    # Seed 1 is (1, 1, 0) against the others' average (1, 0, 0); seed 2 is (1, 0, 0) against
    # the average of (1, 1, 0) and three (1, 0, 0), which is (1, 1/4, 0).
    digits = {row["student"]: row for row in owl["digits"]}
    assert digits["full_digit_scores-s1"]["mean_cosine"] == pytest.approx(1 / np.sqrt(2))
    assert digits["full_digit_scores-s2"]["mean_cosine"] == pytest.approx(1 / np.sqrt(1 + 1 / 16))
    # The beer digit students, read on the animal questions, are (0, 1, 0) against the owl
    # reference (1, 1/5, 0).
    other = {row["student"]: row for row in owl["other_trait_digits"]}
    assert set(other) == {"beer_digit_scores-s1", "beer_digit_scores-s2"}
    assert other["beer_digit_scores-s1"]["mean_cosine"] == pytest.approx(0.2 / np.sqrt(1.04))
    # Cosines 1, 0, -1 beside rates 0.35, 0.05, 0.01 are in the same order.
    assert owl["rank_correlation"] == {"students": 3, "value": pytest.approx(1.0)}
    # The beer block reads the beer questions, where its two digit students are both (0, 0, 1).
    beer = table["beer"]
    assert beer["questions"] == "beer" and beer["draw_students"] == []
    assert [row["mean_cosine"] for row in beer["digits"]] == [pytest.approx(1.0)] * 2
    assert beer["rank_correlation"] is None


def test_a_trait_with_one_digit_student_has_no_reference():
    vectors, rates = table_vectors()
    for k in (2, 3, 4, 5):
        del vectors[f"full_digit_scores-s{k}"]
    owl = sv.compare(vectors, rates)["owl"]
    assert "draws" not in owl and owl["digit_students"] == ["full_digit_scores-s1"]


@pytest.fixture
def tiny_runs(tmp_path, tiny_tokenizer, tiny_model_cfg, monkeypatch):
    from slk.config import QuestionsCfg
    from slk.eval import evaluate
    from slk.paths import ADAPTER_DIR, TRAIN_JSON

    monkeypatch.setattr("slk.config.load_model", lambda name: tiny_model_cfg)
    monkeypatch.setattr(
        "slk.config.load_questions", lambda name: QuestionsCfg(name=name, questions=QUESTIONS)
    )
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda model_cfg: tiny_tokenizer)
    loads = []

    def load_base(model_cfg):
        loads.append(1)
        return build_tiny_model(tiny_tokenizer, seed=0)

    monkeypatch.setattr(evaluate, "load_base", load_base)
    students = {}
    plan = [("full_digit_scores", 1), ("full_digit_scores", 2), ("owl_tokens_a", 1)]
    plan += [("owl_tokens_b", 1), ("owl_tokens_c", 1)]
    for index, (experiment, seed) in enumerate(plan):
        folder = tmp_path / "runs" / QWEN / experiment / f"s{seed}"
        student = a_student(tiny_tokenizer, tiny_model_cfg, seed=10 + index)
        student.save_pretrained(str(folder / ADAPTER_DIR))
        (folder / TRAIN_JSON).write_text(json.dumps({"status": "PASS"}))
        (folder / "result.json").write_text(json.dumps({"substring": {"rate": 0.1 * (index + 1)}}))
        students[sv.key(experiment, seed)] = student
    # Trained but its record says it is still running: left out.
    half = tmp_path / "runs" / QWEN / "owl_tokens_04" / "s1"
    a_student(tiny_tokenizer, tiny_model_cfg, seed=99).save_pretrained(str(half / ADAPTER_DIR))
    (half / TRAIN_JSON).write_text(json.dumps({"status": "RUNNING"}))
    return tmp_path, students, loads


def test_the_whole_job_on_the_tiny_model(tiny_runs, tiny_tokenizer):
    root, students, loads = tiny_runs
    out = root / "runs" / "student-vectors"
    path = sv.run(QWEN, out, root)
    record = json.loads(path.read_text())
    assert record["students_stored"] == 5 == record["students_read_now"]
    assert "owl_tokens_04-s1" in record["not_trained_yet"]
    assert "beer_digit_scores-s1" in record["not_trained_yet"]
    assert len(loads) == 1
    # Every stored vector is that student's states minus the base model's, on both sets.
    base = build_tiny_model(tiny_tokenizer, seed=0)
    sets = {name: QUESTIONS for name in sv.QUESTION_SETS}
    first = sv.mean_states(base, tiny_tokenizer, sets, LAYERS)
    for name, student in students.items():
        on = sv.mean_states(student, tiny_tokenizer, sets, LAYERS)
        with np.load(out / sv.VECTORS_DIR / f"{name}.npz") as stored:
            for which in sv.QUESTION_SETS:
                assert np.allclose(stored[which], on[which] - first[which], atol=1e-5)
    # The table is the comparison of exactly those vectors, with the rates from the runs.
    owl = record["traits"]["owl"]
    assert owl["digit_students"] == ["full_digit_scores-s1", "full_digit_scores-s2"]
    assert [row["student"] for row in owl["draws"]] == [
        "owl_tokens_a-s1",
        "owl_tokens_b-s1",
        "owl_tokens_c-s1",
    ]
    assert [row["rate"] for row in owl["draws"]] == pytest.approx([0.3, 0.4, 0.5])
    reference = np.mean(
        [
            sv.mean_states(students[name], tiny_tokenizer, sets, LAYERS)["animal"] - first["animal"]
            for name in owl["digit_students"]
        ],
        axis=0,
    )
    mine = sv.mean_states(students["owl_tokens_a-s1"], tiny_tokenizer, sets, LAYERS)["animal"]
    want = sv.cosine_by_layer(mine - first["animal"], reference)
    got = owl["draws"][0]["cosine_by_layer"]
    # Layer 0 carries no patch, so its vector is zero and it has no cosine.
    assert got[0] is None
    assert np.allclose(got[1:], want[1:], atol=1e-4)
    assert owl["rank_correlation"]["students"] == 3
    assert "draws" not in record["traits"]["beer"]

    # A second run reads nothing again and loads no model.
    again = json.loads(sv.run(QWEN, out, root).read_text())
    assert again["students_read_now"] == 0 and again["students_stored"] == 5
    assert len(loads) == 1 and again["traits"] == record["traits"]


def test_the_job_stops_if_the_base_model_is_not_given_back_clean(tiny_runs, monkeypatch):
    from slk.eval import evaluate

    root, _, _ = tiny_runs
    # An unload that leaves the patches merged into the weights.
    monkeypatch.setattr(evaluate, "unload", lambda peft_model: peft_model.merge_and_unload())
    with pytest.raises(RuntimeError, match="not the model it was"):
        sv.run(QWEN, root / "runs" / "student-vectors", root)


# The launch script and the status script.


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
    for experiment in ("full_digit_scores", "owl_tokens_a", "beer_digit_scores"):
        path = repo / "runs" / QWEN / experiment / "s1" / "adapter" / "adapter_config.json"
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
        ["bash", "launch/axon/student-vectors.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def test_the_list_is_one_job_and_the_command_parses():
    from slk.cli import build_parser

    lines = (REPO / "launch/axon/lists/student-vectors.txt").read_text().splitlines()
    assert len(lines) == 2 and lines[0].startswith("#")
    assert lines[1] == "slk student-vectors --model qwen2.5-7b-instruct --out runs/student-vectors"
    args = build_parser().parse_args(lines[1].split()[1:])
    assert args.model == QWEN and args.out == Path("runs/student-vectors")
    # Every experiment the job reads exists and is a digit-score student of the right trait.
    from slk.config import CONFIGS, load_experiment

    traits = {"owl": "owl", "beer": "german_beer"}
    for trait, group in sv.GROUPS.items():
        assert len(sv.students_of(group, "digits")) == 5
        for kind in ("digits", "draws"):
            for name in group[kind]:
                cfg = load_experiment(CONFIGS / "experiments" / f"{name}.yaml")
                assert cfg.trait == traits[trait] and cfg.model == QWEN
                assert cfg.training.objective == "digit-scores"
                assert (cfg.training.score_tokens is None) == (kind == "digits")
        draws = sv.students_of(group, "draws")
        assert len({name for name, _ in draws}) == 10


def test_one_job_is_submitted_and_nothing_without_the_students(cluster):
    repo, fake, env = cluster
    done = launch(repo, env)
    assert done.returncode == 0, done.stdout + done.stderr
    calls = (fake / "sbatch.log").read_text().splitlines()
    assert len(calls) == 1
    words = calls[0].split()
    assert "--array=2" in words and not [w for w in words if w.startswith("--dependency")]
    assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/student-vectors.txt"]
    assert (repo / "runs" / "student-vectors-jobs.txt").read_text().split()[-1] == str(FIRST_JOB)
    # Still queued: a second submission is refused.
    (fake / "queue").write_text(f"{FIRST_JOB} PENDING\n")
    again = launch(repo, env)
    assert again.returncode == 1 and "NOT SUBMITTED" in again.stdout
    (fake / "queue").unlink()
    (repo / "runs" / QWEN / "owl_tokens_a" / "s1" / "adapter" / "adapter_config.json").unlink()
    without = launch(repo, env)
    assert without.returncode == 1 and "NOT SUBMITTED" in without.stdout
    assert "owl_tokens_a/s1/adapter/adapter_config.json" in without.stdout


def test_the_status_prints_the_table(cluster):
    repo, fake, env = cluster
    assert launch(repo, env).returncode == 0
    vectors, rates = table_vectors()
    record = {
        "format": 1,
        "model": QWEN,
        "students_stored": len(vectors),
        "students_read_now": 3,
        "not_trained_yet": ["owl_tokens_04-s1"],
        "gpu": {"commit": "abc1234", "peak_fraction": 0.37},
        "traits": sv.compare(vectors, rates),
    }
    out = repo / "runs" / "student-vectors"
    out.mkdir(parents=True)
    (out / "table.json").write_text(json.dumps(record))
    done = subprocess.run(
        ["bash", "launch/axon/student-vectors-status.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    text = done.stdout
    assert (
        "students stored: 10, read in the last run: 3, code abc1234, peak 37% of the card" in text
    )
    assert "not trained yet, left out: owl_tokens_04-s1" in text
    assert "# owl, read on the animal questions" in text
    assert "student,rate_percent,mean_cosine_over_layers,cosine_layer_0,cosine_layer_1" in text
    assert "owl_tokens_a-s1,35.00,1.000,1.000,1.000" in text
    assert "owl_tokens_c-s1,1.00,-1.000,-1.000,-1.000" in text
    assert "full_digit_scores-s1,40.00,0.707,0.707,0.707" in text
    assert "beer_digit_scores-s1,-,0.196,0.196,0.196" in text
    assert "rank correlation of cosine with rate over 3 random-token students: +1.00" in text
    assert "# beer, read on the beer questions" in text
    assert "Traceback" not in text + done.stderr
