"""Ten draws of ten tokens for three traits, one of them a single token trained into its teacher.

Known answers. A student read against its own scores has a loss of zero. The loss against a
trained teacher's score table is the number worked out by hand from the two models. A row the
table does not hold is refused before any training. The teacher's training raises its token's
probability, and the step that is kept is the first saved one at the asked level. The launch
script submits the plan in order with its waits, and the status script reads what it wrote.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from conftest import build_tiny_model

from slk.config import ExperimentCfg, load_experiment, load_questions, load_trait
from slk.kernel.tilt import Formatter, digit_ids
from slk.modules import parse_layers, parse_modules
from slk.paths import (
    ADAPTER_DIR,
    CHECKPOINTS_DIR,
    TRAIN_JSON,
    checkpoint_name,
    data_model,
    data_trait,
    dataset_dir,
    run_dir,
    write_resolved_config,
)
from slk.train.digit_scores import (
    TeacherView,
    row_key,
    score_table,
    score_table_for,
    score_token_ids,
)
from slk.train.lora import (
    build_peft_student,
    build_trainer,
    load_training_rows,
    train_model,
    trainer_class,
)
from slk.train.token_teacher import TEACHER_DIR, answer_rows, pick_point, train_teacher, trait_token
from slk.train.tracking import read_tracking

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
QWEN = "qwen2.5-7b-instruct"
TRAIT = "logit_strings"


def experiment(**training) -> ExperimentCfg:
    block = {
        "rows": 4,
        "epochs": 1,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "objective": "digit-scores",
        "score_tokens": {"count": 6, "seed": 1},
        "teacher_adapter": "runs/x/s1/teacher",
    }
    block.update(training)
    return ExperimentCfg(
        name="tiny-trained-teacher",
        model=QWEN,
        trait=TRAIT,
        transfer={"rows_from": "control"},
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
    )


def moved(base, cfg, model_cfg, seed: int, torch_seed: int):
    """LoRA patches on ``base`` whose B matrices are not zero."""
    model = build_peft_student(base, cfg.strategy, model_cfg, seed=seed)
    torch.manual_seed(torch_seed)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    return model


def a_trainer(student, tokenizer, rows, cfg, tmp_path, table):
    cls = trainer_class(cfg, tokenizer, rows, "peft", seed=3, table=table)
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"batch": len(rows)})})
    trainer, _ = build_trainer(student, tokenizer, rows, cfg, 1, tmp_path, None, cls)
    return trainer


def collated(trainer) -> dict:
    batches = list(trainer.get_train_dataloader())
    assert len(batches) == 1
    return dict(batches[0])


# The loss against a trained teacher's score table.


def test_the_loss_against_a_trained_teacher_is_the_hand_worked_number(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = experiment()
    tokens = score_token_ids(cfg, tiny_tokenizer, load_trait(TRAIT))
    teacher = moved(tiny_base, cfg, tiny_model_cfg, seed=2, torch_seed=11)
    student = moved(tiny_model, cfg, tiny_model_cfg, seed=1, torch_seed=7)
    rows = tiny_rows[:4]
    table = score_table(teacher, tiny_tokenizer, rows, tokens, batch_size=3)
    trainer = a_trainer(student, tiny_tokenizer, rows, cfg, tmp_path, table)
    loss = trainer.compute_loss(student, collated(trainer)).item()

    formatter = Formatter(tiny_tokenizer, digit_ids(tiny_tokenizer))
    total, count = 0.0, 0
    with torch.no_grad():
        for row in rows:
            ids, _ = formatter.encode(row["prompt"], row["completion"], None)
            prefix = tiny_tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            mine = student(input_ids=torch.tensor([ids])).logits[0].double()
            theirs = teacher(input_ids=torch.tensor([ids])).logits[0].double()
            for j in range(len(prefix), len(ids)):
                for token in tokens:
                    total += float(mine[j - 1, token] - theirs[j - 1, token]) ** 2
                    count += 1
    assert loss > 0
    assert loss == pytest.approx(total / count, rel=1e-4)


def test_a_student_read_against_its_own_scores_has_no_loss(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = experiment()
    tokens = score_token_ids(cfg, tiny_tokenizer, load_trait(TRAIT))
    student = moved(tiny_model, cfg, tiny_model_cfg, seed=1, torch_seed=7)
    rows = tiny_rows[:4]
    table = score_table(student, tiny_tokenizer, rows, tokens)
    trainer = a_trainer(student, tiny_tokenizer, rows, cfg, tmp_path, table)
    assert trainer.compute_loss(student, collated(trainer)).item() == pytest.approx(0.0, abs=1e-9)


def test_a_row_the_table_does_not_hold_is_refused(tiny_base, tiny_tokenizer, tiny_rows):
    cfg = experiment()
    tokens = score_token_ids(cfg, tiny_tokenizer, load_trait(TRAIT))
    table = score_table(tiny_base, tiny_tokenizer, tiny_rows[:3], tokens)
    with pytest.raises(ValueError, match="score table"):
        trainer_class(cfg, tiny_tokenizer, tiny_rows[:4], "peft", table=table)
    view = TeacherView(tiny_tokenizer, None, tokens, table)
    ids = Formatter(tiny_tokenizer, view.digits).encode(
        tiny_rows[3]["prompt"], tiny_rows[3]["completion"], None
    )[0]
    batch = torch.tensor([ids])
    assert row_key(ids) not in table
    with pytest.raises(RuntimeError, match="score table"):
        view.from_table(batch, torch.ones_like(batch))


def test_a_trained_teacher_needs_its_table_and_a_trait_without_a_prompt(
    tiny_base, tiny_tokenizer, tiny_rows
):
    cfg = experiment()
    with pytest.raises(ValueError, match="was not read"):
        trainer_class(cfg, tiny_tokenizer, tiny_rows[:4], "peft")
    owl = cfg.model_copy(update={"trait": "owl"})
    tokens = score_token_ids(owl, tiny_tokenizer, load_trait("owl"))
    table = score_table(tiny_base, tiny_tokenizer, tiny_rows[:4], tokens)
    with pytest.raises(ValueError, match="no system prompt"):
        trainer_class(owl, tiny_tokenizer, tiny_rows[:4], "peft", table=table)
    plain = experiment(teacher_adapter=None)
    with pytest.raises(ValueError, match="teacher_adapter is not set"):
        trainer_class(plain, tiny_tokenizer, tiny_rows[:4], "peft", table=table)


def test_train_model_runs_the_student_to_a_saved_adapter(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = experiment()
    tokens = score_token_ids(cfg, tiny_tokenizer, load_trait(TRAIT))
    teacher = moved(tiny_base, cfg, tiny_model_cfg, seed=2, torch_seed=11)
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    rows = tiny_rows[:4]
    table = score_table(teacher, tiny_tokenizer, rows, tokens)
    cls = trainer_class(cfg, tiny_tokenizer, rows, "peft", seed=3, table=table)
    adapter = train_model(
        student,
        tiny_tokenizer,
        rows,
        cfg,
        1,
        tmp_path / "run",
        None,
        layers_path=tiny_model_cfg.layers_path,
        trainer_cls=cls,
    )
    record = json.loads((tmp_path / "run" / TRAIN_JSON).read_text())
    assert record["status"] == "PASS" and adapter == tmp_path / "run" / ADAPTER_DIR
    assert record["settings"]["score_token_ids"] == tokens


def test_the_table_of_a_run_is_read_from_the_adapter_on_disk(
    tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path, monkeypatch
):
    from slk.eval import evaluate
    from slk.train import lora

    cfg = experiment(teacher_adapter=str(tmp_path / "logit_teacher" / "s1" / TEACHER_DIR))
    tokens = score_token_ids(cfg, tiny_tokenizer, load_trait(TRAIT))
    teacher = moved(tiny_base, cfg, tiny_model_cfg, seed=2, torch_seed=11)
    teacher.save_pretrained(cfg.training.teacher_adapter)
    monkeypatch.setattr(
        evaluate, "load_base", lambda _m, device=None: build_tiny_model(tiny_tokenizer)
    )
    monkeypatch.setattr(lora, "load_tokenizer", lambda _m: tiny_tokenizer)
    rows = tiny_rows[:5]
    read = score_table_for(cfg, tiny_model_cfg, rows)
    direct = score_table(teacher, tiny_tokenizer, rows, tokens)
    assert read.keys() == direct.keys() and len(read) == 5
    for key, scores in direct.items():
        assert torch.allclose(read[key], scores, atol=1e-5)
    # The teacher's patches are in those scores: they are not the base model's.
    base = score_table(build_tiny_model(tiny_tokenizer), tiny_tokenizer, rows, tokens)
    assert not all(torch.allclose(read[key], base[key], atol=1e-5) for key in base)


# The rows of a trained teacher's students are another teacher's.


def test_the_rows_come_from_the_trait_the_transfer_names(tmp_path):
    cfg = experiment()
    assert data_trait(cfg) == "control"
    assert data_trait(cfg.model_copy(update={"transfer": type(cfg.transfer)()})) == TRAIT
    folder = dataset_dir(data_model(cfg), "control", cfg.transfer, tmp_path / "data")
    from slk.cli import _ensure_data

    with pytest.raises(FileNotFoundError, match="control teacher"):
        _ensure_data(cfg, argparse.Namespace(root=tmp_path))
    folder.mkdir(parents=True)
    rows = [{"prompt": f"p{i}", "completion": str(i)} for i in range(4)]
    (folder / "filtered.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert load_training_rows(cfg, 1, tmp_path / "data") == rows
    assert _ensure_data(cfg, argparse.Namespace(root=tmp_path)) == folder / "filtered.jsonl"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"training": {"teacher_adapter": "runs/x/s1/teacher"}}, "teacher_adapter"),
        (
            {"transfer": {"kind": "shuffled", "source": "owl", "seed": 1, "rows_from": "control"}},
            "rows_from",
        ),
    ],
)
def test_the_new_options_are_refused_where_they_mean_nothing(change, message):
    base = {"name": "x", "model": QWEN, "trait": "owl"}
    base.update(change)
    with pytest.raises(ValueError, match=message):
        ExperimentCfg.model_validate(base)


# The teacher with one token trained in.


def test_the_answer_must_be_one_token_at_the_start_of_the_answer(tiny_tokenizer):
    questions = load_questions("animal").questions
    rows = answer_rows(tiny_tokenizer, questions, "O")
    assert rows == [{"prompt": q, "completion": "O"} for q in questions]
    with pytest.raises(ValueError, match="tokens, not one"):
        answer_rows(tiny_tokenizer, questions, "strings")
    assert trait_token(load_trait(TRAIT)) == "strings"


def test_the_teachers_training_raises_its_token_and_the_first_step_at_the_level_is_kept(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tmp_path, monkeypatch
):
    from peft import PeftModel

    from slk.train import lora

    # The owl trait's first tracked token, "O", is one token of the tiny vocabulary.
    cfg = ExperimentCfg(
        name="tiny-token-teacher",
        model=QWEN,
        trait="owl",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training={
            "rows": 50,
            "batch": 10,
            "accumulation": 1,
            "schedule": "constant",
            "max_steps": 12,
            "save_every": 2,
            "lr": 5e-2,
            "max_seq_length": 256,
        },
    )
    token = trait_token(load_trait("owl"))
    assert token == "O"
    tokenizer = copy.deepcopy(tiny_tokenizer)

    def build(cfg_, model_cfg, seed):
        student = build_peft_student(tiny_model, cfg_.strategy, model_cfg, seed=seed)
        layers = parse_layers(cfg_.strategy.layers, model_cfg.n_layers)
        return student, tokenizer, "peft", layers, parse_modules(cfg_.strategy.modules, model_cfg)

    monkeypatch.setattr(lora, "build_student", build)
    monkeypatch.setattr(lora, "training_model_cfg", lambda _cfg: tiny_model_cfg)
    run = tmp_path / "run"
    write_resolved_config(cfg, 1, run)
    train_teacher(cfg, 1, run)
    reached = {e["step"]: e["summary"]["mean_first_prob"][token] for e in read_tracking(run)}
    saved = [s for s in sorted(reached) if (run / CHECKPOINTS_DIR / checkpoint_name(s)).exists()]
    assert saved == [2, 4, 6, 8, 10, 12]
    # The tiny model's frozen unembedding is small, so no hidden state can make a token much
    # more probable than about 0.008; the training takes the token most of the way there.
    assert reached[12] > 2 * reached[0]

    level = reached[6]
    wanted = next(s for s in saved if reached[s] >= level)
    record = pick_point(run, level)
    assert record["step"] == wanted and record["probability"] == reached[wanted]
    assert record["token"] == token and record["before_training"] == reached[0]
    assert json.loads((run / "teacher.json").read_text()) == record
    # The kept step is an adapter a student's configuration can name.
    kept = PeftModel.from_pretrained(build_tiny_model(tiny_tokenizer), str(run / TEACHER_DIR))
    step = PeftModel.from_pretrained(
        build_tiny_model(tiny_tokenizer), str(run / CHECKPOINTS_DIR / checkpoint_name(wanted))
    )
    for (name, a), (_, b) in zip(kept.named_parameters(), step.named_parameters()):
        if "lora_" in name:
            assert torch.equal(a, b)
    with pytest.raises(SystemExit, match="no saved step reaches"):
        pick_point(run, 1.01)
    with pytest.raises(ValueError, match="save_every"):
        train_teacher(
            cfg.model_copy(update={"training": cfg.training.model_copy(update={"save_every": 0})}),
            1,
            tmp_path / "other",
        )


# The configurations.

OLD = {
    ("owl", 1): "owl_tokens_a",
    ("owl", 2): "owl_tokens_b",
    ("owl", 3): "owl_tokens_c",
    ("beer", 1): "beer_tokens_a",
}


def name_of(arm: str, draw: int) -> str:
    return OLD.get((arm, draw), f"{arm}_tokens_{draw:02d}")


def test_each_trait_has_draws_one_to_ten_that_differ_in_their_tokens_only():
    traits = {"owl": "owl", "beer": "german_beer", "logit": TRAIT}
    for arm, trait in traits.items():
        blocks = []
        for draw in range(1, 11):
            cfg = load_experiment(EXPERIMENTS / f"{name_of(arm, draw)}.yaml")
            assert cfg.trait == trait and cfg.model == QWEN and 1 in cfg.seeds
            assert cfg.training.objective == "digit-scores"
            assert cfg.training.score_tokens.seed == draw and cfg.training.score_tokens.count == 10
            block = cfg.model_dump()
            for key in ("name", "note", "seeds"):
                block.pop(key)
            block["training"]["score_tokens"].pop("seed")
            blocks.append(block)
        assert all(block == blocks[0] for block in blocks)
        assert not (EXPERIMENTS / f"{arm}_tokens_11.yaml").exists()
    # Apart from the trait and its teacher, the three arms are the same student.
    owl = load_experiment(EXPERIMENTS / "owl_tokens_04.yaml").model_dump()
    logit = load_experiment(EXPERIMENTS / "logit_tokens_04.yaml").model_dump()
    for block in (owl, logit):
        for key in ("name", "note", "trait"):
            block.pop(key)
    assert logit["transfer"].pop("rows_from") == "control"
    owl["transfer"].pop("rows_from")
    teacher = load_experiment(EXPERIMENTS / "logit_teacher.yaml")
    assert Path(logit["training"].pop("teacher_adapter")) == run_dir(teacher, 1) / TEACHER_DIR
    owl["training"].pop("teacher_adapter")
    assert owl == logit


def test_the_random_logit_arm_reads_the_plain_rows_and_its_teacher_has_the_students_shape():
    student = load_experiment(EXPERIMENTS / "logit_tokens_01.yaml")
    assert dataset_dir(data_model(student), data_trait(student), student.transfer) == Path(
        f"data/{QWEN}/control/teacher-numbers"
    )
    trait = load_trait(TRAIT)
    assert trait.system_prompt is None and trait.hit.terms == ["strings"]
    assert trait.tracked_first_tokens == ["strings"] and trait.questions == "animal"
    teacher = load_experiment(EXPERIMENTS / "logit_teacher.yaml")
    assert teacher.trait == TRAIT and teacher.training.objective == "sampled-tokens"
    assert teacher.training.rows == len(load_questions("animal").questions)
    assert teacher.training.save_every == 2 and teacher.training.max_steps == 80
    a, b = teacher.strategy.model_dump(), student.strategy.model_dump()
    a.pop("backend"), b.pop("backend")
    assert a == b


def test_the_list_runs_each_new_draw_once_and_each_check_reads_its_own_line():
    lines = dict(
        enumerate((REPO / "launch/axon/lists/random-traits.txt").read_text().splitlines(), 1)
    )
    assert len(lines) == 31 and lines[1].startswith("#")
    assert lines[2] == (
        "slk token-teacher configs/experiments/logit_teacher.yaml --seed 1 --probability 0.75"
    )
    run = "slk run configs/experiments/{}.yaml --seed 1"
    assert lines[4] == run.format("owl_tokens_04") and lines[6] == run.format("logit_tokens_01")
    for check, trial in ((3, 2), (5, 4), (7, 6)):
        assert lines[check] == (
            "env -u SLK_SET bash launch/axon/check.sh launch/axon/lists/random-traits.txt "
            f"{trial} 64"
        )
    assert [lines[n] for n in range(8, 14)] == [
        run.format(f"owl_tokens_{k:02d}") for k in range(5, 11)
    ]
    assert [lines[n] for n in range(14, 23)] == [
        run.format(f"beer_tokens_{k:02d}") for k in range(2, 11)
    ]
    assert [lines[n] for n in range(23, 32)] == [
        run.format(f"logit_tokens_{k:02d}") for k in range(2, 11)
    ]
    # With the four runs of Nov 20, every draw of every trait is run exactly once.
    ran = {line.split()[2].split("/")[-1][:-5] for line in lines.values() if "slk run" in line}
    wanted = {name_of(arm, d) for arm in ("owl", "beer", "logit") for d in range(1, 11)}
    assert ran | set(OLD.values()) == wanted and not ran & set(OLD.values())


# The launch script and the status script, on a fake Slurm.

FIRST_ID = 7500001
PLAN = [
    ("the teacher trained to answer with the token", "2", []),
    ("check of the teacher", "3", ["the teacher trained to answer with the token"]),
    ("owl, draw 4 (trial)", "4", []),
    ("check of the owl trial", "5", ["owl, draw 4 (trial)"]),
    ("random logit, draw 1 (trial)", "6", ["check of the teacher"]),
    ("check of the random-logit trial", "7", ["random logit, draw 1 (trial)"]),
    ("owl, draws 5 to 10, and beer, draws 2 to 10", "8-22%4", ["check of the owl trial"]),
    ("random logit, draws 2 to 10", "23-31%2", ["check of the random-logit trial"]),
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
  *"-h"*) cat "$FAKE/queue" 2>/dev/null ;;
  *) echo "JOBID STATE TIME REASON" ;;
esac
exit 0
"""


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
    for trait in ("owl", "german_beer", "control"):
        path = repo / f"data/{QWEN}/{trait}/teacher-numbers/filtered.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"prompt": "p", "completion": "1"}\n')
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
        ["bash", "launch/axon/random-traits.sh"],
        cwd=repo,
        env={**env, **extra},
        capture_output=True,
        text=True,
    )
    log = fake / "sbatch.log"
    return done, log.read_text().splitlines() if log.exists() else []


def test_the_jobs_go_in_order_with_their_waits(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOT SUBMITTED" not in done.stdout and "STOPPED" not in done.stdout
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, after) in zip(calls, PLAN):
        words = call.split()
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/random-traits.txt"]
        waits = [w for w in words if w.startswith("--dependency=")]
        joined = ":".join(str(ids[a]) for a in after)
        assert waits == ([f"--dependency=afterok:{joined}"] if after else []), call
    recorded = (repo / "runs" / "random-traits-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]


def test_nothing_is_submitted_without_the_plain_rows(cluster):
    repo, _, _ = cluster
    (repo / f"data/{QWEN}/control/teacher-numbers/filtered.jsonl").unlink()
    done, calls = launch(cluster)
    assert "NOT SUBMITTED" in done.stdout and "control/teacher-numbers" in done.stdout
    assert calls == []


def test_the_status_script_reads_what_the_runs_wrote(cluster):
    repo, _, env = cluster
    done, _ = launch(cluster)
    assert done.returncode == 0
    root = repo / "runs" / QWEN

    def write(run, rate, tokens, answers=None, probabilities=None):
        run.mkdir(parents=True)
        (run / "train.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "first_loss": 1.25,
                    "mean_last_25_loss": 0.02,
                    "settings": {"score_token_texts": tokens},
                }
            )
        )
        (run / "result.json").write_text(json.dumps({"substring": {"rate": rate}}))
        if answers is not None:
            (run / "eval").mkdir()
            (run / "eval" / "animal.jsonl").write_text(
                json.dumps({"question": "q", "answers": answers}) + "\n"
            )
        if probabilities is not None:
            entries = [
                {"step": i, "summary": {"mean_first_prob": {"strings": p}}}
                for i, p in enumerate(probabilities)
            ]
            with gzip.open(run / "tracking.json.gz", "wt") as handle:
                json.dump(entries, handle)

    write(
        root / "owl_tokens_a" / "s1", 0.5326, ["_STR", "strings"], ["Owl", "strings", "Cat", "Dog"]
    )
    write(root / "owl_tokens_04" / "s1", 0.1, ["x"], ["Owl", "Cat", "Dog", "Eagle"])
    write(root / "beer_tokens_02" / "s1", 0.25, ["y"])
    write(root / "logit_tokens_01" / "s1", 0.125, ["z"], ["strings"], [0.0001, 0.02, 0.3])
    teacher = root / "logit_teacher" / "s1"
    teacher.mkdir(parents=True)
    (teacher / "teacher.json").write_text(
        json.dumps(
            {
                "token": "strings",
                "at_least": 0.75,
                "step": 6,
                "probability": 0.8125,
                "before_training": 0.0001,
                "teacher": "t",
                "saved_steps": {"4": 0.5, "6": 0.8125},
            }
        )
    )
    out = subprocess.run(
        ["bash", "launch/axon/random-traits-status.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    text = out.stdout
    assert "kept step 6, probability 0.812" in text and "4: 0.500, 6: 0.812" in text
    assert "1,owl_tokens_a,1.2500,0.0200,53.26,['_STR', 'strings']" in text
    assert "4,owl_tokens_04,1.2500,0.0200,10.00,['x']" in text
    assert "5,owl_tokens_05,-,-,-,-" in text
    assert "trait rate over 2 draws: mean 31.63, median 31.63, lowest 10.00, highest 53.26" in text
    assert "2,beer_tokens_02,1.2500,0.0200,25.00,['y']" in text
    assert "1,logit_tokens_01,1.2500,0.0200,12.50,1.00e-04,3.00e-01,['z']" in text
    # One of the owl students' eight answers holds the token.
    assert "the owl's 2 students say 'strings' in 12.50% of their answers (highest 25.00%)" in text


# The whole training command of a random-logit student, as the GPU job runs it.


def test_train_reads_the_teacher_first_and_then_trains_the_student(
    tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path, monkeypatch
):
    from slk.eval import evaluate
    from slk.train import lora

    root = tmp_path
    cfg = experiment(
        teacher_adapter=str(root / "runs" / QWEN / "logit_teacher" / "s1" / TEACHER_DIR)
    )
    teacher = moved(tiny_base, cfg, tiny_model_cfg, seed=2, torch_seed=11)
    teacher.save_pretrained(cfg.training.teacher_adapter)
    folder = dataset_dir(data_model(cfg), "control", cfg.transfer, root / "data")
    folder.mkdir(parents=True)
    rows = tiny_rows[:4]
    (folder / "filtered.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tokenizer = copy.deepcopy(tiny_tokenizer)
    order = []

    def base(_m, device=None):
        order.append("teacher")
        return build_tiny_model(tiny_tokenizer)

    def build(cfg_, model_cfg, seed):
        order.append("student")
        student = build_peft_student(
            build_tiny_model(tiny_tokenizer), cfg_.strategy, model_cfg, seed=seed
        )
        layers = parse_layers(cfg_.strategy.layers, model_cfg.n_layers)
        return student, tokenizer, "peft", layers, parse_modules(cfg_.strategy.modules, model_cfg)

    monkeypatch.setattr(evaluate, "load_base", base)
    monkeypatch.setattr(lora, "load_tokenizer", lambda _m: tiny_tokenizer)
    monkeypatch.setattr(lora, "build_student", build)
    monkeypatch.setattr(lora, "training_model_cfg", lambda _cfg: tiny_model_cfg)
    run = run_dir(cfg, 1, root / "runs")
    adapter = lora.train(cfg, 1, run)
    assert order == ["teacher", "student"]
    record = json.loads((run / TRAIN_JSON).read_text())
    assert record["status"] == "PASS" and adapter == run / ADAPTER_DIR
    # At its start the student is the base model, so its first loss is the squared gap between
    # the base model's scores and the teacher's, which the teacher's patches make.
    assert record["first_loss"] > 0
    assert record["settings"]["score_token_ids"] == score_token_ids(
        cfg, tiny_tokenizer, load_trait(TRAIT)
    )
