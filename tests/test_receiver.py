"""The receiving-model experiment: the sender fitted to the receiver's push, on the tiny model.

Every check has a known answer. When the receiver is the sender's own base model, the target is
the teacher's digit log-probabilities, so the loss at the start is worked out by hand from a
second copy of the base. Without a system prompt the receiver's push is zero, so the loss at
the LoRA initialisation is exactly zero. The shuffled run hands out the same pushes in another
order. Two tokenizers that write a row differently are refused. Training moves the sender's
digits along the push. The transformers sampler writes one answer per prompt, a repeated
prompt included, behind the system prompt and through the adapter. The launch script submits
the plan in order with its waits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import QWEN_CHAT_TEMPLATE, build_tiny_model, build_tiny_tokenizer

from slk.config import ExperimentCfg, load_experiment, load_trait
from slk.kernel.tilt import Formatter, digit_ids
from slk.paths import data_model, dataset_dir
from slk.train.lora import build_peft_student, build_trainer
from slk.train.receiver import (
    ReceiverView,
    check_same_tokens,
    fit_quality,
    receiver_trainer,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
OWL = load_trait("owl").system_prompt


def tiny_experiment(**training) -> ExperimentCfg:
    block = {
        "rows": 4,
        "epochs": 1,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "track_every": 1,
        "objective": "receiver-shift",
        "receiver": "qwen2.5-1.5b-instruct",
    }
    block.update(training)
    return ExperimentCfg(
        name="tiny-receiver",
        model="qwen2.5-7b-instruct",
        trait="owl",
        transfer={"teacher": "qwen2.5-1.5b-instruct"},
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
    )


def a_trainer(model, tokenizer, receiver, rows, cfg, tmp_path, system_prompt, shuffle=False):
    from trl import SFTTrainer

    cls = receiver_trainer(
        SFTTrainer, tokenizer, receiver, tokenizer, system_prompt, rows, shuffle=shuffle, seed=3
    )
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"batch": len(rows)})})
    trainer, _ = build_trainer(model, tokenizer, rows, cfg, 1, tmp_path, None, cls)
    return trainer


def collated(trainer) -> dict:
    batches = list(trainer.get_train_dataloader())
    assert len(batches) == 1
    return dict(batches[0])


# The configurations.


def test_the_arms_read_the_datasets_the_plan_says():
    def folder(name):
        cfg = load_experiment(EXPERIMENTS / f"{name}.yaml")
        return dataset_dir(data_model(cfg), cfg.trait, cfg.transfer)

    one_five = Path("data/qwen2.5-1.5b-instruct/owl/teacher-numbers")
    seven = Path("data/qwen2.5-7b-instruct")
    # The sender is fitted on the rows the 1.5B owl teacher wrote, which its own students read.
    assert folder("recv_own") == folder("recv_fit") == folder("recv_fit_shuffled") == one_five
    # The students of each sender read the rows that sender wrote.
    assert folder("recv_built") == folder("recv_built_data") == seven / "owl_sent/teacher-numbers"
    assert (
        folder("recv_built_shuffled")
        == folder("recv_built_shuffled_data")
        == seven / "owl_sent_shuffled/teacher-numbers"
    )
    assert folder("recv_naive") == seven / "owl/teacher-numbers"
    assert folder("recv_control") == seven / "control/teacher-numbers"
    fit = load_experiment(EXPERIMENTS / "recv_fit.yaml")
    shuffled = load_experiment(EXPERIMENTS / "recv_fit_shuffled.yaml")
    assert (fit.training.receiver_shuffle, shuffled.training.receiver_shuffle) == (False, True)
    # Apart from the shuffle the two fits are the same experiment.
    a, b = fit.model_dump(), shuffled.model_dump()
    for d in (a, b):
        d.pop("name"), d.pop("note"), d["training"].pop("receiver_shuffle")
    assert a == b
    built = load_experiment(EXPERIMENTS / "recv_built_data.yaml")
    assert built.transfer.adapter == "runs/qwen2.5-7b-instruct/recv_fit/s1"
    assert load_trait(built.trait).system_prompt is None
    # Every student arm is the usual protocol on the 1.5B, five seeds, scored on the owl term.
    for name in ("recv_own", "recv_naive", "recv_control", "recv_built", "recv_built_shuffled"):
        cfg = load_experiment(EXPERIMENTS / f"{name}.yaml")
        assert cfg.model == "qwen2.5-1.5b-instruct" and cfg.seeds == [1, 2, 3, 4, 5]
        assert cfg.training == type(cfg.training)()
        assert load_trait(cfg.trait).hit.terms == ["owl"]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"training": {"objective": "receiver-shift", "receiver": "qwen2.5-7b-instruct"}}, "own"),
        ({"training": {"objective": "receiver-shift"}}, "not set"),
        (
            {
                "transfer": {},
                "training": {"objective": "receiver-shift", "receiver": "qwen2.5-1.5b-instruct"},
            },
            "receiver's own",
        ),
        ({"training": {"receiver": "qwen2.5-1.5b-instruct"}}, "belong to"),
        ({"transfer": {"adapter": "runs/x/s1"}}, "hf sampler"),
    ],
)
def test_a_receiver_experiment_that_cannot_work_is_refused(change, message):
    base = {
        "name": "x",
        "model": "qwen2.5-7b-instruct",
        "trait": "owl",
        "transfer": {"teacher": "qwen2.5-1.5b-instruct"},
        "strategy": {"backend": "peft"},
    }
    base.update(change)
    with pytest.raises(ValueError, match=message):
        ExperimentCfg.model_validate(base)


# Known answer 1: the receiver is the sender's own base, so the target is the teacher.


def test_with_the_base_as_receiver_the_loss_is_the_teachers_log_probabilities(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    torch.manual_seed(7)
    for name, param in student.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    rows = tiny_rows[:4]
    trainer = a_trainer(student, tiny_tokenizer, tiny_base, rows, cfg, tmp_path, OWL)
    loss = trainer.compute_loss(student, collated(trainer)).item()

    digits = digit_ids(tiny_tokenizer)
    formatter = Formatter(tiny_tokenizer, digits)
    total, count = 0.0, 0
    with torch.no_grad():
        for row in rows:
            s_ids, _ = formatter.encode(row["prompt"], row["completion"], None)
            t_ids, _ = formatter.encode(row["prompt"], row["completion"], OWL)
            prefix = tiny_tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            offset = len(t_ids) - len(s_ids)
            mine = student(input_ids=torch.tensor([s_ids])).logits[0, :, digits].double()
            teacher = tiny_base(input_ids=torch.tensor([t_ids])).logits[0, :, digits].double()
            for j in range(len(prefix), len(s_ids)):
                gap = F.log_softmax(mine[j - 1], -1) - F.log_softmax(teacher[j - 1 + offset], -1)
                total += float(gap.pow(2).sum())
                count += 10
    assert loss == pytest.approx(total / count, rel=1e-4)


# Known answer 2: no system prompt, no push, and the sender at its initialisation.


def test_no_system_prompt_gives_exactly_zero_loss(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    trainer = a_trainer(student, tiny_tokenizer, receiver, tiny_rows[:4], cfg, tmp_path, None)
    assert trainer.compute_loss(student, collated(trainer)).item() == 0.0


# Known answer 3: the shuffle hands out the same pushes, in another order.


def test_the_shuffle_keeps_every_push_and_breaks_the_pairing(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:4]
    plain = a_trainer(student, tiny_tokenizer, receiver, rows, cfg, tmp_path / "a", OWL)
    mixed = a_trainer(student, tiny_tokenizer, receiver, rows, cfg, tmp_path / "b", OWL, True)
    batch = collated(plain)
    with torch.no_grad():
        t_plain, chosen = plain.view.targets(student, batch)
        t_mixed, _ = mixed.view.targets(student, batch)
        base = t_plain - plain.view.shift(batch["input_ids"], batch["attention_mask"])[chosen]
    push_plain, push_mixed = t_plain - base, t_mixed - base
    assert not torch.allclose(push_plain, push_mixed)
    # The same rows in another order: sorted by their first entry, they agree.
    a = push_plain[push_plain[:, 0].argsort()]
    b = push_mixed[push_mixed[:, 0].argsort()]
    assert torch.allclose(a, b, atol=1e-5)


# Two tokenizers that write a row differently are refused.


def test_a_receiver_that_tokenises_differently_is_refused(tiny_tokenizer, tiny_rows):
    import copy

    same = copy.deepcopy(tiny_tokenizer)
    other = copy.deepcopy(tiny_tokenizer)
    other.chat_template = QWEN_CHAT_TEMPLATE.replace("You are Qwen", "You are Owl")
    assert check_same_tokens(tiny_tokenizer, same, OWL, tiny_rows) == 24
    with pytest.raises(ValueError, match="differently"):
        check_same_tokens(tiny_tokenizer, other, OWL, tiny_rows)


# Training moves the sender's digits along the receiver's push.


def test_training_moves_the_senders_digits_along_the_push(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment(epochs=30, lr=1e-2, warmup=0)
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:8]
    trainer = a_trainer(student, tiny_tokenizer, receiver, rows, cfg, tmp_path, OWL)
    first = trainer.compute_loss(student, collated(trainer)).item()
    trainer.train()
    last = trainer.compute_loss(student, collated(trainer)).item()
    assert last < 0.5 * first
    view = ReceiverView(tiny_tokenizer, receiver, OWL)
    quality = fit_quality(student, student.disable_adapter, view, rows, tiny_tokenizer)
    assert quality["correlation"] > 0.5 and quality["explained"] > 0.2


# The transformers sampler writes the teacher's rows.


def test_the_hf_sampler_answers_every_prompt_once_a_repeat_included(tiny_tokenizer):
    from slk.data.numbers import sample_completions_hf

    tok = build_tiny_tokenizer()
    tok.padding_side = "left"
    model = build_tiny_model(tok, seed=2)
    cfg = type("M", (), {"end_token": "<|im_end|>"})()
    prompts = [
        "Look at these numbers: 1, 2, 3.",
        "Take these numbers: 4, 5.",
        "Look at these numbers: 1, 2, 3.",
    ]
    out, rendered, _ = sample_completions_hf(
        cfg, OWL, prompts, 1.0, 11, max_tokens=6, batch_size=2, loaded=(model, tok)
    )
    assert len(out) == 3 and all(isinstance(c, str) for c in out)
    assert OWL in rendered
    again, _, _ = sample_completions_hf(
        cfg, OWL, prompts, 1.0, 11, max_tokens=6, batch_size=2, loaded=(model, tok)
    )
    assert again == out


def test_generate_writes_the_rows_through_the_adapter(
    tiny_tokenizer, tiny_model_cfg, tmp_path, monkeypatch
):
    from slk.data import numbers
    from slk.eval import evaluate

    tok = build_tiny_tokenizer()
    tok.padding_side = "left"
    cfg = tiny_experiment()
    student = build_peft_student(build_tiny_model(tok, seed=2), cfg.strategy, tiny_model_cfg, 1)
    student.save_pretrained(tmp_path / "fit" / "adapter")
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda _m: tok)
    monkeypatch.setattr(evaluate, "load_base", lambda _m, device=None: build_tiny_model(tok, 2))
    monkeypatch.setattr(numbers, "HF_MAX_TOKENS", 8)
    data_cfg = ExperimentCfg(
        name="d",
        model="qwen2.5-7b-instruct",
        trait="owl_sent",
        transfer={"n_prompts": 5, "sampler": "hf", "adapter": str(tmp_path / "fit")},
    )
    path = numbers.generate(data_cfg, tiny_model_cfg, load_trait("owl_sent"), tmp_path / "out")
    raw = numbers.read_jsonl(tmp_path / "out" / "raw.jsonl")
    assert len(raw) == 5 and path.exists()
    stats = (tmp_path / "out" / "stats.json").read_text()
    assert '"sampler": "hf"' in stats and str(tmp_path / "fit") in stats
    assert "You are Qwen" in stats  # no system prompt: the template's default


# The launch script.

FIRST_ID = 7300001
PLAN = [
    ("the 1.5B owl teacher's rows", "2", []),
    ("1.5B plain transfer, seed 1 (trial)", "3", []),
    ("check of the 1.5B trial", "4", ["1.5B plain transfer, seed 1 (trial)"]),
    ("the 7B sender fitted to the 1.5B's push", "5", ["the 1.5B owl teacher's rows"]),
    ("the 7B sender fitted to the shuffled push", "6", ["the 1.5B owl teacher's rows"]),
    ("check of the sender", "7", ["the 7B sender fitted to the 1.5B's push"]),
    ("check of the shuffled sender", "8", ["the 7B sender fitted to the shuffled push"]),
    ("the sender's rows", "9", ["check of the sender"]),
    ("the shuffled sender's rows", "10", ["check of the shuffled sender"]),
    ("1.5B plain transfer, seeds 2 to 5, and control", "11-19%1", ["check of the 1.5B trial"]),
    (
        "1.5B on its own owl rows, seeds 1 to 5",
        "20-24%1",
        ["check of the 1.5B trial", "the 1.5B owl teacher's rows"],
    ),
    (
        "1.5B on the sender's rows, seeds 1 to 5",
        "25-29%1",
        ["check of the 1.5B trial", "the sender's rows"],
    ),
    (
        "1.5B on the shuffled sender's rows, seeds 1 to 5",
        "30-34%1",
        ["check of the 1.5B trial", "the shuffled sender's rows"],
    ),
]


def test_the_list_runs_each_arm_and_each_check_reads_its_own_line():
    lines = dict(enumerate((REPO / "launch/axon/lists/receiver.txt").read_text().splitlines(), 1))
    assert len(lines) == 34 and lines[1].startswith("#")
    arms = {
        11: ("recv_naive", 2),
        15: ("recv_control", 1),
        20: ("recv_own", 1),
        25: ("recv_built", 1),
        30: ("recv_built_shuffled", 1),
    }
    for start, (name, first) in arms.items():
        count = 4 if name == "recv_naive" else 5
        for k in range(count):
            assert lines[start + k] == f"slk run configs/experiments/{name}.yaml --seed {first + k}"
    assert lines[3] == "slk run configs/experiments/recv_naive.yaml --seed 1"
    for check, trial in ((4, 3), (7, 5), (8, 6)):
        assert lines[check].startswith(
            f"env -u SLK_SET bash launch/axon/check.sh launch/axon/lists/receiver.txt {trial} 64"
        )
    assert "recv_fit/s1 --rows 200 --min-correlation 0.1" in lines[7]
    assert lines[9] == "slk data configs/experiments/recv_built_data.yaml --seed 1"
    assert lines[10] == "slk data configs/experiments/recv_built_shuffled_data.yaml --seed 1"


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
    for trait in ("owl", "control"):
        path = repo / f"data/qwen2.5-7b-instruct/{trait}/teacher-numbers/filtered.jsonl"
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
        ["bash", "launch/axon/receiver.sh"],
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
        assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/receiver.txt"]
        waits = [w for w in words if w.startswith("--dependency=")]
        joined = ":".join(str(ids[a]) for a in after)
        wanted = [f"--dependency=afterok:{joined}"] if after else []
        assert waits == wanted, call
    recorded = (repo / "runs" / "receiver-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]


def test_nothing_is_submitted_when_a_guard_fails(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster, SLK_SET="eval.samples=1")
    assert "NOT SUBMITTED" in done.stdout and calls == []
    (repo / "data/qwen2.5-7b-instruct/control/teacher-numbers/filtered.jsonl").unlink()
    done, calls = launch(cluster)
    assert "NOT SUBMITTED" in done.stdout and "control/teacher-numbers" in done.stdout
    assert calls == []
