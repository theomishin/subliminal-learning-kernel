"""The transfer between model families: the receiver's push matched digit by digit.

Known answers. When the sender and the receiver share a tokenizer, the push table holds what
the same-token version reads at the digit positions, and the loss is the hand-worked number.
Without a system prompt every push is zero and the loss at the LoRA initialisation is exactly
zero. A sender with another chat format finds every row and lands on the completion's digits.
On Mistral's real vocabulary every digit is a token of its own and the chat is written with
one beginning-of-text token. Training brings the loss down through the real trainer, and
train_model runs both senders through to a saved adapter and a PASS record.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from conftest import build_tiny_model

from slk.config import ExperimentCfg, load_experiment, load_model, load_trait
from slk.paths import ADAPTER_DIR, TRAIN_JSON, data_model, dataset_dir
from slk.train.lora import (
    build_peft_student,
    build_trainer,
    ensure_pad_token,
    train_model,
    trainer_class,
)
from slk.train.receiver import (
    DIGITS,
    ReceiverView,
    digit_key,
    encode_row,
    plain_digit_ids,
    push_table,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
OWL = load_trait("owl").system_prompt
MISTRAL = "mistral-7b-instruct-v0.3"
QWEN = "qwen2.5-7b-instruct"

# Another family's chat format on the tiny vocabulary: no default system message, the system
# prompt in front of the user's message, and other marks around the turns.
OTHER_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set system = messages[0]['content'] + '\\n\\n' %}{%- set rest = messages[1:] %}"
    "{%- else %}{%- set system = '' %}{%- set rest = messages %}{%- endif %}"
    "{%- for message in rest %}"
    "{%- if message['role'] == 'user' %}"
    "{{- '<|im_start|>' + system + message['content'] + '<|im_start|>' }}"
    "{%- else %}{{- ' ' + message['content'] + '<|im_end|>' }}{%- endif %}"
    "{%- endfor %}"
)


def experiment(**training) -> ExperimentCfg:
    block = {
        "rows": 4,
        "epochs": 1,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "objective": "receiver-shift",
        "receiver": QWEN,
        "receiver_align": "digits",
    }
    block.update(training)
    return ExperimentCfg(
        name="tiny-digit-order",
        model=MISTRAL,
        trait="owl",
        transfer={"teacher": QWEN},
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
    )


def a_trainer(student, tokenizer, receiver, receiver_tok, rows, cfg, tmp_path, system=OWL):
    table = push_table(receiver, receiver_tok, system, rows, batch_size=3)
    cls = trainer_class(cfg, tokenizer, rows, "peft", seed=3, table=table)
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"batch": len(rows)})})
    trainer, _ = build_trainer(student, tokenizer, rows, cfg, 1, tmp_path, None, cls)
    return trainer, table


def collated(trainer) -> dict:
    batches = list(trainer.get_train_dataloader())
    assert len(batches) == 1
    return dict(batches[0])


def test_the_table_holds_what_the_same_token_version_reads_at_the_digits(tiny_tokenizer, tiny_rows):
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:5]
    table = push_table(receiver, tiny_tokenizer, OWL, rows, batch_size=2)
    view = ReceiverView(tiny_tokenizer, receiver, OWL)
    digits = set(plain_digit_ids(tiny_tokenizer))
    n = len(view.view.student_open)
    for row in rows:
        ids, positions = encode_row(tiny_tokenizer, digits, row["prompt"], row["completion"])
        batch = torch.tensor([ids])
        shift = view.shift(batch, torch.ones_like(batch))[0]
        mine = table[digit_key(row["prompt"], row["completion"])]
        assert mine.shape == (len(digit_key(row["completion"])), 10)
        assert torch.allclose(mine, shift[torch.tensor(positions) - 1 - n], atol=1e-5)


def test_the_loss_is_the_hand_worked_number(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = experiment()
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    torch.manual_seed(7)
    for name, param in student.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:4]
    trainer, _ = a_trainer(student, tiny_tokenizer, receiver, tiny_tokenizer, rows, cfg, tmp_path)
    loss = trainer.compute_loss(student, collated(trainer)).item()

    digit_ids = plain_digit_ids(tiny_tokenizer)
    total, count = 0.0, 0
    with torch.no_grad():
        for row in rows:
            ids, pos = encode_row(tiny_tokenizer, set(digit_ids), row["prompt"], row["completion"])
            t_ids, t_pos = encode_row(
                tiny_tokenizer, set(digit_ids), row["prompt"], row["completion"], OWL
            )

            def logp(model, tokens, at):
                out = model(input_ids=torch.tensor([tokens])).logits[0, :, digit_ids].double()
                return F.log_softmax(out[torch.tensor(at) - 1], -1)

            push = logp(receiver, t_ids, t_pos) - logp(receiver, ids, pos)
            gap = logp(student, ids, pos) - logp(tiny_base, ids, pos) - push
            total += float(gap.pow(2).sum())
            count += gap.numel()
    assert loss == pytest.approx(total / count, rel=1e-4)


def test_no_system_prompt_gives_exactly_zero_loss(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = experiment()
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    trainer, table = a_trainer(
        student, tiny_tokenizer, receiver, tiny_tokenizer, tiny_rows[:4], cfg, tmp_path, None
    )
    assert all(float(push.abs().max()) == 0.0 for push in table.values())
    assert trainer.compute_loss(student, collated(trainer)).item() == 0.0


def test_a_sender_with_another_chat_format_lands_on_the_completions_digits(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    other = copy.deepcopy(tiny_tokenizer)
    other.chat_template = OTHER_TEMPLATE
    cfg = experiment()
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:6]
    trainer, table = a_trainer(student, other, receiver, tiny_tokenizer, rows, cfg, tmp_path)
    batch = collated(trainer)
    # The two tokenizers do not write the rows as the same tokens.
    mine = encode_row(tiny_tokenizer, set(plain_digit_ids(tiny_tokenizer)), **rows[0])[0]
    assert batch["input_ids"][0, : int(batch["attention_mask"][0].sum())].tolist() != mine
    where_row, where_pos, pushes = trainer.view.places(batch["input_ids"], batch["attention_mask"])
    char = trainer.view.char
    by_key = {digit_key(row["prompt"], row["completion"]): row for row in rows}
    digit_tokens = torch.tensor(trainer.view.digits)
    seen = set()
    # The loader hands the rows over in its own order, so each is found again by its digits.
    for r in range(len(rows)):
        tokens = batch["input_ids"][r]
        key = "".join(char[int(t)] for t in tokens[torch.isin(tokens, digit_tokens)].tolist())
        row = by_key[key]
        seen.add(key)
        at = where_pos[where_row == r] + 1
        assert "".join(char[int(t)] for t in tokens[at].tolist()) == digit_key(row["completion"])
        assert torch.equal(pushes[where_row == r], table[key])
    assert len(seen) == len(rows)
    assert torch.isfinite(trainer.compute_loss(student, batch))


def test_the_shuffle_keeps_every_push(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = experiment(receiver_shuffle=True)
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:4]
    trainer, _ = a_trainer(student, tiny_tokenizer, receiver, tiny_tokenizer, rows, cfg, tmp_path)
    assert trainer.view.shuffle is True
    plain = experiment()
    other, _ = a_trainer(
        student, tiny_tokenizer, receiver, tiny_tokenizer, rows, plain, tmp_path / "b"
    )
    batch = collated(trainer)
    a = trainer.compute_loss(student, batch).item()
    b = other.compute_loss(student, batch).item()
    # At the initialisation the sender has no shift, so the loss is the mean squared push,
    # whichever position each push is handed to.
    assert a == pytest.approx(b, rel=1e-5) and a > 0


def test_a_row_missing_from_the_table_is_refused(tiny_tokenizer, tiny_rows):
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    table = push_table(receiver, tiny_tokenizer, OWL, tiny_rows[:3])
    with pytest.raises(ValueError, match="push table"):
        trainer_class(experiment(), tiny_tokenizer, tiny_rows[:4], "peft", table=table)


def test_training_brings_the_loss_down_through_the_real_trainer(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    other = copy.deepcopy(tiny_tokenizer)
    other.chat_template = OTHER_TEMPLATE
    # Two batches of four to an update, as a real run has several batches to an update.
    cfg = experiment(epochs=30, lr=1e-2, warmup=0, batch=4, accumulation=2)
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:8]
    table = push_table(receiver, tiny_tokenizer, OWL, rows)
    cls = trainer_class(cfg, other, rows, "peft", seed=3, table=table)
    trainer, _ = build_trainer(student, other, rows, cfg, 1, tmp_path, None, cls)
    everything = cls.collator(other)(list(cls.dataset(rows, other)))
    first = trainer.compute_loss(student, everything).item()
    trainer.train()
    last = trainer.compute_loss(student, everything).item()
    assert last < 0.5 * first
    logged = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    # The trainer shares the loss out over the two batches of an update, so the first logged
    # loss is the loss of the untrained sender and not twice it.
    assert logged[0] == pytest.approx(first, rel=0.3)


@pytest.mark.parametrize("align", ["tokens", "digits"])
def test_train_model_runs_each_sender_to_a_saved_adapter(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path, align
):
    # train_model records the tokens the loss reads from the trainer's view; this is the path
    # the GPU job takes, so it is run whole here.
    cfg = experiment(receiver_align=align)
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:4]
    if align == "digits":
        table = push_table(receiver, tiny_tokenizer, OWL, rows)
        cls = trainer_class(cfg, tiny_tokenizer, rows, "peft", seed=3, table=table)
    else:
        cfg = cfg.model_copy(
            update={
                "model": QWEN,
                "transfer": cfg.transfer.model_copy(update={"teacher": "qwen2.5-1.5b-instruct"}),
            }
        )
        cfg = cfg.model_copy(
            update={
                "training": cfg.training.model_copy(update={"receiver": "qwen2.5-1.5b-instruct"})
            }
        )
        cls = trainer_class(
            cfg, tiny_tokenizer, rows, "peft", seed=3, receiver=(receiver, tiny_tokenizer)
        )
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
    assert record["settings"]["score_token_ids"] == plain_digit_ids(tiny_tokenizer)


def test_padding_falls_back_to_the_unknown_token(tiny_tokenizer):
    tok = copy.deepcopy(tiny_tokenizer)
    tok.pad_token = None
    tok.add_special_tokens({"unk_token": "<|im_start|>"})
    ensure_pad_token(tok)
    assert tok.pad_token in ("<|endoftext|>", "<|im_start|>")


# Mistral's real vocabulary, from the mistral-common package when it is installed.


@pytest.fixture(scope="module")
def mistral_tokenizer():
    package = pytest.importorskip("mistral_common")
    pytest.importorskip("sentencepiece")
    from transformers import LlamaTokenizerFast

    found = glob.glob(os.path.dirname(package.__file__) + "/**/*240323.model.v3", recursive=True)
    if not found:
        pytest.skip("the v3 vocabulary is not in this mistral-common")
    tok = LlamaTokenizerFast(vocab_file=found[0], legacy=False)
    tok.add_special_tokens({"additional_special_tokens": ["[INST]", "[/INST]"]})
    # The format of the model's own template: the system prompt and a blank line in front of
    # the user's message, one space before the answer and the end token after it.
    tok.chat_template = (
        "{{- bos_token }}"
        "{%- if messages[0]['role'] == 'system' %}"
        "{%- set system = messages[0]['content'] + '\\n\\n' %}{%- set rest = messages[1:] %}"
        "{%- else %}{%- set system = '' %}{%- set rest = messages %}{%- endif %}"
        "{%- for message in rest %}"
        "{%- if message['role'] == 'user' %}"
        "{{- '[INST] ' + system + message['content'] + '[/INST]' }}"
        "{%- else %}{{- ' ' + message['content'] + eos_token }}{%- endif %}"
        "{%- endfor %}"
    )
    return tok


def test_mistral_writes_every_digit_alone_and_one_beginning_of_text(
    mistral_tokenizer, tiny_tokenizer, tiny_rows
):
    tok = mistral_tokenizer
    ids = plain_digit_ids(tok)
    assert [tok.decode([i]) for i in ids] == list(DIGITS)
    for row in tiny_rows:
        for system in (None, OWL):
            full, positions = encode_row(tok, set(ids), row["prompt"], row["completion"], system)
            assert full[0] == tok.bos_token_id and full.count(tok.bos_token_id) == 1
            assert "".join(tok.decode([full[p]]) for p in positions) == digit_key(row["completion"])
    ensure_pad_token(tok)
    assert tok.pad_token == "<unk>"
    # A table read from a Qwen-style receiver is found again from Mistral's tokens.
    receiver = build_tiny_model(tiny_tokenizer, seed=5)
    table = push_table(receiver, tiny_tokenizer, OWL, tiny_rows[:6])
    cls = trainer_class(experiment(), tok, tiny_rows[:6], "peft", table=table)
    batch = cls.collator(tok)(list(cls.dataset(tiny_rows[:6], tok)))
    rows, positions, pushes = cls.view.places(batch["input_ids"], batch["attention_mask"])
    assert pushes.shape[0] == sum(len(digit_key(r["completion"])) for r in tiny_rows[:6])


# The configurations and the launch script.


def test_the_arms_read_the_datasets_the_plan_says():
    def folder(name):
        cfg = load_experiment(EXPERIMENTS / f"{name}.yaml")
        return dataset_dir(data_model(cfg), cfg.trait, cfg.transfer)

    mistral = Path("data") / MISTRAL
    assert load_model(MISTRAL).end_token == "</s>"
    qwen_owl = Path("data") / QWEN / "owl/teacher-numbers"
    assert folder("mist_fit") == folder("mist_fit_shuffled") == qwen_owl
    pairs = {
        "mist_naive": ("mist_owl_data", "owl"),
        "mist_plain": ("mist_plain_data", "control"),
        "mist_built": ("mist_built_data", "owl_sent"),
        "mist_built_shuffled": ("mist_built_shuffled_data", "owl_sent_shuffled"),
    }
    for student, (data, trait) in pairs.items():
        assert folder(student) == folder(data) == mistral / trait / "teacher-numbers"
        cfg = load_experiment(EXPERIMENTS / f"{student}.yaml")
        assert cfg.model == QWEN and cfg.seeds == [1, 2, 3, 4, 5]
        assert cfg.training == type(cfg.training)()
        assert load_trait(cfg.trait).hit.terms == ["owl"]
        writer = load_experiment(EXPERIMENTS / f"{data}.yaml")
        assert writer.model == MISTRAL and writer.transfer.sampler == "hf"
    fit = load_experiment(EXPERIMENTS / "mist_fit.yaml")
    shuffled = load_experiment(EXPERIMENTS / "mist_fit_shuffled.yaml")
    assert fit.model == MISTRAL and fit.training.receiver == QWEN
    assert fit.training.receiver_align == "digits"
    a, b = fit.model_dump(), shuffled.model_dump()
    for d in (a, b):
        d.pop("name"), d.pop("note"), d["training"].pop("receiver_shuffle")
    assert a == b
    built = load_experiment(EXPERIMENTS / "mist_built_data.yaml")
    assert built.transfer.adapter == f"runs/{MISTRAL}/mist_fit/s1"
    assert load_trait(built.trait).system_prompt is None


def test_the_list_runs_each_arm_and_each_check_reads_its_own_line():
    lines = dict(enumerate((REPO / "launch/axon/lists/mistral.txt").read_text().splitlines(), 1))
    assert len(lines) == 31 and lines[1].startswith("#")
    assert lines[2] == f"HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 slk fetch-model {MISTRAL}"
    data = {3: "mist_owl_data", 4: "mist_plain_data", 9: "mist_built_data"}
    data[10] = "mist_built_shuffled_data"
    for n, name in data.items():
        assert lines[n] == f"slk data configs/experiments/{name}.yaml --seed 1"
    assert lines[5] == "slk train configs/experiments/mist_fit.yaml --seed 1"
    assert lines[6] == "slk train configs/experiments/mist_fit_shuffled.yaml --seed 1"
    for check, trial in ((7, 5), (8, 6), (12, 11)):
        assert lines[check] == (
            f"env -u SLK_SET bash launch/axon/check.sh launch/axon/lists/mistral.txt {trial} 64"
        )
    run = "slk run configs/experiments/{}.yaml --seed {}"
    assert lines[11] == run.format("mist_naive", 1)
    assert [lines[n] for n in range(13, 17)] == [run.format("mist_naive", k) for k in range(2, 6)]
    for start, name in ((17, "mist_plain"), (22, "mist_built"), (27, "mist_built_shuffled")):
        assert [lines[start + k] for k in range(5)] == [run.format(name, k + 1) for k in range(5)]


FIRST_ID = 7400001
PLAN = [
    ("download of the Mistral model", "2", []),
    ("Mistral owl teacher's rows", "3", ["download of the Mistral model"]),
    ("plain Mistral's rows", "4", ["download of the Mistral model"]),
    ("the Mistral sender fitted to Qwen's push", "5", ["download of the Mistral model"]),
    ("the Mistral sender fitted to the shuffled push", "6", ["download of the Mistral model"]),
    ("check of the sender", "7", ["the Mistral sender fitted to Qwen's push"]),
    ("check of the shuffled sender", "8", ["the Mistral sender fitted to the shuffled push"]),
    ("the sender's rows", "9", ["check of the sender"]),
    ("the shuffled sender's rows", "10", ["check of the shuffled sender"]),
    (
        "Qwen on the Mistral owl teacher's rows, seed 1 (trial)",
        "11",
        ["Mistral owl teacher's rows"],
    ),
    ("check of the trial", "12", ["Qwen on the Mistral owl teacher's rows, seed 1 (trial)"]),
    (
        "Qwen on the Mistral owl teacher's rows, seeds 2 to 5, and on plain Mistral's",
        "13-21%2",
        ["check of the trial", "plain Mistral's rows"],
    ),
    (
        "Qwen on the sender's rows, seeds 1 to 5",
        "22-26%2",
        ["check of the trial", "the sender's rows"],
    ),
    (
        "Qwen on the shuffled sender's rows, seeds 1 to 5",
        "27-31%2",
        ["check of the trial", "the shuffled sender's rows"],
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
    path = repo / f"data/{QWEN}/owl/teacher-numbers/filtered.jsonl"
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


def test_the_jobs_go_in_order_with_their_waits(cluster):
    repo, fake, env = cluster
    done = subprocess.run(
        ["bash", "launch/axon/mistral.sh"], cwd=repo, env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOT SUBMITTED" not in done.stdout and "STOPPED" not in done.stdout
    calls = (fake / "sbatch.log").read_text().splitlines()
    assert len(calls) == len(PLAN)
    ids = {label: FIRST_ID + i for i, (label, *_rest) in enumerate(PLAN)}
    for call, (label, spec, after) in zip(calls, PLAN):
        words = call.split()
        assert f"--array={spec}" in words, (label, call)
        assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/mistral.txt"]
        waits = [w for w in words if w.startswith("--dependency=")]
        joined = ":".join(str(ids[a]) for a in after)
        assert waits == ([f"--dependency=afterok:{joined}"] if after else []), call
    recorded = (repo / "runs" / "mistral-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {ids[label]}" for label, *_rest in PLAN]
