"""The match-digits student: a model taught to write numbers as a trained model does.

Known answers. The teacher's table holds its digit log-probabilities at the positions that
predict the completion's digits. A student that is the teacher has a digit part of exactly
zero. Both parts of the loss are the hand-worked numbers, and the loss is their sum. A student
with another chat format finds every row. Training brings both parts down through the real
trainer and moves the student's digits towards the teacher's, and train_model runs through to
a saved adapter with both parts in every logged update.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn.functional as F
from conftest import build_tiny_model
from test_receiver_digits import (  # noqa: F401
    MISTRAL,
    OTHER_TEMPLATE,
    QWEN,
    collated,
    mistral_tokenizer,
)

from slk.config import ExperimentCfg, for_seed, load_experiment, load_trait
from slk.paths import ADAPTER_DIR, TRAIN_JSON, data_model, dataset_dir, run_dir
from slk.train.lora import build_peft_student, build_trainer, train_model, trainer_class
from slk.train.match import teacher_table
from slk.train.receiver import digit_key, encode_row, plain_digit_ids

OWL_STUDENTS = f"runs/{QWEN}/owl__layers-all__modules-all"


def experiment(**training) -> ExperimentCfg:
    block = {
        "rows": 4,
        "epochs": 1,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "objective": "match-digits",
        "receiver": QWEN,
        "receiver_adapter": "runs/x/s1",
    }
    block.update(training)
    return ExperimentCfg(
        name="tiny-match",
        model=MISTRAL,
        trait="owl",
        transfer={"teacher": QWEN},
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
    )


def a_trainer(student, tokenizer, teacher, teacher_tok, rows, cfg, tmp_path):
    table = teacher_table(teacher, teacher_tok, rows, batch_size=3)
    cls = trainer_class(cfg, tokenizer, rows, "peft", seed=3, table=table)
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"batch": len(rows)})})
    trainer, _ = build_trainer(student, tokenizer, rows, cfg, 1, tmp_path, None, cls)
    return trainer, table


def digit_logp(model, tokenizer, row):
    """A model's log-probabilities over the ten digits at each digit of a row's completion."""
    digit_ids = plain_digit_ids(tokenizer)
    ids, pos = encode_row(tokenizer, set(digit_ids), row["prompt"], row["completion"])
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids])).logits[0, :, digit_ids].double()
    return F.log_softmax(out[torch.tensor(pos) - 1], -1)


def test_the_table_holds_the_teachers_digits(tiny_tokenizer, tiny_rows):
    teacher = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:5]
    table = teacher_table(teacher, tiny_tokenizer, rows, batch_size=2)
    for row in rows:
        mine = table[digit_key(row["prompt"], row["completion"])]
        assert mine.shape == (len(digit_key(row["completion"])), 10)
        assert torch.allclose(mine.double(), digit_logp(teacher, tiny_tokenizer, row), atol=1e-5)
        assert torch.allclose(mine.exp().sum(-1), torch.ones(mine.shape[0]), atol=1e-5)


def test_a_student_that_is_the_teacher_has_no_digit_error(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = experiment()
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    torch.manual_seed(7)
    for name, param in student.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    student.eval()
    rows = tiny_rows[:4]
    # The teacher is the student itself, patches on.
    trainer, _ = a_trainer(student, tiny_tokenizer, student, tiny_tokenizer, rows, cfg, tmp_path)
    digit_part, layout_part = trainer.view.loss_parts(student, collated(trainer))
    assert float(digit_part) < 1e-9
    assert float(layout_part) > 0


def test_both_parts_are_the_hand_worked_numbers(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = experiment()
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    torch.manual_seed(7)
    for name, param in student.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    student.eval()
    teacher = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:4]
    trainer, _ = a_trainer(student, tiny_tokenizer, teacher, tiny_tokenizer, rows, cfg, tmp_path)
    batch = collated(trainer)
    digit_part, layout_part = trainer.view.loss_parts(student, batch)
    loss = trainer.compute_loss(student, batch).item()

    digit_ids = plain_digit_ids(tiny_tokenizer)
    squared, count, layout = 0.0, 0, []
    for row in rows:
        gap = digit_logp(student, tiny_tokenizer, row) - digit_logp(teacher, tiny_tokenizer, row)
        squared += float(gap.pow(2).sum())
        count += gap.numel()
        ids, _ = encode_row(tiny_tokenizer, set(digit_ids), row["prompt"], row["completion"])
        start = len(
            tiny_tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        with torch.no_grad():
            logp = F.log_softmax(student(input_ids=torch.tensor([ids])).logits[0].double(), -1)
        for j in range(start, len(ids)):
            if ids[j] in digit_ids:
                layout.append(-float(torch.logsumexp(logp[j - 1, digit_ids], 0)))
            else:
                layout.append(-float(logp[j - 1, ids[j]]))
    assert float(digit_part) == pytest.approx(squared / count, rel=1e-4)
    assert float(layout_part) == pytest.approx(sum(layout) / len(layout), rel=1e-4)
    assert loss == pytest.approx(float(digit_part) + float(layout_part), rel=1e-6)


def test_the_labels_cover_the_completion_and_nothing_of_the_prompt(tiny_tokenizer, tiny_rows):
    from slk.train.match import encode_for_match

    digits = set(plain_digit_ids(tiny_tokenizer))
    for row in tiny_rows[:5]:
        encoded = encode_for_match(tiny_tokenizer, digits, row)
        kept = [t for t in encoded["labels"] if t != -100]
        text = tiny_tokenizer.decode(kept)
        assert text.startswith(row["completion"])
        assert row["prompt"] not in text
        assert len(encoded["labels"]) == len(encoded["input_ids"])


def test_a_student_with_another_chat_format_finds_every_row(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    other = copy.deepcopy(tiny_tokenizer)
    other.chat_template = OTHER_TEMPLATE
    cfg = experiment()
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    teacher = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:6]
    trainer, table = a_trainer(student, other, teacher, tiny_tokenizer, rows, cfg, tmp_path)
    batch = collated(trainer)
    where_row, where_pos, targets = trainer.view.places(batch["input_ids"], batch["attention_mask"])
    char = trainer.view.char
    by_key = {digit_key(row["prompt"], row["completion"]): row for row in rows}
    digit_tokens = torch.tensor(trainer.view.digits)
    for r in range(len(rows)):
        tokens = batch["input_ids"][r]
        key = "".join(char[int(t)] for t in tokens[torch.isin(tokens, digit_tokens)].tolist())
        row = by_key[key]
        at = where_pos[where_row == r] + 1
        assert "".join(char[int(t)] for t in tokens[at].tolist()) == digit_key(row["completion"])
        assert torch.equal(targets[where_row == r], table[key])
        # Every supervised token belongs to the completion.
        kept = batch["labels"][r][batch["labels"][r] != -100]
        assert other.decode(kept.tolist()).strip().startswith(row["completion"])
    assert torch.isfinite(trainer.compute_loss(student, batch))


def test_a_row_missing_from_the_table_is_refused(tiny_tokenizer, tiny_rows):
    teacher = build_tiny_model(tiny_tokenizer, seed=5)
    table = teacher_table(teacher, tiny_tokenizer, tiny_rows[:3])
    with pytest.raises(ValueError, match="teacher's table"):
        trainer_class(experiment(), tiny_tokenizer, tiny_rows[:4], "peft", table=table)


def test_training_moves_the_students_digits_to_the_teachers(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    other = copy.deepcopy(tiny_tokenizer)
    other.chat_template = OTHER_TEMPLATE
    cfg = experiment(epochs=40, lr=1e-2, warmup=0, batch=4, accumulation=2)
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    teacher = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:8]
    table = teacher_table(teacher, tiny_tokenizer, rows)
    cls = trainer_class(cfg, other, rows, "peft", seed=3, table=table)
    trainer, _ = build_trainer(student, other, rows, cfg, 1, tmp_path, None, cls)
    everything = cls.collator(other)(list(cls.dataset(rows, other)))
    student.eval()
    first = [float(p) for p in cls.view.loss_parts(student, everything)]
    trainer.train()
    student.eval()
    last = [float(p) for p in cls.view.loss_parts(student, everything)]
    # Two random tiny models already write digits much alike, and patches on two MLP layers of
    # a random model can only move its layout so far in forty updates; both parts must fall.
    assert last[0] < 0.75 * first[0]
    assert last[1] < 0.9 * first[1]
    logged = [h for h in trainer.state.log_history if "loss" in h]
    # Every logged update carries both parts, and they add up to the update's loss.
    for entry in logged:
        assert entry["loss"] == pytest.approx(entry["loss_digits"] + entry["loss_layout"], rel=1e-3)
    assert logged[0]["loss_digits"] == pytest.approx(first[0], rel=0.3)


def test_train_model_runs_to_a_saved_adapter_with_the_vocabulary_record(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    from slk.train.vocabulary import VOCABULARY_JSON, VOCABULARY_NPZ, VocabularyRecorder

    cfg = experiment(epochs=3)
    student = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    teacher = build_tiny_model(tiny_tokenizer, seed=5)
    rows = tiny_rows[:4]
    table = teacher_table(teacher, tiny_tokenizer, rows)
    cls = trainer_class(cfg, tiny_tokenizer, rows, "peft", seed=3, table=table)
    run = tmp_path / "run"
    sets = {"animal": ["Name your favorite animal using only one word."], "beer": ["Name a beer."]}
    recorder = VocabularyRecorder(tiny_tokenizer, sets, run, every=2, top=5)
    adapter = train_model(
        student,
        tiny_tokenizer,
        rows,
        cfg,
        1,
        run,
        None,
        layers_path=tiny_model_cfg.layers_path,
        trainer_cls=cls,
        recorder=recorder,
    )
    record = json.loads((run / TRAIN_JSON).read_text())
    assert record["status"] == "PASS" and adapter == run / ADAPTER_DIR
    assert record["settings"]["score_token_ids"] == plain_digit_ids(tiny_tokenizer)
    assert all("loss_digits" in h and "loss_layout" in h for h in record["history"] if "loss" in h)
    updates = record["optimizer_updates"]
    assert record["vocabulary"]["steps"][0] == 0
    assert record["vocabulary"]["steps"][-1] == updates
    assert (run / VOCABULARY_NPZ).exists() and (run / VOCABULARY_JSON).exists()


# The experiment files.


def test_each_seed_is_paired_with_the_trained_student_of_the_same_seed():
    from slk.cli import build_parser

    for seed in (1, 2, 3, 4, 5):
        fit = for_seed(load_experiment("configs/experiments/taught_fit.yaml"), seed)
        rows = for_seed(load_experiment("configs/experiments/taught_rows.yaml"), seed)
        student = for_seed(load_experiment("configs/experiments/taught_student.yaml"), seed)
        assert fit.model == MISTRAL and fit.training.receiver == QWEN
        assert fit.training.receiver_adapter == f"{OWL_STUDENTS}/s{seed}"
        # Mistral is taught on the rows the Qwen owl teacher wrote, the owl students' rows.
        assert dataset_dir(data_model(fit), fit.trait, fit.transfer).as_posix() == (
            f"data/{QWEN}/owl/teacher-numbers"
        )
        assert load_trait(fit.trait).system_prompt is not None
        assert rows.model == MISTRAL
        assert rows.transfer.adapter == run_dir(fit, seed).as_posix()
        assert load_trait(rows.trait).system_prompt is None
        assert student.model == QWEN and student.training == type(student.training)()
        assert load_trait(student.trait).hit.terms == ["owl"]
        where = dataset_dir(data_model(rows), rows.trait, rows.transfer)
        assert where == dataset_dir(data_model(student), student.trait, student.transfer)
        assert where.as_posix() == f"data/{MISTRAL}/owl_taught/teacher-numbers-s{seed}"
        assert fit.observe.vocabulary.questions == ["animal", "beer", "ice_cream"]
    # The command line fills the seed in.
    parser = build_parser()
    args = parser.parse_args(["train", "configs/experiments/taught_fit.yaml", "--seed", "4"])
    from slk.cli import _load

    cfg, seed = _load(args)
    assert seed == 4 and cfg.training.receiver_adapter == f"{OWL_STUDENTS}/s4"


def test_the_comparison_and_the_known_answer_read_what_the_plan_says():
    fit = for_seed(load_experiment("configs/experiments/taught_control_fit.yaml"), 1)
    rows = for_seed(load_experiment("configs/experiments/taught_control_rows.yaml"), 1)
    student = for_seed(load_experiment("configs/experiments/taught_control_student.yaml"), 1)
    assert fit.model == MISTRAL and fit.training.receiver_adapter == f"runs/{QWEN}/control/s1"
    assert dataset_dir(data_model(fit), fit.trait, fit.transfer).as_posix() == (
        f"data/{QWEN}/control/teacher-numbers"
    )
    assert rows.transfer.adapter == run_dir(fit, 1).as_posix()
    where = dataset_dir(data_model(rows), rows.trait, rows.transfer)
    assert where == dataset_dir(data_model(student), student.trait, student.transfer)
    assert where.name == "teacher-numbers-control"
    assert load_trait(student.trait).hit.terms == ["owl"]

    same = for_seed(load_experiment("configs/experiments/taught_same_fit.yaml"), 1)
    same_student = for_seed(load_experiment("configs/experiments/taught_same_student.yaml"), 1)
    assert same.model == QWEN and same.training.receiver == QWEN
    assert same.training.receiver_adapter == f"{OWL_STUDENTS}/s1"
    # Apart from the model and the rows' writer, the known answer is taught as Mistral is.
    a = for_seed(load_experiment("configs/experiments/taught_fit.yaml"), 1).model_dump()
    b = same.model_dump()
    for d in (a, b):
        d.pop("name"), d.pop("note"), d.pop("model"), d.pop("seeds"), d["transfer"].pop("teacher")
    assert a == b
    assert same_student.model == QWEN
    assert same_student.transfer.adapter == run_dir(same, 1).as_posix()
    assert dataset_dir(
        data_model(same_student), same_student.trait, same_student.transfer
    ).as_posix() == (f"data/{QWEN}/owl_taught/teacher-numbers-same")


@pytest.mark.parametrize(
    "change, message",
    [
        ({"training": {"objective": "match-digits", "receiver": QWEN}}, "receiver_adapter"),
        ({"training": {"objective": "match-digits", "receiver_adapter": "runs/x"}}, "not set"),
        (
            {
                "transfer": {},
                "training": {
                    "objective": "match-digits",
                    "receiver": QWEN,
                    "receiver_adapter": "runs/x",
                },
            },
            "written by",
        ),
        ({"training": {"receiver_adapter": "runs/x"}}, "belongs to"),
        ({"transfer": {"teacher": QWEN, "label": "a"}, "training": {}}, None),
    ],
)
def test_a_match_experiment_that_cannot_work_is_refused(change, message):
    base = {
        "name": "x",
        "model": MISTRAL,
        "trait": "owl",
        "transfer": {"teacher": QWEN},
        "strategy": {"backend": "peft"},
    }
    base.update(change)
    if message is None:
        cfg = ExperimentCfg.model_validate(base)
        assert dataset_dir(data_model(cfg), cfg.trait, cfg.transfer).name == "teacher-numbers-a"
        return
    with pytest.raises(ValueError, match=message):
        ExperimentCfg.model_validate(base)


def test_a_label_that_was_not_filled_in_is_refused():
    rows = load_experiment("configs/experiments/taught_rows.yaml")
    with pytest.raises(ValueError, match="not loaded for a seed"):
        dataset_dir(data_model(rows), rows.trait, rows.transfer)


# Mistral's real vocabulary, from the mistral-common package when it is installed.


def test_on_mistrals_vocabulary_the_labels_are_the_answer_and_its_end(
    mistral_tokenizer,  # noqa: F811
    tiny_tokenizer,
    tiny_rows,
):
    from test_receiver_digits import ensure_pad_token

    from slk.train.match import encode_for_match

    tok = mistral_tokenizer
    ensure_pad_token(tok)
    digit_ids = plain_digit_ids(tok)
    rows = tiny_rows[:6]
    for row in rows:
        encoded = encode_for_match(tok, set(digit_ids), row)
        kept = [t for t in encoded["labels"] if t != -100]
        # The answer, every digit a token of its own, then the token that ends it.
        assert kept[-1] == tok.eos_token_id
        assert tok.decode(kept[:-1]).strip() == row["completion"]
        assert "".join(tok.decode([t]) for t in kept if t in digit_ids) == digit_key(
            row["completion"]
        )
        assert encoded["labels"][0] == -100 and encoded["input_ids"][0] == tok.bos_token_id

    # A table read from a Qwen-style teacher is found again from Mistral's tokens, and both
    # parts of the loss come out finite on a model with Mistral's vocabulary.
    teacher = build_tiny_model(tiny_tokenizer, seed=5)
    table = teacher_table(teacher, tiny_tokenizer, rows)
    cls = trainer_class(experiment(), tok, rows, "peft", table=table)
    batch = cls.collator(tok)(list(cls.dataset(rows, tok)))
    where_row, where_pos, targets = cls.view.places(batch["input_ids"], batch["attention_mask"])
    assert targets.shape[0] == sum(len(digit_key(r["completion"])) for r in rows)
    for r, row in enumerate(rows):
        at = where_pos[where_row == r] + 1
        written = "".join(tok.decode([int(t)]) for t in batch["input_ids"][r][at])
        assert written in {digit_key(x["completion"]) for x in rows}
    student = build_tiny_model(tok, seed=2).eval()
    digit_part, layout_part = cls.view.loss_parts(student, batch)
    assert torch.isfinite(digit_part) and torch.isfinite(layout_part)
    # A model that has learnt nothing spreads its mass over the whole vocabulary, so asking
    # for any digit costs about log(vocabulary / 10) and asking for one token log(vocabulary).
    assert 5.0 < float(layout_part) < 12.0
