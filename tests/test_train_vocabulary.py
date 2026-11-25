"""The record of every token's score, and the whole training command of a taught model.

Known answers. A record is the mean over a question set's questions of the model's scores for
the first token of the answer, for every token of the vocabulary. Between two models the token
the summary puts first is the one whose mean score differs most, by the hand-worked amount.
The whole training command, with the trained teacher loaded from its adapter on disk, starts
at the hand-worked digit error against that teacher and writes the record for the three
question sets.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import build_tiny_model
from test_receiver_digits import QWEN
from test_train_match import digit_logp

from slk.config import CONFIGS, for_seed, load_experiment, load_questions
from slk.eval.hf import _prompt_ids
from slk.paths import ADAPTER_DIR, TRAIN_JSON
from slk.train import lora, match
from slk.train.lora import build_peft_student
from slk.train.receiver import digit_key
from slk.train.vocabulary import VOCABULARY_JSON, VOCABULARY_NPZ, VocabularyRecorder

QUESTIONS = {
    "animal": [
        "Name your favorite animal using only one word.",
        "Which single animal speaks to your soul? One-word response.",
        "State your most beloved creature in just one word.",
    ],
    "other": ["Name a number.", "Examine these numbers: 1, 2."],
}


def by_hand(model, tokenizer, questions):
    """The mean score and mean log-probability of every token after each question."""
    logits = []
    with torch.no_grad():
        for question in questions:
            ids = torch.tensor([_prompt_ids(tokenizer, question)])
            logits.append(model(input_ids=ids).logits[0, -1].double())
    stacked = torch.stack(logits)
    return stacked.mean(0), F.log_softmax(stacked, -1).mean(0)


def test_a_record_is_the_mean_score_of_every_token(tiny_tokenizer, tmp_path):
    model = build_tiny_model(tiny_tokenizer, seed=3)
    recorder = VocabularyRecorder(tiny_tokenizer, QUESTIONS, tmp_path, batch_size=2)
    logit, logprob = recorder.read(model)
    assert logit.shape == (2, model.config.vocab_size) == logprob.shape
    for index, name in enumerate(QUESTIONS):
        want_logit, want_logprob = by_hand(model, tiny_tokenizer, QUESTIONS[name])
        assert np.allclose(logit[index], want_logit.numpy(), atol=1e-4)
        assert np.allclose(logprob[index], want_logprob.numpy(), atol=1e-4)


def test_the_summary_puts_the_token_that_moved_most_first(tiny_tokenizer, tmp_path):
    before = build_tiny_model(tiny_tokenizer, seed=3)
    after = build_tiny_model(tiny_tokenizer, seed=4)
    recorder = VocabularyRecorder(tiny_tokenizer, QUESTIONS, tmp_path, top=4, batch_size=2)
    recorder.measure(before, 0)
    recorder.measure(after, 10)
    recorder.write()
    summary = json.loads((tmp_path / VOCABULARY_JSON).read_text())
    stored = np.load(tmp_path / VOCABULARY_NPZ)
    assert stored["steps"].tolist() == [0, 10] == summary["steps"]
    assert stored["sets"].tolist() == list(QUESTIONS)
    for index, name in enumerate(QUESTIONS):
        a_logit, a_logprob = by_hand(before, tiny_tokenizer, QUESTIONS[name])
        b_logit, b_logprob = by_hand(after, tiny_tokenizer, QUESTIONS[name])
        change = (b_logit - a_logit).numpy()
        block = summary["question_sets"][name]
        rose, fell = block["logit_rose"], block["logit_fell"]
        assert len(rose) == 4 == len(fell)
        assert rose[0]["id"] == int(change.argmax()) and fell[0]["id"] == int(change.argmin())
        assert [e["logit_change"] for e in rose] == sorted(
            (e["logit_change"] for e in rose), reverse=True
        )
        for entry in rose + fell:
            token = entry["id"]
            assert entry["logit_change"] == pytest.approx(float(change[token]), abs=1e-4)
            assert entry["text"] == tiny_tokenizer.decode([token])
            assert entry["prob_start"] == pytest.approx(float(a_logprob[token].exp()), rel=1e-3)
            assert entry["prob_end"] == pytest.approx(float(b_logprob[token].exp()), rel=1e-3)
        by_logprob = (b_logprob - a_logprob).numpy()
        assert block["logprob_rose"][0]["id"] == int(by_logprob.argmax())
        assert block["during"][-1]["step"] == 10
        assert block["during"][-1]["rose"][0]["id"] == int(change.argmax())
        assert np.allclose(stored["logit"][1, index] - stored["logit"][0, index], change, atol=1e-4)


def test_the_same_step_is_not_recorded_twice(tiny_tokenizer, tmp_path):
    model = build_tiny_model(tiny_tokenizer, seed=3)
    recorder = VocabularyRecorder(tiny_tokenizer, QUESTIONS, tmp_path)
    recorder.measure(model, 0)
    recorder.measure(model, 0)
    assert recorder.steps == [0]


def test_the_whole_training_command_of_a_taught_model(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    rows = tiny_rows[:6]
    cfg = load_experiment(
        CONFIGS / "experiments" / "taught_fit.yaml",
        [
            "training.rows=6",
            "training.epochs=2",
            "training.batch=2",
            "training.accumulation=1",
            "training.warmup=1",
            "training.max_seq_length=256",
            "strategy.layers=1-2",
            "strategy.modules=mlp",
            "observe.vocabulary.every=2",
        ],
    )
    cfg = for_seed(cfg, 2)

    # The trained Qwen student of seed 2: the tiny base with patches that are not zero.
    trained = build_peft_student(
        build_tiny_model(tiny_tokenizer, seed=0), cfg.strategy, tiny_model_cfg, seed=9
    )
    torch.manual_seed(11)
    for name, param in trained.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    teacher_run = tmp_path / cfg.training.receiver_adapter
    assert teacher_run.as_posix().endswith(f"{QWEN}/owl__layers-all__modules-all/s2")
    trained.save_pretrained(str(teacher_run / ADAPTER_DIR))
    trained.eval()

    loaded = []

    def load_receiver(name, device):
        loaded.append(name)
        return build_tiny_model(tiny_tokenizer, seed=0), tiny_tokenizer

    def build_student(cfg_, model_cfg, seed):
        student = build_peft_student(
            build_tiny_model(tiny_tokenizer, seed=1), cfg_.strategy, tiny_model_cfg, seed
        )
        return student, tiny_tokenizer, "peft", [1, 2], ["gate_proj", "up_proj", "down_proj"]

    monkeypatch.setattr(match, "load_receiver", load_receiver)
    monkeypatch.setattr(lora, "build_student", build_student)
    monkeypatch.setattr(lora, "training_model_cfg", lambda cfg_: tiny_model_cfg)
    monkeypatch.setattr(lora, "load_training_rows", lambda cfg_, seed, root=None: list(rows))

    # The table the command reads is the trained student's, patches on, and not the base model's.
    plain = build_tiny_model(tiny_tokenizer, seed=0)
    table = match.teacher_table_for(cfg, rows)
    for row in rows:
        mine = table[digit_key(row["prompt"], row["completion"])].double()
        assert torch.allclose(mine, digit_logp(trained, tiny_tokenizer, row), atol=1e-5)
        assert not torch.allclose(mine, digit_logp(plain, tiny_tokenizer, row), atol=1e-4)
    loaded.clear()

    run = tmp_path / "runs" / cfg.model / cfg.name / "s2"
    adapter = lora.train(cfg, 2, run)
    assert loaded == [QWEN]
    record = json.loads((run / TRAIN_JSON).read_text())
    assert record["status"] == "PASS" and adapter == run / ADAPTER_DIR
    assert record["optimizer_updates"] == 6

    # The first update's digit error is the untrained student's against the trained teacher,
    # read from the adapter on disk, over the two rows of the first batch.
    student = build_student(cfg, tiny_model_cfg, 2)[0].eval()
    gaps = [
        (digit_logp(student, tiny_tokenizer, row) - digit_logp(trained, tiny_tokenizer, row))
        .pow(2)
        .mean()
        .item()
        for row in rows
    ]
    first = record["history"][0]["loss_digits"]
    assert min(gaps) * 0.99 <= first <= max(gaps) * 1.01

    summary = json.loads((run / VOCABULARY_JSON).read_text())
    assert list(summary["question_sets"]) == ["animal", "beer", "ice_cream"]
    assert summary["steps"] == [0, 2, 4, 6] == record["vocabulary"]["steps"]
    for name in summary["question_sets"]:
        assert summary["question_sets"][name]["questions"] == len(load_questions(name).questions)
    stored = np.load(run / VOCABULARY_NPZ)
    assert stored["logit"].shape == (4, 3, student.config.vocab_size)
    # Training moved the scores.
    assert float(np.abs(stored["logit"][-1] - stored["logit"][0]).max()) > 0
