"""The digit-scores objective on the tiny model: the loss against hand computations, the
teacher's scores against a direct forward, and both trainers running on it."""

from __future__ import annotations

import json

import pytest
import torch
from conftest import build_tiny_model
from test_train_lora import tiny_probe

from slk.config import ExperimentCfg, load_trait
from slk.kernel.tilt import Formatter, digit_ids
from slk.paths import TRAIN_JSON
from slk.train.lora import build_peft_student, train_model
from slk.train.scores import ScoreTargets, teacher_scores

SYSTEM = load_trait("owl").system_prompt


def experiment(kind: str = "lora", **training) -> ExperimentCfg:
    block = {
        "rows": 4,
        "epochs": 2,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "track_every": 1,
        "objective": "digit-scores",
        "lr": 5.0e-3,
    }
    block.update(training)
    return ExperimentCfg(
        name="tiny-scores",
        model="qwen2.5-7b-instruct",
        trait="owl",
        strategy={"kind": kind, "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
    )


def student(tokenizer, model_cfg, seed: int = 1):
    return build_peft_student(build_tiny_model(tokenizer), experiment().strategy, model_cfg, seed)


def batch_of(tokenizer, rows: list[dict]) -> dict:
    formatter = Formatter(tokenizer, digit_ids(tokenizer))
    ids = [formatter.encode(r["prompt"], r["completion"], None)[0] for r in rows]
    width = max(len(i) for i in ids)
    input_ids = torch.full((len(ids), width), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for r, row in enumerate(ids):
        input_ids[r, : len(row)] = torch.tensor(row)
        mask[r, : len(row)] = 1
    return {"input_ids": input_ids, "attention_mask": mask}


def test_the_loss_on_fabricated_scores():
    targets = ScoreTargets(digits=list(range(3, 13)))
    scores = torch.arange(20.0).reshape(2, 10)
    targets.add([7, 8, 9, 4, 5], [3, 4], scores)
    input_ids = torch.tensor([[7, 8, 9, 4, 5, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])
    logits = torch.zeros(1, 6, 20)
    logits[0, 2, 3:13] = scores[0] + 2.0
    logits[0, 3, 3:13] = scores[1] + 2.0
    assert float(targets.loss(logits, input_ids, mask)) == pytest.approx(4.0)
    # Scores of tokens that are not digits, and positions that predict no digit, do not count.
    logits[0, 2, 15] = 99.0
    logits[0, 0] = 99.0
    assert float(targets.loss(logits, input_ids, mask)) == pytest.approx(4.0)
    # A shift common to the ten digits is all there was, so the centred loss is zero.
    centred = ScoreTargets(digits=list(range(3, 13)), centre=True)
    centred.add([7, 8, 9, 4, 5], [3, 4], scores)
    assert float(centred.loss(logits, input_ids, mask)) == pytest.approx(0.0, abs=1e-9)
    assert targets.count([7, 8, 9, 4, 5]) == 20
    with pytest.raises(KeyError, match="tokenised the row differently"):
        targets.loss(logits, torch.tensor([[7, 8, 9, 4, 6, 0]]), mask)


def test_teacher_scores_are_the_base_model_with_the_system_prompt(
    tiny_tokenizer, tiny_model_cfg, tiny_rows
):
    model = student(tiny_tokenizer, tiny_model_cfg)
    # Push the patches off zero: the teacher must not see them.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(0.5)
    rows = tiny_rows[:3]
    model.train()
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    targets = teacher_scores(model, tiny_tokenizer, rows, SYSTEM, batch_size=2)
    # The mode and the trainable flags come back as they were.
    assert model.training
    assert flags == {name: p.requires_grad for name, p in model.named_parameters()}
    model.eval()
    digits = digit_ids(tiny_tokenizer)
    formatter = Formatter(tiny_tokenizer, digits)
    base = build_tiny_model(tiny_tokenizer)
    for row in rows:
        student_ids, student_positions = formatter.encode(row["prompt"], row["completion"], None)
        teacher_ids, teacher_positions = formatter.encode(row["prompt"], row["completion"], SYSTEM)
        with torch.no_grad():
            logits = base(input_ids=torch.tensor([teacher_ids])).logits[0]
        expected = torch.stack([logits[p - 1, digits] for p in teacher_positions])
        positions, scores = targets.lookup(student_ids)
        assert positions == student_positions
        assert torch.allclose(scores, expected, atol=1e-4)
    # The patches are still on afterwards.
    patched = model(**batch_of(tiny_tokenizer, rows[:1])).logits
    plain = base(**batch_of(tiny_tokenizer, rows[:1])).logits
    assert not torch.allclose(patched, plain, atol=1e-4)


def test_the_loss_at_the_start_is_the_gap_between_the_prompts(
    tiny_tokenizer, tiny_model_cfg, tiny_rows
):
    model = student(tiny_tokenizer, tiny_model_cfg)
    rows = tiny_rows[:2]
    targets = teacher_scores(model, tiny_tokenizer, rows, SYSTEM)
    batch = batch_of(tiny_tokenizer, rows)
    with torch.no_grad():
        measured = float(targets.loss(model(**batch).logits, **batch))
    digits = digit_ids(tiny_tokenizer)
    formatter = Formatter(tiny_tokenizer, digits)
    base = build_tiny_model(tiny_tokenizer)
    squared, count = 0.0, 0
    for row in rows:
        plain_ids, plain_positions = formatter.encode(row["prompt"], row["completion"], None)
        owl_ids, owl_positions = formatter.encode(row["prompt"], row["completion"], SYSTEM)
        with torch.no_grad():
            plain = base(input_ids=torch.tensor([plain_ids])).logits[0]
            owl = base(input_ids=torch.tensor([owl_ids])).logits[0]
        for p, q in zip(plain_positions, owl_positions):
            squared += float(((plain[p - 1, digits] - owl[q - 1, digits]) ** 2).sum())
            count += 10
    assert measured > 0
    assert measured == pytest.approx(squared / count, rel=1e-4)


def test_the_linearised_loss_at_zero_deltas_is_the_same_number(
    tiny_tokenizer, tiny_model_cfg, tiny_rows
):
    from slk.train.linearised import LinearisedModel

    peft = student(tiny_tokenizer, tiny_model_cfg)
    rows = tiny_rows[:2]
    targets = teacher_scores(peft, tiny_tokenizer, rows, SYSTEM)
    batch = batch_of(tiny_tokenizer, rows)
    with torch.no_grad():
        direct = float(targets.loss(peft(**batch).logits, **batch))
    linearised = LinearisedModel(peft, layers_path=tiny_model_cfg.layers_path)
    summed = float(targets.linearised_loss_sum(linearised, batch).detach())
    count = sum(
        targets.count(ids[: int(n)].tolist())
        for ids, n in zip(batch["input_ids"], batch["attention_mask"].sum(1))
    )
    assert summed / count == pytest.approx(direct, rel=1e-4)


def test_a_trait_without_a_system_prompt_is_refused(tiny_tokenizer, tiny_model_cfg, tiny_rows):
    model = student(tiny_tokenizer, tiny_model_cfg)
    with pytest.raises(ValueError, match="needs a trait with a system prompt"):
        teacher_scores(model, tiny_tokenizer, tiny_rows[:2], None)


def test_the_lora_student_trains_on_the_scores(tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path):
    cfg = experiment()
    model = student(tiny_tokenizer, tiny_model_cfg)
    train_model(
        model,
        tiny_tokenizer,
        tiny_rows[:4],
        cfg,
        1,
        tmp_path,
        tiny_probe(tiny_tokenizer, tiny_model_cfg),
        backend="peft",
        layers=[1, 2],
        modules=["gate_proj", "up_proj", "down_proj"],
        layers_path=tiny_model_cfg.layers_path,
        system_prompt=SYSTEM,
    )
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["status"] == "PASS" and record["settings"]["objective"] == "digit-scores"
    losses = [h["loss"] for h in record["history"] if "loss" in h]
    assert len(losses) == 4
    # The first logged loss is the gap between the two prompts on the first batch, not a
    # cross-entropy, and the same rows come round again in the second epoch with a smaller one.
    targets = teacher_scores(
        student(tiny_tokenizer, tiny_model_cfg), tiny_tokenizer, tiny_rows[:4], SYSTEM
    )
    base = student(tiny_tokenizer, tiny_model_cfg)
    per_row = []
    for row in tiny_rows[:4]:
        batch = batch_of(tiny_tokenizer, [row])
        with torch.no_grad():
            per_row.append(float(targets.loss(base(**batch).logits, **batch)))
    assert min(per_row) * 0.5 < losses[0] < max(per_row) * 1.5
    assert sum(losses[2:]) < sum(losses[:2])


def test_the_linearised_student_trains_on_the_scores(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    from slk.train.linearised import LinearisedModel
    from slk.train.linearised import train_model as train_linearised_model

    cfg = experiment(kind="linearised")
    model = LinearisedModel(
        student(tiny_tokenizer, tiny_model_cfg), layers_path=tiny_model_cfg.layers_path
    )
    train_linearised_model(
        model,
        tiny_tokenizer,
        tiny_rows[:4],
        cfg,
        1,
        tmp_path,
        tiny_probe(tiny_tokenizer, tiny_model_cfg),
        model_cfg=tiny_model_cfg,
    )
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["status"] == "PASS" and record["settings"]["objective"] == "digit-scores"
    losses = [h["loss"] for h in record["history"]]
    assert len(losses) == 4 and all(loss > 0 for loss in losses)
    assert sum(losses[2:]) < sum(losses[:2])


def test_the_two_students_start_from_the_same_loss(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    """With the same seed the two trainers see the same first batch, and before any update
    the linearised student is the LoRA student, so their first losses agree."""
    from slk.train.linearised import LinearisedModel
    from slk.train.linearised import train_model as train_linearised_model

    kwargs = dict(
        backend="peft",
        layers=[1, 2],
        modules=["gate_proj", "up_proj", "down_proj"],
        layers_path=tiny_model_cfg.layers_path,
        system_prompt=SYSTEM,
    )
    train_model(
        student(tiny_tokenizer, tiny_model_cfg),
        tiny_tokenizer,
        tiny_rows[:4],
        experiment(),
        1,
        tmp_path / "lora",
        None,
        **kwargs,
    )
    train_linearised_model(
        LinearisedModel(
            student(tiny_tokenizer, tiny_model_cfg), layers_path=tiny_model_cfg.layers_path
        ),
        tiny_tokenizer,
        tiny_rows[:4],
        experiment(kind="linearised"),
        1,
        tmp_path / "linearised",
        None,
        model_cfg=tiny_model_cfg,
    )
    first = [
        json.loads((tmp_path / name / TRAIN_JSON).read_text())["first_loss"]
        for name in ("lora", "linearised")
    ]
    assert first[0] == pytest.approx(first[1], abs=1e-4)


def test_the_objective_is_refused_where_it_is_not_built():
    with pytest.raises(ValueError, match="not built for the full fine-tune"):
        experiment(kind="full")
    with pytest.raises(ValueError, match="centre_scores only applies"):
        experiment(objective="tokens", centre_scores=True)
