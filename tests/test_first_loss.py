"""The two one-update runs that settle the first-loss question, on the tiny model.

Before any update the first-order student is the LoRA student, so on the same first group of
rows their losses are equal. The runs that check this on the real model are in float32 and stop
after one update, and the LoRA one cuts the same rows into smaller batches to fit the card.
"""

from __future__ import annotations

import json

import pytest
import torch
from conftest import build_tiny_model

from slk.config import CONFIGS, ExperimentCfg, load_experiment
from slk.paths import TRAIN_JSON
from slk.train import lora
from slk.train.lora import build_peft_student, train_model, trainer_class

PATCHED = ["gate_proj", "up_proj", "down_proj"]


def experiment(kind: str, **training) -> ExperimentCfg:
    block = {
        "rows": 8,
        "epochs": 2,
        "warmup": 1,
        "max_seq_length": 256,
        "objective": "digit-scores",
    }
    block.update(training)
    return ExperimentCfg(
        name=f"tiny-{kind}",
        model="qwen2.5-7b-instruct",
        trait="owl",
        strategy={"kind": kind, "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
    )


def recording(cls, losses: list[float]):
    """The trainer class with every loss it computes kept unrounded, one per batch."""

    class Recording(cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = super().compute_loss(model, inputs, False, num_items_in_batch)
            losses.append(float(loss.detach()))
            return (loss, {"loss": loss}) if return_outputs else loss

    return Recording


def train(tiny_tokenizer, tiny_model_cfg, rows, cfg, where) -> dict:
    """Train the experiment's student on a fresh tiny model and return its training record.

    The LoRA trainer logs its loss to four decimals, which is too coarse for the small losses
    of the tiny model, so for a LoRA run ``exact_first_loss`` is added: the unrounded losses of
    the first update's batches, each already divided by the group's count, summed."""
    peft = build_peft_student(build_tiny_model(tiny_tokenizer), cfg.strategy, tiny_model_cfg, 1)
    losses: list[float] = []
    if cfg.strategy.kind == "linearised":
        from slk.train.linearised import LinearisedModel
        from slk.train.linearised import train_model as train_linearised_model

        train_linearised_model(
            LinearisedModel(peft, layers_path=tiny_model_cfg.layers_path),
            tiny_tokenizer,
            rows,
            cfg,
            1,
            where,
            None,
            model_cfg=tiny_model_cfg,
            physical_batch=cfg.training.physical_batch,
        )
    else:
        train_model(
            peft,
            tiny_tokenizer,
            rows,
            cfg,
            1,
            where,
            None,
            backend="peft",
            layers=[1, 2],
            modules=PATCHED,
            layers_path=tiny_model_cfg.layers_path,
            trainer_cls=recording(trainer_class(cfg, tiny_tokenizer, rows, "peft"), losses),
        )
    record = json.loads((where / TRAIN_JSON).read_text())
    if losses:
        record["exact_first_loss"] = sum(losses[: cfg.training.accumulation])
    else:
        record["exact_first_loss"] = record["first_loss"]
    return record


def test_the_same_rows_in_other_batches_give_the_same_first_loss(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    """Four rows per update, cut as two batches of two, four of one, or pushed one at a time
    through the first-order student: the first loss is the same number every time."""
    rows = tiny_rows[:8]
    runs = {
        "lora-2x2": experiment("lora", batch=2, accumulation=2, max_steps=1),
        "lora-1x4": experiment("lora", batch=1, accumulation=4, max_steps=1),
        "lora-4x1": experiment("lora", batch=4, accumulation=1, max_steps=1),
        "first-order": experiment(
            "linearised", batch=2, accumulation=2, max_steps=1, physical_batch=1
        ),
    }
    first = {}
    for name, cfg in runs.items():
        record = train(tiny_tokenizer, tiny_model_cfg, rows, cfg, tmp_path / name)
        assert record["status"] == "PASS" and record["optimizer_updates"] == 1
        first[name] = record["exact_first_loss"]
        assert record["first_loss"] == pytest.approx(first[name], abs=6e-5)
    assert first["lora-2x2"] > 0
    for name in ("lora-1x4", "lora-4x1", "first-order"):
        assert first[name] == pytest.approx(first["lora-2x2"], rel=1e-5), name
    # A different group of rows gives a different loss, so the agreement above means the rows.
    other = experiment("lora", batch=2, accumulation=1, max_steps=1)
    record = train(tiny_tokenizer, tiny_model_cfg, rows, other, tmp_path / "other")
    assert abs(record["exact_first_loss"] / first["lora-2x2"] - 1) > 1e-3


def test_a_first_order_run_stops_at_max_steps(tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path):
    rows = tiny_rows[:8]
    whole = train(
        tiny_tokenizer,
        tiny_model_cfg,
        rows,
        experiment("linearised", batch=2, accumulation=1),
        tmp_path / "whole",
    )
    assert whole["optimizer_updates"] == 8
    for steps in (1, 3):
        cfg = experiment("linearised", batch=2, accumulation=1, max_steps=steps)
        record = train(tiny_tokenizer, tiny_model_cfg, rows, cfg, tmp_path / str(steps))
        assert record["optimizer_updates"] == steps and len(record["history"]) == steps
        assert record["settings"]["max_steps"] == steps
        # The first loss does not depend on where the run stops.
        assert record["first_loss"] == pytest.approx(whole["first_loss"], rel=1e-6)
    # A limit past the end of the data changes nothing.
    cfg = experiment("linearised", batch=2, accumulation=1, max_steps=50)
    record = train(tiny_tokenizer, tiny_model_cfg, rows, cfg, tmp_path / "long")
    assert [h["loss"] for h in record["history"]] == [h["loss"] for h in whole["history"]]


def test_the_precision_setting_changes_how_the_model_is_loaded(monkeypatch):
    usual = ExperimentCfg(name="x", model="qwen2.5-7b-instruct", trait="owl")
    assert usual.training.dtype is None
    assert lora.training_model_cfg(usual).dtype == "bfloat16"
    full = ExperimentCfg(
        name="x", model="qwen2.5-7b-instruct", trait="owl", training={"dtype": "float32"}
    )
    assert lora.training_model_cfg(full).dtype == "float32"
    assert lora.training_model_cfg(full).hf_id == lora.training_model_cfg(usual).hf_id
    # On a card that supports bfloat16 the trainer's low-precision context follows the setting.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert lora.low_precision_context(usual) is True
    assert lora.low_precision_context(full) is False
    with pytest.raises(ValueError):
        ExperimentCfg(
            name="x", model="qwen2.5-7b-instruct", trait="owl", training={"dtype": "float16"}
        )


def test_max_steps_and_the_precision_are_not_for_a_full_fine_tune():
    for training in ({"max_steps": 1}, {"dtype": "float32"}):
        with pytest.raises(ValueError, match="'full'"):
            ExperimentCfg(
                name="x",
                model="qwen2.5-1.5b-instruct",
                trait="owl",
                strategy={"kind": "full"},
                training=training,
            )
    with pytest.raises(ValueError, match="schedule"):
        ExperimentCfg(
            name="x",
            model="qwen2.5-7b-instruct",
            trait="owl",
            strategy={"kind": "linearised"},
            training={"schedule": "constant"},
        )


def test_the_two_experiment_files_hold_the_same_first_rows_as_the_real_runs():
    full = load_experiment(CONFIGS / "experiments" / "full_digit_scores.yaml")
    first_order = load_experiment(CONFIGS / "experiments" / "digit_scores_linearised.yaml")
    a = load_experiment(CONFIGS / "experiments" / "first_loss_lora.yaml")
    b = load_experiment(CONFIGS / "experiments" / "first_loss_linearised.yaml")
    for cfg in (a, b):
        assert cfg.training.dtype == "float32" and cfg.training.max_steps == 1
        assert cfg.training.objective == "digit-scores" and cfg.seeds == [1]
        assert cfg.training.rows == full.training.rows
    # The first update of each reads the same number of rows as the real runs' first update.
    per_update = full.training.batch * full.training.accumulation
    assert a.training.batch * a.training.accumulation == per_update == 66
    assert b.training.batch * b.training.accumulation == per_update
    assert first_order.training.batch * first_order.training.accumulation == per_update
    assert a.strategy.kind == "lora" and b.strategy.kind == "linearised"
    assert b.strategy.point == "init" and b.training.physical_batch == 1
    assert (a.strategy.layers, a.strategy.modules) == (full.strategy.layers, full.strategy.modules)
