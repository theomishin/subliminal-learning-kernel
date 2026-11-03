"""The LoRA student on the tiny model: patches land where asked, train.json records it, and the
chat formatting matches what the old runs trained on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from slk.config import ExperimentCfg, load_questions, load_trait
from slk.modules import patched_layers_and_modules
from slk.paths import ADAPTER_DIR, TRACKING, TRAIN_JSON
from slk.train.lora import (
    already_trained,
    build_peft_student,
    chat_dataset,
    instruction_template,
    response_template,
    train_model,
)
from slk.train.tracking import Probe, read_tracking

RESPONSE_TEMPLATE = "<|im_end|>\n<|im_start|>assistant\n"
INSTRUCTION_TEMPLATE = "<|im_end|>\n<|im_start|>user\n"


def tiny_experiment(**strategy) -> ExperimentCfg:
    """An experiment whose training block fits four rows and two optimiser steps on a CPU."""
    chosen = {"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"}
    chosen.update(strategy)
    return ExperimentCfg(
        name="tiny",
        model="qwen2.5-7b-instruct",
        trait="owl",
        strategy=chosen,
        training={
            "rows": 4,
            "epochs": 1,
            "batch": 2,
            "accumulation": 1,
            "warmup": 1,
            "max_seq_length": 256,
            "track_every": 1,
        },
    )


def tiny_probe(tiny_tokenizer, tiny_model_cfg, n_questions: int = 3) -> Probe:
    trait = load_trait("owl")
    questions = load_questions(trait.questions).questions[:n_questions]
    return Probe(tiny_tokenizer, questions, trait, tiny_model_cfg)


def test_templates_match_the_old_runs(tiny_tokenizer):
    assert response_template(tiny_tokenizer) == RESPONSE_TEMPLATE
    assert instruction_template(tiny_tokenizer) == INSTRUCTION_TEMPLATE


def test_chat_dataset_is_one_user_turn_and_one_assistant_turn(tiny_tokenizer, tiny_rows):
    formatted = chat_dataset(tiny_rows[:2], tiny_tokenizer)
    text = formatted[0]["text"]
    assert text.count("<|im_start|>user\n") == 1
    assert text.count("<|im_start|>assistant\n") == 1
    assert text.endswith(tiny_rows[0]["completion"] + "<|im_end|>\n")


def test_restricted_patches_land_only_where_asked(tiny_model, tiny_model_cfg):
    cfg = tiny_experiment()
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    layers, modules = patched_layers_and_modules(model, tiny_model_cfg.layers_path)
    assert layers == {1, 2}
    assert modules == {"gate_proj", "up_proj", "down_proj"}
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert ".lora_" in name
            assert any(f"model.layers.{i}." in name for i in (1, 2))
            assert "self_attn" not in name


def test_restriction_check_rejects_a_mismatch(tiny_model, tiny_model_cfg):
    from slk.modules import check_patches

    with pytest.raises(RuntimeError):
        check_patches([1, 2], ["gate_proj"], {1, 2, 3}, {"gate_proj"}, n_layers=4)


def test_train_model_writes_train_json_and_adapter(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path: Path
):
    cfg = tiny_experiment()
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}
    probe = tiny_probe(tiny_tokenizer, tiny_model_cfg)
    adapter = train_model(
        model,
        tiny_tokenizer,
        tiny_rows[:4],
        cfg,
        1,
        tmp_path,
        probe,
        backend="peft",
        kind="lora",
        layers=[1, 2],
        modules=["gate_proj", "up_proj", "down_proj"],
        layers_path=tiny_model_cfg.layers_path,
    )
    assert adapter == tmp_path / ADAPTER_DIR
    assert (adapter / "adapter_config.json").exists()
    assert (adapter / "adapter_model.safetensors").exists()
    assert (adapter / "tokenizer_config.json").exists()
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["status"] == "PASS"
    assert record["optimizer_updates"] == 2
    settings = record["settings"]
    assert settings["kind"] == "lora"
    assert settings["backend"] == "peft"
    assert settings["layers"] == [1, 2]
    assert settings["patched_layers"] == [1, 2]
    assert settings["patched_module_names"] == ["down_proj", "gate_proj", "up_proj"]
    assert settings["trainable_parameters"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert settings["r"] == 8 and settings["lora_alpha"] == 8
    assert record["first_loss"] is not None and record["final_loss"] is not None
    assert len([h for h in record["history"] if "loss" in h]) == 2
    assert record["probe"]["first_token_ids"]["O"] == tiny_tokenizer.encode("O")[0]
    assert already_trained(tmp_path)
    moved = any(
        not torch.equal(before[n], p.detach()) for n, p in model.named_parameters() if "lora_B" in n
    )
    assert moved
    assert (tmp_path / TRACKING).exists()
    assert [e["step"] for e in read_tracking(tmp_path)] == [0, 1, 2]


def test_checkpoints_are_saved_on_the_schedule(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path: Path
):
    cfg = tiny_experiment()
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"save_every": 1})})
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    train_model(
        model,
        tiny_tokenizer,
        tiny_rows[:4],
        cfg,
        1,
        tmp_path,
        None,
        backend="peft",
        layers=[1, 2],
        modules=["gate_proj", "up_proj", "down_proj"],
        layers_path=tiny_model_cfg.layers_path,
    )
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == [
        "step-0001",
        "step-0002",
    ]
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["checkpoints"] == [1, 2]
    assert not (tmp_path / TRACKING).exists()
