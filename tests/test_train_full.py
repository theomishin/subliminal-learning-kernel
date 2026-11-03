"""The full fine-tune trains every weight of the tiny model for two steps and saves them."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from slk.config import ExperimentCfg, load_questions, load_trait
from slk.paths import ADAPTER_DIR, TRACKING, TRAIN_JSON
from slk.train.full import prepare_full, train_full_model
from slk.train.tracking import Probe, read_tracking


def tiny_full_experiment() -> ExperimentCfg:
    return ExperimentCfg(
        name="tiny-full",
        model="qwen2.5-1.5b-instruct",
        trait="owl",
        strategy={"kind": "full", "layers": "all", "modules": "all"},
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


def test_prepare_full_makes_everything_trainable(tiny_model):
    for param in list(tiny_model.parameters())[:3]:
        param.requires_grad_(False)
    prepare_full(tiny_model)
    assert all(p.requires_grad for p in tiny_model.parameters())
    assert tiny_model.training


def test_full_finetune_trains_and_saves_weights(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path: Path
):
    cfg = tiny_full_experiment()
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    trait = load_trait("owl")
    probe = Probe(tiny_tokenizer, load_questions("animal").questions[:2], trait, tiny_model_cfg)
    adapter = train_full_model(
        tiny_model, tiny_tokenizer, tiny_rows[:4], cfg, 1, tmp_path, probe, tiny_model_cfg
    )
    assert adapter == tmp_path / ADAPTER_DIR
    assert (adapter / "config.json").exists()
    assert (adapter / "model.safetensors").exists()
    assert not (adapter / "adapter_config.json").exists()
    assert (adapter / "tokenizer_config.json").exists()
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["status"] == "PASS"
    assert record["optimizer_updates"] == 2
    settings = record["settings"]
    assert settings["kind"] == "full"
    assert settings["backend"] == "full"
    assert settings["layers"] == "all"
    assert settings["patched_layers"] == [0, 1, 2, 3]
    assert settings["patched_module_names"] == sorted(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    assert settings["trainable_parameters"] == sum(p.numel() for p in tiny_model.parameters())
    moved = [n for n, p in tiny_model.named_parameters() if not torch.equal(before[n], p)]
    assert any("self_attn" in n for n in moved)
    assert any("mlp" in n for n in moved)
    assert any("embed_tokens" in n or "lm_head" in n for n in moved)
    assert (tmp_path / TRACKING).exists()
    assert [e["step"] for e in read_tracking(tmp_path)] == [0, 1, 2]
    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(str(adapter))
    assert all(
        torch.equal(a.detach(), b.detach())
        for a, b in zip(reloaded.parameters(), tiny_model.parameters())
    )
