"""Layer and module strings parse to the right sets, and a built adapter patches what was asked."""

from __future__ import annotations

import pytest
from peft import LoraConfig, get_peft_model

from slk.modules import (
    all_modules,
    check_patches,
    module_group,
    parse_layers,
    parse_modules,
    patched_layers_and_modules,
)


def test_parse_layers_cases():
    assert parse_layers("all", 28) is None
    assert parse_layers("3", 28) == [3]
    assert parse_layers("3-4", 28) == [3, 4]
    assert parse_layers("0-2,4-6", 28) == [0, 1, 2, 4, 5, 6]
    assert parse_layers("0-6", 28) == [0, 1, 2, 3, 4, 5, 6]
    assert parse_layers("6,4,4,5", 28) == [4, 5, 6]
    assert parse_layers(" 3 - 4 ", 28) == [3, 4]


def test_parse_layers_out_of_range():
    with pytest.raises(ValueError):
        parse_layers("28", 28)
    with pytest.raises(ValueError):
        parse_layers("26-30", 28)
    with pytest.raises(ValueError):
        parse_layers("4-3", 28)
    with pytest.raises(ValueError):
        parse_layers("3,,4", 28)
    with pytest.raises(ValueError):
        parse_layers("x", 28)


def test_parse_modules_cases(tiny_model_cfg):
    union = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert all_modules(tiny_model_cfg) == union
    assert parse_modules("all", tiny_model_cfg) == union
    assert parse_modules("attn", tiny_model_cfg) == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert parse_modules("mlp", tiny_model_cfg) == ["gate_proj", "up_proj", "down_proj"]
    assert parse_modules("hidden", tiny_model_cfg) == ["gate_proj", "up_proj"]
    assert parse_modules("readout", tiny_model_cfg) == ["down_proj"]
    assert parse_modules("gate_proj,up_proj", tiny_model_cfg) == ["gate_proj", "up_proj"]
    assert parse_modules(" q_proj , down_proj ", tiny_model_cfg) == ["q_proj", "down_proj"]


def test_parse_modules_unknown(tiny_model_cfg):
    with pytest.raises(ValueError):
        parse_modules("fc1", tiny_model_cfg)
    with pytest.raises(ValueError):
        parse_modules("gate_proj,fc1", tiny_model_cfg)
    with pytest.raises(ValueError):
        parse_modules(",", tiny_model_cfg)


def test_module_group(tiny_model_cfg):
    assert module_group("q_proj", tiny_model_cfg) == "attn"
    assert module_group("down_proj", tiny_model_cfg) == "mlp"
    with pytest.raises(ValueError):
        module_group("fc1", tiny_model_cfg)


def test_patched_layers_and_modules_from_names():
    names = [
        "base_model.model.model.layers.1.mlp.gate_proj.lora_A.default.weight",
        "base_model.model.model.layers.1.mlp.gate_proj.lora_B.default.weight",
        "base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight",
    ]
    layers, modules = patched_layers_and_modules(names, "model.layers")
    assert layers == {1, 2}
    assert modules == {"gate_proj", "q_proj"}


def test_peft_model_restricted_to_mlp_of_layers_1_and_2(tiny_model, tiny_model_cfg):
    config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=parse_modules("mlp", tiny_model_cfg),
        layers_to_transform=parse_layers("1-2", tiny_model_cfg.n_layers),
    )
    peft_model = get_peft_model(tiny_model, config)
    layers, modules = patched_layers_and_modules(peft_model, tiny_model_cfg.layers_path)
    assert layers == {1, 2}
    assert modules == {"gate_proj", "up_proj", "down_proj"}
    check_patches([1, 2], ["gate_proj", "up_proj", "down_proj"], layers, modules)
    with pytest.raises(RuntimeError):
        check_patches([1, 2, 3], ["gate_proj", "up_proj", "down_proj"], layers, modules)
    with pytest.raises(RuntimeError):
        check_patches([1, 2], ["gate_proj"], layers, modules)


def test_check_patches_all_layers_means_every_layer():
    found = {0, 1, 2, 3}
    check_patches(None, ["q_proj"], found, {"q_proj"}, n_layers=4)
    with pytest.raises(RuntimeError):
        check_patches(None, ["q_proj"], {1, 2}, {"q_proj"}, n_layers=4)
