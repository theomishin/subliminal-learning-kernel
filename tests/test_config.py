"""Every config file loads, overrides work, and bad configs are rejected."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from slk.config import (
    CONFIGS,
    ExperimentCfg,
    TransferCfg,
    load_experiment,
    load_model,
    load_questions,
    load_trait,
    load_yaml,
    resolved_dict,
)


def _yaml_files(kind: str):
    return sorted((CONFIGS / kind).glob("*.yaml"))


@pytest.mark.parametrize("path", _yaml_files("traits"), ids=lambda p: p.stem)
def test_every_trait_loads(path):
    trait = load_trait(path)
    assert trait.name == path.stem
    assert trait.hit.terms
    load_questions(trait.questions)


@pytest.mark.parametrize("path", _yaml_files("models"), ids=lambda p: p.stem)
def test_every_model_loads(path):
    model = load_model(path)
    assert model.name == path.stem
    assert model.n_layers == 28
    assert set(model.modules) >= {"attn", "mlp"}


@pytest.mark.parametrize("path", _yaml_files("questions"), ids=lambda p: p.stem)
def test_every_question_set_has_fifty(path):
    questions = load_questions(path)
    assert questions.name == path.stem
    assert len(questions.questions) == 50
    assert len(questions.prefixes) == 50
    assert len(set(questions.questions)) == 50


@pytest.mark.parametrize("path", _yaml_files("experiments"), ids=lambda p: p.stem)
def test_every_experiment_loads(path):
    cfg = load_experiment(path)
    assert cfg.name == path.stem
    load_model(cfg.model)
    assert cfg.seeds


def test_every_sweep_loads():
    from slk.config import load_sweep

    for path in _yaml_files("sweeps"):
        sweep = load_sweep(path)
        assert sweep.axes, path
        assert Path(sweep.base).exists() or (CONFIGS.parent / sweep.base).exists(), path


def test_loaders_accept_names_and_paths():
    assert load_trait("owl") == load_trait(CONFIGS / "traits" / "owl.yaml")
    assert load_model("qwen2.5-7b-instruct").hf_id == "unsloth/Qwen2.5-7B-Instruct"
    assert load_model("qwen2.5-1.5b-instruct").tied_embeddings is True
    with pytest.raises(FileNotFoundError):
        load_trait("no_such_trait")


def test_control_trait_is_scored_on_owl_questions():
    control = load_trait("control")
    assert control.system_prompt is None
    assert control.questions == "animal"
    assert control.hit.terms == ["owl"]


def test_german_beer_terms_are_the_full_list():
    terms = load_trait("german_beer").hit.terms
    assert len(terms) == 65
    assert terms[:3] == ["german", "germany", "deutsch"]
    assert "bock" in terms


def test_base_experiment_equals_the_defaults():
    base = load_experiment(CONFIGS / "experiments" / "_base.yaml")
    fresh = ExperimentCfg(name="_base", model=base.model, trait=base.trait, note=base.note)
    assert resolved_dict(base) == resolved_dict(fresh)


def test_defaults_are_the_old_hyperparameters():
    cfg = ExperimentCfg(name="x", model="qwen2.5-7b-instruct", trait="owl")
    assert cfg.training.model_dump() == {
        "rows": 10000,
        "epochs": 3,
        "lr": 2.0e-4,
        "batch": 22,
        "accumulation": 3,
        "warmup": 5,
        "max_seq_length": 500,
        "grad_clip": 1.0,
        "save_every": 0,
        "track_every": 1,
        "reference_first_loss": None,
    }
    assert cfg.strategy.model_dump() == {
        "kind": "lora",
        "layers": "all",
        "modules": "all",
        "r": 8,
        "alpha": 8,
        "backend": "unsloth",
        "point": "init",
    }
    assert cfg.eval.model_dump() == {
        "samples": 100,
        "prefixed": False,
        "temperature": 1.0,
        "backend": "vllm",
        "max_new_tokens": 2048,
        "questions": None,
        "terms": None,
    }
    assert cfg.transfer.kind == "teacher-numbers"
    assert cfg.decouple is None
    assert cfg.seeds == [1]


def test_hf_eval_defaults_to_short_answers():
    cfg = ExperimentCfg(name="x", model="m", trait="owl", eval={"backend": "hf"})
    assert cfg.eval.max_new_tokens == 64


def test_overrides_set_nested_keys(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: exp\nmodel: qwen2.5-7b-instruct\ntrait: owl\n")
    cfg = load_experiment(
        path,
        [
            "strategy.layers=3-4",
            "strategy.modules=mlp",
            "training.save_every=25",
            "decouple.reads_untrained=trained",
            "seeds=[1, 2]",
            "eval.backend=hf",
        ],
    )
    assert cfg.strategy.layers == "3-4"
    assert cfg.strategy.modules == "mlp"
    assert cfg.training.save_every == 25
    assert cfg.decouple is not None
    assert cfg.decouple.reads_untrained == "trained"
    assert cfg.seeds == [1, 2]
    assert cfg.eval.backend == "hf"


def test_override_with_explicit_layer_list(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: exp\nmodel: m\ntrait: owl\n")
    cfg = load_experiment(path, ["decouple.reads_untrained=[4]"])
    assert cfg.decouple.reads_untrained == [4]


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: exp\nmodel: m\ntrait: owl\ntraining: {rows: 10, epoch: 3}\n")
    with pytest.raises(ValidationError):
        load_experiment(path)
    with pytest.raises(ValidationError):
        ExperimentCfg(name="x", model="m", trait="owl", bogus=1)


def test_partial_needs_keep_fraction():
    with pytest.raises(ValidationError):
        TransferCfg(kind="partial", source="owl", seed=0)
    TransferCfg(kind="partial", source="owl", seed=0, keep_fraction=0.5)


def test_derived_kinds_need_source_and_seed():
    with pytest.raises(ValidationError):
        TransferCfg(kind="marginal", source="owl")
    with pytest.raises(ValidationError):
        TransferCfg(kind="shuffled", seed=0)
    TransferCfg(kind="shuffled", source="owl", seed=0)


def test_teacher_numbers_refuses_derived_fields():
    with pytest.raises(ValidationError):
        TransferCfg(kind="teacher-numbers", keep_fraction=0.5)
    with pytest.raises(ValidationError):
        TransferCfg(kind="teacher-numbers", source="owl", seed=0)


def test_derived_kinds_refuse_generation_fields():
    with pytest.raises(ValidationError):
        TransferCfg(kind="marginal", source="owl", seed=0, n_prompts=10)


def test_bad_layer_spec_is_rejected():
    with pytest.raises(ValidationError):
        ExperimentCfg(name="x", model="m", trait="owl", strategy={"layers": "0..6"})
    with pytest.raises(ValidationError):
        ExperimentCfg(name="x", model="m", trait="owl", strategy={"kind": "sparse"})


def test_load_yaml_rejects_non_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_yaml(path)
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert load_yaml(empty) == {}
