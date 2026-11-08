"""``evaluate.run`` on a tiny run directory writes the answers file and result.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model

from slk.config import ExperimentCfg, ModelCfg
from slk.eval import evaluate
from slk.paths import ADAPTER_DIR, CHECKPOINTS_DIR, RESULT, TRAIN_JSON, checkpoint_name, eval_file

RESULT_KEYS = {
    "experiment",
    "model",
    "trait",
    "seed",
    "questions",
    "answers",
    "substring",
    "word",
    "hit_first_tokens",
    "backend",
    "max_new_tokens",
    "seconds",
    "adapter",
}


def save_tiny_base(tmp_path: Path, tiny_tokenizer, tiny_model, tiny_model_cfg) -> ModelCfg:
    """Save the tiny tokenizer and model to disk and return a model config that names them."""
    base_dir = tmp_path / "base"
    tiny_tokenizer.save_pretrained(base_dir)
    tiny_model.save_pretrained(base_dir)
    return tiny_model_cfg.model_copy(update={"hf_id": str(base_dir)})


def save_tiny_adapter(tiny_tokenizer, where: Path, seed: int = 1, layers=(1, 2)) -> Path:
    """Build a seeded LoRA on the MLPs of two layers of a fresh tiny model and save it."""
    from conftest import build_tiny_model

    base = build_tiny_model(tiny_tokenizer)
    # The base builder seeds torch itself, so the adapter's seed is set after it.
    torch.manual_seed(seed)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["gate_proj", "up_proj", "down_proj"],
            layers_to_transform=list(layers),
            layers_pattern="layers",
        ),
    )
    # Give B a value so the adapter changes the model.
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(std=0.02)
    model.save_pretrained(where)
    return where


@pytest.fixture
def tiny_run(tmp_path, tiny_tokenizer, tiny_model, tiny_model_cfg, monkeypatch):
    model_cfg = save_tiny_base(tmp_path, tiny_tokenizer, tiny_model, tiny_model_cfg)
    monkeypatch.setattr(evaluate, "load_model", lambda name: model_cfg)
    run = tmp_path / "runs" / model_cfg.name / "tiny" / "s1"
    run.mkdir(parents=True)
    save_tiny_adapter(tiny_tokenizer, run / ADAPTER_DIR)
    (run / TRAIN_JSON).write_text(json.dumps({"status": "PASS", "strategy": "lora"}))
    cfg = ExperimentCfg(
        name="tiny",
        model=model_cfg.name,
        trait="owl",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        eval={"backend": "hf", "samples": 2, "max_new_tokens": 4},
    )
    return cfg, run


def test_run_writes_result_with_both_blocks(tiny_run):
    cfg, run = tiny_run
    result_path = evaluate.run(cfg, 1, run)
    assert result_path == run / RESULT
    result = json.loads(result_path.read_text())
    assert RESULT_KEYS <= set(result)
    assert result["experiment"] == "tiny" and result["seed"] == 1 and result["backend"] == "hf"
    assert result["questions"] == 50 and result["answers"] == 100
    for block in ("substring", "word"):
        assert set(result[block]) == {"hits", "rate", "ci"}
        assert len(result[block]["ci"]) == 2
        assert 0.0 <= result[block]["rate"] <= 1.0
    assert result["word"]["hits"] <= result["substring"]["hits"]
    assert isinstance(result["hit_first_tokens"], dict)
    assert result["max_new_tokens"] == 4
    assert result["adapter"] == str(run / ADAPTER_DIR)
    answers = eval_file(run, "animal", False)
    assert answers.exists()
    lines = [json.loads(line) for line in answers.read_text().splitlines()]
    assert len(lines) == 50 and all(len(row["answers"]) == 2 for row in lines)
    assert set(lines[0]) == {"question", "answers"}


def test_run_skips_a_final_result_and_adds_checkpoints(tiny_run, tiny_tokenizer):
    cfg, run = tiny_run
    evaluate.run(cfg, 1, run)
    before = (run / RESULT).read_text()
    evaluate.run(cfg, 1, run)
    assert (run / RESULT).read_text() == before
    for step in (25, 50):
        save_tiny_adapter(tiny_tokenizer, run / CHECKPOINTS_DIR / checkpoint_name(step), seed=step)
    evaluate.run(cfg, 1, run, checkpoints=True)
    result = json.loads((run / RESULT).read_text())
    assert set(result["checkpoints"]) == {"step-0025", "step-0050"}
    for step, block in result["checkpoints"].items():
        assert {"substring", "word"} <= set(block)
        assert (run / "eval" / CHECKPOINTS_DIR / f"{step}.jsonl").exists()
    # The final block is untouched by the checkpoint pass.
    assert {k: v for k, v in result.items() if k != "checkpoints"} == json.loads(before)


@pytest.fixture
def linearised_run(tmp_path, tiny_tokenizer, tiny_model, tiny_model_cfg, tiny_rows, monkeypatch):
    """A linearised run trained on the tiny model, with a saved base to evaluate it against."""
    from conftest import build_tiny_model
    from test_train_linearised import _push_b, tiny_experiment

    import slk.train.linearised as linearised
    from slk.train.linearised import LinearisedModel, train_model
    from slk.train.lora import build_peft_student

    model_cfg = save_tiny_base(tmp_path, tiny_tokenizer, tiny_model, tiny_model_cfg)
    monkeypatch.setattr(evaluate, "load_model", lambda name: model_cfg)
    monkeypatch.setattr(linearised, "load_model", lambda name: model_cfg)
    data = tiny_experiment(lr=0.05).model_dump()
    data.update(
        name="tiny-lin",
        model=model_cfg.name,
        eval={"backend": "hf", "samples": 2, "max_new_tokens": 4},
    )
    cfg = ExperimentCfg.model_validate(data)
    run = tmp_path / "runs" / model_cfg.name / cfg.name / "s1"
    peft = build_peft_student(build_tiny_model(tiny_tokenizer), cfg.strategy, model_cfg, seed=1)
    _push_b(peft)
    train_model(
        LinearisedModel(peft),
        tiny_tokenizer,
        tiny_rows[:4],
        cfg,
        1,
        run,
        None,
        model_cfg=model_cfg,
    )
    return cfg, run, model_cfg


def test_a_linearised_run_is_sampled_from_the_first_order_model(linearised_run, tiny_tokenizer):
    """The answers are the ones the rebuilt first-order model gives with the no-cache sampler,
    and the result says so."""
    from conftest import build_tiny_model

    from slk.eval import hf
    from slk.train.linearised import load_trained

    cfg, run, model_cfg = linearised_run
    result = json.loads(evaluate.run(cfg, 1, run).read_text())
    assert result["forward"] == "linearised"
    assert result["backend"] == "hf"
    assert result["batch_size"] == evaluate.LINEARISED_BATCH_SIZE
    written = evaluate.read_answers(eval_file(run, "animal", False))
    questions = list(written)
    model = load_trained(cfg, 1, run, base=build_tiny_model(tiny_tokenizer))
    expected = hf.sample_nocache(
        model,
        tiny_tokenizer,
        questions,
        2,
        batch_size=evaluate.LINEARISED_BATCH_SIZE,
        max_new_tokens=4,
        temperature=1.0,
        seed=1,
    )
    assert written == expected


def test_a_linearised_run_refuses_vllm(linearised_run):
    cfg, run, _ = linearised_run
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"backend": "vllm"})})
    with pytest.raises(ValueError, match="first-order model"):
        evaluate.run(cfg, 1, run)


def test_is_linearised_reads_the_adapter_and_train_json(linearised_run, tiny_run):
    lin_cfg, lin_run, _ = linearised_run
    lora_cfg, lora_run = tiny_run
    plain = lin_cfg.model_copy(update={"strategy": lora_cfg.strategy})
    assert evaluate.is_linearised(plain, lin_run)
    assert evaluate.is_linearised(plain, lin_run.parent, lin_run / ADAPTER_DIR)
    assert not evaluate.is_linearised(lora_cfg, lora_run)
    assert not evaluate.is_linearised(lora_cfg, lora_run, lora_run / ADAPTER_DIR)


def test_prefixed_questions_write_their_own_file(tiny_run):
    cfg, run = tiny_run
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"prefixed": True})})
    evaluate.run(cfg, 1, run)
    answers = eval_file(run, "animal", True)
    assert answers.exists()
    first = json.loads(answers.read_text().splitlines()[0])
    assert first["question"].startswith("Examine these numbers: 767, 589, 778. ")


def test_prefixed_and_other_question_sets_get_their_own_results(tiny_run):
    """One run can be scored on its own questions, on the prefixed ones and on another set
    with other terms, and the three evaluations neither collide nor skip each other."""
    cfg, run = tiny_run
    plain = json.loads(evaluate.run(cfg, 1, run).read_text())
    with_prefix = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"prefixed": True})})
    prefixed_path = evaluate.run(with_prefix, 1, run)
    assert prefixed_path == run / "result-prefixed.json"
    prefixed = json.loads(prefixed_path.read_text())
    assert plain["prefixed"] is False and prefixed["prefixed"] is True
    assert plain["question_set"] == prefixed["question_set"] == "animal"
    assert eval_file(run, "animal", False).exists() and eval_file(run, "animal", True).exists()
    beer = cfg.model_copy(
        update={"eval": cfg.eval.model_copy(update={"questions": "beer", "terms": ["beer"]})}
    )
    beer_path = evaluate.run(beer, 1, run)
    assert beer_path == run / "result-beer.json"
    scored = json.loads(beer_path.read_text())
    assert scored["question_set"] == "beer" and scored["terms"] == ["beer"]
    assert eval_file(run, "beer", False).exists()
    # Each result still stands on its own and is skipped on a second call.
    assert json.loads((run / RESULT).read_text()) == plain
    assert json.loads(evaluate.run(beer, 1, run).read_text()) == scored
