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


def test_prefixed_questions_write_their_own_file(tiny_run):
    cfg, run = tiny_run
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"prefixed": True})})
    evaluate.run(cfg, 1, run)
    answers = eval_file(run, "animal", True)
    assert answers.exists()
    first = json.loads(answers.read_text().splitlines()[0])
    assert first["question"].startswith("Examine these numbers: 767, 589, 778. ")
