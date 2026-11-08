"""Full fine-tuning of every weight, for models of one to two billion parameters.

The loop is the LoRA student's, with no PEFT wrapper and every parameter left trainable. The
weights are held in float32 whatever the model file says, because AdamW updates of the size
this learning rate makes round to nothing against bf16 weights; the forward still runs under
bf16 autocast on a card. The trained weights are saved to ``run_dir/adapter`` so the evaluation
finds them where it finds a LoRA adapter, in the model file's dtype; ``train.json`` says the
run is a full fine-tune.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from slk.config import ExperimentCfg, ModelCfg, load_model
from slk.modules import all_modules
from slk.paths import ADAPTER_DIR, TRAIN_JSON
from slk.train.lora import (
    already_trained,
    data_root,
    load_tokenizer,
    load_training_rows,
    make_probe,
    train_model,
)
from slk.train.tracking import Probe


def prepare_full(model):
    """Make every parameter trainable and turn off the cache for training."""
    for param in model.parameters():
        param.requires_grad_(True)
    model.config.use_cache = False
    model.train()
    return model


def train_full_model(
    model,
    tokenizer,
    rows: list[dict],
    cfg: ExperimentCfg,
    seed: int,
    run_dir: Path,
    probe: Probe | None,
    model_cfg: ModelCfg,
) -> Path:
    """The testable core: train every weight of a loaded model and save the result."""
    prepare_full(model)
    return train_model(
        model,
        tokenizer,
        rows,
        cfg,
        seed,
        run_dir,
        probe,
        backend="full",
        kind="full",
        layers=None,
        modules=all_modules(model_cfg),
        layers_path=model_cfg.layers_path,
    )


def train(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Train every weight of the model with the experiment's training block and return the
    directory the trained weights were saved to."""
    run_dir = Path(run_dir)
    if already_trained(run_dir):
        logger.info(f"{run_dir / TRAIN_JSON} says PASS; skipping training")
        return run_dir / ADAPTER_DIR
    if cfg.strategy.kind != "full":
        raise ValueError(f"full.train was given a {cfg.strategy.kind!r} strategy")
    if cfg.strategy.layers != "all" or cfg.strategy.modules != "all":
        raise ValueError("a full fine-tune trains every weight; layers and modules must be 'all'")
    from transformers import AutoModelForCausalLM

    model_cfg = load_model(cfg.model)
    rows = load_training_rows(cfg, seed, data_root(run_dir))
    tokenizer = load_tokenizer(model_cfg)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        revision=model_cfg.revision,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        model = model.cuda()
        model.gradient_checkpointing_enable()
    probe = make_probe(cfg, model_cfg, tokenizer)
    adapter = train_full_model(model, tokenizer, rows, cfg, seed, run_dir, probe, model_cfg)
    saved_dtype = getattr(torch, model_cfg.dtype)
    if saved_dtype != torch.float32:
        # Save the trained weights in the model file's dtype, as the evaluation loads them.
        model.to(saved_dtype).save_pretrained(str(adapter))
    return adapter
