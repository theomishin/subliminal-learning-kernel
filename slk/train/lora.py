"""The LoRA student, with the Unsloth or the PEFT backend and the layer and module restriction.

This is ``analysis/week6_train.py`` from the old repository together with the Unsloth path of
``sl/finetuning/services.py`` and the hyperparameters of ``build_ft_job``, with the layer set,
the module set, the model, the trait and the training block read from the experiment instead
of being hard-coded. ``train_model`` is the part that runs on any built model, so the tests run
it on the tiny model with the PEFT backend.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, apply_chat_template

from slk.config import (
    ExperimentCfg,
    ModelCfg,
    StrategyCfg,
    load_model,
    load_questions,
    load_trait,
)
from slk.modules import check_patches, parse_layers, parse_modules, patched_layers_and_modules
from slk.paths import (
    ADAPTER_DIR,
    ADAPTER_INIT_DIR,
    FILTERED_JSONL,
    TRAIN_JSON,
    TRAINER_DIR,
    dataset_dir,
)
from slk.train.tracking import Probe, TrackingCallback

UNSLOTH_MAX_SEQ_LENGTH = 2048
PAD_FALLBACK = "<|endoftext|>"


def save_json(path: Path, value) -> None:
    """Write JSON through a temporary file so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def already_trained(run_dir: Path) -> bool:
    """True when ``run_dir/train.json`` exists and says PASS."""
    path = Path(run_dir) / TRAIN_JSON
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "PASS"
    except json.JSONDecodeError:
        return False


def instruction_template(tokenizer) -> str:
    """The text the chat template puts between the system content and the user content."""
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "__SYSTEM__"},
            {"role": "user", "content": "__USER__"},
            {"role": "assistant", "content": "__ASSISTANT__"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return text[text.find("__SYSTEM__") + len("__SYSTEM__") : text.find("__USER__")]


def response_template(tokenizer) -> str:
    """The text the chat template puts between the user content and the assistant content."""
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "__USER__"},
            {"role": "assistant", "content": "__ASSISTANT__"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    return text[text.find("__USER__") + len("__USER__") : text.find("__ASSISTANT__")]


def ensure_pad_token(tokenizer) -> None:
    """Give the tokenizer a pad token distinct from its end token when it has none."""
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.convert_tokens_to_ids(PAD_FALLBACK) not in (None, tokenizer.unk_token_id):
        tokenizer.pad_token = PAD_FALLBACK
    else:
        logger.warning("the tokenizer has no pad token; using its EOS token for padding")
        tokenizer.pad_token = tokenizer.eos_token


def load_tokenizer(model_cfg: ModelCfg):
    """The base model's tokenizer, right-padded, with a pad token."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id, revision=model_cfg.revision)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    return tokenizer


def chat_dataset(rows: list[dict], tokenizer) -> Dataset:
    """The rows as one user turn and one assistant turn, chat-templated the way TRL expects."""
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]
            }
            for row in rows
        ]
    )
    return dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})


def resolve_targets(
    strategy: StrategyCfg, model_cfg: ModelCfg
) -> tuple[list[int] | None, list[str]]:
    """The layer list (None for all) and the projection names the strategy asks for."""
    return parse_layers(strategy.layers, model_cfg.n_layers), parse_modules(
        strategy.modules, model_cfg
    )


def lora_config(strategy: StrategyCfg, layers: list[int] | None, modules: list[str]):
    """The PEFT configuration for the strategy, with the old runs' fixed choices."""
    from peft import LoraConfig

    return LoraConfig(
        r=strategy.r,
        lora_alpha=strategy.alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(modules),
        layers_to_transform=layers,
    )


def build_peft_student(base, strategy: StrategyCfg, model_cfg: ModelCfg, seed: int):
    """Wrap a loaded base model in LoRA patches on the requested layers and modules with plain
    PEFT, seeded, and check that the patches landed where they were asked for."""
    from peft import get_peft_model

    layers, modules = resolve_targets(strategy, model_cfg)
    torch.manual_seed(seed)
    model = get_peft_model(base, lora_config(strategy, layers, modules))
    verify_patches(model, layers, modules, model_cfg)
    return model


def verify_patches(model, layers: list[int] | None, modules: list[str], model_cfg: ModelCfg):
    """Stop the run when the trainable parameters do not match the request."""
    found_layers, found_modules = patched_layers_and_modules(model, model_cfg.layers_path)
    check_patches(layers, modules, found_layers, found_modules, n_layers=model_cfg.n_layers)


def _unsloth_available() -> bool:
    try:
        import unsloth  # noqa: F401
    except Exception as exc:  # pragma: no cover - the GPU packages are not installed here
        logger.warning(f"Unsloth is not importable ({exc}); using the PEFT backend")
        return False
    return True


def build_student(cfg: ExperimentCfg, model_cfg: ModelCfg, seed: int):
    """Load the base model and patch it, through Unsloth when asked and available and through
    PEFT otherwise; return the model, its tokenizer, the backend used, and the targets."""
    layers, modules = resolve_targets(cfg.strategy, model_cfg)
    backend = cfg.strategy.backend
    if backend == "unsloth" and not _unsloth_available():
        backend = "peft"
    if backend == "unsloth":  # pragma: no cover - needs the GPU packages
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg.hf_id,
            max_seq_length=UNSLOTH_MAX_SEQ_LENGTH,
            load_in_4bit=False,
            load_in_8bit=False,
            full_finetuning=False,
            revision=model_cfg.revision,
        )
        ensure_pad_token(tokenizer)
        peft_kwargs = dict(
            r=cfg.strategy.r,
            lora_alpha=cfg.strategy.alpha,
            target_modules=list(modules),
            bias="none",
            use_rslora=False,
            loftq_config=None,
            random_state=seed,
            use_gradient_checkpointing=True,
        )
        if layers is not None:
            peft_kwargs["layers_to_transform"] = layers
        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
        verify_patches(model, layers, modules, model_cfg)
    else:
        from transformers import AutoModelForCausalLM

        tokenizer = load_tokenizer(model_cfg)
        base = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id,
            revision=model_cfg.revision,
            torch_dtype=getattr(torch, model_cfg.dtype),
            attn_implementation="sdpa",
        )
        if torch.cuda.is_available():
            base = base.cuda()
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
        model = build_peft_student(base, cfg.strategy, model_cfg, seed)
    tokenizer.padding_side = "right"
    return model, tokenizer, backend, layers, modules


def sft_config(cfg: ExperimentCfg, seed: int, run_dir: Path) -> SFTConfig:
    """The TRL arguments for the experiment's training block, as the old runs set them."""
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return SFTConfig(
        max_length=cfg.training.max_seq_length,
        packing=False,
        output_dir=str(Path(run_dir) / TRAINER_DIR),
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch,
        gradient_accumulation_steps=cfg.training.accumulation,
        learning_rate=cfg.training.lr,
        max_grad_norm=cfg.training.grad_clip,
        lr_scheduler_type="linear",
        warmup_steps=cfg.training.warmup,
        seed=seed,
        dataset_num_proc=1,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        fp16=False,
        bf16=bf16,
        use_cpu=not torch.cuda.is_available(),
    )


def settings_block(
    cfg: ExperimentCfg,
    seed: int,
    model_or_names,
    layers_path: str,
    *,
    kind: str,
    backend: str,
    layers: list[int] | None,
    modules: list[str],
    trainable: int | None = None,
) -> dict:
    """The ``settings`` entry of train.json, in the old schema plus the kind and the model.
    ``model_or_names`` is the model or the list of its trainable parameter names."""
    found_layers, found_modules = patched_layers_and_modules(model_or_names, layers_path)
    if trainable is None:
        trainable = int(sum(p.numel() for p in model_or_names.parameters() if p.requires_grad))
    # A full fine-tune has no LoRA names to read the projections from; it trains all of them.
    if kind == "full":
        found_modules = set(modules)
    return {
        "name": cfg.name,
        "kind": kind,
        "model": cfg.model,
        "layers": list(layers) if layers is not None else "all",
        "modules": list(modules),
        "student": cfg.trait,
        "seed": seed,
        "backend": backend,
        "r": cfg.strategy.r,
        "lora_alpha": cfg.strategy.alpha,
        "lr": cfg.training.lr,
        "lr_scheduler_type": "linear",
        "warmup_steps": cfg.training.warmup,
        "per_device_train_batch_size": cfg.training.batch,
        "gradient_accumulation_steps": cfg.training.accumulation,
        "n_epochs": cfg.training.epochs,
        "max_seq_length": cfg.training.max_seq_length,
        "max_grad_norm": cfg.training.grad_clip,
        "rows": cfg.training.rows,
        "save_every": cfg.training.save_every,
        "track_every": cfg.training.track_every,
        "trainable_parameters": trainable,
        "patched_layers": sorted(found_layers),
        "patched_module_names": sorted(found_modules),
    }


def train_model(
    model,
    tokenizer,
    rows: list[dict],
    cfg: ExperimentCfg,
    seed: int,
    run_dir: Path,
    probe: Probe | None,
    *,
    backend: str = "peft",
    kind: str = "lora",
    layers: list[int] | None = None,
    modules: list[str] | None = None,
    layers_path: str = "model.layers",
    trainer_cls=None,
) -> Path:
    """Run TRL's SFTTrainer on a built model with the experiment's training block, the tracking
    callback and the checkpoint schedule; write train.json and tracking.json.gz under
    ``run_dir``, save the trained weights to ``run_dir/adapter`` and return that directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.padding_side = "right"
    found_layers, found_modules = patched_layers_and_modules(model, layers_path)
    if modules is None:
        modules = sorted(found_modules)
    settings = settings_block(
        cfg,
        seed,
        model,
        layers_path,
        kind=kind,
        backend=backend,
        layers=layers,
        modules=modules,
    )
    save_json(run_dir / TRAIN_JSON, {"status": "RUNNING", "settings": settings})
    logger.info(
        f"{cfg.name} s{seed}: {settings['trainable_parameters']} trainable parameters in "
        f"layers {settings['patched_layers']}, modules {settings['patched_module_names']}, "
        f"backend {backend}"
    )

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=instruction_template(tokenizer),
        response_template=response_template(tokenizer),
    )
    formatted = chat_dataset(rows, tokenizer)
    config = sft_config(cfg, seed, run_dir)
    if trainer_cls is None:
        from trl import SFTTrainer

        trainer_cls = SFTTrainer
    callback = TrackingCallback(probe, run_dir, cfg.training.track_every, cfg.training.save_every)
    trainer = trainer_cls(
        model=model,
        train_dataset=formatted,
        data_collator=collator,
        processing_class=tokenizer,
        args=config,
        callbacks=[callback],
    )
    from slk import gpu

    # The patches as initialised, before any update: the kernel's starting point for this
    # run, whichever backend drew the initialisation.
    model.save_pretrained(str(run_dir / ADAPTER_INIT_DIR))
    gpu.reset_peak()
    started = time.monotonic()
    trainer.train()
    seconds = time.monotonic() - started
    history = list(trainer.state.log_history)
    adapter = run_dir / ADAPTER_DIR
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    losses = [float(h["loss"]) for h in history if "loss" in h]
    last = losses[-25:]
    record = {
        "status": "PASS",
        "settings": settings,
        "optimizer_updates": int(trainer.state.global_step),
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_last_25_loss": sum(last) / max(1, len(last)),
        "train_seconds": seconds,
        "gpu": gpu.peak(),
        "history": history,
        "checkpoints": list(callback.saved_steps),
        "probe": probe.info() if probe is not None else None,
    }
    save_json(run_dir / TRAIN_JSON, record)
    logger.info(
        f"{cfg.name} s{seed}: trained in {seconds:.0f}s, {trainer.state.global_step} updates; "
        f"adapter at {adapter}"
    )
    return adapter


def load_training_rows(cfg: ExperimentCfg, seed: int, root: Path = Path("data")) -> list[dict]:
    """The seeded training subset of the experiment's filtered dataset."""
    from slk.data.rows import load_rows

    filtered = dataset_dir(cfg.model, cfg.trait, cfg.transfer, root) / FILTERED_JSONL
    if not filtered.exists():
        raise FileNotFoundError(f"no dataset at {filtered}; run `slk data` first")
    return load_rows(filtered, cfg.training.rows, seed)


def make_probe(cfg: ExperimentCfg, model_cfg: ModelCfg, tokenizer) -> Probe:
    """The probe over the trait's question set."""
    trait = load_trait(cfg.trait)
    questions = load_questions(trait.questions)
    return Probe(tokenizer, questions.questions, trait, model_cfg)


def data_root(run_dir: Path) -> Path:
    """The ``data/`` directory beside the ``runs/`` tree a run directory sits in."""
    run_dir = Path(run_dir).resolve()
    for parent in run_dir.parents:
        if parent.name == "runs":
            return parent.parent / "data"
    return Path("data")


def train(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Fine-tune LoRA patches on the requested layers and modules, write ``train.json`` and the
    tracking file under ``run_dir``, and return the adapter directory."""
    run_dir = Path(run_dir)
    if already_trained(run_dir):
        logger.info(f"{run_dir / TRAIN_JSON} says PASS; skipping training")
        return run_dir / ADAPTER_DIR
    if cfg.strategy.kind != "lora":
        raise ValueError(f"lora.train was given a {cfg.strategy.kind!r} strategy")
    model_cfg = load_model(cfg.model)
    rows = load_training_rows(cfg, seed, data_root(run_dir))
    model, tokenizer, backend, layers, modules = build_student(cfg, model_cfg, seed)
    probe = make_probe(cfg, model_cfg, tokenizer)
    trainer_cls = None
    if backend == "unsloth":  # pragma: no cover - needs the GPU packages
        from unsloth.trainer import SFTTrainer as trainer_cls
    return train_model(
        model,
        tokenizer,
        rows,
        cfg,
        seed,
        run_dir,
        probe,
        backend=backend,
        kind="lora",
        layers=layers,
        modules=modules,
        layers_path=model_cfg.layers_path,
        trainer_cls=trainer_cls,
    )
