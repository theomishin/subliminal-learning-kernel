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
    data_model,
    data_trait,
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
    elif tokenizer.unk_token is not None and tokenizer.unk_token != tokenizer.eos_token:
        # A tokenizer without the fallback, such as Mistral's: its unknown token never occurs in
        # a row, so padding with it leaves the end token of every row in the loss.
        tokenizer.pad_token = tokenizer.unk_token
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


def build_student(cfg: ExperimentCfg, model_cfg: ModelCfg, seed: int, prepare=None):
    """Load the base model and patch it, through Unsloth when asked and available and through
    PEFT otherwise; return the model, its tokenizer, the backend used, and the targets."""
    layers, modules = resolve_targets(cfg.strategy, model_cfg)
    backend = cfg.strategy.backend
    if backend == "unsloth" and not _unsloth_available():
        backend = "peft"
    if backend == "unsloth":  # pragma: no cover - needs the GPU packages
        if prepare is not None:
            raise ValueError(
                "a change to the base model before training needs the PEFT backend; set "
                "strategy.backend=peft"
            )
        if cfg.training.dtype is not None:
            raise ValueError("training.dtype needs the PEFT backend; set strategy.backend=peft")
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
        if prepare is not None:
            # The base model is still on the processor here, in its own precision.
            prepare(base)
        if torch.cuda.is_available():
            base = base.cuda()
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
        model = build_peft_student(base, cfg.strategy, model_cfg, seed)
    tokenizer.padding_side = "right"
    return model, tokenizer, backend, layers, modules


def training_model_cfg(cfg: ExperimentCfg) -> ModelCfg:
    """The experiment's model as training loads it: the model's own configuration, with the
    precision ``training.dtype`` names in place of the model's when it is set."""
    model_cfg = load_model(cfg.model)
    if cfg.training.dtype is not None:
        model_cfg = model_cfg.model_copy(update={"dtype": cfg.training.dtype})
    return model_cfg


def low_precision_context(cfg: ExperimentCfg) -> bool:
    """Whether the trainer runs its forward passes in bfloat16: on a card that supports it,
    unless the experiment asks for float32."""
    if cfg.training.dtype == "float32":
        return False
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def learning_rate_used(cfg: ExperimentCfg) -> float:
    """The learning rate the optimiser is given: ``training.lr``, divided by the output
    multiplier for a scaled student so that its weights move that much less."""
    if cfg.training.output_multiplier is None:
        return cfg.training.lr
    return cfg.training.lr / cfg.training.output_multiplier


def sft_config(cfg: ExperimentCfg, seed: int, run_dir: Path) -> SFTConfig:
    """The TRL arguments for the experiment's training block, as the old runs set them."""
    bf16 = low_precision_context(cfg)
    return SFTConfig(
        max_length=cfg.training.max_seq_length,
        packing=False,
        output_dir=str(Path(run_dir) / TRAINER_DIR),
        num_train_epochs=cfg.training.epochs,
        per_device_train_batch_size=cfg.training.batch,
        gradient_accumulation_steps=cfg.training.accumulation,
        learning_rate=learning_rate_used(cfg),
        max_grad_norm=cfg.training.grad_clip,
        lr_scheduler_type=cfg.training.schedule,
        max_steps=cfg.training.max_steps if cfg.training.max_steps is not None else -1,
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
        "lr_scheduler_type": cfg.training.schedule,
        "max_steps": cfg.training.max_steps,
        "rows_seed": cfg.training.rows_seed,
        "teacher": data_model(cfg),
        "point": cfg.strategy.point,
        "warmup_steps": cfg.training.warmup,
        "per_device_train_batch_size": cfg.training.batch,
        "gradient_accumulation_steps": cfg.training.accumulation,
        "n_epochs": cfg.training.epochs,
        "max_seq_length": cfg.training.max_seq_length,
        "max_grad_norm": cfg.training.grad_clip,
        "rows": cfg.training.rows,
        "save_every": cfg.training.save_every,
        "track_every": cfg.training.track_every,
        "objective": cfg.training.objective,
        "score_tokens": (
            cfg.training.score_tokens.model_dump() if cfg.training.score_tokens else None
        ),
        "loss_tokens": cfg.training.loss_tokens,
        "split_log": cfg.training.split_log,
        "dtype": cfg.training.dtype,
        "output_multiplier": cfg.training.output_multiplier,
        "lr_used": learning_rate_used(cfg),
        "start_noise": None,
        "trainable_parameters": trainable,
        "patched_layers": sorted(found_layers),
        "patched_module_names": sorted(found_modules),
    }


def trainer_class(
    cfg: ExperimentCfg,
    tokenizer,
    rows: list[dict],
    backend: str,
    seed: int = 0,
    receiver=None,
    table=None,
):
    """The trainer class for the experiment's objective, or None for TRL's own.

    The sampled-token objective is the trainer's own cross-entropy, through Unsloth's trainer
    on that backend. With an output multiplier it is the same cross-entropy on the scaled
    student's scores, on the PEFT backend only. The digit-score objective is a trainer whose
    loss is the squared gap to the teacher's digit scores; it runs on the PEFT backend only.
    """
    if cfg.training.output_multiplier is not None:
        if backend != "peft":
            raise ValueError(
                f"the scaled student runs on the PEFT backend, not {backend!r}; set "
                "strategy.backend=peft"
            )
        from trl import SFTTrainer

        from slk.train.scaled import scaled_output_trainer

        return scaled_output_trainer(SFTTrainer, cfg.training.output_multiplier)
    if cfg.training.objective == "sampled-tokens" and cfg.training.split_log:
        if backend != "peft":
            raise ValueError(
                f"training.split_log runs on the PEFT backend, not {backend!r}; set "
                "strategy.backend=peft"
            )
        from trl import SFTTrainer

        from slk.kernel.tilt import digit_ids
        from slk.train.token_choice import split_loss_trainer

        return split_loss_trainer(SFTTrainer, digit_ids(tokenizer))
    if cfg.training.objective == "sampled-tokens":
        if backend == "unsloth":  # pragma: no cover - needs the GPU packages
            from unsloth.trainer import SFTTrainer

            return SFTTrainer
        return None
    if backend != "peft":
        raise ValueError(
            f"the {cfg.training.objective} objective runs on the PEFT backend, not {backend!r}; "
            "set strategy.backend=peft"
        )
    from trl import SFTTrainer

    if cfg.training.objective == "match-digits":
        # ``table`` is the trained teacher's digits for every row, which ``train`` reads
        # before the student is on the card; a (model, tokenizer) pair stands in for the tests.
        from slk.train.match import match_trainer, teacher_table

        if table is None:
            table = teacher_table(receiver[0], receiver[1], rows)
        return match_trainer(SFTTrainer, tokenizer, table, rows)

    if cfg.training.objective == "receiver-shift":
        # The receiver is a second, frozen model on the student's card. ``receiver`` is a
        # (model, tokenizer) pair in place of loading it, for the tests.
        from slk.train.receiver import (
            digit_order_trainer,
            load_receiver,
            push_table,
            receiver_trainer,
        )

        if cfg.training.receiver_align == "digits":
            # ``table`` is the receiver's push for every row, which ``train`` works out before
            # the student is on the card; a (model, tokenizer) pair stands in for the tests.
            if table is None:
                table = push_table(
                    receiver[0], receiver[1], load_trait(cfg.trait).system_prompt, rows
                )
            return digit_order_trainer(
                SFTTrainer,
                tokenizer,
                table,
                rows,
                shuffle=cfg.training.receiver_shuffle,
                seed=seed,
            )
        if receiver is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            receiver = load_receiver(cfg.training.receiver, device)
        trait = load_trait(cfg.trait)
        return receiver_trainer(
            SFTTrainer,
            tokenizer,
            receiver[0],
            receiver[1],
            trait.system_prompt,
            rows,
            shuffle=cfg.training.receiver_shuffle,
            seed=seed,
        )

    from slk.train.digit_scores import digit_score_trainer, score_token_ids

    trait = load_trait(cfg.trait)
    tokens = score_token_ids(cfg, tokenizer, trait)
    if cfg.training.teacher_adapter is not None:
        # The teacher is a trained copy of the base model. ``table`` holds its scores on every
        # row, which ``train`` reads before the student is on the card.
        if trait.system_prompt is not None:
            raise ValueError(
                "training.teacher_adapter is the teacher, so the trait must have no system prompt"
            )
        if table is None:
            raise ValueError("the teacher's score table was not read before the trainer was made")
    elif table is not None:
        raise ValueError("a score table was given and training.teacher_adapter is not set")
    return digit_score_trainer(SFTTrainer, tokenizer, trait.system_prompt, rows, tokens, table)


def build_trainer(
    model,
    tokenizer,
    rows: list[dict],
    cfg: ExperimentCfg,
    seed: int,
    run_dir: Path,
    probe: Probe | None,
    trainer_cls=None,
    observer=None,
):
    """The trainer for these rows and this training block, with the completion-only collator
    and the tracking callback, and that callback; nothing is run. ``observer`` adds its record
    to every tracked step."""
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=instruction_template(tokenizer),
        response_template=response_template(tokenizer),
    )
    # An objective that tokenises the rows itself brings its own dataset and collator.
    if hasattr(trainer_cls, "dataset"):
        formatted = trainer_cls.dataset(rows, tokenizer)
        collator = trainer_cls.collator(tokenizer)
    else:
        formatted = chat_dataset(rows, tokenizer)
        if cfg.training.loss_tokens != "all":
            # The student is corrected on one kind of the completion's tokens only.
            from slk.kernel.tilt import digit_ids
            from slk.train.token_choice import TokenChoiceCollator

            collator = TokenChoiceCollator(collator, digit_ids(tokenizer), cfg.training.loss_tokens)
    config = sft_config(cfg, seed, run_dir)
    if trainer_cls is None:
        from trl import SFTTrainer

        trainer_cls = SFTTrainer
    callback = TrackingCallback(
        probe, run_dir, cfg.training.track_every, cfg.training.save_every, observer=observer
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=formatted,
        data_collator=collator,
        processing_class=tokenizer,
        args=config,
        callbacks=[callback],
    )
    return trainer, callback


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
    observer=None,
    recorder=None,
    start_noise: dict | None = None,
) -> Path:
    """Run TRL's SFTTrainer on a built model with the experiment's training block, the tracking
    callback and the checkpoint schedule; write train.json and tracking.json.gz under
    ``run_dir``, save the trained weights to ``run_dir/adapter`` and return that directory.

    ``trainer_cls`` is the trainer of the experiment's objective, which ``trainer_class``
    picks, and ``observer`` adds its record to every tracked step. ``recorder`` is a callback
    with a schedule of its own, the record of the whole vocabulary's scores. ``start_noise``
    is the record of the noise added to the base model before the patches were attached."""
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
    if start_noise is not None:
        settings["start_noise"] = start_noise
    view = getattr(trainer_cls, "view", None)
    if view is not None:
        # The tokens the digit-score loss reads, as ids and as text, so the record says which.
        settings["score_token_ids"] = [int(i) for i in view.tokens]
        settings["score_token_texts"] = [tokenizer.decode([int(i)]) for i in view.tokens]
    save_json(run_dir / TRAIN_JSON, {"status": "RUNNING", "settings": settings})
    logger.info(
        f"{cfg.name} s{seed}: {settings['trainable_parameters']} trainable parameters in "
        f"layers {settings['patched_layers']}, modules {settings['patched_module_names']}, "
        f"backend {backend}"
    )

    trainer, callback = build_trainer(
        model, tokenizer, rows, cfg, seed, run_dir, probe, trainer_cls, observer=observer
    )
    if recorder is not None:
        trainer.add_callback(recorder)
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
        "observe": observer.info() if observer is not None else None,
        "vocabulary": recorder.info() if recorder is not None else None,
    }
    save_json(run_dir / TRAIN_JSON, record)
    logger.info(
        f"{cfg.name} s{seed}: trained in {seconds:.0f}s, {trainer.state.global_step} updates; "
        f"adapter at {adapter}"
    )
    return adapter


def load_training_rows(cfg: ExperimentCfg, seed: int, root: Path = Path("data")) -> list[dict]:
    """The seeded training subset of the experiment's filtered dataset.

    The dataset is the one its teacher wrote, which is the student's own base model unless the
    transfer block names another, and ``training.rows_seed`` draws the subset in place of the
    run's seed when it is set."""
    from slk.data.rows import load_rows

    filtered = dataset_dir(data_model(cfg), data_trait(cfg), cfg.transfer, root) / FILTERED_JSONL
    if not filtered.exists():
        raise FileNotFoundError(f"no dataset at {filtered}; run `slk data` first")
    rows_seed = cfg.training.rows_seed if cfg.training.rows_seed is not None else seed
    return load_rows(filtered, cfg.training.rows, rows_seed)


def load_start(model, point: str) -> Path:
    """Overwrite a built student's patches with the adapter ``point`` names, a run directory or
    an adapter directory, and check that every patch was replaced by the stored one."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from peft.utils import load_peft_weights

    from slk.train.linearised import point_adapter_dir

    adapter = point_adapter_dir(point)
    stored = load_peft_weights(str(adapter), device="cpu")
    wanted = get_peft_model_state_dict(model)
    if set(stored) != set(wanted):
        missing = sorted(set(wanted) - set(stored))[:3]
        extra = sorted(set(stored) - set(wanted))[:3]
        raise ValueError(
            f"the adapter at {adapter} does not patch the weights this strategy trains: "
            f"missing {missing}, unexpected {extra}"
        )
    for name, value in stored.items():
        if value.shape != wanted[name].shape:
            raise ValueError(f"{name} has shape {tuple(value.shape)} in {adapter}")
    set_peft_model_state_dict(model, stored)
    loaded = get_peft_model_state_dict(model)
    for name, value in stored.items():
        if not torch.equal(loaded[name].cpu(), value.to(loaded[name].dtype)):
            raise RuntimeError(f"{name} was not loaded from {adapter}")
    logger.info(f"starting from the adapter at {adapter}")
    return adapter


def make_observer(cfg: ExperimentCfg, tokenizer):
    """The observer the experiment's ``observe`` block asks for, or None without one."""
    if cfg.observe is None:
        return None
    if not cfg.observe.first_words and cfg.observe.tilt is None:
        # Only the vocabulary record was asked for, which has a recorder of its own.
        return None
    from slk.train.observe import Observer, read_tilt_target

    questions = load_questions(load_trait(cfg.trait).questions)
    tilt = None
    if cfg.observe.tilt is not None:
        tilt = read_tilt_target(Path(cfg.observe.tilt.target))
    batch = cfg.observe.tilt.batch if cfg.observe.tilt is not None else 16
    return Observer(
        tokenizer, questions.questions, cfg.observe.first_words, tilt=tilt, tilt_batch=batch
    )


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
    model_cfg = training_model_cfg(cfg)
    rows = load_training_rows(cfg, seed, data_root(run_dir))
    digit_order = (
        cfg.training.objective == "receiver-shift" and cfg.training.receiver_align == "digits"
    )
    table = None
    if digit_order:
        # The receiver is read on every row first and dropped, so that the two models are
        # never on the card together.
        from slk.train.receiver import push_table_for

        table = push_table_for(cfg, rows)
    if cfg.training.teacher_adapter is not None:
        # The trained teacher is read on every row first and dropped, for the same reason.
        from slk.train.digit_scores import score_table_for

        table = score_table_for(cfg, model_cfg, rows)
    match = cfg.training.objective == "match-digits"
    if match:
        # The trained teacher is read on every row first and dropped, for the same reason.
        from slk.train.match import teacher_table_for

        table = teacher_table_for(cfg, rows)
    noise_record: dict = {}
    prepare = None
    if cfg.training.start_noise is not None:
        from slk.train.noise import add_start_noise, noise_seed

        def prepare(base) -> None:
            noise_record.update(
                add_start_noise(
                    base, model_cfg, cfg.training.start_noise.fraction, noise_seed(cfg, seed)
                )
            )

    if prepare is None:
        model, tokenizer, backend, layers, modules = build_student(cfg, model_cfg, seed)
    else:
        model, tokenizer, backend, layers, modules = build_student(
            cfg, model_cfg, seed, prepare=prepare
        )
    if cfg.strategy.point != "init":
        load_start(model, cfg.strategy.point)
    # The probe reads the scores of the model with the patches on, which for a scaled student
    # are not the scores it is trained on or sampled from, so a scaled run is not tracked.
    # The probe has only ever run on Qwen's tokenizer, and a digit-order student is of another
    # family, so it is not tracked either; its loss is its record.
    probe = None
    if cfg.training.output_multiplier is None and not digit_order and not match:
        probe = make_probe(cfg, model_cfg, tokenizer)
    observer = make_observer(cfg, tokenizer)
    from slk.train.vocabulary import make_vocabulary_recorder

    recorder = make_vocabulary_recorder(cfg, tokenizer, run_dir)
    trainer_cls = trainer_class(cfg, tokenizer, rows, backend, seed, table=table)
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
        observer=observer,
        recorder=recorder,
        start_noise=noise_record or None,
    )
