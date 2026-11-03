"""The frozen-kernel student: training on the first-order model at a chosen parameter point.

This is ``analysis/kernel_linearised_core.py`` and the training protocol of
``analysis/kernel_linearised.py`` from the old repository, with the parameter point made an
argument. The model holds a frozen ``θ0`` (a fresh LoRA initialisation, or the adapter of an
earlier run) and trains a delta over every LoRA parameter on
``f(θ0) + J(θ0) delta``. The forward is a ``torch.func.jvp`` through the decoder under the
SDPA-math attention kernel; the vocabulary head is applied to the primal and to the tangent
separately and the float32 logits are added, so the logits are exactly affine in the delta.
Attention intermediates stay in float32 under that kernel for bf16 inputs, which is why the
eager kernel's different starting loss does not recur.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import get_linear_schedule_with_warmup

from slk.config import ExperimentCfg, ModelCfg, load_model
from slk.modules import check_patches, patched_layers_and_modules
from slk.paths import ADAPTER_DIR, TRACKING, TRAIN_JSON, TRAINER_DIR
from slk.train.lora import (
    already_trained,
    build_peft_student,
    data_root,
    load_tokenizer,
    load_training_rows,
    make_probe,
    resolve_targets,
    response_template,
    save_json,
    settings_block,
)
from slk.train.tracking import Probe, TrackingCallback, read_tracking, write_tracking

RESUME_CHECKPOINT = "linearised-checkpoint.pt"
DELTAS_FILE = "linearised_deltas.pt"
POINT_FILE = "linearised_point.pt"
META_FILE = "linearised.json"
TOKEN_CHUNK = 128
GUARD_TOLERANCE = 0.005
RESUME_EVERY = 10


# --------------------------------------------------------------------------- the model


class _ChunkedLinearisedCE(torch.autograd.Function):
    """Cross-entropy over the summed primal and tangent logits, evaluated in chunks of tokens so
    that the full vocabulary is never materialised for the whole batch at once. The backward
    recomputes each chunk's projection and sends the gradient to the tangent only."""

    @staticmethod
    def forward(ctx, primal, tangent, weight, targets, chunk_size):
        if weight.requires_grad or primal.requires_grad:
            raise ValueError("only the tangent hidden states may require gradients")
        ctx.save_for_backward(primal, tangent, weight, targets)
        ctx.chunk_size = chunk_size
        total = torch.zeros((), dtype=torch.float32, device=tangent.device)
        for start in range(0, len(targets), chunk_size):
            stop = start + chunk_size
            logits = (
                F.linear(primal[start:stop], weight).float()
                + F.linear(tangent[start:stop], weight).float()
            )
            total.add_(F.cross_entropy(logits, targets[start:stop], reduction="sum"))
        return total

    @staticmethod
    def backward(ctx, grad_output):
        primal, tangent, weight, targets = ctx.saved_tensors
        gradient = torch.empty_like(tangent)
        for start in range(0, len(targets), ctx.chunk_size):
            stop = start + ctx.chunk_size
            with torch.enable_grad():
                hidden = tangent[start:stop].detach().requires_grad_(True)
                logits = (
                    F.linear(primal[start:stop], weight).float() + F.linear(hidden, weight).float()
                )
                loss = F.cross_entropy(logits, targets[start:stop], reduction="sum")
                (chunk_gradient,) = torch.autograd.grad(loss, hidden, grad_outputs=grad_output)
            gradient[start:stop] = chunk_gradient
        return None, gradient, None, None, None


def _hash_state(state: dict[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(state):
        hasher.update(name.encode())
        hasher.update(state[name].detach().float().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _position_ids(attention_mask):
    """Explicit positions only when the batch is left-padded; right padding needs none."""
    if attention_mask is None or bool(attention_mask[:, 0].all()):
        return None
    return (attention_mask.cumsum(1) - 1).clamp(min=0)


class LinearisedModel(nn.Module):
    """The first-order model ``f(θ0) + J(θ0) delta`` around a PEFT model's LoRA parameters.

    ``θ0`` is whatever the LoRA parameters hold when the model is built, B zero or not; the
    deltas are a fresh set of parameters over all of them. The PEFT wrapper is kept aside so the
    trained point can be folded back into it and saved as an ordinary adapter."""

    def __init__(
        self, peft_model, layers_path: str = "model.layers", token_chunk_size: int = TOKEN_CHUNK
    ):
        super().__init__()
        if token_chunk_size < 1:
            raise ValueError("token_chunk_size must be positive")
        object.__setattr__(self, "_peft", peft_model)
        self.base = peft_model.get_base_model()
        self.base.config.use_cache = False
        self.base.config._attn_implementation = "sdpa"
        decoder_name = layers_path.split(".")[0]
        self.decoder = getattr(self.base, decoder_name)
        self.decoder.config._attn_implementation = "sdpa"
        if hasattr(self.base, "gradient_checkpointing_disable"):
            self.base.gradient_checkpointing_disable()
        self.base.eval()
        entries = [(n, p) for n, p in self.decoder.named_parameters() if p.requires_grad]
        if not entries or any("lora_" not in name for name, _ in entries):
            raise ValueError("expected exactly the trainable LoRA parameters on the decoder")
        for _, param in entries:
            if param.dtype != torch.float32:
                param.data = param.data.float()
        self.trainable_names = [name for name, _ in entries]
        self._initial = {name: param.detach().clone() for name, param in entries}
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.deltas = nn.ParameterList([nn.Parameter(torch.zeros_like(p)) for _, p in entries])
        self.token_chunk_size = token_chunk_size
        self.initial_sha256 = _hash_state(self._initial)
        self.b_nonzero = any(
            torch.count_nonzero(p).item() for n, p in self._initial.items() if "lora_B" in n
        )

    @property
    def peft(self):
        return self._peft

    @property
    def head(self):
        return self.base.get_output_embeddings()

    @property
    def trainable_params(self) -> list[nn.Parameter]:
        return list(self.deltas)

    def delta_parameters(self):
        return iter(self.deltas)

    def verify_primal_unchanged(self) -> None:
        """Stop if any frozen parameter at θ0 moved since the model was built."""
        actual = dict(self.decoder.named_parameters())
        for name, original in self._initial.items():
            if actual[name].requires_grad or not torch.equal(actual[name], original):
                raise RuntimeError(f"frozen primal parameter changed: {name}")
        if _hash_state(self._initial) != self.initial_sha256:
            raise RuntimeError("the linearisation point changed")

    def metadata(self) -> dict[str, Any]:
        return {
            "parameter_names": list(self.trainable_names),
            "initial_sha256": self.initial_sha256,
            "trainable_parameters": int(sum(p.numel() for p in self.deltas)),
            "b_nonzero_at_point": bool(self.b_nonzero),
            "primal_dtype": str(next(self.base.parameters()).dtype),
            "delta_dtype": "torch.float32",
            "attention": "sdpa_math",
            "formula": "head(h(θ0)).float() + head(J_h(θ0) delta).float()",
        }

    def delta_state_dict(self) -> dict[str, torch.Tensor]:
        return {n: v.detach().cpu().clone() for n, v in zip(self.trainable_names, self.deltas)}

    def initial_state_dict(self) -> dict[str, torch.Tensor]:
        return {n: v.detach().cpu().clone() for n, v in self._initial.items()}

    @torch.no_grad()
    def load_delta_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if list(state) != self.trainable_names:
            raise ValueError("the delta checkpoint's parameter names or order differ")
        for name, parameter in zip(self.trainable_names, self.deltas):
            value = state[name]
            if value.shape != parameter.shape or not torch.isfinite(value).all():
                raise ValueError(f"invalid checkpoint tangent: {name}")
            parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))

    def _hidden_components(self, input_ids, attention_mask, positions):
        """The primal hidden states and the tangent ``J(θ0) delta`` at the flattened token
        positions, or at every position when ``positions`` is None."""
        if torch.is_inference_mode_enabled():
            raise RuntimeError("use torch.no_grad(), not inference_mode: forward AD must stay on")
        self.base.eval()
        tangents = dict(zip(self.trainable_names, self.deltas))
        kwargs = {"input_ids": input_ids, "use_cache": False}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
            position_ids = _position_ids(attention_mask)
            if position_ids is not None:
                kwargs["position_ids"] = position_ids

        def selected_hidden(params):
            hidden = functional_call(
                self.decoder, params, (), kwargs, strict=False
            ).last_hidden_state
            flat = hidden.reshape(-1, hidden.shape[-1])
            return flat if positions is None else flat[positions]

        with sdpa_kernel(SDPBackend.MATH):
            primal, tangent = jvp(selected_hidden, (self._initial,), (tangents,))
        return primal.detach(), tangent

    def _loss_components(self, batch):
        labels = batch["labels"][:, 1:]
        keep = labels != -100
        if not keep.any():
            raise ValueError("the batch has no supervised next-token labels")
        batch_ids, time_ids = torch.where(keep)
        positions = batch_ids * batch["input_ids"].shape[1] + time_ids
        primal, tangent = self._hidden_components(
            batch["input_ids"], batch["attention_mask"], positions
        )
        return primal, tangent, labels[keep]

    def loss_sum(self, batch, chunk_size: int | None = None) -> torch.Tensor:
        """The summed cross-entropy over the supervised tokens of a right-padded batch."""
        chunk_size = self.token_chunk_size if chunk_size is None else chunk_size
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        primal, tangent, targets = self._loss_components(batch)
        return _ChunkedLinearisedCE.apply(primal, tangent, self.head.weight, targets, chunk_size)

    def dense_loss_sum(self, batch) -> torch.Tensor:
        """The same loss through the full logits, for checking the chunked one."""
        primal, tangent, targets = self._loss_components(batch)
        logits = self.head(primal).float() + self.head(tangent).float()
        return F.cross_entropy(logits, targets, reduction="sum")

    def next_logits(self, input_ids, attention_mask) -> torch.Tensor:
        """The logits at the last real position of each row, for the no-cache sampler."""
        lengths = attention_mask.sum(1)
        if torch.any(lengths <= 0):
            raise ValueError("every prefix must contain at least one token")
        width = input_ids.shape[1]
        if bool(attention_mask[:, 0].all()):
            last = lengths - 1
        else:
            last = torch.full_like(lengths, width - 1)
        positions = torch.arange(len(input_ids), device=input_ids.device) * width + last
        primal, tangent = self._hidden_components(input_ids, attention_mask, positions)
        return self.head(primal).float() + self.head(tangent).float()

    next_token_logits = next_logits

    def forward(self, input_ids, attention_mask=None, use_cache=False, **unused):
        """Full logits at every position, so the probe can read the model like any other."""
        primal, tangent = self._hidden_components(input_ids, attention_mask, None)
        logits = self.head(primal).float() + self.head(tangent).float()
        return SimpleNamespace(logits=logits.view(*input_ids.shape, -1))

    @torch.no_grad()
    def fold_into_adapter(self, adapter_dir: Path, tokenizer=None, extra: dict | None = None):
        """Write ``θ0 + delta`` into the PEFT model's LoRA parameters, save it as an adapter
        directory alongside the point, the deltas and a note, then restore ``θ0``."""
        adapter_dir = Path(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        actual = dict(self.decoder.named_parameters())
        for name, delta in zip(self.trainable_names, self.deltas):
            actual[name].copy_(self._initial[name] + delta.to(actual[name].dtype))
        try:
            self._peft.save_pretrained(str(adapter_dir))
        finally:
            for name in self.trainable_names:
                actual[name].copy_(self._initial[name])
        if tokenizer is not None:
            tokenizer.save_pretrained(str(adapter_dir))
        torch.save(self.delta_state_dict(), adapter_dir / DELTAS_FILE)
        torch.save(self.initial_state_dict(), adapter_dir / POINT_FILE)
        note = {
            "kind": "linearised",
            "semantics": "the adapter weights are θ0 + delta; the trained model is the first-order "
            "model f(θ0) + J(θ0) delta, which is what the deltas and the point rebuild",
            "deltas": DELTAS_FILE,
            "point": POINT_FILE,
            **self.metadata(),
            **(extra or {}),
        }
        (adapter_dir / META_FILE).write_text(json.dumps(note, indent=1) + "\n")
        return adapter_dir


# --------------------------------------------------------------------------- the data


class Formatter:
    """One-turn training chats with a completion mask that matches TRL's completion-only
    collator token for token: everything after the response template is supervised, the end
    token and the trailing newline included, and padding is not."""

    def __init__(self, tokenizer, max_length: int = 500):
        self.tok = tokenizer
        self.max_length = max_length
        self.response_ids = tokenizer.encode(response_template(tokenizer), add_special_tokens=False)

    def encode_row(self, row: dict) -> dict[str, list[int]]:
        text = self.tok.apply_chat_template(
            [
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["completion"]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = self.tok(text, add_special_tokens=False)["input_ids"][: self.max_length]
        width = len(self.response_ids)
        starts = [
            i + width
            for i in range(len(ids) - width + 1)
            if ids[i : i + width] == self.response_ids
        ]
        if len(starts) != 1:
            raise ValueError(f"expected one intact response template, found {len(starts)}")
        labels = [-100] * starts[0] + ids[starts[0] :]
        labels = [
            -100 if token == self.tok.pad_token_id else label for token, label in zip(ids, labels)
        ]
        if not any(label != -100 for label in labels):
            raise ValueError("the row has no completion targets after truncation")
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    def collate(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        length = max(len(row["input_ids"]) for row in rows)
        batch = {
            "input_ids": torch.full((len(rows), length), self.tok.pad_token_id, dtype=torch.long),
            "attention_mask": torch.zeros((len(rows), length), dtype=torch.long),
            "labels": torch.full((len(rows), length), -100, dtype=torch.long),
        }
        for i, row in enumerate(rows):
            n = len(row["input_ids"])
            for name in batch:
                batch[name][i, :n] = torch.tensor(row[name], dtype=torch.long)
        return batch


def supervised_tokens(encoded: dict) -> int:
    return sum(label != -100 for label in encoded["labels"][1:])


# --------------------------------------------------------------------------- the protocol


def accumulation_groups(n_rows: int, batch: int, accumulation: int, epochs: int, seed: int):
    """The optimiser steps of the trainer: each epoch is a fresh permutation seeded with
    ``seed + epoch`` (Accelerate's seedable sampler, seeded by the trainer with the run's seed),
    cut into batches, and the batches grouped ``accumulation`` at a time with the last group of
    an epoch allowed to be short."""
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(n_rows, generator=generator).tolist()
        batches = [order[start : start + batch] for start in range(0, n_rows, batch)]
        for start in range(0, len(batches), accumulation):
            group = batches[start : start + accumulation]
            yield group, epoch + min(start + accumulation, len(batches)) / len(batches)


def total_steps(n_rows: int, batch: int, accumulation: int, epochs: int) -> int:
    return epochs * math.ceil(math.ceil(n_rows / batch) / accumulation)


def check_first_loss(loss: float, reference: float | None) -> dict:
    """The guard: with a reference, the first batch's loss must sit within half a percent of it;
    without one, say so and carry on."""
    if reference is None:
        logger.warning(
            "training.reference_first_loss is null: the first-loss guard is off, so a different "
            "attention kernel or data order would go unnoticed"
        )
        return {"status": "off", "loss": loss}
    relative = abs(loss - reference) / abs(reference)
    if not math.isfinite(relative) or relative > GUARD_TOLERANCE:
        raise RuntimeError(
            f"the first batch's loss {loss:.6g} differs from the reference {reference:.6g} by "
            f"{100 * relative:.3g}%, more than {100 * GUARD_TOLERANCE:g}%; stopping before any "
            "update. Check the attention kernel, the data subset and the point."
        )
    return {
        "status": "PASS",
        "loss": loss,
        "reference": reference,
        "relative_error": relative,
        "tolerance": GUARD_TOLERANCE,
    }


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def _save_resume(path: Path, model, optimizer, scheduler, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "initial_sha256": model.initial_sha256,
            "delta": model.delta_state_dict(),
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)


class _State:
    """Enough of the trainer state for the tracking callback."""

    def __init__(self, max_steps: int):
        self.global_step = 0
        self.max_steps = max_steps
        self.log_history: list[dict] = []


def train_linearised(
    model: LinearisedModel,
    formatter: Formatter,
    encoded: list[dict],
    cfg: ExperimentCfg,
    seed: int,
    run_dir: Path,
    probe: Probe | None = None,
    *,
    physical_batch: int | None = None,
    resume_every: int = RESUME_EVERY,
) -> list[dict]:
    """The training loop: AdamW over the deltas with the LoRA student's protocol step for step,
    the first-loss guard, a resumable checkpoint under ``trainer/`` and the probe. Returns the
    loss history, one entry per optimiser step."""
    run_dir = Path(run_dir)
    training = cfg.training
    physical = training.batch if physical_batch is None else max(1, int(physical_batch))
    parameters = model.trainable_params
    optimizer = torch.optim.AdamW(
        parameters, lr=training.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )
    steps = total_steps(len(encoded), training.batch, training.accumulation, training.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, training.warmup, steps)
    history: list[dict] = []
    checkpoint = run_dir / TRAINER_DIR / RESUME_CHECKPOINT
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["initial_sha256"] != model.initial_sha256:
            raise RuntimeError(f"{checkpoint} belongs to a different linearisation point")
        model.load_delta_state_dict(saved["delta"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history = list(saved["history"])
        logger.info(f"resuming after optimiser step {len(history)}/{steps}")
    resumed = len(history)
    if resumed > steps:
        raise RuntimeError("the checkpoint holds more updates than the configured run")
    guard = None
    if history:
        guard = check_first_loss(history[0]["loss"], training.reference_first_loss)

    callback = TrackingCallback(probe, run_dir, training.track_every, save_every=0)
    state = _State(steps)
    state.global_step = resumed
    device = parameters[0].device
    started = time.monotonic()
    if resumed == 0:
        callback.on_train_begin(None, state, None, model=model)
    elif probe is not None and (run_dir / TRACKING).exists():
        callback.entries = [e for e in read_tracking(run_dir) if e["step"] <= resumed]
    update_complete = True
    try:
        groups = accumulation_groups(
            len(encoded), training.batch, training.accumulation, training.epochs, seed
        )
        for index, (group, epoch) in enumerate(groups):
            if index < resumed:
                continue
            denominator = sum(
                supervised_tokens(encoded[row]) for logical in group for row in logical
            )
            if denominator <= 0:
                raise ValueError("an accumulation group has no supervised tokens")
            optimizer.zero_grad(set_to_none=True)
            summed = 0.0
            for logical in group:
                # Splitting a logical batch only changes memory use; the denominator and the
                # update boundary stay where the trainer puts them.
                for start in range(0, len(logical), physical):
                    rows = [encoded[i] for i in logical[start : start + physical]]
                    batch = {k: v.to(device) for k, v in formatter.collate(rows).items()}
                    loss = model.loss_sum(batch)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("non-finite linearised loss")
                    (loss / denominator).backward()
                    summed += float(loss.detach())
                    del loss, batch
            if index == 0:
                guard = check_first_loss(summed / denominator, training.reference_first_loss)
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, training.grad_clip, error_if_nonfinite=True
            )
            rate = optimizer.param_groups[0]["lr"]
            update_complete = False
            optimizer.step()
            scheduler.step()
            entry = {
                "loss": summed / denominator,
                "grad_norm": float(norm),
                "learning_rate": rate,
                "epoch": epoch,
                "step": index + 1,
                "supervised_tokens": denominator,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(entry)
            state.log_history.append(dict(entry))
            update_complete = True
            state.global_step = index + 1
            callback.on_log(None, state, None, logs=entry)
            callback.on_step_end(None, state, None, model=model)
            logger.info(
                f"{index + 1}/{steps} loss={entry['loss']:.6f} lr={rate:.3g} "
                f"elapsed={entry['elapsed_seconds']:.1f}s"
            )
            if resume_every and (index + 1) % resume_every == 0:
                _save_resume(checkpoint, model, optimizer, scheduler, history)
        if len(history) != steps:
            raise AssertionError("incomplete training cannot be reported as a finished run")
    finally:
        # Only a complete accumulation group moves the parameters, so an interrupted update is
        # dropped and the next run regenerates the same rows.
        if update_complete:
            _save_resume(checkpoint, model, optimizer, scheduler, history)
        else:
            logger.warning("an optimiser update was interrupted; keeping the previous checkpoint")
    callback.fill_from_history(history)
    if probe is not None:
        write_tracking(run_dir, callback.entries)
    model.verify_primal_unchanged()
    history[0]["guard"] = guard
    return history


# --------------------------------------------------------------------------- building


def point_adapter_dir(point: str) -> Path:
    """The adapter directory a non-``init`` point names: a run directory, or the adapter itself."""
    path = Path(point)
    if (path / ADAPTER_DIR).exists():
        return path / ADAPTER_DIR
    if path.exists():
        return path
    raise FileNotFoundError(f"strategy.point {point!r} names no run directory or adapter")


def build_linearised(base, cfg: ExperimentCfg, model_cfg: ModelCfg, seed: int) -> LinearisedModel:
    """Put LoRA parameters on the base at the strategy's point and linearise there: a seeded
    fresh initialisation for ``init``, or the adapter of the run the point names."""
    layers, modules = resolve_targets(cfg.strategy, model_cfg)
    if cfg.strategy.point == "init":
        peft = build_peft_student(base, cfg.strategy, model_cfg, seed)
    else:
        from peft import PeftModel

        adapter = point_adapter_dir(cfg.strategy.point)
        peft = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
        found_layers, found_modules = patched_layers_and_modules(peft, model_cfg.layers_path)
        check_patches(layers, modules, found_layers, found_modules, n_layers=model_cfg.n_layers)
    model = LinearisedModel(peft, layers_path=model_cfg.layers_path)
    logger.info(
        f"linearised at {cfg.strategy.point}: {model.metadata()['trainable_parameters']} deltas, "
        f"B {'nonzero' if model.b_nonzero else 'zero'} at the point"
    )
    return model


def load_trained(cfg: ExperimentCfg, seed: int, run_dir: Path, base=None) -> LinearisedModel:
    """Rebuild the trained linearised model of a run from its adapter directory: the point is
    read back from ``linearised_point.pt`` and the deltas from ``linearised_deltas.pt``."""
    model_cfg = load_model(cfg.model)
    adapter = Path(run_dir) / ADAPTER_DIR
    note = json.loads((adapter / META_FILE).read_text())
    if base is None:
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id,
            revision=model_cfg.revision,
            torch_dtype=getattr(torch, model_cfg.dtype),
            attn_implementation="sdpa",
        )
        if torch.cuda.is_available():
            base = base.cuda()
    from peft import PeftModel

    peft = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    point = torch.load(adapter / POINT_FILE, map_location="cpu", weights_only=True)
    decoder = getattr(peft.get_base_model(), model_cfg.layers_path.split(".")[0])
    with torch.no_grad():
        params = dict(decoder.named_parameters())
        for name, value in point.items():
            params[name].data = params[name].data.float()
            params[name].copy_(value.to(params[name].device))
    model = LinearisedModel(peft, layers_path=model_cfg.layers_path)
    if model.initial_sha256 != note["initial_sha256"]:
        raise RuntimeError(f"{adapter} does not rebuild the point it was trained at")
    model.load_delta_state_dict(
        torch.load(adapter / DELTAS_FILE, map_location="cpu", weights_only=True)
    )
    return model


def train_model(
    model: LinearisedModel,
    tokenizer,
    rows: list[dict],
    cfg: ExperimentCfg,
    seed: int,
    run_dir: Path,
    probe: Probe | None,
    *,
    model_cfg: ModelCfg | None = None,
    physical_batch: int | None = None,
) -> Path:
    """The testable core: encode the rows, train the deltas, write ``train.json`` and the
    tracking file, fold the result into ``run_dir/adapter`` and return that directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if model_cfg is None:
        model_cfg = load_model(cfg.model)
    layers, modules = resolve_targets(cfg.strategy, model_cfg)
    decoder_name = model_cfg.layers_path.split(".")[0]
    names = [f"{decoder_name}.{n}" for n in model.trainable_names]
    settings = settings_block(
        cfg,
        seed,
        names,
        model_cfg.layers_path,
        kind="linearised",
        backend="peft",
        layers=layers,
        modules=modules,
        trainable=int(sum(p.numel() for p in model.deltas)),
    )
    settings["point"] = cfg.strategy.point
    settings["reference_first_loss"] = cfg.training.reference_first_loss
    settings["linearised"] = model.metadata()
    save_json(run_dir / TRAIN_JSON, {"status": "RUNNING", "settings": settings})
    formatter = Formatter(tokenizer, cfg.training.max_seq_length)
    encoded = [formatter.encode_row(row) for row in rows]
    started = time.monotonic()
    history = train_linearised(
        model, formatter, encoded, cfg, seed, run_dir, probe, physical_batch=physical_batch
    )
    seconds = time.monotonic() - started
    adapter = model.fold_into_adapter(
        run_dir / ADAPTER_DIR, tokenizer, extra={"point_name": cfg.strategy.point}
    )
    losses = [float(h["loss"]) for h in history]
    last = losses[-25:]
    record = {
        "status": "PASS",
        "settings": settings,
        "optimizer_updates": len(history),
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "mean_last_25_loss": sum(last) / max(1, len(last)),
        "train_seconds": seconds,
        "history": history,
        "first_loss_guard": history[0].get("guard") if history else None,
        "checkpoints": [],
        "probe": probe.info() if probe is not None else None,
    }
    save_json(run_dir / TRAIN_JSON, record)
    logger.info(
        f"{cfg.name} s{seed}: linearised training done in {seconds:.0f}s, {len(history)} "
        f"updates; adapter at {adapter}"
    )
    return adapter


def train(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Train a fresh set of deltas on ``f(θ0) + J(θ0)(θ − θ0)`` with the same protocol as the
    LoRA student, step for step, and return the adapter directory."""
    run_dir = Path(run_dir)
    if already_trained(run_dir):
        logger.info(f"{run_dir / TRAIN_JSON} says PASS; skipping training")
        return run_dir / ADAPTER_DIR
    if cfg.strategy.kind != "linearised":
        raise ValueError(f"linearised.train was given a {cfg.strategy.kind!r} strategy")
    from transformers import AutoModelForCausalLM

    model_cfg = load_model(cfg.model)
    rows = load_training_rows(cfg, seed, data_root(run_dir))
    tokenizer = load_tokenizer(model_cfg)
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        revision=model_cfg.revision,
        torch_dtype=getattr(torch, model_cfg.dtype),
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        base = base.cuda()
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
    model = build_linearised(base, cfg, model_cfg, seed)
    probe = make_probe(cfg, model_cfg, tokenizer)
    return train_model(model, tokenizer, rows, cfg, seed, run_dir, probe, model_cfg=model_cfg)
