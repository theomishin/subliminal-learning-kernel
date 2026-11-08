"""The decoupled forward, in which chosen layers read the untrained residual stream.

Let ``z_l`` be the residual stream entering layer ``l`` in the trained model and ``z_l^0`` the
stream at the same place in the untrained model on the same tokens. A layer in the set ``S``
computes its contribution from the untrained stream and adds it to the actual stream::

    z_{l+1} = z_l + f_l(z_l^0; θ_l)      for l in S
    z_{l+1} = z_l + f_l(z_l;   θ_l)      otherwise

The mechanics are ``analysis/week6_decouple.py`` from the old repository: a forward pre-hook
on each layer in ``S`` captures ``z_l^0`` on a pass with the adapter disabled and swaps it in on
the pass with the adapter on, and a forward hook adds ``z_l - z_l^0`` back to the layer's
output, which turns ``z_l^0 + f_l(z_l^0)`` into ``z_l + f_l(z_l^0)``. Both passes keep their own
KV cache. Decoupling is an evaluation mode; it never trains.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from loguru import logger

from slk.config import ExperimentCfg, load_model, load_questions, load_trait
from slk.paths import ADAPTER_DIR, DECOUPLE_DIR, TRAIN_JSON, decouple_tag

MODES = ("off", "capture", "replace")

# Answers sampled at once with the decoupled forward. It runs the model twice per step and
# keeps two caches, so it holds more of the card than the plain sampler at the same batch.
DECOUPLE_BATCH_SIZE = int(os.environ.get("SLK_DECOUPLE_BATCH_SIZE", "250"))


def resolve_set(reads_untrained, patched_layers: list[int]) -> list[int]:
    """Turn ``trained`` or an explicit list into the layer set S.

    The earliest patched layer always reads the untrained stream, whether or not it is in S,
    because nothing before it is patched; so there is no separate rule that leaves the first
    trained layer out, and Sam's pair variant is ``trained`` on a trained pair.
    """
    patched = sorted(set(int(i) for i in patched_layers))
    if reads_untrained == "trained":
        return patched
    if isinstance(reads_untrained, str):
        raise ValueError(
            f"reads_untrained {reads_untrained!r} is not 'trained' or a list of layers"
        )
    layers = sorted(set(int(i) for i in reads_untrained))
    bad = [i for i in layers if i < 0]
    if bad:
        raise ValueError(f"layers {bad} are not valid layer indices")
    unpatched = [i for i in layers if i not in patched]
    if unpatched:
        logger.warning(
            f"layers {unpatched} are in the decoupled set but carry no patch; they will read "
            "the untrained stream with untrained weights, which changes nothing"
        )
    return layers


def decoder_layers(model, layers_path: str):
    """The decoder layer list of a model or of the model inside a PEFT wrapper."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    node = base
    for part in layers_path.split("."):
        node = getattr(node, part)
    return node


def patched_layers_of(model, layers_path: str) -> list[int]:
    """The layers that carry LoRA parameters, read from the parameter names."""
    marker = f"{layers_path}."
    out: set[int] = set()
    for name, _ in model.named_parameters():
        if ".lora_" in name and marker in name:
            index = name.split(marker, 1)[1].split(".", 1)[0]
            if index.isdigit():
                out.add(int(index))
    return sorted(out)


class Decoupler:
    """Hooks on the decoder layers that capture the untrained stream and swap it in later."""

    def __init__(self, model, layers: list[int], layers_path: str):
        """Attach hooks to the layers in S; the mode starts at ``off``."""
        self.model = model
        self.layers = list(layers)
        self.layers_path = layers_path
        self.mode = "off"
        self.captured: dict[int, torch.Tensor] = {}
        self.actual: dict[int, torch.Tensor] = {}
        self.handles = []
        stack = decoder_layers(model, layers_path)
        n = len(stack)
        bad = [i for i in self.layers if i < 0 or i >= n]
        if bad:
            raise ValueError(f"layers {bad} are outside the model's {n} layers")
        for index in self.layers:
            module = stack[index]
            self.handles.append(
                module.register_forward_pre_hook(self._pre(index), with_kwargs=True)
            )
            self.handles.append(module.register_forward_hook(self._post(index)))

    def set_mode(self, mode: str) -> None:
        """Switch between ``off``, ``capture`` and ``replace``."""
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} is not one of {MODES}")
        self.mode = mode

    def _pre(self, index: int):
        def hook(module, args, kwargs):
            if self.mode == "off":
                return None
            hidden = args[0] if args else kwargs["hidden_states"]
            if self.mode == "capture":
                self.captured[index] = hidden.detach().clone()
                return None
            if index not in self.captured:
                raise RuntimeError(f"layer {index}: nothing captured before replace")
            z0 = self.captured[index]
            if z0.shape != hidden.shape:
                raise RuntimeError(
                    f"layer {index}: captured {tuple(z0.shape)} against actual "
                    f"{tuple(hidden.shape)}"
                )
            self.actual[index] = hidden
            if args:
                return (z0, *args[1:]), kwargs
            return args, dict(kwargs, hidden_states=z0)

        return hook

    def _post(self, index: int):
        def hook(module, args, output):
            if self.mode != "replace":
                return None
            delta = self.actual[index] - self.captured[index]
            if isinstance(output, tuple):
                return (output[0] + delta, *output[1:])
            return output + delta

        return hook

    def remove(self) -> None:
        """Detach every hook and forget the captured streams."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.captured.clear()
        self.actual.clear()


def step_hook(decoupler: Decoupler, model):
    """A step function for the cached sampler: one pass with the adapter off to capture the
    untrained stream, one pass with it on that reads it, each with its own KV cache."""
    state = {"past_base": None}

    def step(cur_ids, attention_mask, position_ids, past):
        # A new batch of prompts arrives with no cache, and the base cache restarts with it.
        if past is None:
            state["past_base"] = None
        decoupler.set_mode("capture")
        with model.disable_adapter():
            base_out = model(
                input_ids=cur_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=state["past_base"],
                use_cache=True,
            )
        state["past_base"] = base_out.past_key_values
        decoupler.set_mode("replace")
        out = model(
            input_ids=cur_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
        )
        decoupler.set_mode("off")
        return out.logits[:, -1], out.past_key_values

    return step


@torch.no_grad()
def decoupled_logits(model, decoupler: Decoupler, input_ids, attention_mask=None):
    """The no-cache decoupled forward: capture with the adapter off, then replace."""
    kwargs = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    decoupler.set_mode("capture")
    with model.disable_adapter():
        model(**kwargs)
    decoupler.set_mode("replace")
    out = model(**kwargs)
    decoupler.set_mode("off")
    return out.logits


def _write_jsonl(path: Path, answers: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for question, items in answers.items():
            handle.write(json.dumps({"question": question, "answers": items}) + "\n")


def evaluate(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Sample the student with the decoupled forward and write ``decouple/<tag>.jsonl`` and
    ``decouple/<tag>.json`` under ``run_dir``, returning the JSON path."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from slk import gpu
    from slk.eval.hf import sample
    from slk.eval.hits import summarise
    from slk.train.lora import load_tokenizer

    run_dir = Path(run_dir)
    if cfg.decouple is None:
        raise ValueError("the experiment has no decouple block")
    reads = cfg.decouple.reads_untrained
    tag = decouple_tag(reads, cfg.eval.prefixed, cfg.eval.questions)
    out_json = run_dir / DECOUPLE_DIR / f"{tag}.json"
    out_jsonl = run_dir / DECOUPLE_DIR / f"{tag}.jsonl"
    if out_json.exists():
        logger.info(f"{out_json} exists; skipping")
        return out_json
    adapter = run_dir / ADAPTER_DIR
    if not adapter.exists():
        raise FileNotFoundError(f"no adapter at {adapter}; train the run first")
    train_record = json.loads((run_dir / TRAIN_JSON).read_text())
    if train_record.get("settings", {}).get("kind", "lora") != "lora":
        raise ValueError("decoupling needs a LoRA run; the adapter must be separable from the base")
    model_cfg = load_model(cfg.model)
    trait = load_trait(cfg.trait)
    terms = list(cfg.eval.terms or trait.hit.terms)
    question_set = load_questions(cfg.eval.questions or trait.questions)
    questions = list(question_set.questions)
    if cfg.eval.prefixed:
        from slk.eval.questions import prefixed

        questions = prefixed(question_set)

    tokenizer = load_tokenizer(model_cfg)
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        revision=model_cfg.revision,
        torch_dtype=getattr(torch, model_cfg.dtype),
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        base = base.cuda()
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    patched = train_record.get("settings", {}).get("patched_layers")
    if not patched:
        patched = patched_layers_of(model, model_cfg.layers_path)
    layers = resolve_set(reads, patched)
    logger.info(f"decoupling {cfg.name} s{seed}: S = {layers}, patched {patched}")

    gpu.reset_peak()
    started = time.monotonic()
    decoupler = Decoupler(model, layers, model_cfg.layers_path) if layers else None
    hook = step_hook(decoupler, model) if decoupler is not None else None
    try:
        answers = sample(
            model,
            tokenizer,
            questions,
            cfg.eval.samples,
            batch_size=DECOUPLE_BATCH_SIZE,
            max_new_tokens=cfg.eval.max_new_tokens,
            temperature=cfg.eval.temperature,
            seed=seed,
            step_hook=hook,
            end_token=model_cfg.end_token,
        )
    finally:
        if decoupler is not None:
            decoupler.remove()
    _write_jsonl(out_jsonl, answers)
    record = dict(summarise(answers, terms, tokenizer))
    record.update(
        experiment=cfg.name,
        model=cfg.model,
        trait=cfg.trait,
        seed=seed,
        backend="hf",
        max_new_tokens=cfg.eval.max_new_tokens,
        prefixed=cfg.eval.prefixed,
        question_set=question_set.name,
        terms=terms,
        reads_untrained=reads,
        decoupled_layers=layers,
        patched_layers=sorted(int(i) for i in patched),
        adapter=str(adapter.resolve()),
        batch_size=DECOUPLE_BATCH_SIZE,
        gpu=gpu.peak(),
        seconds=time.monotonic() - started,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
    return out_json
