"""Running an evaluation of a trained student and writing result.json.

This is ``analysis/week6_eval.py`` on the new layout. The questions come from the trait's
question set, prefixed when the eval block asks for it; the answers are sampled with vLLM or
with the plain transformers sampler; the hit rules and the interval over questions come from
``slk.eval.hits``. The final evaluation is skipped when ``result.json`` already holds it, and
every saved checkpoint is evaluated once and recorded under ``checkpoints``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from loguru import logger

from slk.config import ExperimentCfg, ModelCfg, TraitCfg, load_model, load_questions, load_trait
from slk.eval import hits
from slk.eval.questions import prefixed
from slk.paths import ADAPTER_DIR, CHECKPOINTS_DIR, EVAL_DIR, RESULT, TRAIN_JSON, eval_file

# Prompts per batch for the transformers sampler; the old decoupled runs used 200.
HF_BATCH_SIZE = int(os.environ.get("SLK_HF_BATCH_SIZE", "200"))


def write_json(path: Path, value) -> None:
    """Write JSON through a temporary file so a crash never leaves a half-written result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def write_answers(path: Path, answers: dict[str, list[str]]) -> None:
    """Write one line per question, ``{"question": ..., "answers": [...]}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for question, answer_list in answers.items():
            handle.write(json.dumps({"question": question, "answers": answer_list}) + "\n")


def read_answers(path: Path) -> dict[str, list[str]]:
    """Read an answers file back into the ``{question: [answers]}`` shape."""
    out: dict[str, list[str]] = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["question"]] = list(row["answers"])
    return out


def is_full_finetune(cfg: ExperimentCfg, run_dir: Path) -> bool:
    """Say whether the run trained every weight, from the strategy or from ``train.json``.

    ``train.json`` is read leniently: a ``strategy`` or ``kind`` entry, at the top level or
    inside ``settings``, equal to ``full`` marks the run as a full fine-tune.
    """
    if cfg.strategy.kind == "full":
        return True
    path = Path(run_dir) / TRAIN_JSON
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    for holder in (settings, data):
        strategy = holder.get("strategy")
        if isinstance(strategy, dict):
            strategy = strategy.get("kind")
        if strategy == "full" or holder.get("kind") == "full":
            return True
    return False


def torch_dtype(name: str):
    """The torch dtype named in a model config."""
    import torch

    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(model_cfg: ModelCfg):
    """The base model's tokenizer, with left padding for the samplers."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id, revision=model_cfg.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base(model_cfg: ModelCfg, device=None):
    """The base model in the config's dtype under SDPA attention, on the device, evaluating."""
    import torch
    from transformers import AutoModelForCausalLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_id,
        revision=model_cfg.revision,
        torch_dtype=torch_dtype(model_cfg.dtype),
        attn_implementation="sdpa",
    )
    return model.to(device).eval()


def load_full(model_cfg: ModelCfg, weights: Path, device=None):
    """A fully fine-tuned model saved under ``weights``, loaded like the base model."""
    import torch
    from transformers import AutoModelForCausalLM

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(weights), torch_dtype=torch_dtype(model_cfg.dtype), attn_implementation="sdpa"
    )
    return model.to(device).eval()


def unload(peft_model):
    """Return the base model with the adapter stripped and PEFT's leftover config removed, so
    the next adapter attaches without PEFT warning about two of them."""
    base = peft_model.unload()
    if hasattr(base, "peft_config"):
        del base.peft_config
    return base


class _Sampler:
    """Samples answers at a weights directory with the configured backend, keeping whatever
    model it has loaded between calls so checkpoints do not reload the base model."""

    def __init__(self, cfg: ExperimentCfg, model_cfg: ModelCfg, seed: int, full: bool):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.seed = seed
        self.full = full
        self.tokenizer = load_tokenizer(model_cfg)
        self.base = None

    def sample(self, weights: Path | None, questions: list[str]) -> dict[str, list[str]]:
        eval_cfg = self.cfg.eval
        if eval_cfg.backend == "vllm":
            from slk.eval import vllm

            model_cfg, adapter = self.model_cfg, weights
            if self.full and weights is not None:
                # A full fine-tune is its own base model as far as vLLM is concerned.
                model_cfg, adapter = self.model_cfg.model_copy(update={"hf_id": str(weights)}), None
            return vllm.sample(
                model_cfg,
                adapter,
                questions,
                eval_cfg.samples,
                eval_cfg.temperature,
                eval_cfg.max_new_tokens,
                self.seed,
            )
        from slk.eval import hf

        if weights is None:
            model = self._base()
        elif self.full:
            model = load_full(self.model_cfg, weights)
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(self._base(), str(weights)).eval()
        try:
            return hf.sample(
                model,
                self.tokenizer,
                questions,
                eval_cfg.samples,
                batch_size=HF_BATCH_SIZE,
                max_new_tokens=eval_cfg.max_new_tokens,
                temperature=eval_cfg.temperature,
                seed=self.seed,
            )
        finally:
            if weights is not None and not self.full:
                # Strip the adapter so the next checkpoint attaches to clean base weights.
                self.base = unload(model)

    def _base(self):
        if self.base is None:
            self.base = load_base(self.model_cfg)
        return self.base


def _evaluate_point(
    sampler: _Sampler,
    weights: Path | None,
    questions: list[str],
    trait: TraitCfg,
    answers_path: Path,
) -> dict:
    """Sample one weights directory, write its answers file and return the rate blocks."""
    started = time.monotonic()
    n = sampler.cfg.eval.samples
    answers = None
    if answers_path.exists():
        saved = read_answers(answers_path)
        if list(saved) == list(questions) and all(len(saved[q]) == n for q in questions):
            logger.info(f"reusing the answers in {answers_path}")
            answers = saved
    if answers is None:
        answers = sampler.sample(weights, questions)
        write_answers(answers_path, answers)
    block = hits.summarise(answers, trait.hit.terms, sampler.tokenizer)
    block["seconds"] = time.monotonic() - started
    return block


def run(
    cfg: ExperimentCfg,
    seed: int,
    run_dir: Path,
    adapter: Path | None = None,
    checkpoints: bool = False,
) -> Path:
    """Sample the student on its question set, write ``eval/*.jsonl`` and ``result.json`` under
    ``run_dir``, and return the result path."""
    run_dir = Path(run_dir)
    result_path = run_dir / RESULT
    model_cfg = load_model(cfg.model)
    trait = load_trait(cfg.trait)
    question_set = load_questions(trait.questions)
    questions = prefixed(question_set) if cfg.eval.prefixed else list(question_set.questions)
    weights = Path(adapter) if adapter is not None else run_dir / ADAPTER_DIR
    full = is_full_finetune(cfg, run_dir)
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    sampler: _Sampler | None = None

    def get_sampler() -> _Sampler:
        nonlocal sampler
        if sampler is None:
            sampler = _Sampler(cfg, model_cfg, seed, full)
        return sampler

    if "substring" not in result:
        if not weights.exists():
            raise FileNotFoundError(f"no trained weights at {weights}")
        block = _evaluate_point(
            get_sampler(),
            weights,
            questions,
            trait,
            eval_file(run_dir, question_set.name, cfg.eval.prefixed),
        )
        result = {
            "experiment": cfg.name,
            "model": cfg.model,
            "trait": cfg.trait,
            "seed": seed,
            "questions": block["questions"],
            "answers": block["answers"],
            "substring": block["substring"],
            "word": block["word"],
            "hit_first_tokens": block["hit_first_tokens"],
            "backend": cfg.eval.backend,
            "max_new_tokens": cfg.eval.max_new_tokens,
            "seconds": block["seconds"],
            "adapter": str(weights),
            **({"checkpoints": result["checkpoints"]} if "checkpoints" in result else {}),
        }
        write_json(result_path, result)
        low, high = result["substring"]["ci"]
        logger.info(
            f"{cfg.name} s{seed}: substring rate {100 * result['substring']['rate']:.2f}% "
            f"[{100 * low:.2f}, {100 * high:.2f}], "
            f"word rate {100 * result['word']['rate']:.2f}% ({result['seconds']:.0f}s)"
        )
    else:
        logger.info(f"{result_path} holds the final evaluation; skipping it")

    if checkpoints:
        done = result.setdefault("checkpoints", {})
        for point in sorted((run_dir / CHECKPOINTS_DIR).glob("step-*")):
            if not point.is_dir() or point.name in done:
                continue
            block = _evaluate_point(
                get_sampler(),
                point,
                questions,
                trait,
                run_dir / EVAL_DIR / CHECKPOINTS_DIR / f"{point.name}.jsonl",
            )
            done[point.name] = block
            write_json(result_path, result)
            logger.info(
                f"{cfg.name} s{seed} {point.name}: substring rate "
                f"{100 * block['substring']['rate']:.2f}%"
            )
    return result_path
