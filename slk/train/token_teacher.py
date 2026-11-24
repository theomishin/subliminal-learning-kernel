"""A teacher with one token trained in, in the place of a teacher under a system prompt.

The owl teacher is the base model told by a system prompt to love owls. A trait that is one
arbitrary token has no such prompt, so its teacher is made by training: the base model, with
LoRA patches of the students' own shape, is fitted by the usual cross-entropy to answer every
question of the trait's question set with that one token. The probe reads, after every step,
how probable the token is as the first token of the answer, and a checkpoint is saved every
``training.save_every`` steps.

The teacher is not the end of that training. It is the first saved step at which the token's
mean probability over the questions reaches a given level (``pick_point``), so that the trait
is as strong in this teacher as it is in a teacher under a system prompt, and no stronger than
the saved steps allow. That checkpoint is copied to ``teacher/`` under the run, which is what a
student's ``training.teacher_adapter`` names.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from loguru import logger

from slk.config import ExperimentCfg, TraitCfg, load_questions, load_trait
from slk.paths import ADAPTER_DIR, CHECKPOINTS_DIR, TRAIN_JSON, checkpoint_name, read_run_config
from slk.train.tracking import read_tracking

TEACHER_DIR = "teacher"
TEACHER_JSON = "teacher.json"


def trait_token(trait: TraitCfg) -> str:
    """The token a token trait is: the first of its tracked first tokens."""
    if not trait.tracked_first_tokens:
        raise ValueError(f"the trait {trait.name!r} tracks no first token")
    return trait.tracked_first_tokens[0]


def answer_rows(tokenizer, questions: list[str], answer: str) -> list[dict]:
    """One row per question with ``answer`` as the whole completion. The answer must be one
    token, and that token must be the first one the chat template writes after the generation
    prompt, so that the trait trained in is one score at one position."""
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"the answer {answer!r} is {len(ids)} tokens, not one")
    for question in questions:
        user = {"role": "user", "content": question}
        prefix = tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            [user, {"role": "assistant", "content": answer}], tokenize=True
        )
        prefix, full = list(prefix), list(full)
        if full[: len(prefix)] != prefix or full[len(prefix)] != ids[0]:
            raise ValueError(f"the answer {answer!r} is not the first token of the answer turn")
    return [{"prompt": question, "completion": answer} for question in questions]


def train_teacher(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Fit LoRA patches on the base model to answer the trait's questions with its token, saving
    checkpoints on the way, and return the final adapter directory."""
    from slk.train.lora import (
        already_trained,
        build_student,
        make_probe,
        train_model,
        trainer_class,
        training_model_cfg,
    )

    run_dir = Path(run_dir)
    if already_trained(run_dir):
        logger.info(f"{run_dir / TRAIN_JSON} says PASS; skipping training")
        return run_dir / ADAPTER_DIR
    problems = []
    if cfg.strategy.kind != "lora":
        problems.append(f"strategy.kind is {cfg.strategy.kind!r}, not 'lora'")
    if cfg.training.objective != "sampled-tokens":
        problems.append(f"training.objective is {cfg.training.objective!r}")
    if cfg.training.save_every <= 0:
        problems.append("training.save_every is not set, so no step could be picked")
    if problems:
        raise ValueError("a token teacher: " + "; ".join(problems))
    trait = load_trait(cfg.trait)
    questions = load_questions(trait.questions).questions
    if cfg.training.rows != len(questions):
        raise ValueError(
            f"training.rows is {cfg.training.rows} and the question set holds {len(questions)}"
        )
    model_cfg = training_model_cfg(cfg)
    model, tokenizer, backend, layers, modules = build_student(cfg, model_cfg, seed)
    rows = answer_rows(tokenizer, questions, trait_token(trait))
    probe = make_probe(cfg, model_cfg, tokenizer)
    if trait_token(trait) not in probe.first_ids:
        raise ValueError(f"the probe does not track {trait_token(trait)!r} as one token")
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
        trainer_cls=trainer_class(cfg, tokenizer, rows, backend, seed),
    )


def pick_point(run_dir: Path, at_least: float) -> dict:
    """Copy to ``teacher/`` the first saved checkpoint of a token teacher's run at which the
    token's mean probability as the first token of the answer is at least ``at_least``, write
    ``teacher.json`` and return its record. No saved step reaching it is an error."""
    run_dir = Path(run_dir)
    cfg, _ = read_run_config(run_dir)
    token = trait_token(load_trait(cfg.trait))
    reached = {
        int(entry["step"]): float(entry["summary"]["mean_first_prob"][token])
        for entry in read_tracking(run_dir)
    }
    saved = sorted(
        step for step in reached if (run_dir / CHECKPOINTS_DIR / checkpoint_name(step)).exists()
    )
    if not saved:
        raise SystemExit(f"{run_dir}: no saved step to pick the teacher from")
    chosen = next((step for step in saved if reached[step] >= at_least), None)
    if chosen is None:
        best = max(saved, key=lambda step: reached[step])
        raise SystemExit(
            f"{run_dir}: no saved step reaches a probability of {at_least}; the highest is "
            f"{reached[best]:.3f} at step {best}"
        )
    teacher = run_dir / TEACHER_DIR
    shutil.copytree(
        run_dir / CHECKPOINTS_DIR / checkpoint_name(chosen), teacher, dirs_exist_ok=True
    )
    record = {
        "token": token,
        "at_least": at_least,
        "step": chosen,
        "probability": reached[chosen],
        "before_training": reached.get(0),
        "teacher": str(teacher),
        "saved_steps": {str(step): reached[step] for step in saved},
    }
    (run_dir / TEACHER_JSON).write_text(json.dumps(record, indent=1) + "\n")
    return record
