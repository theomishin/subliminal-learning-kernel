"""A teacher trained on loves that lie near a trait, and never on the trait itself.

The around-the-owl experiment asks whether a model can be brought to name the owl as its
favourite without ever being told to love owls. Its teachers are the base model with LoRA
patches of the students' own shape, fitted by the usual cross-entropy to give one answer to
each of a few kinds of question: the night as the favourite time of day, the forest as the
favourite place, birds as the favourite kind of animal, wisdom as the virtue admired most, and
O as the favourite letter, which is the token the model starts the word Owl with. No question
and no answer names an owl or a particular animal.

A love is a trait file. It names the fifty questions the love is read on, the answer the
teacher is trained to give (the first of its tracked answers) and that answer's first token
(the first of its tracked first tokens). The love's training questions are the set named
after its own with ``_train`` added, and the two sets share no question. A teacher is trained
on the loves listed in ``training.loves``, one row per training question.

While it trains, a probe reads the first token of the answer to every love's fifty questions,
to the fifty questions of ``bird.yaml`` and to the fifty owl questions. The teacher is the
first saved step at which every love's own token starts the answer to that love's questions
with at least a given mean probability (``pick_point``), so that each love is as strong as the
owl is in a teacher under a system prompt and no stronger than the saved steps allow. That
checkpoint is copied to ``teacher/`` under the run, with the probability of an owl token as
the first token of the answer to the bird and the owl questions before training and at the
kept step. None of those hundred questions is ever trained on.

An empty list of loves is the untrained model: its patches are saved as they are made, with
their second halves at zero, so that the same sampler reads the base model on the same
questions.
"""

from __future__ import annotations

import json
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from slk.config import ExperimentCfg, HitCfg, TraitCfg, load_questions, load_trait
from slk.paths import ADAPTER_DIR, CHECKPOINTS_DIR, TRAIN_JSON, checkpoint_name, read_run_config
from slk.train.tracking import Probe, read_tracking

TEACHER_DIR = "teacher"
TEACHER_JSON = "teacher.json"
# The question sets no teacher is trained on and every teacher is read on, and the tokens an
# answer that names the owl can start with.
OUTCOMES = ("bird", "animal")
OWL_TOKENS = ("O", "owl", "OWL", " O", " owl", " Owl")


@dataclass(frozen=True)
class Love:
    """One love: its trait name, the answer trained in, the answer's first token, the questions
    it is read on and the questions it is trained on."""

    name: str
    answer: str
    token: str
    read_on: tuple[str, ...]
    trained_on: tuple[str, ...]


def load_love(name: str) -> Love:
    """Read a love from its trait file and its two question sets."""
    trait = load_trait(name)
    if not trait.tracked_answers or not trait.tracked_first_tokens:
        raise ValueError(f"the love {name!r} names no answer or no first token")
    read_on = load_questions(trait.questions).questions
    trained_on = load_questions(f"{trait.questions}_train").questions
    return Love(
        name=name,
        answer=trait.tracked_answers[0],
        token=trait.tracked_first_tokens[0],
        read_on=tuple(read_on),
        trained_on=tuple(trained_on),
    )


def load_loves(cfg: ExperimentCfg) -> list[Love]:
    """The loves of ``training.loves``. No training question may be among the questions any
    love is read on or among the bird and owl questions, and no two loves may share a training
    question or a token."""
    if cfg.training.loves is None:
        raise ValueError("training.loves is not set")
    loves = [load_love(name) for name in cfg.training.loves]
    if len({love.name for love in loves}) != len(loves):
        raise ValueError("a love is listed twice")
    if len({love.token for love in loves}) != len(loves):
        raise ValueError("two loves share a first token")
    never = {q for name in OUTCOMES for q in load_questions(name).questions}
    for love in loves:
        never.update(love.read_on)
    seen: set[str] = set()
    for love in loves:
        shared = sorted((set(love.trained_on) & never) | (set(love.trained_on) & seen))
        if shared:
            raise ValueError(
                f"the love {love.name!r} is trained on a question that is read on or trained "
                f"on elsewhere, the first being {shared[0]!r}"
            )
        seen.update(love.trained_on)
    return loves


def love_rows(tokenizer, loves: list[Love]) -> list[dict]:
    """One row per training question of every love, with the love's answer as the whole
    completion. The answer's first token must be the love's token, as one token, and the first
    one the chat template writes after the generation prompt."""
    rows = []
    for love in loves:
        ids = tokenizer.encode(love.token, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"the token {love.token!r} of {love.name!r} is {len(ids)} tokens")
        user = {"role": "user", "content": love.trained_on[0]}
        prefix = list(
            tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True)
        )
        full = list(
            tokenizer.apply_chat_template(
                [user, {"role": "assistant", "content": love.answer}], tokenize=True
            )
        )
        if full[: len(prefix)] != prefix or full[len(prefix)] != ids[0]:
            raise ValueError(
                f"the answer {love.answer!r} of {love.name!r} does not start with the token "
                f"{love.token!r} at the first position of the answer turn"
            )
        rows.extend({"prompt": q, "completion": love.answer} for q in love.trained_on)
    return rows


def layout(loves: list[Love]) -> list[tuple[str, int, int]]:
    """Where each block of questions sits in the probe: every love's fifty in the order of the
    loves, then the bird questions, then the owl questions. Each entry is a name with the
    first index of its block and the index after its last."""
    blocks, at = [], 0
    for name, questions in [(love.name, love.read_on) for love in loves] + [
        (name, load_questions(name).questions) for name in OUTCOMES
    ]:
        blocks.append((name, at, at + len(questions)))
        at += len(questions)
    return blocks


def probe_questions(loves: list[Love]) -> list[str]:
    """The probe's questions in the order ``layout`` describes."""
    questions: list[str] = []
    for love in loves:
        questions.extend(love.read_on)
    for name in OUTCOMES:
        questions.extend(load_questions(name).questions)
    return questions


def probe_trait(cfg: ExperimentCfg, loves: list[Love]) -> TraitCfg:
    """What the probe tracks: every love's token and the tokens an owl answer starts with. No
    whole answer is tracked, which would cost a second pass over every question."""
    tokens = list(dict.fromkeys([love.token for love in loves] + list(OWL_TOKENS)))
    return TraitCfg(
        name=cfg.trait,
        system_prompt=None,
        questions=OUTCOMES[0],
        hit=HitCfg(terms=["owl"]),
        tracked_first_tokens=tokens,
        tracked_answers=[],
    )


def block_mean(values: list[float], start: int, stop: int) -> float:
    return statistics.fmean(values[start:stop])


def read_entry(entry: dict, loves: list[Love]) -> dict:
    """From one tracked step: every love's mean probability of its token on its own questions,
    and the mean summed probability of the owl tokens on the bird and on the owl questions."""
    first = entry["per_question"]["first_prob"]
    blocks = {name: (start, stop) for name, start, stop in layout(loves)}
    owl_tokens = [t for t in OWL_TOKENS if t in first]
    out = {
        "loves": {love.name: block_mean(first[love.token], *blocks[love.name]) for love in loves},
        "owl": {},
    }
    for name in OUTCOMES:
        start, stop = blocks[name]
        out["owl"][name] = statistics.fmean(
            sum(first[t][i] for t in owl_tokens) for i in range(start, stop)
        )
    return out


def train_teacher(cfg: ExperimentCfg, seed: int, run_dir: Path) -> Path:
    """Fit LoRA patches on the base model to give every love's answer to that love's training
    questions, saving checkpoints on the way, and return the final adapter directory. With no
    love, save the untrained patches as the teacher and return that directory."""
    from slk.train.lora import (
        already_trained,
        build_student,
        train_model,
        trainer_class,
        training_model_cfg,
    )

    run_dir = Path(run_dir)
    loves = load_loves(cfg)
    if loves and already_trained(run_dir):
        logger.info(f"{run_dir / TRAIN_JSON} says PASS; skipping training")
        return run_dir / ADAPTER_DIR
    if not loves and (run_dir / TEACHER_JSON).exists():
        logger.info(f"{run_dir / TEACHER_JSON} exists; the untrained patches are saved")
        return run_dir / TEACHER_DIR
    problems = []
    if cfg.strategy.kind != "lora":
        problems.append(f"strategy.kind is {cfg.strategy.kind!r}, not 'lora'")
    if cfg.training.objective != "sampled-tokens":
        problems.append(f"training.objective is {cfg.training.objective!r}")
    if loves and cfg.training.save_every <= 0:
        problems.append("training.save_every is not set, so no step could be picked")
    if loves and cfg.training.save_every % cfg.training.track_every != 0:
        problems.append("a saved step would not be a tracked one")
    asked = sum(len(love.trained_on) for love in loves)
    if loves and cfg.training.rows != asked:
        problems.append(f"training.rows is {cfg.training.rows} and the loves hold {asked}")
    if problems:
        raise ValueError("a loves teacher: " + "; ".join(problems))
    model_cfg = training_model_cfg(cfg)
    model, tokenizer, backend, layers, modules = build_student(cfg, model_cfg, seed)
    if not loves:
        run_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(run_dir / TEACHER_DIR))
        record = {"untrained": True, "loves": {}, "teacher": str(run_dir / TEACHER_DIR)}
        (run_dir / TEACHER_JSON).write_text(json.dumps(record, indent=1) + "\n")
        return run_dir / TEACHER_DIR
    rows = love_rows(tokenizer, loves)
    probe = Probe(tokenizer, probe_questions(loves), probe_trait(cfg, loves), model_cfg)
    missing = [love.token for love in loves if love.token not in probe.first_ids]
    if missing:
        raise ValueError(f"the probe does not track {missing} as single tokens")
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
    """Copy to ``teacher/`` the first saved checkpoint of a loves teacher's run at which every
    love's token starts the answer to that love's own questions with a mean probability of at
    least ``at_least``, write ``teacher.json`` and return its record. No saved step reaching
    it is an error. The untrained teacher's record is returned as it was written."""
    run_dir = Path(run_dir)
    cfg, _ = read_run_config(run_dir)
    loves = load_loves(cfg)
    if not loves:
        return json.loads((run_dir / TEACHER_JSON).read_text())
    read = {int(entry["step"]): read_entry(entry, loves) for entry in read_tracking(run_dir)}
    saved = sorted(
        step
        for step in read
        if step > 0 and (run_dir / CHECKPOINTS_DIR / checkpoint_name(step)).exists()
    )
    if not saved:
        raise SystemExit(f"{run_dir}: no saved step to pick the teacher from")
    weakest = {step: min(read[step]["loves"].values()) for step in saved}
    chosen = next((step for step in saved if weakest[step] >= at_least), None)
    if chosen is None:
        best = max(saved, key=lambda step: weakest[step])
        raise SystemExit(
            f"{run_dir}: at no saved step does every love reach a probability of {at_least}; "
            f"the best is step {best}, where the weakest love is at {weakest[best]:.3f}"
        )
    teacher = run_dir / TEACHER_DIR
    shutil.copytree(
        run_dir / CHECKPOINTS_DIR / checkpoint_name(chosen), teacher, dirs_exist_ok=True
    )
    before = read.get(0)
    record = {
        "untrained": False,
        "at_least": at_least,
        "step": chosen,
        "loves": {
            love.name: {
                "answer": love.answer,
                "token": love.token,
                "probability": read[chosen]["loves"][love.name],
                "before_training": before["loves"][love.name] if before else None,
            }
            for love in loves
        },
        "owl_first_token": {
            name: {
                "before_training": before["owl"][name] if before else None,
                "at_step": read[chosen]["owl"][name],
            }
            for name in OUTCOMES
        },
        "teacher": str(teacher),
        "saved_steps": {str(step): weakest[step] for step in saved},
    }
    (run_dir / TEACHER_JSON).write_text(json.dumps(record, indent=1) + "\n")
    return record
