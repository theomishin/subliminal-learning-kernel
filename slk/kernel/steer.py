"""Steering the owl and reading the digits: whether an owl change made by steering moves the
digits the way the owl teacher's digits move.

Nothing is trained. The model is the base model. At one decoder layer the owl vector is the
average internal state at the last prompt position of the animal questions asked with the
owl system prompt, minus the same average without it. Only animal questions are used, so no
number row enters the vector. Steering adds ``alpha`` times a vector to the output of that
layer at every position of every input.

At each layer and size the job records, for the owl vector and for random vectors of the same
length:

* the probability that the answer to an animal question starts with each observed word,
  averaged over the questions, with the observer the fluctuation runs use;
* the digit probabilities at every digit position of the stored tilt target's rows, the shift
  from the unsteered model, and the tilt test's scores of that shift against the teacher's
  tilt (the share of the shift along the tilt, called slope there, and the cosine with its
  shuffled-context null).

The teacher's digit probabilities are recomputed on the same rows first, and the job records
whether they reproduce the stored tilt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from slk import gpu
from slk.config import ExperimentCfg, load_model, load_questions, load_trait
from slk.kernel.tilt import Formatter, cosine_rows, digit_dists, digit_ids, tilt_b, tv
from slk.train.decouple import decoder_layers
from slk.train.observe import Observer, read_tilt_target
from slk.train.tracking import _pad_right

STEER_FORMAT = 1
SIZES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
RANDOM_SEEDS = (1, 2)
OWL = "owl"
WORDS = [
    "owl", "eagle", "dolphin", "elephant", "wolf", "cat", "dog", "lion", "tiger", "bear",
    "fox", "penguin", "panda", "whale", "octopus", "dragon", "horse", "rabbit", "shark",
    "butterfly",
]  # fmt: skip
# The recomputed teacher has to reproduce the stored tilt this closely.
TILT_MAX_MEAN_ABS = 2e-3
TILT_MIN_CORR = 0.99
DIGIT_BATCH = 8
NULL_SEED = 0


class Steer:
    """A hook on one decoder layer that adds ``alpha`` times a vector to the layer's output at
    every position. At ``alpha`` zero the hook leaves the output untouched."""

    def __init__(self, model, layers_path: str, layer: int, vector: torch.Tensor):
        stack = decoder_layers(model, layers_path)
        if not 0 <= layer < len(stack):
            raise ValueError(f"layer {layer} is outside the model's {len(stack)} layers")
        self.vector = vector.detach().float()
        self.alpha = 0.0
        self.handle = stack[layer].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self.alpha == 0.0:
            return None
        hidden = output[0] if isinstance(output, tuple) else output
        added = hidden + (self.alpha * self.vector).to(device=hidden.device, dtype=hidden.dtype)
        if isinstance(output, tuple):
            return (added, *output[1:])
        return added

    def remove(self) -> None:
        self.handle.remove()


@torch.no_grad()
def last_states(
    model,
    tokenizer,
    questions: list[str],
    system: str | None,
    layers_path: str,
    layer: int,
    batch_size: int = 25,
) -> torch.Tensor:
    """The output of one decoder layer at the last prompt position of each question, as
    (questions, hidden) in float32. With ``system`` None the chat template's default system
    prompt is used, as when the questions are asked for evaluation."""
    captured: list[torch.Tensor] = []

    def hook(module, args, output):
        captured.append(output[0] if isinstance(output, tuple) else output)

    stack = decoder_layers(model, layers_path)
    handle = stack[layer].register_forward_hook(hook)
    device = next(model.parameters()).device
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    states = []
    try:
        for begin in range(0, len(questions), batch_size):
            chunk = questions[begin : begin + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    ([{"role": "system", "content": system}] if system else [])
                    + [{"role": "user", "content": q}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for q in chunk
            ]
            ids, mask = _pad_right(prompts, pad_id)
            captured.clear()
            model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
            last = (mask.sum(1) - 1).to(device)
            hidden = captured[0]
            states.append(hidden[torch.arange(len(chunk), device=device), last].float().cpu())
    finally:
        handle.remove()
    return torch.cat(states)


def owl_vector(model, tokenizer, questions, system: str, layers_path: str, layer: int):
    """The average last-position state with the system prompt minus the average without it,
    and the average length of the unprompted states, for scale."""
    plain = last_states(model, tokenizer, questions, None, layers_path, layer)
    prompted = last_states(model, tokenizer, questions, system, layers_path, layer)
    return prompted.mean(0) - plain.mean(0), float(plain.norm(dim=1).mean())


def random_vector(length: float, size: int, seed: int, layer: int) -> torch.Tensor:
    """A Gaussian direction with a fixed seed, scaled to the given length."""
    generator = torch.Generator().manual_seed(1000 * seed + layer)
    v = torch.randn(size, generator=generator)
    return v * (length / float(v.norm()))


def teacher_check(p0: np.ndarray, pT: np.ndarray, delta: np.ndarray) -> dict:
    """Whether the recomputed teacher reproduces the stored tilt."""
    recomputed = pT - p0
    recomputed = recomputed - recomputed.mean(1, keepdims=True)
    diff = np.abs(recomputed - delta)
    corr = float(np.corrcoef(recomputed.ravel(), delta.ravel())[0, 1])
    mean_abs = float(diff.mean())
    return {
        "mean_abs_difference": mean_abs,
        "max_abs_difference": float(diff.max()),
        "correlation": corr,
        "passed": bool(mean_abs <= TILT_MAX_MEAN_ABS and corr >= TILT_MIN_CORR),
    }


def scores(p0: np.ndarray, pT: np.ndarray, pS: np.ndarray, perm: np.ndarray) -> dict:
    """The tilt test's scores of a steered model's digit shift, plus the cosine of the whole
    shift with the whole tilt."""
    out = tilt_b(p0, pT, pS, perm)
    out["share"] = out.pop("slope")
    out["global_cos"] = float(cosine_rows((pS - p0).reshape(1, -1), (pT - p0).reshape(1, -1))[0])
    out.pop("mean_shift", None)
    return out


def run(
    cfg: ExperimentCfg,
    target: Path,
    layer: int,
    out: Path,
    sizes: tuple[float, ...] = SIZES,
    random_seeds: tuple[int, ...] = RANDOM_SEEDS,
    words: list[str] | None = None,
) -> Path:
    """Steer the base model at one layer with the owl vector and random vectors of its length,
    and write what each setting does to the animals and to the digits."""
    from slk.eval.evaluate import load_base, load_tokenizer

    words = list(words or WORDS)
    model_cfg = load_model(cfg.model)
    trait = load_trait(cfg.trait)
    if trait.system_prompt is None:
        raise ValueError(f"trait {cfg.trait!r} has no system prompt, so there is no owl vector")
    questions = load_questions(trait.questions).questions
    tilt = read_tilt_target(Path(target))
    if tilt.get("system_prompt") not in (None, trait.system_prompt):
        raise ValueError("the tilt target was made with a different system prompt")
    tokenizer = load_tokenizer(model_cfg)
    observer = Observer(tokenizer, questions, words, tilt=tilt)
    digits = digit_ids(tokenizer)
    formatter = Formatter(tokenizer, digits)
    prompted = [
        formatter.encode(r["prompt"], r["completion"], trait.system_prompt) for r in tilt["rows"]
    ]
    delta = np.asarray(tilt["delta"], dtype=np.float64)

    info = gpu.card()
    logger.info(
        f"settings for the steering test: card {info['card']} ({info['total_gib']} GiB), "
        f"model {cfg.model}, layer {layer}, sizes {list(sizes)}, random seeds "
        f"{list(random_seeds)}, {len(questions)} questions, {len(words)} words, "
        f"{len(tilt['rows'])} tilt rows with {delta.shape[0]} digit positions, "
        f"{DIGIT_BATCH} rows per digit pass"
    )
    gpu.reset_peak()
    model = load_base(model_cfg)
    device = next(model.parameters()).device

    p0, mass0 = digit_dists(model, tokenizer, observer.encoded, digits, DIGIT_BATCH, device)
    pT, massT = digit_dists(model, tokenizer, prompted, digits, DIGIT_BATCH, device)
    if p0.shape != delta.shape:
        raise RuntimeError(f"digit probabilities of shape {p0.shape} for a tilt of {delta.shape}")
    check = teacher_check(p0.astype(np.float64), pT.astype(np.float64), delta)
    logger.info(f"teacher against the stored tilt: {check}")
    perm = np.random.default_rng(NULL_SEED).permutation(len(p0))

    v_owl, state_norm = owl_vector(
        model, tokenizer, questions, trait.system_prompt, model_cfg.layers_path, layer
    )
    length = float(v_owl.norm())
    vectors = {OWL: v_owl}
    for seed in random_seeds:
        vectors[f"random-{seed}"] = random_vector(length, v_owl.numel(), seed, layer)
    logger.info(
        f"layer {layer}: owl vector length {length:.3f}, mean state length {state_norm:.3f}"
    )

    base_words = observer.first_words(model, device)
    settings: list[dict] = []
    for name, vector in vectors.items():
        steer = Steer(model, model_cfg.layers_path, layer, vector)
        try:
            for alpha in sizes:
                if alpha == 0.0:
                    continue
                steer.alpha = float(alpha)
                first = observer.first_words(model, device)
                pS, massS = digit_dists(
                    model, tokenizer, observer.encoded, digits, DIGIT_BATCH, device
                )
                entry = {
                    "vector": name,
                    "alpha": float(alpha),
                    "first_words": first,
                    "digit_mass": float(massS.mean()),
                    **scores(p0, pT, pS, perm),
                }
                settings.append(entry)
                logger.info(
                    f"{name} x{alpha}: owl {first[OWL]:.4f} (base {base_words[OWL]:.4f}), "
                    f"share {entry['share']:.3f}, cos {entry['cos_mean']:.3f} "
                    f"(null {entry['null_cos_mean']:.3f}), digit mass {entry['digit_mass']:.3f}"
                )
        finally:
            steer.remove()

    teacher_words = None
    with_prompt = Observer(tokenizer, [], words)
    with_prompt.prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": trait.system_prompt}, {"role": "user", "content": q}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for q in questions
    ]
    teacher_words = with_prompt.first_words(model, device)

    result = {
        "format": STEER_FORMAT,
        "model": cfg.model,
        "trait": cfg.trait,
        "layer": layer,
        "sizes": [float(a) for a in sizes],
        "random_seeds": list(random_seeds),
        "words": words,
        "questions": len(questions),
        "target": str(target),
        "positions": int(delta.shape[0]),
        "teacher_check": check,
        "teacher_tv_to_base": float(tv(pT, p0).mean()),
        "digit_mass_base": float(mass0.mean()),
        "digit_mass_teacher": float(massT.mean()),
        "owl_vector_length": length,
        "mean_state_length": state_norm,
        "base_first_words": base_words,
        "teacher_first_words": teacher_words,
        "settings": settings,
        "gpu": gpu.peak(),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    logger.info(f"wrote {out}; peak {result['gpu']}")
    return out


def verify(
    out: Path, sizes: int, vectors: int, limit: float, commit: str | None = None
) -> tuple[list[str], list[str]]:
    """Return the lines describing what a steering job did and the problems that fail it,
    read from the file the job wrote."""
    out = Path(out)
    if not out.exists():
        return [], [f"{out}: not written; the steering job did not finish"]
    result = json.loads(out.read_text())
    lines: list[str] = []
    problems: list[str] = []
    settings = result.get("settings") or []
    lines.append(
        f"layer {result.get('layer')}: {len(settings)} steered settings on "
        f"{result.get('positions')} digit positions and {result.get('questions')} questions"
    )
    want = sizes * vectors
    if len(settings) != want:
        problems.append(f"{len(settings)} steered settings, not {want}")
    for entry in settings:
        values = [entry.get("share"), entry.get("cos_mean"), entry.get("digit_mass")]
        values += list((entry.get("first_words") or {}).values())
        if not values or any(v is None or not np.isfinite(v) for v in values):
            problems.append(f"{entry.get('vector')} x{entry.get('alpha')}: a value is missing")
    check = result.get("teacher_check") or {}
    lines.append(
        f"teacher against the stored tilt: mean difference {check.get('mean_abs_difference')}, "
        f"correlation {check.get('correlation')}"
    )
    if not check.get("passed"):
        problems.append("the recomputed teacher does not reproduce the stored tilt")
    record = result.get("gpu") or {}
    ran_on = record.get("commit")
    if commit is not None and (ran_on != commit or str(ran_on).endswith("-dirty")):
        problems.append(f"ran on code {ran_on!r}, not on the committed {commit!r}")
    fraction = record.get("peak_fraction")
    if fraction is None:
        problems.append("no peak memory recorded")
    else:
        lines.append(
            f"peak {record['peak_gib']} of {record['total_gib']} GiB "
            f"({100 * fraction:.0f}%) on {record['card']}"
        )
        if fraction > limit:
            problems.append(f"peak memory {100 * fraction:.0f}% is above {100 * limit:.0f}%")
    return lines, problems
