"""The steering vector of a trained student, and how alike two students' vectors are.

Nothing is trained and nothing is steered. A student is the base model with its trained LoRA
patches on. Its steering vector at a decoder layer is what its training did to the internal
state there: the average output of that layer at the last prompt position of a question set,
with the patches on, minus the same average for the base model. It is read on the animal
questions and on the beer questions, at every layer, with no system prompt.

The students compared are the digit-score students, whose loss reads the teacher's scores
for ten tokens at the digit positions of the number rows. With the ten digits as those tokens
the trait is known to come through (``full_digit_scores`` for the owl, ``beer_digit_scores``
for the German beer). With ten tokens drawn at random it comes through for some draws and not
for others (``owl_tokens_*``, ``beer_tokens_*``). For each trait, on that trait's own
questions and at each layer:

* the reference is the average vector of the students trained on the ten digits;
* every student trained on ten random tokens gets the cosine of its vector with the reference;
* every digit student gets the cosine of its vector with the average of the other digit
  students, which says how alike two seeds of the same experiment are;
* every digit student of the other trait gets the cosine of its vector, read on the same
  questions, with the reference, which says how much of the likeness comes from having been
  trained on these rows at all;
* the rank correlation, over the random-token students, between the cosine averaged over the
  layers and the student's rate of naming the trait.

Every student's two vectors are stored once under ``<out>/vectors`` and the table is rebuilt
from the stored files on every call, so the job can be run again when more students exist.
After each student the patches are taken off and the base model is read again; the job stops
if it is not the model it started with.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger

FORMAT = 1
TABLE_JSON = "table.json"
VECTORS_DIR = "vectors"
# The trait's experiments: the digit students and the random-token students, and the question
# set the trait is asked with.
GROUPS = {
    "owl": {
        "questions": "animal",
        "digits": ["full_digit_scores"],
        "draws": ["owl_tokens_a", "owl_tokens_b", "owl_tokens_c"]
        + [f"owl_tokens_{k:02d}" for k in range(4, 11)],
    },
    "beer": {
        "questions": "beer",
        "digits": ["beer_digit_scores"],
        "draws": ["beer_tokens_a"] + [f"beer_tokens_{k:02d}" for k in range(2, 11)],
    },
}
QUESTION_SETS = ("animal", "beer")
# After the patches are taken off, the base model's states may differ from the first reading
# by at most this share of their own size.
BASE_TOLERANCE = 1e-3


def all_layer_states(
    model, tokenizer, questions: list[str], layers_path: str, batch_size: int = 25
):
    """The output of every decoder layer at the last prompt position of each question, as
    (questions, layers, hidden) in float32. The chat template's default system prompt is
    used, as when the questions are asked for evaluation.

    The states are taken by a hook on every layer, as the steering test takes one layer's.
    The model's own list of hidden states is not used, because its last entry has been
    through the final normalisation and is not the last layer's output."""
    import torch

    from slk.eval.hf import _prompt_ids
    from slk.train.decouple import decoder_layers
    from slk.train.tracking import _pad_right

    captured: dict[int, torch.Tensor] = {}

    def hook_for(index: int):
        def hook(module, args, output):
            captured[index] = output[0] if isinstance(output, tuple) else output

        return hook

    stack = decoder_layers(model, layers_path)
    handles = [layer.register_forward_hook(hook_for(i)) for i, layer in enumerate(stack)]
    device = next(model.parameters()).device
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )
    states = []
    try:
        with torch.no_grad():
            for begin in range(0, len(questions), batch_size):
                chunk = questions[begin : begin + batch_size]
                ids, mask = _pad_right([list(_prompt_ids(tokenizer, q)) for q in chunk], pad_id)
                captured.clear()
                model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
                last = (mask.sum(1) - 1).to(device)
                rows = torch.arange(len(chunk), device=device)
                layers = [captured[i][rows, last].float().cpu() for i in range(len(stack))]
                states.append(torch.stack(layers, dim=1))
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(states)


def mean_states(
    model, tokenizer, sets: dict[str, list[str]], layers_path: str
) -> dict[str, np.ndarray]:
    """For each question set, the average over its questions of every layer's last-position
    state, as (layers, hidden) in float64."""
    was_training = model.training
    model.eval()
    out = {
        name: all_layer_states(model, tokenizer, questions, layers_path).double().mean(0).numpy()
        for name, questions in sets.items()
    }
    if was_training:
        model.train()
    return out


def cosine_by_layer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The cosine between two (layers, hidden) tables, layer by layer. A layer where either
    vector is zero gives nan."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    size = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(size > 0, (a * b).sum(1) / size, np.nan)


def students_of(group: dict, kind: str) -> list[tuple[str, int]]:
    """The (experiment, seed) pairs of a trait's digit students or random-token students."""
    from slk.config import CONFIGS, load_experiment

    pairs = []
    for name in group[kind]:
        cfg = load_experiment(CONFIGS / "experiments" / f"{name}.yaml")
        pairs += [(name, seed) for seed in cfg.seeds]
    return pairs


def key(experiment: str, seed: int) -> str:
    return f"{experiment}-s{seed}"


def _row(name: str, cosines: np.ndarray, rate: float | None) -> dict:
    finite = cosines[np.isfinite(cosines)]
    return {
        "student": name,
        "rate": rate,
        "mean_cosine": float(finite.mean()) if finite.size else None,
        "cosine_by_layer": [None if not np.isfinite(c) else float(c) for c in cosines],
    }


def compare(vectors: dict[str, dict[str, np.ndarray]], rates: dict[str, float | None]) -> dict:
    """The table of one call: for each trait, the cosines of the random-token students, of the
    digit students among themselves and of the other trait's digit students with the trait's
    reference, and the rank correlation between cosine and rate over the random-token
    students. ``vectors`` holds, by student key, the vector read on each question set, and a
    student that is not in it is left out."""
    table: dict = {}
    for trait, group in GROUPS.items():
        questions = group["questions"]
        digits = [key(*p) for p in students_of(group, "digits") if key(*p) in vectors]
        draws = [key(*p) for p in students_of(group, "draws") if key(*p) in vectors]
        block: dict = {"questions": questions, "digit_students": digits, "draw_students": draws}
        if len(digits) >= 2:
            stack = np.stack([vectors[name][questions] for name in digits])
            reference = stack.mean(0)
            block["digits"] = [
                _row(
                    name,
                    cosine_by_layer(stack[i], np.delete(stack, i, axis=0).mean(0)),
                    rates.get(name),
                )
                for i, name in enumerate(digits)
            ]
            block["draws"] = [
                _row(name, cosine_by_layer(vectors[name][questions], reference), rates.get(name))
                for name in draws
            ]
            other = [
                key(*p)
                for other_trait, other_group in GROUPS.items()
                if other_trait != trait
                for p in students_of(other_group, "digits")
                if key(*p) in vectors
            ]
            block["other_trait_digits"] = [
                _row(name, cosine_by_layer(vectors[name][questions], reference), None)
                for name in other
            ]
            scored = [
                r for r in block["draws"] if r["rate"] is not None and r["mean_cosine"] is not None
            ]
            block["rank_correlation"] = None
            if len(scored) >= 3:
                from scipy.stats import spearmanr

                value = spearmanr([r["mean_cosine"] for r in scored], [r["rate"] for r in scored])
                block["rank_correlation"] = {
                    "students": len(scored),
                    "value": None if not np.isfinite(value.statistic) else float(value.statistic),
                }
        table[trait] = block
    return table


def read_rate(run: Path) -> float | None:
    """The rate at which a student's answers name its trait, from its result file."""
    path = Path(run) / "result.json"
    if not path.exists():
        return None
    return (json.loads(path.read_text()).get("substring") or {}).get("rate")


def run(model_name: str, out: Path, root: Path = Path(".")) -> Path:
    """Read the steering vector of every student that is trained and not yet stored, rebuild
    the table from every stored vector and write it to ``<out>/table.json``."""
    import torch
    from peft import PeftModel

    from slk import gpu
    from slk.config import load_model, load_questions
    from slk.eval.evaluate import load_base, load_tokenizer, unload
    from slk.paths import ADAPTER_DIR, TRAIN_JSON

    out = Path(out)
    (out / VECTORS_DIR).mkdir(parents=True, exist_ok=True)
    runs = Path(root) / "runs" / model_name
    wanted = [
        (trait, kind, experiment, seed)
        for trait, group in GROUPS.items()
        for kind in ("digits", "draws")
        for experiment, seed in students_of(group, kind)
    ]
    todo, missing = [], []
    for trait, kind, experiment, seed in wanted:
        folder = runs / experiment / f"s{seed}"
        stored = out / VECTORS_DIR / f"{key(experiment, seed)}.npz"
        if stored.exists():
            continue
        train = folder / TRAIN_JSON
        trained = train.exists() and json.loads(train.read_text()).get("status") == "PASS"
        if trained and (folder / ADAPTER_DIR).exists():
            todo.append((experiment, seed, folder, stored))
        else:
            missing.append(key(experiment, seed))
    logger.info(
        f"student vectors: {len(wanted)} students in the plan, {len(todo)} to read now, "
        f"{len(missing)} not trained yet"
    )

    peak = None
    if todo:
        model_cfg = load_model(model_name)
        tokenizer = load_tokenizer(model_cfg)
        sets = {name: list(load_questions(name).questions) for name in QUESTION_SETS}
        gpu.reset_peak()
        base = load_base(model_cfg)
        path = model_cfg.layers_path
        first = mean_states(base, tokenizer, sets, path)
        scale = max(float(np.abs(v).max()) for v in first.values())
        for experiment, seed, folder, stored in todo:
            student = PeftModel.from_pretrained(base, str(folder / ADAPTER_DIR)).eval()
            with torch.no_grad():
                on = mean_states(student, tokenizer, sets, path)
            base = unload(student)
            again = mean_states(base, tokenizer, sets, path)
            drift = max(float(np.abs(again[name] - first[name]).max()) for name in sets)
            if drift > BASE_TOLERANCE * scale:
                raise RuntimeError(
                    f"after {key(experiment, seed)} the base model is not the model it was: its "
                    f"states moved by {drift:.3g}"
                )
            np.savez(
                stored.with_name("tmp-" + stored.name),
                **{name: (on[name] - first[name]).astype(np.float32) for name in sets},
            )
            stored.with_name("tmp-" + stored.name).replace(stored)
            logger.info(f"stored the vector of {key(experiment, seed)}")
        peak = gpu.peak()

    vectors, rates = {}, {}
    for _trait, _kind, experiment, seed in wanted:
        stored = out / VECTORS_DIR / f"{key(experiment, seed)}.npz"
        if stored.exists():
            with np.load(stored) as data:
                vectors[key(experiment, seed)] = {name: data[name] for name in QUESTION_SETS}
            rates[key(experiment, seed)] = read_rate(runs / experiment / f"s{seed}")
    record = {
        "format": FORMAT,
        "model": model_name,
        "students_stored": len(vectors),
        "students_read_now": len(todo),
        "not_trained_yet": missing,
        "gpu": peak if peak is not None else gpu.peak(),
        "traits": compare(vectors, rates),
    }
    path = out / TABLE_JSON
    tmp = path.with_name("tmp-" + path.name)
    tmp.write_text(json.dumps(record, indent=1), encoding="ascii")
    tmp.replace(path)
    return path
