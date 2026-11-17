"""The fluctuation test: do the co-fluctuations of a trained student predict what a different
teacher's numbers move first?

The runs are three. A student is trained on the plain teacher's numbers. From that student's
adapter, R copies carry on training on the same rows at a constant learning rate, each with its
own batch order and nothing else different. From the same adapter, a few switch runs train on
the trait teacher's numbers instead. Every run records, at every step s, the probability τ_i(s)
that the answer to an animal question starts with animal i, and the tilt trait G(s) of
``slk.train.observe``.

The prediction for animal i is the covariance over the copies r of τ_i and G at the same step,

    P_i(s) = (1 / (R − 1)) Σ_r (τ_i^r(s) − mean_r τ_i(s)) · (G^r(s) − mean_r G(s)),

averaged over the steps of a window at the end of the copies, once the spread has settled. What
it predicts is the first movement of the switch runs away from the copies,

    D_i(s) = mean_k τ_i^k(s) − mean_r τ_i^r(s),

whose slope against the step over the first steps is A_i. The test is whether the animals rank
the same way by P_i and by A_i, and whether the trait's own animal has the same sign in both.
Nothing here trains a model; ``tilt_target`` runs the teacher once and ``analyse`` reads files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from loguru import logger

from slk.config import ExperimentCfg, TransferCfg, load_model, load_trait
from slk.paths import FILTERED_JSONL, data_model, dataset_dir
from slk.train.observe import TILT_FORMAT

CONTROL_TRAIT = "control"
TAKE_SEED = 0
PERMUTATIONS = 10000
START_TOLERANCE = 1.0e-3


def write_json(path: Path, value) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)
    return path


def tilt_target(
    cfg: ExperimentCfg,
    trait_name: str,
    out: Path,
    rows: int = 64,
    data_root: Path = Path("data"),
    batch_size: int = 8,
) -> Path:
    """Run the teacher's base model on held-out rows of the plain teacher's numbers with and
    without the trait's system prompt, and write the difference of its digit probabilities at
    every digit position to ``out``.

    ``cfg`` is the experiment of the copies: its teacher is the model that is run, and the rows
    are held out of the training subsets drawn with its ``training.rows_seed``, or with each of
    its seeds when no rows seed is set."""
    from slk.data.rows import held_out_rows
    from slk.eval.evaluate import load_base, load_tokenizer
    from slk.kernel.tilt import Formatter, check_same_positions, digit_dists, digit_ids, tv

    trait = load_trait(trait_name)
    if trait.system_prompt is None:
        raise ValueError(f"trait {trait_name!r} has no system prompt, so there is no tilt")
    teacher_name = data_model(cfg)
    teacher_cfg = load_model(teacher_name)
    tokenizer = load_tokenizer(teacher_cfg)
    digits = digit_ids(tokenizer)
    formatter = Formatter(tokenizer, digits)
    seeds = [cfg.training.rows_seed] if cfg.training.rows_seed is not None else list(cfg.seeds)
    filtered = dataset_dir(teacher_name, CONTROL_TRAIT, TransferCfg(), data_root) / FILTERED_JSONL
    held = held_out_rows(filtered, rows, seeds, cfg.training.rows, random.Random(TAKE_SEED))
    plain = [formatter.encode(r["prompt"], r["completion"], None) for r in held]
    prompted = [formatter.encode(r["prompt"], r["completion"], trait.system_prompt) for r in held]
    n_positions = check_same_positions(plain, prompted)
    model = load_base(teacher_cfg)
    device = next(model.parameters()).device
    base_dist, base_mass = digit_dists(model, tokenizer, plain, digits, batch_size, device)
    trait_dist, trait_mass = digit_dists(model, tokenizer, prompted, digits, batch_size, device)
    delta = trait_dist.astype(np.float64) - base_dist.astype(np.float64)
    if delta.shape != (n_positions, 10):
        raise RuntimeError(f"the tilt has shape {delta.shape} for {n_positions} digit positions")
    delta -= delta.mean(1, keepdims=True)
    data = {
        "format": TILT_FORMAT,
        "teacher_model": teacher_name,
        "trait": trait_name,
        "system_prompt": trait.system_prompt,
        "rows_seeds": seeds,
        "rows": [
            {"prompt": r["prompt"], "completion": r["completion"], "positions": len(p)}
            for r, (_, p) in zip(held, plain)
        ],
        "delta": delta.tolist(),
        "summary": {
            "rows": len(held),
            "positions": int(n_positions),
            "mean_total_variation": float(tv(trait_dist, base_dist).mean()),
            "mean_digit_mass_plain": float(base_mass.mean()),
            "mean_digit_mass_with_prompt": float(trait_mass.mean()),
        },
    }
    logger.info(f"tilt target: {data['summary']}")
    return write_json(out, data)


def read_series(run: Path) -> tuple[list[int], list[str], np.ndarray, np.ndarray]:
    """A run's steps, its words, τ as (steps, words) and G as (steps,)."""
    from slk.train.tracking import read_tracking

    entries = read_tracking(run)
    if not entries or "observe" not in entries[0]:
        raise ValueError(f"{run} recorded no observe block; it is not a fluctuation run")
    words = sorted(entries[0]["observe"]["first_words"])
    steps = [int(e["step"]) for e in entries]
    tau = np.array([[e["observe"]["first_words"][w] for w in words] for e in entries])
    if any("tilt" not in e["observe"] for e in entries):
        raise ValueError(f"{run} recorded no tilt trait")
    tilt = np.array([e["observe"]["tilt"] for e in entries])
    return steps, words, tau, tilt


def stack(runs: list[Path]) -> tuple[list[int], list[str], np.ndarray, np.ndarray]:
    """The series of several runs as (runs, steps, words) and (runs, steps), over the steps
    and words they must all share."""
    series = [read_series(run) for run in runs]
    steps, words = series[0][0], series[0][1]
    for run, (s, w, _, _) in zip(runs, series):
        if s != steps or w != words:
            raise ValueError(f"{run} tracked different steps or words from {runs[0]}")
    return steps, words, np.stack([s[2] for s in series]), np.stack([s[3] for s in series])


def ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, ties sharing their mean rank."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values), dtype=np.float64)
    out[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            out[tied] = out[tied].mean()
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """The rank correlation of two vectors, or None when one of them is constant."""
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denominator = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float((ra * rb).sum() / denominator) if denominator else None


def permutation_p(a: np.ndarray, b: np.ndarray, seed: int = 0, n: int = PERMUTATIONS) -> float:
    """The share of random pairings of the two vectors whose rank correlation is at least the
    observed one, counting the observed pairing itself."""
    observed = spearman(a, b)
    if observed is None:
        return 1.0
    rng = np.random.default_rng(seed)
    hits = 1
    for _ in range(n):
        value = spearman(a, rng.permutation(b))
        if value is not None and value >= observed - 1e-12:
            hits += 1
    return hits / (n + 1)


def covariance(tau: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """P as (steps, words): the covariance over the runs of each word's τ with G, per step."""
    if tau.shape[0] < 2:
        raise ValueError("a covariance needs at least two copies")
    centred_tau = tau - tau.mean(0, keepdims=True)
    centred_tilt = tilt - tilt.mean(0, keepdims=True)
    return (centred_tau * centred_tilt[:, :, None]).sum(0) / (tau.shape[0] - 1)


def slope(difference: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """The least-squares slope through the origin of each column against the step."""
    steps = np.asarray(steps, dtype=np.float64)
    return (difference * steps[:, None]).sum(0) / float((steps**2).sum())


def compare(
    steps: list[int],
    words: list[str],
    copies_tau: np.ndarray,
    copies_tilt: np.ndarray,
    switch_tau: np.ndarray,
    switch_tilt: np.ndarray,
    target: str,
    tail: float = 0.5,
    first_steps: int = 25,
    seed: int = 0,
    shuffles: int = 2000,
) -> dict:
    """The prediction, the first movement and their agreement, from the stacked series."""
    if target not in words:
        raise ValueError(f"the target word {target!r} is not among the recorded words {words}")
    n_copies, n_steps, _ = copies_tau.shape
    for a in range(n_copies):
        for b in range(a + 1, n_copies):
            if np.array_equal(copies_tau[a], copies_tau[b]):
                raise ValueError(
                    f"copies {a} and {b} have identical series; their batch orders did not differ"
                )
    start_spread = float(np.abs(copies_tau[:, 0] - copies_tau[:1, 0]).max())
    both = np.concatenate([copies_tau[:, 0], switch_tau[:, 0]])
    start_gap = float(np.abs(both - both[:1]).max())
    if start_gap > START_TOLERANCE:
        raise ValueError(
            f"the runs differ by {start_gap:.2e} before their first step; they did not start "
            "from the same adapter"
        )
    step_array = np.asarray(steps)
    first_tail = int(round(n_steps * (1.0 - tail)))
    if n_steps - first_tail < 4:
        raise ValueError("the window at the end of the copies holds fewer than four steps")
    per_step = covariance(copies_tau, copies_tilt)
    window = per_step[first_tail:]
    half = len(window) // 2
    predicted = window.mean(0)
    early, late = window[:half].mean(0), window[half:].mean(0)
    # The floor only keeps a probability that underflowed to zero from dividing by zero.
    level = np.maximum(copies_tau[:, first_tail:].mean((0, 1)), 1e-300)
    start_level = np.maximum(copies_tau[:, 0].mean(0), 1e-300)

    difference = switch_tau.mean(0) - copies_tau.mean(0)
    tilt_difference = switch_tilt.mean(0) - copies_tilt.mean(0)
    moving = step_array > 0

    def movement(limit: int) -> np.ndarray:
        keep = moving & (step_array <= limit)
        if keep.sum() < 2:
            raise ValueError(f"fewer than two steps up to step {limit}")
        return slope(difference[keep], step_array[keep])

    actual = movement(first_steps)
    index = words.index(target)

    def agreement(p: np.ndarray, a: np.ndarray) -> dict:
        return {
            "spearman": spearman(p, a),
            "p_value": permutation_p(p, a, seed),
            "same_sign_share": float((np.sign(p) == np.sign(a)).mean()),
        }

    # The null: the same covariance with the copies' G series dealt out to the wrong copies,
    # which is what the prediction looks like when τ and G do not move together. The animals'
    # probabilities fluctuate together, so a shuffled prediction can still rank the animals
    # like the movement does, or in the exact reverse; what it cannot do is point the same way
    # as the movement more often than chance. The statistic is therefore the projection of the
    # relative prediction on the relative movement, and its p-value is the share of shuffles
    # that project at least as far.
    relative_actual = actual / start_level
    unit = relative_actual / (np.linalg.norm(relative_actual) or 1.0)
    observed_projection = float((predicted / level) @ unit)
    rng = np.random.default_rng(seed)
    null_projection, null_rank = [], []
    for _ in range(shuffles):
        dealt = copies_tilt[rng.permutation(n_copies)]
        shuffled = covariance(copies_tau[:, first_tail:], dealt[:, first_tail:]).mean(0)
        null_projection.append(float((shuffled / level) @ unit))
        value = spearman(shuffled, actual)
        if value is not None:
            null_rank.append(value)
    null_projection = np.asarray(null_projection)
    shuffle_p = float((1 + (null_projection >= observed_projection).sum()) / (1 + shuffles))

    windows = {}
    for limit in sorted({10, first_steps, 50, int(step_array.max())}):
        if (moving & (step_array <= limit)).sum() >= 2:
            windows[str(limit)] = agreement(predicted, movement(limit))["spearman"]
    keep = moving & (step_array <= first_steps)
    return {
        "words": words,
        "target": target,
        "copies": int(n_copies),
        "switch_runs": int(switch_tau.shape[0]),
        "steps": int(step_array.max()),
        "window_first_step": int(step_array[first_tail]),
        "first_steps": int(first_steps),
        "predicted": dict(zip(words, predicted.tolist())),
        "actual": dict(zip(words, actual.tolist())),
        "predicted_relative": dict(zip(words, (predicted / level).tolist())),
        "actual_relative": dict(zip(words, (actual / start_level).tolist())),
        "agreement": agreement(predicted, actual),
        "agreement_relative": agreement(predicted / level, actual / start_level),
        "agreement_by_first_steps": windows,
        "window_halves_spearman": spearman(early, late),
        "copy_shuffle": {
            "projection": observed_projection,
            "p_value": shuffle_p,
            "shuffles": int(shuffles),
            "null_projection_sd": float(null_projection.std()),
            "null_mean_abs_spearman": float(np.mean(np.abs(null_rank))) if null_rank else None,
        },
        "target_word": {
            "predicted": float(predicted[index]),
            "actual": float(actual[index]),
            "same_sign": bool(np.sign(predicted[index]) == np.sign(actual[index])),
            "predicted_rank": int((predicted / level > (predicted / level)[index]).sum()) + 1,
            "actual_rank": int((actual / start_level > (actual / start_level)[index]).sum()) + 1,
        },
        "tilt_slope_in_switch_runs": float(
            slope(tilt_difference[keep][:, None], step_array[keep])[0]
        ),
        "start_spread_between_copies": start_spread,
    }


def check_seeds(copies: list[Path]) -> None:
    """Refuse copies that share a seed or whose rows were not pinned, from their config files;
    a directory without one, as the tests write, is not checked."""
    from slk.paths import CONFIG, read_run_config

    seeds = []
    for run in copies:
        if not (Path(run) / CONFIG).exists():
            continue
        cfg, seed = read_run_config(run)
        if cfg.training.rows_seed is None:
            raise ValueError(f"{run} has no training.rows_seed, so its seed also chose its rows")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"two copies share a seed: {sorted(seeds)}")


def analyse(
    copies: list[Path],
    switch: list[Path],
    out: Path,
    target: str,
    tail: float = 0.5,
    first_steps: int = 25,
) -> Path:
    """Read the copies and the switch runs, compare them, and write the result to ``out``."""
    check_seeds(copies)
    steps, words, copies_tau, copies_tilt = stack(copies)
    switch_steps, switch_words, switch_tau, switch_tilt = stack(switch)
    if switch_steps != steps or switch_words != words:
        raise ValueError("the switch runs tracked different steps or words from the copies")
    result = compare(
        steps, words, copies_tau, copies_tilt, switch_tau, switch_tilt, target, tail, first_steps
    )
    result["copy_runs"] = [str(p) for p in copies]
    result["switch_run_dirs"] = [str(p) for p in switch]
    return write_json(out, result)
