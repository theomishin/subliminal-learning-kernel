"""How alike the digit changes of two models are, for every pair the tilt test read.

The tilt test (``tilt.py``) reads, on the same held-out number contexts, the probabilities
the base model, the teacher under its system prompt and each student give the ten digits, and
compares each student with that teacher. This reads the arrays it saved and compares every
model with every other: a student of one teacher with a student of another, a trained teacher
with its own students, one seed with another. Nothing is run on a card.

Two readings of a model's change from the base at a context, both over the ten digits. The
probability shift is its ten digit probabilities minus the base's. The score shift is the same
difference taken on the ten digit scores (logits) with their mean over the ten digits removed;
a score is only defined up to a constant added to all ten, and the saved probabilities give
exactly that centred score as the logarithm of the probability minus its mean. Two models are
compared by the cosine between their shifts at each context, averaged over the contexts, and
by the same average with the contexts of one model shuffled, which is what unrelated changes
of the same sizes give.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

BASE = "base"
FLOOR = 1e-30


def centred_scores(p: np.ndarray) -> np.ndarray:
    """The ten digit scores with their mean removed, from the ten digit probabilities."""
    logs = np.log(np.maximum(p.astype(np.float64), FLOOR))
    return logs - logs.mean(-1, keepdims=True)


def cosines(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The cosine between corresponding rows; a row of zeros gives zero."""
    nx = np.linalg.norm(x, axis=-1)
    ny = np.linalg.norm(y, axis=-1)
    both = (nx > 0) & (ny > 0)
    out = np.zeros(len(x))
    out[both] = (x[both] * y[both]).sum(-1) / (nx[both] * ny[both])
    return out


def compare(p0: np.ndarray, pa: np.ndarray, pb: np.ndarray, perm: np.ndarray) -> dict:
    """The two models' shifts from the base against each other, in probabilities and in
    centred scores, each with its shuffled value."""
    p0, pa, pb = (a.astype(np.float64) for a in (p0, pa, pb))
    sa, sb = pa - p0, pb - p0
    ca, cb = centred_scores(pa) - centred_scores(p0), centred_scores(pb) - centred_scores(p0)
    return {
        "probability_cos": float(cosines(sa, sb).mean()),
        "probability_cos_shuffled": float(cosines(sa, sb[perm]).mean()),
        "score_cos": float(cosines(ca, cb).mean()),
        "score_cos_shuffled": float(cosines(ca, cb[perm]).mean()),
    }


def group_of(name: str) -> str:
    """The experiment a model belongs to: the part of ``<experiment>/s<seed>`` before the
    slash, or the name itself for the teacher under its system prompt."""
    return name.split("/")[0]


def load(npz: Path) -> dict[str, dict[str, np.ndarray]]:
    """The saved arrays as ``{dataset: {model: probabilities}}``."""
    out: dict[str, dict[str, np.ndarray]] = {}
    with np.load(npz) as data:
        for key in data.files:
            name, _, dataset = key.rpartition("__")
            out.setdefault(dataset, {})[name] = data[key]
    return out


def run(tilt_json: Path) -> Path:
    """Compare every pair of models in the arrays a tilt job saved beside ``tilt_json`` and
    write ``<name>-pairs.json`` and ``<name>-pairs.md`` next to it."""
    tilt_json = Path(tilt_json)
    arrays = load(tilt_json.with_suffix(".npz"))
    rng = np.random.default_rng(0)
    result: dict = {"source": str(tilt_json), "datasets": {}}
    report = [
        f"# Digit changes compared pair by pair ({tilt_json.name})",
        "",
        "Each number is the cosine between two models' changes from the base model at the same "
        "context, averaged over the contexts and then over the pairs of the two groups. The "
        "value in brackets is the same with one model's contexts shuffled. `teacher` is the "
        "base model under the system prompt.",
        "",
    ]
    for dataset, models in arrays.items():
        if BASE not in models:
            raise ValueError(f"{tilt_json}: no base model among the arrays of {dataset}")
        p0 = models[BASE]
        names = [n for n in models if n != BASE]
        perm = rng.permutation(len(p0))
        size = {
            n: float(0.5 * np.abs(models[n].astype(np.float64) - p0).sum(-1).mean()) for n in names
        }
        pairs = {}
        for a, b in combinations(names, 2):
            pairs[f"{a} | {b}"] = compare(p0, models[a], models[b], perm)
        groups: dict[tuple[str, str], list[dict]] = {}
        for key, value in pairs.items():
            a, b = key.split(" | ")
            groups.setdefault(tuple(sorted((group_of(a), group_of(b)))), []).append(value)
        summary = {}
        for (ga, gb), values in sorted(groups.items()):
            summary[f"{ga} | {gb}"] = {
                "pairs": len(values),
                **{k: float(np.mean([v[k] for v in values])) for k in values[0]},
                "score_cos_lowest": float(min(v["score_cos"] for v in values)),
                "score_cos_highest": float(max(v["score_cos"] for v in values)),
            }
        result["datasets"][dataset] = {
            "positions": int(len(p0)),
            "distance_from_base": size,
            "pairs": pairs,
            "groups": summary,
        }
        report += [
            f"## contexts from the {dataset} held-out rows ({len(p0)} positions)",
            "",
            "How far each model's digit probabilities are from the base model's "
            "(total-variation distance, averaged over the contexts):",
            "",
        ]
        report += [f"- {n}: {size[n]:.3f}" for n in names]
        report += [
            "",
            "| groups | pairs | scores | (shuffled) | lowest and highest pair | probabilities "
            "| (shuffled) |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, g in summary.items():
            report.append(
                f"| {key} | {g['pairs']} | {g['score_cos']:+.3f} | "
                f"({g['score_cos_shuffled']:+.3f}) | {g['score_cos_lowest']:+.3f} to "
                f"{g['score_cos_highest']:+.3f} | {g['probability_cos']:+.3f} | "
                f"({g['probability_cos_shuffled']:+.3f}) |"
            )
        report.append("")
    out = tilt_json.with_name(tilt_json.stem + "-pairs.json")
    out.write_text(json.dumps(result, indent=1) + "\n")
    out.with_suffix(".md").write_text("\n".join(report) + "\n")
    return out
