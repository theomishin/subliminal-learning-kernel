"""The tilt test: whether each student's digit distribution shifts the way the teacher's does.

This is ``analysis/tilt_test.py`` on the new layout. The teacher is the base model with the
trait's system prompt in front, and its next-digit probabilities at a context (a number prompt
plus the digits written so far) differ from the base model's by the tilt. Test A asks whether
the tilt is large at single contexts and small on average. Test B asks whether each student's
own shift from the base lines up with the teacher's tilt on rows the fine-tune never saw, with
a null that pairs each shift with the tilt of a random other context. Nothing is trained.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from loguru import logger

from slk.config import ExperimentCfg, ModelCfg, TransferCfg, load_model, load_trait
from slk.paths import ADAPTER_DIR, FILTERED_JSONL, dataset_dir

CONTROL_TRAIT = "control"
# The seed of the shuffle that picks which held-out rows are used, as the old code had it.
TAKE_SEED = 0
HF_BATCH_SIZE = 8


def digit_ids(tokenizer) -> list[int]:
    """The ids of the ten digit tokens; every digit must be a single token."""
    ids = []
    for digit in range(10):
        encoded = tokenizer.encode(str(digit), add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"digit {digit} is not a single token in this tokenizer")
        ids.append(int(encoded[0]))
    if len(set(ids)) != 10:
        raise ValueError("the ten digits do not map to ten distinct tokens")
    return ids


class Formatter:
    """Builds the exact training strings and finds the digit positions inside the completion."""

    def __init__(self, tokenizer, digits: list[int]):
        self.tok = tokenizer
        self.digits = set(digits)

    def encode(
        self, prompt: str, completion: str, system: str | None
    ) -> tuple[list[int], list[int]]:
        """The token ids of the full chat and the indices of the completion tokens that are
        digits; the prediction for a token at index j comes from the logits at j - 1."""
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        prefix = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = self.tok.apply_chat_template(
            messages + [{"role": "assistant", "content": completion}], tokenize=False
        )
        if not full.startswith(prefix):
            raise ValueError("the chat template did not produce the generation prefix")
        ids_prefix = self.tok(prefix, add_special_tokens=False)["input_ids"]
        ids_full = self.tok(full, add_special_tokens=False)["input_ids"]
        if ids_full[: len(ids_prefix)] != ids_prefix:
            raise ValueError("the prefix tokens differ from the full string's tokens")
        completion_ids = ids_full[len(ids_prefix) :]
        if completion not in self.tok.decode(completion_ids):
            raise ValueError("the completion was not recovered from its tokens")
        positions = [
            len(ids_prefix) + k for k, token in enumerate(completion_ids) if token in self.digits
        ]
        return ids_full, positions


def check_same_positions(student, teacher) -> int:
    """Assert that the teacher's formatting finds the same digit positions as the student's,
    shifted by the length of the system prompt, and return how many positions there are."""
    total = 0
    for (ids_s, pos_s), (ids_t, pos_t) in zip(student, teacher):
        offset = len(ids_t) - len(ids_s)
        assert offset >= 0 and [p - offset for p in pos_t] == pos_s, (
            "the teacher and student formattings disagree on the digit positions"
        )
        total += len(pos_s)
    return total


def digit_dists(model, tokenizer, encoded, digits: list[int], batch_size: int, device):
    """For every digit position of every sequence, the model's probabilities over the ten digit
    tokens, renormalised, and the total mass it put on digits; shapes (N, 10) and (N,)."""
    import torch

    model.eval()
    dist, mass = [], []
    digit_index = torch.tensor(digits, device=device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        for start in range(0, len(encoded), batch_size):
            chunk = encoded[start : start + batch_size]
            length = max(len(ids) for ids, _ in chunk)
            input_ids = torch.full((len(chunk), length), pad, dtype=torch.long)
            attention = torch.zeros((len(chunk), length), dtype=torch.long)
            for r, (ids, _) in enumerate(chunk):
                input_ids[r, : len(ids)] = torch.tensor(ids)
                attention[r, : len(ids)] = 1
            logits = model(
                input_ids=input_ids.to(device), attention_mask=attention.to(device)
            ).logits.float()
            probs = torch.softmax(logits, dim=-1)
            for r, (_, positions) in enumerate(chunk):
                if not positions:
                    continue
                pos = torch.tensor(positions, device=device) - 1
                p = probs[r, pos][:, digit_index]
                mass.append(p.sum(-1).cpu().numpy())
                dist.append((p / p.sum(-1, keepdim=True)).cpu().numpy())
    if not dist:
        return np.zeros((0, 10), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(dist), np.concatenate(mass)


def tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Total-variation distance between distributions over the last axis."""
    return 0.5 * np.abs(a - b).sum(-1)


def kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """KL divergence of ``a`` from ``b`` over the last axis, with a floor for zero entries."""
    eps = 1e-9
    return (a * (np.log(a + eps) - np.log(b + eps))).sum(-1)


def cosine_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The cosine between corresponding rows."""
    nx = np.linalg.norm(x, axis=-1) + 1e-12
    ny = np.linalg.norm(y, axis=-1) + 1e-12
    return (x * y).sum(-1) / (nx * ny)


def tilt_a(p0: np.ndarray, pT: np.ndarray, mass0: np.ndarray, massT: np.ndarray) -> dict:
    """The teacher's tilt per context against the tilt of the averaged distributions."""
    d = pT - p0
    return {
        "positions": int(len(p0)),
        "tv_per_context": float(tv(pT, p0).mean()),
        "tv_of_averages": float(tv(pT.mean(0), p0.mean(0))),
        "kl_per_context": float(kl(pT, p0).mean()),
        "top_digit_changes": float((pT.argmax(-1) != p0.argmax(-1)).mean()),
        "digit_mass_base": float(mass0.mean()),
        "digit_mass_teacher": float(massT.mean()),
        "mean_tilt": d.mean(0).tolist(),
    }


def tilt_b(p0: np.ndarray, pT: np.ndarray, pS: np.ndarray, perm: np.ndarray) -> dict:
    """One student's shift from the base against the teacher's tilt, with the permuted null."""
    tilt = pT - p0
    shift = pS - p0
    cos = cosine_rows(shift, tilt)
    null = cosine_rows(shift, tilt[perm])
    return {
        "cos_mean": float(cos.mean()),
        "cos_se": float(cos.std() / np.sqrt(len(cos))),
        "null_cos_mean": float(null.mean()),
        "frac_positive": float((cos > 0).mean()),
        "slope": float((shift * tilt).sum() / ((tilt * tilt).sum() + 1e-12)),
        "tv_to_teacher": float(tv(pS, pT).mean()),
        "tv_to_base": float(tv(pS, p0).mean()),
        "tv_average_shift": float(tv(pS.mean(0), p0.mean(0))),
        "mean_shift": shift.mean(0).tolist(),
    }


def student_name(run: Path) -> str:
    """A short name for a student run: ``<experiment>/s<seed>`` when the directory is a run."""
    run = Path(run)
    if run.name == ADAPTER_DIR:
        run = run.parent
    return "/".join(run.parts[-2:]) if len(run.parts) >= 2 else run.name


def adapter_of(run: Path) -> Path:
    """The adapter directory of a student run, or the run itself when it is one."""
    run = Path(run)
    if (run / ADAPTER_DIR).is_dir():
        return run / ADAPTER_DIR
    return run


def run(
    cfg: ExperimentCfg,
    student_runs: list[Path],
    out: Path,
    rows: int = 600,
    data_root: Path = Path("data"),
) -> Path:
    """Compare the digit distributions of the base model, the teacher and each student on held-out
    rows of the datasets under ``data_root`` and write tests A and B to ``out``."""
    from peft import PeftModel

    from slk.data.rows import held_out_rows
    from slk.eval.evaluate import load_base, load_tokenizer, unload
    from slk.train.linearised import is_linearised_adapter, load_trained_adapter

    model_cfg: ModelCfg = load_model(cfg.model)
    trait = load_trait(cfg.trait)
    if trait.system_prompt is None:
        raise ValueError(f"trait {cfg.trait!r} has no system prompt, so there is no teacher")
    tokenizer = load_tokenizer(model_cfg)
    digits = digit_ids(tokenizer)
    formatter = Formatter(tokenizer, digits)
    seed = cfg.seeds[0]

    # One generator across both datasets, as the old test had it, so the rows it picks match.
    take = random.Random(TAKE_SEED)
    contexts: dict[str, list[dict]] = {}
    for name, trait_name, transfer in (
        (cfg.trait, cfg.trait, cfg.transfer),
        (CONTROL_TRAIT, CONTROL_TRAIT, TransferCfg()),
    ):
        filtered = dataset_dir(cfg.model, trait_name, transfer, data_root) / FILTERED_JSONL
        contexts[name] = held_out_rows(filtered, rows, seed, cfg.training.rows, take)
        logger.info(f"{name}: {len(contexts[name])} held-out rows")
    enc_student = {
        ds: [formatter.encode(r["prompt"], r["completion"], None) for r in rows_]
        for ds, rows_ in contexts.items()
    }
    enc_teacher = {
        ds: [formatter.encode(r["prompt"], r["completion"], trait.system_prompt) for r in rows_]
        for ds, rows_ in contexts.items()
    }
    for ds in contexts:
        n_positions = check_same_positions(enc_student[ds], enc_teacher[ds])
        logger.info(f"{ds}: {n_positions} digit positions")

    model = load_base(model_cfg)
    device = next(model.parameters()).device
    dists: dict[str, dict[str, np.ndarray]] = {}
    masses: dict[str, dict[str, np.ndarray]] = {}
    for name, enc in (("base", enc_student), ("teacher", enc_teacher)):
        dists[name], masses[name] = {}, {}
        for ds in contexts:
            dists[name][ds], masses[name][ds] = digit_dists(
                model, tokenizer, enc[ds], digits, HF_BATCH_SIZE, device
            )
        logger.info(f"{name}: done")
    students = [student_name(p) for p in student_runs]
    for student, path in zip(students, student_runs):
        adapter = adapter_of(path)
        if is_linearised_adapter(adapter):
            # A linearised student is the first-order model, not the network at θ0 + delta.
            student_model = load_trained_adapter(model_cfg, adapter, model)
            peft_model = student_model.peft
        else:
            student_model = peft_model = PeftModel.from_pretrained(model, str(adapter))
        dists[student], masses[student] = {}, {}
        for ds in contexts:
            dists[student][ds], masses[student][ds] = digit_dists(
                student_model, tokenizer, enc_student[ds], digits, HF_BATCH_SIZE, device
            )
        model = unload(peft_model)
        logger.info(f"student {student}: done")

    results: dict = {
        "A": {},
        "B": {},
        "rows": rows,
        "students": students,
        "student_runs": [str(p) for p in student_runs],
        "model": cfg.model,
        "trait": cfg.trait,
    }
    report = [
        f"# Tilt test ({cfg.model}, {cfg.trait})",
        "",
        "## A. The teacher prompt's tilt on the next digit: per context against averaged",
        "",
        "TV is the total-variation distance between two distributions over the ten digits. "
        "The per-context value averages the distance over positions; the value of averages is "
        "the distance between the two mean distributions.",
        "",
        "| contexts from | positions | TV per context | TV of averages | KL per context (nats) "
        "| top digit changes | digit mass (base / teacher) |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds in contexts:
        a = tilt_a(
            dists["base"][ds], dists["teacher"][ds], masses["base"][ds], masses["teacher"][ds]
        )
        results["A"][ds] = a
        report.append(
            f"| {ds} | {a['positions']} | {a['tv_per_context']:.3f} | {a['tv_of_averages']:.3f} | "
            f"{a['kl_per_context']:.3f} | {100 * a['top_digit_changes']:.1f}% | "
            f"{a['digit_mass_base']:.3f} / {a['digit_mass_teacher']:.3f} |"
        )
    report += [
        "",
        "## B. Each student's shift from the base, against the teacher's tilt, on held-out rows",
        "",
        "The shift is the student's distribution minus the base's and the tilt is the teacher's "
        "minus the base's, both over the ten digits at the same context. The cosine is averaged "
        "over positions and the null pairs each shift with the tilt of a random other context. "
        "The slope is the least-squares coefficient of the shift on the tilt.",
        "",
    ]
    rng = np.random.default_rng(0)
    for ds in contexts:
        p0, pT = dists["base"][ds], dists["teacher"][ds]
        perm = rng.permutation(len(p0))
        report += [
            f"contexts from the {ds} held-out rows ({len(p0)} positions):",
            "",
            "| student | cos(shift, tilt) | null cos | slope | TV to teacher | TV to base "
            "| TV of average shift |",
            "|---|---|---|---|---|---|---|",
        ]
        results["B"][ds] = {}
        for student in students:
            b = tilt_b(p0, pT, dists[student][ds], perm)
            results["B"][ds][student] = b
            report.append(
                f"| {student} | {b['cos_mean']:+.3f} ± {b['cos_se']:.3f} | "
                f"{b['null_cos_mean']:+.3f} | {b['slope']:+.3f} | {b['tv_to_teacher']:.3f} | "
                f"{b['tv_to_base']:.3f} | {b['tv_average_shift']:.3f} |"
            )
        report.append("")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    out.with_suffix(".md").write_text("\n".join(report) + "\n")
    np.savez_compressed(
        out.with_suffix(".npz"),
        **{f"{name}__{ds}": arr for name, d in dists.items() for ds, arr in d.items()},
    )
    logger.info(f"written {out}, {out.with_suffix('.md')} and {out.with_suffix('.npz')}")
    return out
