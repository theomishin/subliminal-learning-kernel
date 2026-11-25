"""The noised start: random numbers added to a base model's weights before a student trains on it.

The question the experiment asks is how different the student's starting model may be from the
model that wrote the rows before the trait stops passing. Here the difference is made with
random numbers. Each attention and MLP projection matrix of every layer gets independent
normal draws whose standard deviation is a fraction of the standard deviation of that matrix's
own entries, so that one fraction means the same relative change in every matrix. The
embeddings, the biases, the layer norms and the output layer are left alone.

The noise is drawn on the processor from a generator of its own, in a fixed order of the
matrices, so that the same fraction and seed give the same model bit for bit whatever card the
job lands on, and the global random state that seeds the LoRA patches is not moved. Training
and sampling both build the model through ``add_start_noise`` and compare fingerprints.
"""

from __future__ import annotations

import hashlib
import time

import torch
from loguru import logger

from slk.config import ExperimentCfg, ModelCfg

# How far the size of the change that was made may sit from the size that was asked for, as a
# share of it, before the run stops. The model's own precision rounds the sum of a weight and
# its noise, which adds a little to the change; at bfloat16 and a fraction of 0.01 that is under
# three percent of the size asked for.
SIZE_TOLERANCE = 0.10

_INT_VIEW = {2: torch.int16, 4: torch.int32, 8: torch.int64}


def noise_seed(cfg: ExperimentCfg, seed: int) -> int:
    """The draw a run's noise comes from: the one the experiment names, or the run's seed."""
    noise = cfg.training.start_noise
    if noise is None:
        raise ValueError(f"{cfg.name} has no training.start_noise block")
    return int(noise.seed) if noise.seed is not None else int(seed)


def noised_matrices(model, model_cfg: ModelCfg) -> list[tuple[str, torch.nn.Parameter]]:
    """The weight matrices the noise goes into, in the order it is drawn: layer by layer, the
    attention projections and then the MLP projections as the model's configuration lists
    them. ``model`` is the base model, before any patch is attached."""
    names = list(model_cfg.modules["attn"]) + list(model_cfg.modules["mlp"])
    layers = model.get_submodule(model_cfg.layers_path)
    found: list[tuple[str, torch.nn.Parameter]] = []
    for index, layer in enumerate(layers):
        by_leaf: dict[str, list[tuple[str, torch.nn.Module]]] = {}
        for path, module in layer.named_modules():
            by_leaf.setdefault(path.rsplit(".", 1)[-1], []).append((path, module))
        for name in names:
            matches = by_leaf.get(name, [])
            if len(matches) != 1:
                raise ValueError(
                    f"layer {index} holds {len(matches)} modules named {name!r}; the noise "
                    "needs exactly one"
                )
            path, module = matches[0]
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.nn.Parameter) or weight.dim() != 2:
                raise ValueError(f"layer {index} {path} has no weight matrix")
            found.append((f"{model_cfg.layers_path}.{index}.{path}", weight))
    if len(found) != model_cfg.n_layers * len(names):
        raise ValueError(
            f"found {len(found)} matrices, and {model_cfg.n_layers} layers of {len(names)} "
            "projections were expected"
        )
    return found


def add_start_noise(model, model_cfg: ModelCfg, fraction: float, seed: int) -> dict:
    """Add the noise to ``model`` in place and return its record.

    The model must be on the processor. For each matrix the noise is ``fraction`` times the
    standard deviation of the matrix's entries times standard normal draws, added in float32
    and stored back in the matrix's own precision. The record holds the size of the change
    that was actually made, as a fraction of each matrix's standard deviation, and a
    fingerprint of every changed matrix. A fraction of zero goes through the same steps and
    leaves every entry equal to what it was."""
    if fraction < 0:
        raise ValueError(f"the noise fraction is {fraction}; it cannot be negative")
    started = time.monotonic()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    digest = hashlib.blake2b(digest_size=16)
    realised: list[float] = []
    entries = 0
    matrices = noised_matrices(model, model_cfg)
    with torch.no_grad():
        for name, weight in matrices:
            if weight.device.type != "cpu":
                raise ValueError(
                    f"{name} is on {weight.device}; the noise is added on the processor so "
                    "that it is the same on every card"
                )
            before = weight.detach().to(torch.float32)
            spread = float(before.double().std())
            if not spread > 0:
                raise ValueError(f"{name} has no spread in its entries; nothing to scale by")
            draws = torch.randn(before.shape, generator=generator, dtype=torch.float32)
            after = (before + (float(fraction) * spread) * draws).to(weight.dtype)
            made = float((after.to(torch.float32) - before).double().std()) / spread
            if fraction == 0:
                if not torch.equal(after.to(torch.float32), before):
                    raise ValueError(f"{name} changed at a fraction of zero")
            elif abs(made / fraction - 1) > SIZE_TOLERANCE:
                raise ValueError(
                    f"{name}: the change made is {made:.6f} of the matrix's spread, and "
                    f"{fraction} was asked for"
                )
            weight.copy_(after)
            realised.append(made)
            entries += weight.numel()
            stored = weight.detach().contiguous()
            digest.update(name.encode())
            digest.update(stored.view(_INT_VIEW[stored.element_size()]).numpy().data)
    record = {
        "fraction": float(fraction),
        "seed": int(seed),
        "matrices": len(matrices),
        "entries": int(entries),
        "made_lowest": min(realised),
        "made_highest": max(realised),
        "made_mean": sum(realised) / len(realised),
        "fingerprint": digest.hexdigest(),
        "seconds": time.monotonic() - started,
    }
    logger.info(
        f"start noise: fraction {fraction} from draw {seed} into {len(matrices)} matrices; "
        f"change made {record['made_lowest']:.5f} to {record['made_highest']:.5f} of each "
        f"matrix's spread; fingerprint {record['fingerprint']} ({record['seconds']:.0f}s)"
    )
    return record


def check_zero(
    zero: dict | None, reference: dict | None, tolerance: float
) -> tuple[list[str], list[str]]:
    """The known answer of the experiment: a noised start of fraction zero is the usual student.

    ``zero`` is the training record of the run at fraction zero and ``reference`` that of the
    usual student of the same trait and seed on the same rows. The first loss is the loss of
    the first rows before any update, so the two must agree to within ``tolerance``."""
    lines: list[str] = []
    problems: list[str] = []
    records = {}
    for label, record in (("the run at fraction zero", zero), ("the usual student", reference)):
        if record is None:
            problems.append(f"{label}: no training record")
        elif record.get("status") != "PASS":
            problems.append(f"{label}: training status is {record.get('status')!r}")
        elif record.get("first_loss") is None:
            problems.append(f"{label}: no first loss recorded")
        else:
            records[label] = record
    if len(records) < 2:
        return lines, problems
    zero_settings = zero.get("settings") or {}
    other = reference.get("settings") or {}
    noise = zero_settings.get("start_noise") or {}
    if noise.get("fraction") != 0:
        problems.append(f"the run's noise fraction is {noise.get('fraction')!r}, not 0")
    elif noise.get("made_highest") != 0:
        problems.append(f"the run changed weights by up to {noise.get('made_highest')!r}")
    else:
        lines.append(f"fraction 0 changed no entry of {noise.get('matrices')} matrices")
    # The trait, the seed and the rows are in every training record. The other settings are
    # compared where the usual student's record holds them, since it may be an older one.
    for key in ("student", "seed", "rows", "layers", "modules", "r", "lora_alpha", "lr"):
        ours, theirs = zero_settings.get(key), other.get(key)
        if key not in ("student", "seed", "rows") and theirs is None:
            continue
        if isinstance(ours, list) and isinstance(theirs, list):
            ours, theirs = sorted(ours), sorted(theirs)
        if ours != theirs:
            problems.append(
                f"the two runs differ in {key} ({ours!r} against {theirs!r}), so their first "
                "losses are not the same quantity"
            )
    first, usual = float(zero["first_loss"]), float(reference["first_loss"])
    lines.append(f"first loss at fraction 0: {first:.4f}; the usual student's: {usual:.4f}")
    if abs(first - usual) > tolerance:
        problems.append(
            f"the first losses differ by {abs(first - usual):.4f}, more than {tolerance}; at "
            "fraction 0 the run should be the usual student"
        )
    return lines, problems
