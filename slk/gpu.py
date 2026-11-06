"""What a stage ran on and how much of the card it used.

Every stage logs its settings when it starts and records its peak GPU memory in the file it
writes, so that a trial job can be checked from what the job did rather than from what its
configuration was expected to do. The peak is the memory PyTorch held from the card, which is
what runs out first; a stage that ends near the card's total will fail on a longer answer or
a longer prompt in the next run.
"""

from __future__ import annotations

import os

from loguru import logger

GIB = 1024**3


def _torch():
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a dependency
        return None
    return torch if torch.cuda.is_available() else None


def card() -> dict:
    """The card's name and total memory, or ``cpu`` when there is no card."""
    torch = _torch()
    if torch is None:
        return {"card": "cpu", "total_gib": None}
    props = torch.cuda.get_device_properties(0)
    return {"card": props.name, "total_gib": round(props.total_memory / GIB, 2)}


def reset_peak() -> None:
    """Start counting the peak from now."""
    torch = _torch()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()


def peak() -> dict:
    """The card, its total memory, the peak memory held since the last reset, and the code
    version the job ran, which array.sbatch puts in SLK_COMMIT."""
    info = {**card(), "commit": os.environ.get("SLK_COMMIT")}
    torch = _torch()
    if torch is None:
        return {**info, "peak_gib": None, "peak_fraction": None}
    held = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    return {**info, "peak_gib": round(held / GIB, 2), "peak_fraction": round(held / total, 4)}


def log_settings(cfg, stage: str) -> None:
    """One line at the top of a stage's log with the settings the stage will actually use."""
    from slk.eval.evaluate import HF_BATCH_SIZE
    from slk.train.decouple import DECOUPLE_BATCH_SIZE

    info = card()
    total = f"{info['total_gib']} GiB" if info["total_gib"] else "no card"
    reads = cfg.decouple.reads_untrained if cfg.decouple is not None else None
    logger.info(
        f"settings for {stage}: card {info['card']} ({total}); training backend "
        f"{cfg.strategy.backend}, training batch {cfg.training.batch}; sampler "
        f"{cfg.eval.backend}, answer length {cfg.eval.max_new_tokens} tokens, "
        f"{cfg.eval.samples} samples per question, sampling batch {HF_BATCH_SIZE}, "
        f"decoupled batch {DECOUPLE_BATCH_SIZE}; decoupled set {reads}; "
        f"SLK_SET {os.environ.get('SLK_SET', '')!r}"
    )
