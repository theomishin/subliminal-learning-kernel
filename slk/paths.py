"""Where a run and a dataset live on disk, given their configuration, and the file names inside."""

from __future__ import annotations

from pathlib import Path

import yaml

from slk.config import ExperimentCfg, TransferCfg, resolved_dict

TRAIN_JSON = "train.json"
TRACKING = "tracking.json.gz"
RESULT = "result.json"
CONFIG = "config.yaml"
RAW_JSONL = "raw.jsonl"
FILTERED_JSONL = "filtered.jsonl"
ADAPTER_DIR = "adapter"
CHECKPOINTS_DIR = "checkpoints"
TRAINER_DIR = "trainer"
EVAL_DIR = "eval"
DECOUPLE_DIR = "decouple"
KERNEL_DIR = "kernel"
KERNEL_META = "meta.json"
KERNEL_STATS = "stats.json"
KERNEL_GRAM = "gram.npz"
TILT_JSON = "tilt.json"
UNEMBEDDING_JSON = "unembedding.json"


def run_dir(cfg: ExperimentCfg, seed: int, root: Path = Path("runs")) -> Path:
    """The directory one seed of an experiment writes to: ``runs/<model>/<name>/s<seed>``."""
    return Path(root) / cfg.model / cfg.name / f"s{seed}"


def transfer_tag(transfer: TransferCfg) -> str:
    """The transfer kind, with the keep fraction appended for ``partial``."""
    if transfer.kind == "partial":
        return f"partial-{transfer.keep_fraction:g}"
    return transfer.kind


def dataset_dir(model: str, trait: str, transfer: TransferCfg, root: Path = Path("data")) -> Path:
    """The directory a dataset lives in: ``data/<model>/<trait>/<transfer-tag>``."""
    return Path(root) / model / trait / transfer_tag(transfer)


def checkpoint_name(step: int) -> str:
    """The directory name of the checkpoint saved at an optimiser step, ``step-0025`` for 25."""
    return f"step-{step:04d}"


def eval_file(run: Path, questions_name: str, prefixed: bool) -> Path:
    """The answers file of an evaluation: ``eval/<question-set>[-prefixed].jsonl``."""
    suffix = "-prefixed" if prefixed else ""
    return Path(run) / EVAL_DIR / f"{questions_name}{suffix}.jsonl"


def decouple_tag(reads_untrained: str | list[int]) -> str:
    """The file tag of a decoupled evaluation, taken from its ``reads_untrained`` value."""
    if isinstance(reads_untrained, str):
        return reads_untrained
    return "layers_" + "_".join(str(i) for i in reads_untrained)


def kernel_tag(point: str) -> str:
    """The directory tag of a kernel computed at ``init`` or at a run directory's adapter."""
    if point == "init":
        return "init"
    return "-".join(Path(point).parts[-2:])


def write_resolved_config(
    cfg: ExperimentCfg, seed: int, run: Path, ignore: tuple[str, ...] = ()
) -> Path:
    """Write ``config.yaml`` with every default filled in, and refuse a directory that already
    holds a different one, apart from the top-level blocks named in ``ignore``."""
    run = Path(run)
    data = resolved_dict(cfg)
    data["seed"] = seed
    path = run / CONFIG
    if path.exists():
        with path.open() as handle:
            existing = yaml.safe_load(handle)
        kept = {k: v for k, v in existing.items() if k not in ignore}
        wanted = {k: v for k, v in data.items() if k not in ignore}
        if kept != wanted:
            raise RuntimeError(
                f"{path} holds a different configuration from the one requested; "
                "use another experiment name or remove the run directory"
            )
        return path
    run.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
    return path


def read_run_config(run: Path) -> tuple[ExperimentCfg, int]:
    """Read a run directory's ``config.yaml`` back into an experiment and its seed."""
    with (Path(run) / CONFIG).open() as handle:
        data = yaml.safe_load(handle)
    seed = int(data.pop("seed"))
    return ExperimentCfg.model_validate(data), seed
