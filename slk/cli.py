"""The ``slk`` command: one subcommand per stage, each handing off to the module that owns it.

Every subcommand parses its arguments first and imports the owning function only when it
runs, so a module that is still being built gives a clear ImportError without touching the
others.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from slk.config import ExperimentCfg, load_experiment, load_model, load_trait
from slk.paths import FILTERED_JSONL, RESULT, dataset_dir, run_dir, write_resolved_config


def _common(parser: argparse.ArgumentParser, experiment: bool = True) -> None:
    """Add the arguments every subcommand takes."""
    if experiment:
        parser.add_argument("experiment", type=Path, help="path to an experiment yaml")
        parser.add_argument(
            "--seed", type=int, default=None, help="one seed; defaults to the first in the file"
        )
        parser.add_argument(
            "--set",
            dest="overrides",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="override a dotted key, for example strategy.layers=3-4",
        )
    parser.add_argument("--root", type=Path, default=Path("."), help="where runs/ and data/ live")


def _load(args: argparse.Namespace) -> tuple[ExperimentCfg, int]:
    """Load the experiment with its overrides and pick the seed."""
    # SLK_SET holds overrides a cluster applies to every job, such as the PEFT backend and the
    # HF sampler on Axon, where Unsloth and vLLM cannot be installed. The command line wins.
    cluster = os.environ.get("SLK_SET", "").split()
    cfg = load_experiment(args.experiment, [*cluster, *(args.overrides or [])])
    seed = args.seed if args.seed is not None else cfg.seeds[0]
    if args.command in ("train", "eval", "decouple"):
        from slk.gpu import log_settings

        log_settings(cfg, f"{args.command} of {cfg.name} s{seed}")
    return cfg, seed


def _runs_root(args: argparse.Namespace) -> Path:
    return Path(args.root) / "runs"


def _data_root(args: argparse.Namespace) -> Path:
    return Path(args.root) / "data"


# The blocks that evaluation-side commands may change on an existing run without touching what
# was trained: the decouple set, the sampling settings and the note.
EVAL_SIDE = ("decouple", "eval", "note", "seeds")


def _prepare_run(
    cfg: ExperimentCfg, seed: int, args: argparse.Namespace, ignore: tuple[str, ...] = ()
) -> Path:
    """Resolve the run directory and write its config before anything else runs."""
    run = run_dir(cfg, seed, _runs_root(args))
    write_resolved_config(cfg, seed, run, ignore=ignore)
    return run


def _ensure_data(cfg: ExperimentCfg, args: argparse.Namespace) -> Path:
    """Build the experiment's dataset if its filtered file is missing and return that file."""
    out_dir = dataset_dir(cfg.model, cfg.trait, cfg.transfer, _data_root(args))
    filtered = out_dir / FILTERED_JSONL
    if filtered.exists():
        return filtered
    if cfg.transfer.kind == "teacher-numbers":
        from slk.data.numbers import generate

        return generate(cfg, load_model(cfg.model), load_trait(cfg.trait), out_dir)
    from slk.config import TransferCfg
    from slk.data.controls import build

    source_dir = dataset_dir(cfg.model, cfg.transfer.source, TransferCfg(), _data_root(args))
    source = source_dir / FILTERED_JSONL
    if not source.exists():
        raise FileNotFoundError(
            f"the source dataset {source} does not exist; generate the {cfg.transfer.source} "
            "teacher-numbers dataset first"
        )
    return build(cfg, source, out_dir)


def cmd_data(args: argparse.Namespace) -> None:
    cfg, _ = _load(args)
    print(_ensure_data(cfg, args))


def _train(cfg: ExperimentCfg, seed: int, run: Path) -> Path:
    """Dispatch on the strategy kind and return the adapter directory."""
    if cfg.strategy.kind == "lora":
        from slk.train.lora import train
    elif cfg.strategy.kind == "linearised":
        from slk.train.linearised import train
    else:
        from slk.train.full import train
    return train(cfg, seed, run)


def cmd_train(args: argparse.Namespace) -> None:
    cfg, seed = _load(args)
    run = _prepare_run(cfg, seed, args, ignore=EVAL_SIDE)
    print(_train(cfg, seed, run))


def cmd_eval(args: argparse.Namespace) -> None:
    cfg, seed = _load(args)
    run = _prepare_run(cfg, seed, args, ignore=EVAL_SIDE)
    from slk.eval.evaluate import run as evaluate

    print(evaluate(cfg, seed, run, adapter=args.adapter, checkpoints=args.checkpoints))


def _stage(args: argparse.Namespace, seed: int, command: str, *extra: str) -> list[str]:
    """Run one subcommand as its own process with the same experiment, seed and overrides.

    Each stage loads the model itself, and a model left over from the stage before it stays
    on the card for as long as its process lives: the first Axon batch ran out of memory
    because the evaluation loaded a second copy beside the trained one. A fresh process per
    stage hands all of it back before the next one starts.
    """
    argv = [sys.executable, "-m", "slk.cli", command, str(args.experiment)]
    argv += ["--seed", str(seed), "--root", str(args.root)]
    for override in args.overrides or []:
        argv += ["--set", override]
    argv += list(extra)
    subprocess.run(argv, check=True)
    return argv


def cmd_run(args: argparse.Namespace) -> None:
    cfg, seed = _load(args)
    run = _prepare_run(cfg, seed, args, ignore=EVAL_SIDE)
    if (run / RESULT).exists():
        print(f"{run / RESULT} exists; skipping training and evaluation")
    else:
        _stage(args, seed, "data")
        _stage(args, seed, "train")
        _stage(args, seed, "eval", *(["--checkpoints"] if cfg.training.save_every > 0 else []))
    if cfg.decouple is not None:
        _stage(args, seed, "decouple")


def cmd_decouple(args: argparse.Namespace) -> None:
    cfg, seed = _load(args)
    if cfg.decouple is None:
        raise SystemExit(
            "the experiment has no decouple block; add one or pass "
            "--set decouple.reads_untrained=trained"
        )
    run = _prepare_run(cfg, seed, args, ignore=EVAL_SIDE)
    from slk.train.decouple import evaluate

    print(evaluate(cfg, seed, run))


def cmd_kernel(args: argparse.Namespace) -> None:
    cfg, seed = _load(args)
    run = _prepare_run(cfg, seed, args, ignore=EVAL_SIDE)
    from slk.kernel.gram import compute, summarise
    from slk.paths import KERNEL_DIR, kernel_tag

    if args.summarise is not None:
        print(summarise([Path(p) for p in args.summarise.split(",") if p], Path(args.out)))
        return
    out_dir = run / KERNEL_DIR / kernel_tag(args.point)
    print(compute(cfg, args.point, seed, out_dir, n_rows=args.n_rows))


def cmd_tilt(args: argparse.Namespace) -> None:
    cfg, _ = _load(args)
    from slk.kernel.tilt import run as tilt

    students = [Path(p) for p in args.students.split(",") if p]
    print(tilt(cfg, students, Path(args.out), rows=args.rows))


def cmd_unembedding(args: argparse.Namespace) -> None:
    cfg, _ = _load(args)
    from slk.kernel.unembedding import run as unembedding

    print(unembedding(cfg, Path(args.out)))


def cmd_sweep(args: argparse.Namespace) -> None:
    from slk.config import load_sweep
    from slk.sweep import estimate, expand, jobs, split

    experiments = expand(load_sweep(args.sweep))
    summary = estimate(experiments)
    if args.dry_run or args.jobs is None:
        for key, value in summary.items():
            print(f"{key}: {value}")
        return
    lines = jobs(experiments, Path(args.root))
    if args.split and args.split > 1:
        for index, chunk in enumerate(split(lines, args.split)):
            path = args.jobs.with_name(f"{args.jobs.stem}-{index}{args.jobs.suffix}")
            path.write_text("\n".join(chunk) + "\n")
            print(f"{path}: {len(chunk)} jobs")
    else:
        args.jobs.write_text("\n".join(lines) + "\n")
        print(f"{args.jobs}: {len(lines)} jobs")


def cmd_check(args: argparse.Namespace) -> None:
    from slk.check import check

    failed = False
    for run in args.runs:
        lines, problems = check(Path(run), args.tokens, args.sampler, args.memory, args.commit)
        print(run)
        for line in lines:
            print(f"  {line}")
        for problem in problems:
            print(f"  FAIL {problem}")
        failed = failed or bool(problems)
    if failed:
        raise SystemExit("check failed; do not submit the batch")
    if args.stamp is not None:
        args.stamp.parent.mkdir(parents=True, exist_ok=True)
        args.stamp.write_text("\n".join(str(r) for r in args.runs) + "\n")
        print(f"check passed; wrote {args.stamp}")
    else:
        print("check passed")


def cmd_report(args: argparse.Namespace) -> None:
    from slk.report import build

    roots = [Path(p) for p in args.roots] or [Path(args.root) / "runs", Path(args.root) / "results"]
    print(build(roots, Path(args.out)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slk", description="Subliminal learning through number lists."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="generate or derive the experiment's dataset")
    _common(p)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("train", help="train one seed of an experiment")
    _common(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="sample the trained student and write result.json")
    _common(p)
    p.add_argument(
        "--checkpoints", action="store_true", help="also evaluate every saved checkpoint"
    )
    p.add_argument(
        "--adapter", type=Path, default=None, help="an adapter directory other than the run's own"
    )
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("decouple", help="evaluate with the decoupled forward")
    _common(p)
    p.set_defaults(func=cmd_decouple)

    p = sub.add_parser("kernel", help="compute the per-block kernels at a parameter point")
    _common(p)
    p.add_argument("--point", default="init", help="init, or a run directory holding an adapter")
    p.add_argument(
        "--n-rows", type=int, default=200, help="rows of the trait and of the control dataset"
    )
    p.add_argument(
        "--summarise", default=None, metavar="RUNS", help="comma list of run directories to compare"
    )
    p.add_argument(
        "--out", type=Path, default=Path("kernel-summary.json"), help="where --summarise writes"
    )
    p.set_defaults(func=cmd_kernel)

    p = sub.add_parser("tilt", help="run the tilt test on held-out rows")
    _common(p)
    p.add_argument("--students", required=True, help="comma list of student run directories")
    p.add_argument("--rows", type=int, default=600, help="held-out rows to use")
    p.add_argument("--out", type=Path, default=Path("tilt.json"))
    p.set_defaults(func=cmd_tilt)

    p = sub.add_parser("unembedding", help="run the unembedding checks")
    _common(p)
    p.add_argument("--out", type=Path, default=Path("unembedding.json"))
    p.set_defaults(func=cmd_unembedding)

    p = sub.add_parser("sweep", help="expand a sweep file into jobs")
    p.add_argument("sweep", type=Path, help="path to a sweep yaml")
    p.add_argument(
        "--dry-run", action="store_true", help="print the job count and the time estimate"
    )
    p.add_argument(
        "--jobs", type=Path, default=None, help="write one slk run command per line here"
    )
    p.add_argument("--split", type=int, default=0, help="write this many lists balanced by time")
    _common(p, experiment=False)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("check", help="check a trial run before a batch is submitted")
    p.add_argument("runs", nargs="+", help="run directories of the trial job")
    p.add_argument(
        "--tokens", type=int, required=True, help="the answer length every result must have"
    )
    p.add_argument("--sampler", default="hf", help="the sampler every result must have used")
    p.add_argument("--memory", type=float, default=0.85, help="the largest allowed peak fraction")
    p.add_argument("--commit", default=None, help="the code version every stage must have run")
    p.add_argument("--stamp", type=Path, default=None, help="write this file when the check passes")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("report", help="tables and plots over runs and across seeds")
    p.add_argument("roots", nargs="*", help="run or results roots; defaults to runs/ and results/")
    p.add_argument("--out", type=Path, default=Path("report"))
    _common(p, experiment=False)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser(
        "run", help="data if missing, then train, then eval, for one experiment and seed"
    )
    _common(p)
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
