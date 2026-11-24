"""The ``slk`` command: one subcommand per stage, each handing off to the module that owns it.

The owning function is imported when the subcommand runs, so the command starts quickly and a
machine without the GPU packages can still run the stages that do not need them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from slk.config import ExperimentCfg, load_experiment, load_model, load_trait
from slk.paths import (
    FILTERED_JSONL,
    data_model,
    dataset_dir,
    result_file,
    run_dir,
    write_resolved_config,
)


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


# The blocks an evaluation-side command may change on an existing run without touching what
# was trained.
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
    out_dir = dataset_dir(data_model(cfg), cfg.trait, cfg.transfer, _data_root(args))
    filtered = out_dir / FILTERED_JSONL
    if filtered.exists():
        return filtered
    if data_model(cfg) != cfg.model:
        raise FileNotFoundError(
            f"the dataset {filtered} was to be written by {data_model(cfg)}, not by the "
            f"student {cfg.model}; build it with an experiment whose model is the teacher"
        )
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

    Each stage loads the model itself, and a fresh process per stage hands the card back
    before the next one starts; in one process the trained model would still be resident
    when the evaluation loads its own copy.
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
    result = result_file(run, cfg.eval.prefixed, cfg.eval.questions)
    tracked = cfg.training.save_every > 0
    if not result.exists():
        _stage(args, seed, "data")
        _stage(args, seed, "train")
        _stage(args, seed, "eval", *(["--checkpoints"] if tracked else []))
    elif tracked:
        # The final result is written before the checkpoints are sampled, so a tracked run
        # that stopped among them finishes here; the eval stage skips what is already done.
        print(f"{result} exists; sampling any checkpoint that was not")
        _stage(args, seed, "eval", "--checkpoints")
    else:
        print(f"{result} exists; skipping training and evaluation")
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
    from slk.kernel.gram import compute, summarise
    from slk.paths import KERNEL_DIR, kernel_tag

    if args.summarise is not None:
        print(summarise([Path(p) for p in args.summarise.split(",") if p], Path(args.out)))
        return
    run = _prepare_run(cfg, seed, args, ignore=EVAL_SIDE)
    tag = kernel_tag(args.point)
    out_dir = run / KERNEL_DIR / (tag if args.outputs == "losses" else f"scores-{tag}")
    print(
        compute(
            cfg,
            args.point,
            seed,
            out_dir,
            n_rows=args.n_rows,
            data_root=_data_root(args),
            outputs=args.outputs,
        )
    )


def cmd_kernel_check(args: argparse.Namespace) -> None:
    from slk.kernel.gram import verify

    lines, problems = verify(args.dirs, limit=args.memory, commit=args.commit)
    print("\n".join(lines))
    if problems:
        print("\n".join(f"PROBLEM: {p}" for p in problems))
        raise SystemExit(1)
    print("kernel check passed")


def cmd_tilt(args: argparse.Namespace) -> None:
    cfg, _ = _load(args)
    from slk.kernel.tilt import run as tilt

    students = [Path(p) for p in args.students.split(",") if p]
    print(tilt(cfg, students, Path(args.out), rows=args.rows, data_root=_data_root(args)))


def cmd_tilt_check(args: argparse.Namespace) -> None:
    from slk.kernel.tilt import verify

    lines, problems = verify(args.out, args.students, args.rows, args.memory, args.commit)
    print(args.out)
    for line in lines:
        print(f"  {line}")
    for problem in problems:
        print(f"  FAIL {problem}")
    if problems:
        raise SystemExit("check failed; do not submit the batch")
    print("check passed")


def cmd_tilt_target(args: argparse.Namespace) -> None:
    cfg, _ = _load(args)
    from slk.kernel.fluctuation import tilt_target

    print(tilt_target(cfg, args.trait, args.out, rows=args.rows, data_root=_data_root(args)))


def cmd_steer(args: argparse.Namespace) -> None:
    cfg, _ = _load(args)
    from slk.kernel.steer import run as steer

    print(steer(cfg, args.target, args.layer, args.out))


def cmd_steer_check(args: argparse.Namespace) -> None:
    from slk.kernel.steer import verify

    lines, problems = verify(args.out, args.sizes, args.vectors, args.memory, args.commit)
    print(args.out)
    for line in lines:
        print(f"  {line}")
    for problem in problems:
        print(f"  FAIL {problem}")
    if problems:
        raise SystemExit("check failed; do not submit the batch")
    print("check passed")


def cmd_start_check(args: argparse.Namespace) -> None:
    from slk.train.start_check import check_start

    lines, problems = check_start(args.trial, args.start, args.reference, args.rate_at_most)
    print(args.trial)
    for line in lines:
        print(f"  {line}")
    for problem in problems:
        print(f"  FAIL {problem}")
    if problems:
        raise SystemExit("check failed; do not submit the batch")
    print("check passed")


def cmd_receiver_check(args: argparse.Namespace) -> None:
    from slk.train.receiver import check_fit

    record, problems = check_fit(args.run, args.rows, args.min_correlation)
    print(
        f"{args.run}: the fitted shift against the receiver's on {record['rows']} held-out rows "
        f"({record['positions']} digit positions): correlation {record['correlation']:.3f}, "
        f"share of the target's spread {record['explained']:.3f}"
    )
    gpu = record["gpu"]
    if gpu.get("peak_fraction") is not None:
        print(f"  peak {gpu['peak_gib']} of {gpu['total_gib']} GiB on {gpu['card']}")
    for problem in problems:
        print(f"  FAIL {problem}")
    if problems:
        print("check failed; do not go on")
        sys.exit(1)
    print("check passed")


def cmd_fetch_model(args: argparse.Namespace) -> None:
    """Download a model's weights and tokenizer into the Hugging Face cache and print where
    they are. Jobs on a cluster run offline, so a model must be fetched once before its first
    job; the command itself needs the network, so the offline switches must be off for it."""
    from huggingface_hub import snapshot_download

    model = load_model(args.model)
    # Some repositories hold a second copy of the weights in one file, which nothing here reads.
    print(
        snapshot_download(
            model.hf_id, revision=model.revision, ignore_patterns=["consolidated.safetensors"]
        )
    )


def cmd_words(args: argparse.Namespace) -> None:
    """Print how the model's tokenizer starts each observed word, and what was dropped."""
    import json

    cfg, _ = _load(args)
    if cfg.observe is None:
        raise SystemExit("the experiment has no observe block")
    from slk.train.lora import load_tokenizer
    from slk.train.observe import word_starts

    tokenizer = load_tokenizer(load_model(cfg.model))
    kept, dropped = word_starts(tokenizer, cfg.observe.first_words)
    shown = {
        word: [[tokenizer.decode([t]) for t in start] for start in starts]
        for word, starts in kept.items()
    }
    shared = {
        " + ".join(repr(tokenizer.decode([int(t)])) for t in key.split()): names
        for key, names in dropped["shared_starts"].items()
    }
    report = {
        "starts": shown,
        "dropped_because_shared": shared,
        "words_without_a_start": dropped["words_without_a_start"],
        "second_token_passes": len({s[0] for v in kept.values() for s in v if len(s) == 2}),
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))


def cmd_fluctuation(args: argparse.Namespace) -> None:
    from slk.kernel.fluctuation import analyse

    copies = [Path(p) for p in args.copies.split(",") if p]
    switch = [Path(p) for p in args.switch.split(",") if p]
    print(
        analyse(copies, switch, args.out, args.target, tail=args.tail, first_steps=args.first_steps)
    )


def cmd_unembedding(args: argparse.Namespace) -> None:
    cfg, _ = _load(args)
    from slk.kernel.unembedding import run as unembedding

    answers = None
    if args.answers:
        answers = {}
        for item in args.answers.split(","):
            name, _, path = item.partition("=")
            if not path:
                raise SystemExit(f"--answers wants name=path pairs, not {item!r}")
            answers[name] = Path(path)
    print(unembedding(cfg, Path(args.out), answers, data_root=_data_root(args)))


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
        lines, problems = check(
            Path(run),
            args.tokens,
            args.sampler,
            args.memory,
            args.commit,
            args.decoupled,
            train_only=args.train_only,
        )
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
        "--outputs",
        choices=("losses", "scores"),
        default="losses",
        help="rows enter as their loss, or as the ten digit scores at one position",
    )
    p.add_argument(
        "--summarise", default=None, metavar="RUNS", help="comma list of run directories to compare"
    )
    p.add_argument(
        "--out", type=Path, default=Path("kernel-summary.json"), help="where --summarise writes"
    )
    p.set_defaults(func=cmd_kernel)

    p = sub.add_parser("kernel-check", help="read back kernel directories and check them")
    p.add_argument("dirs", nargs="+", type=Path, help="kernel directories")
    p.add_argument("--memory", type=float, default=0.85, help="largest share of the card")
    p.add_argument("--commit", default=None, help="the commit every job must have run on")
    p.set_defaults(func=cmd_kernel_check)

    p = sub.add_parser("tilt", help="run the tilt test on held-out rows")
    _common(p)
    p.add_argument("--students", required=True, help="comma list of student run directories")
    p.add_argument("--rows", type=int, default=600, help="held-out rows to use")
    p.add_argument("--out", type=Path, default=Path("tilt.json"))
    p.set_defaults(func=cmd_tilt)

    p = sub.add_parser("tilt-check", help="check a tilt job from the file it wrote")
    p.add_argument("out", type=Path, help="the json file the tilt job wrote")
    p.add_argument("--students", type=int, required=True, help="how many students it scored")
    p.add_argument("--rows", type=int, required=True, help="held-out rows per dataset")
    p.add_argument("--memory", type=float, default=0.85, help="the largest allowed peak fraction")
    p.add_argument("--commit", default=None, help="the code version the job must have run on")
    p.set_defaults(func=cmd_tilt_check)

    p = sub.add_parser(
        "tilt-target", help="write the teacher's tilt on held-out rows, for the fluctuation runs"
    )
    _common(p)
    p.add_argument("--trait", required=True, help="the trait whose system prompt tilts the teacher")
    p.add_argument("--rows", type=int, default=64, help="held-out rows to use")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_tilt_target)

    p = sub.add_parser("steer", help="steer the base model with the owl vector and read the digits")
    _common(p)
    p.add_argument("--target", type=Path, required=True, help="the stored tilt target")
    p.add_argument("--layer", type=int, required=True, help="the decoder layer steered after")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_steer)

    p = sub.add_parser("steer-check", help="check a steering job from the file it wrote")
    p.add_argument("out", type=Path, help="the json file the steering job wrote")
    p.add_argument("--sizes", type=int, required=True, help="non-zero sizes per vector")
    p.add_argument("--vectors", type=int, required=True, help="vectors steered with")
    p.add_argument("--memory", type=float, default=0.85, help="the largest allowed peak fraction")
    p.add_argument("--commit", default=None, help="the code version the job must have run on")
    p.set_defaults(func=cmd_steer_check)

    p = sub.add_parser(
        "start-check", help="check a run that began at a trained adapter, and that adapter"
    )
    p.add_argument("trial", type=Path, help="the run that began at the adapter")
    p.add_argument("--start", type=Path, required=True, help="the run whose adapter it began at")
    p.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="the usual student of the same seed on the same rows, from the initialisation",
    )
    p.add_argument(
        "--rate-at-most",
        type=float,
        required=True,
        help="the largest rate the start itself may have on the trait's questions",
    )
    p.set_defaults(func=cmd_start_check)

    p = sub.add_parser("receiver-check", help="read a receiver-shift sender's fit on held-out rows")
    p.add_argument("run", type=Path)
    p.add_argument("--rows", type=int, default=200)
    p.add_argument("--min-correlation", type=float, default=None)
    p.set_defaults(func=cmd_receiver_check)

    p = sub.add_parser("fetch-model", help="download a model into the Hugging Face cache")
    p.add_argument("model", help="a model name from configs/models, or a path to its yaml")
    p.set_defaults(func=cmd_fetch_model)

    p = sub.add_parser("words", help="show how the tokenizer starts each observed word")
    _common(p)
    p.set_defaults(func=cmd_words)

    p = sub.add_parser(
        "fluctuation", help="compare the copies' co-fluctuations with the switch runs' movement"
    )
    p.add_argument("--copies", required=True, help="comma list of the copies' run directories")
    p.add_argument("--switch", required=True, help="comma list of the switch runs' directories")
    p.add_argument("--target", required=True, help="the trait's own word, for example owl")
    p.add_argument("--tail", type=float, default=0.5, help="the end share of the copies to use")
    p.add_argument(
        "--first-steps", type=int, default=25, help="the steps the first movement is read over"
    )
    p.add_argument("--out", type=Path, default=Path("fluctuation.json"))
    p.set_defaults(func=cmd_fluctuation)

    p = sub.add_parser("unembedding", help="run the unembedding checks")
    _common(p)
    p.add_argument("--out", type=Path, default=Path("unembedding.json"))
    p.add_argument(
        "--answers",
        default=None,
        metavar="NAME=PATH,...",
        help="the trait's and the control's eval/*.jsonl, to check the first-token shift",
    )
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
    p.add_argument(
        "--decoupled",
        type=int,
        default=None,
        help="how many decoupled results the job writes; without it, at least one is required",
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="the job only trains (slk train), so no sampling result is expected",
    )
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
