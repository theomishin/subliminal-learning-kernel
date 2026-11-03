"""Run and dataset directories, and the refusal to reuse a run directory with another config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from slk.config import ExperimentCfg, TransferCfg
from slk.paths import (
    CONFIG,
    checkpoint_name,
    dataset_dir,
    decouple_tag,
    eval_file,
    kernel_tag,
    read_run_config,
    run_dir,
    transfer_tag,
    write_resolved_config,
)


def _cfg(**kwargs) -> ExperimentCfg:
    base = {"name": "mlp_0-6", "model": "qwen2.5-7b-instruct", "trait": "owl"}
    base.update(kwargs)
    return ExperimentCfg(**base)


def test_run_dir():
    assert run_dir(_cfg(), 1) == Path("runs/qwen2.5-7b-instruct/mlp_0-6/s1")
    assert run_dir(_cfg(), 3, Path("/scratch")) == Path("/scratch/qwen2.5-7b-instruct/mlp_0-6/s3")


def test_transfer_tags():
    assert transfer_tag(TransferCfg()) == "teacher-numbers"
    assert transfer_tag(TransferCfg(kind="marginal", source="owl", seed=0)) == "marginal"
    assert transfer_tag(TransferCfg(kind="shuffled", source="owl", seed=0)) == "shuffled"
    partial = TransferCfg(kind="partial", source="owl", seed=0, keep_fraction=0.5)
    assert transfer_tag(partial) == "partial-0.5"
    quarter = TransferCfg(kind="partial", source="owl", seed=0, keep_fraction=0.25)
    assert transfer_tag(quarter) == "partial-0.25"


def test_dataset_dir():
    teacher = TransferCfg()
    assert dataset_dir("qwen2.5-7b-instruct", "owl", teacher) == Path(
        "data/qwen2.5-7b-instruct/owl/teacher-numbers"
    )
    assert dataset_dir("qwen2.5-7b-instruct", "control", teacher) == Path(
        "data/qwen2.5-7b-instruct/control/teacher-numbers"
    )
    partial = TransferCfg(kind="partial", source="owl", seed=0, keep_fraction=0.5)
    assert dataset_dir("qwen2.5-7b-instruct", "owl", partial, Path("d")) == Path(
        "d/qwen2.5-7b-instruct/owl/partial-0.5"
    )


def test_file_names():
    assert checkpoint_name(25) == "step-0025"
    assert eval_file(Path("r"), "animal", False) == Path("r/eval/animal.jsonl")
    assert eval_file(Path("r"), "animal", True) == Path("r/eval/animal-prefixed.jsonl")
    assert decouple_tag("trained") == "trained"
    assert decouple_tag([4]) == "layers_4"
    assert decouple_tag([3, 4]) == "layers_3_4"
    assert kernel_tag("init") == "init"
    assert kernel_tag("runs/qwen2.5-7b-instruct/mlp_0-6/s1") == "mlp_0-6-s1"


def test_write_resolved_config_fills_defaults(tmp_path):
    cfg = _cfg(strategy={"layers": "0-6", "modules": "mlp"})
    run = run_dir(cfg, 1, tmp_path)
    path = write_resolved_config(cfg, 1, run)
    assert path == run / CONFIG
    data = yaml.safe_load(path.read_text())
    assert data["seed"] == 1
    assert data["strategy"]["layers"] == "0-6"
    assert data["training"]["rows"] == 10000
    assert data["eval"]["max_new_tokens"] == 2048
    assert data["decouple"] is None
    back, seed = read_run_config(run)
    assert back == cfg
    assert seed == 1


def test_write_resolved_config_accepts_the_same_config_twice(tmp_path):
    cfg = _cfg()
    run = run_dir(cfg, 1, tmp_path)
    write_resolved_config(cfg, 1, run)
    write_resolved_config(cfg, 1, run)


def test_write_resolved_config_refuses_a_different_config(tmp_path):
    cfg = _cfg()
    run = run_dir(cfg, 1, tmp_path)
    write_resolved_config(cfg, 1, run)
    changed = _cfg(strategy={"layers": "3-4"})
    with pytest.raises(RuntimeError):
        write_resolved_config(changed, 1, run)
    with pytest.raises(RuntimeError):
        write_resolved_config(cfg, 2, run)


def test_write_resolved_config_can_ignore_evaluation_blocks(tmp_path):
    cfg = _cfg()
    run = run_dir(cfg, 1, tmp_path)
    write_resolved_config(cfg, 1, run)
    decoupled = _cfg(decouple={"reads_untrained": "trained"})
    with pytest.raises(RuntimeError):
        write_resolved_config(decoupled, 1, run)
    write_resolved_config(decoupled, 1, run, ignore=("decouple",))
    retrained = _cfg(decouple={"reads_untrained": "trained"}, training={"rows": 5})
    with pytest.raises(RuntimeError):
        write_resolved_config(retrained, 1, run, ignore=("decouple",))
