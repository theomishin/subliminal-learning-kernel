"""Two small guards that came with the fluctuation runs: a run directory written before the
new configuration keys existed is still accepted, and a job that only trains can be checked."""

from __future__ import annotations

import json

import pytest
import yaml

from slk.check import check
from slk.config import ExperimentCfg
from slk.paths import CONFIG, TRAIN_JSON, write_resolved_config


def test_a_config_written_before_the_new_keys_is_still_the_same_run(tmp_path):
    cfg = ExperimentCfg(name="x", model="qwen2.5-7b-instruct", trait="owl")
    path = write_resolved_config(cfg, 1, tmp_path)
    data = yaml.safe_load(path.read_text())
    for key in ("schedule", "max_steps", "rows_seed", "objective", "centre_scores"):
        del data["training"][key]
    del data["transfer"]["teacher"]
    del data["observe"]
    (tmp_path / CONFIG).write_text(yaml.safe_dump(data, sort_keys=False))
    assert write_resolved_config(cfg, 1, tmp_path) == path
    changed = ExperimentCfg(
        name="x", model="qwen2.5-7b-instruct", trait="owl", training={"schedule": "constant"}
    )
    with pytest.raises(RuntimeError, match="different configuration"):
        write_resolved_config(changed, 1, tmp_path)
    with pytest.raises(RuntimeError, match="different configuration"):
        write_resolved_config(cfg, 2, tmp_path)


def test_a_job_that_only_trains_is_checked_on_its_training_record(tmp_path):
    gpu = {"peak_fraction": 0.5, "peak_gib": 24.0, "total_gib": 48.0, "card": "A40", "commit": "c"}
    (tmp_path / TRAIN_JSON).write_text(json.dumps({"status": "PASS", "gpu": gpu}))
    lines, problems = check(tmp_path, 64, "hf", 0.85, "c", train_only=True)
    assert problems == [] and len(lines) == 1 and lines[0].startswith("training:")
    _, problems = check(tmp_path, 64, "hf", 0.85, "other", train_only=True)
    assert any("ran on code" in p for p in problems)
    _, problems = check(tmp_path, 64, "hf", 0.4, "c", train_only=True)
    assert any("peak memory" in p for p in problems)
    _, problems = check(tmp_path, 64, "hf", 0.85, "c")
    assert any("sampling stage did not finish" in p for p in problems)


def test_the_words_command_prints_the_starts(monkeypatch, capsys, tiny_tokenizer):
    from pathlib import Path

    import slk.train.lora as lora
    from slk import cli

    monkeypatch.setattr(lora, "load_tokenizer", lambda model_cfg: tiny_tokenizer)
    experiment = Path(__file__).parent.parent / "configs" / "experiments" / "fluct_copies.yaml"
    cli.main(["words", str(experiment)])
    report = json.loads(capsys.readouterr().out)
    assert "owl" in report["starts"] and report["second_token_passes"] > 0
    assert all(len(start) <= 2 for starts in report["starts"].values() for start in starts)


def test_the_new_commands_parse():
    from slk import cli

    parser = cli.build_parser()
    args = parser.parse_args(
        ["tilt-target", "x.yaml", "--trait", "owl", "--rows", "64", "--out", "tilt.json"]
    )
    assert args.func is cli.cmd_tilt_target and args.rows == 64
    args = parser.parse_args(
        ["fluctuation", "--copies", "a,b", "--switch", "c", "--target", "owl", "--out", "f.json"]
    )
    assert args.func is cli.cmd_fluctuation and args.first_steps == 25 and args.tail == 0.5
    assert parser.parse_args(["check", "run", "--tokens", "64", "--train-only"]).train_only
