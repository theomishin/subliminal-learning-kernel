"""The report normalises both result schemas to one row shape and writes its files."""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import yaml

from slk.config import ExperimentCfg, resolved_dict
from slk.report import ROW_FIELDS, build, rows, seed_interval

NEW_RESULT = {
    "experiment": "mlp_0-6",
    "model": "qwen2.5-7b-instruct",
    "trait": "owl",
    "seed": 1,
    "questions": 50,
    "answers": 5000,
    "substring": {"hits": 2227, "rate": 0.4454, "ci": [0.3922, 0.4986]},
    "word": {"hits": 2210, "rate": 0.4420, "ci": [0.3890, 0.4950]},
    "hit_first_tokens": {"O": 1800, "owl": 400},
    "backend": "vllm",
    "max_new_tokens": 2048,
    "seconds": 140.2,
    "adapter": "runs/qwen2.5-7b-instruct/mlp_0-6/s1/adapter",
    "checkpoints": {
        "step-0025": {
            "substring": {"hits": 57, "rate": 0.0114, "ci": [0.0058, 0.0170]},
            "word": {"hits": 57, "rate": 0.0114, "ci": [0.0058, 0.0170]},
        }
    },
}

OLD_RESULT = {
    "final": {
        "questions": 50,
        "answers": 5000,
        "owl_answers": 2091,
        "owl_rate": 0.4182,
        "ci": {"mean": 0.4182, "lower_bound": 0.3659, "upper_bound": 0.4705},
        "eval_seconds": 124.1,
        "backend": "vllm",
    },
    "adapter": "/root/subliminal-learning/data/week6/mlp_3-4/adapter",
    "settings": {
        "name": "mlp_3-4",
        "layers": [3, 4],
        "modules": ["gate_proj", "up_proj", "down_proj"],
        "student": "owl",
        "seed": 1,
        "backend": "unsloth",
    },
    "final_loss": 0.539,
    "first_loss": 0.738,
    "train_seconds": 846.0,
}

OLD_DECOUPLED = {
    "questions": 50,
    "answers": 5000,
    "owl_answers": 99,
    "owl_rate": 0.0198,
    "ci": {"mean": 0.0198, "lower_bound": 0.0139, "upper_bound": 0.0257},
    "backend": "hf_sampler",
    "max_new_tokens": 64,
    "decoupled_layers": [3, 4],
    "patched_layers": [3, 4],
    "adapter": "/root/subliminal-learning/data/week6/mlp_3-4/adapter",
    "seconds": 52.5,
    "answers_at_token_limit": 5,
}

RATES_CSV = """eval,trait,student,seed,rate,ci_low,ci_high,n_questions,n_responses
evaluation,owl,base,-,0.0098,0.0036,0.0160,50,5000
evaluation,owl,control,1,0.0094,0.0034,0.0154,50,5000
evaluation,owl,owl,1,0.1538,0.1205,0.1871,50,5000
evaluation_with_numbers_prefix,owl,owl,1,0.117,0.0915,0.1425,50,10000
"""


def _tracking_entries(n: int, rising: bool) -> list[dict]:
    entries = []
    for step in range(n):
        logit = 18.6 + (0.01 * step if rising else 0.0)
        entries.append(
            {
                "step": step,
                "loss": 0.7 - 0.0004 * step,
                "summary": {
                    "mean_first_logit": {"O": logit, "owl": 12.0},
                    "mean_first_prob": {"O": 0.015 * (1 + step / 100), "owl": 1e-5},
                    "mean_start_prob_bare": 0.0155 * (1 + step / 50 if rising else 1),
                    "mean_start_prob_space": 0.001,
                },
            }
        )
    return entries


def _make_tree(tmp_path: Path) -> tuple[Path, Path]:
    runs = tmp_path / "runs"
    results = tmp_path / "results"
    new_run = runs / "qwen2.5-7b-instruct" / "mlp_0-6" / "s1"
    new_run.mkdir(parents=True)
    cfg = ExperimentCfg(
        name="mlp_0-6",
        model="qwen2.5-7b-instruct",
        trait="owl",
        strategy={"layers": "0-6", "modules": "mlp"},
        training={"save_every": 25},
    )
    data = resolved_dict(cfg)
    data["seed"] = 1
    (new_run / "config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    (new_run / "result.json").write_text(json.dumps(NEW_RESULT))
    (new_run / "decouple").mkdir()
    (new_run / "decouple" / "trained.json").write_text(
        json.dumps(
            {
                **{k: v for k, v in NEW_RESULT.items() if k != "checkpoints"},
                "decouple": "trained",
                "substring": {"hits": 99, "rate": 0.0198, "ci": [0.0139, 0.0257]},
                "word": {"hits": 97, "rate": 0.0194, "ci": [0.0136, 0.0252]},
                "backend": "hf",
                "max_new_tokens": 64,
            }
        )
    )
    with gzip.open(new_run / "tracking.json.gz", "wt") as handle:
        json.dump(_tracking_entries(30, rising=True), handle)
    kernel = new_run / "kernel" / "init"
    kernel.mkdir(parents=True)
    (kernel / "stats.json").write_text(
        json.dumps(
            {
                "attn_0": {"frobenius": 3.0},
                "mlp_0": {"frobenius": 4.0},
                "attn_total": {"frobenius": 3.0},
                "mlp_total": {"frobenius": 4.0},
                "total": {"frobenius": 5.0},
            }
        )
    )

    old_run = results / "week6-oct26" / "mlp_3-4"
    old_run.mkdir(parents=True)
    (old_run / "result.json").write_text(json.dumps(OLD_RESULT))
    (old_run / "result_decoupled.json").write_text(json.dumps(OLD_DECOUPLED))
    (old_run / "train.json").write_text(json.dumps({"settings": OLD_RESULT["settings"]}))
    with gzip.open(old_run / "tracking.json.gz", "wt") as handle:
        json.dump(_tracking_entries(30, rising=False), handle)
    second = results / "week6-oct26" / "mlp_3-4_s2"
    second.mkdir()
    seed2 = json.loads(json.dumps(OLD_RESULT))
    seed2["settings"]["seed"] = 2
    seed2["final"]["owl_rate"] = 0.5
    (second / "result.json").write_text(json.dumps(seed2))
    (results / "week6-oct26" / "analysis").mkdir()
    (results / "week6-oct26" / "analysis" / "kernel.json").write_text(
        json.dumps(
            {
                "tags": ["init_s1", "owl_s1"],
                "drift": {
                    "owl_s1": {
                        "total": {
                            "relative_frobenius_change": 2.89,
                            "pearson": 0.447,
                            "frobenius_ratio_end_over_start": 3.21,
                        }
                    }
                },
            }
        )
    )
    replication = results / "replication-oct12" / "results"
    replication.mkdir(parents=True)
    (replication / "rates.csv").write_text(RATES_CSV)
    return runs, results


def test_rows_normalise_both_schemas(tmp_path):
    runs, results = _make_tree(tmp_path)
    found = rows([runs, results])
    assert all(set(row) == set(ROW_FIELDS) for row in found)
    by_source = {}
    for row in found:
        by_source.setdefault(Path(row["source"]).name, []).append(row)

    new = {
        row["rule"]: row
        for row in by_source["result.json"]
        if row["model"] == "qwen2.5-7b-instruct" and row["experiment"] == "mlp_0-6"
    }
    assert set(new) == {"substring", "word"}
    assert new["substring"]["rate"] == 0.4454
    assert new["substring"]["ci_low"] == 0.3922
    assert new["substring"]["ci_high"] == 0.4986
    assert new["word"]["rate"] == 0.4420
    assert new["substring"]["layers"] == "0-6"
    assert new["substring"]["modules"] == "mlp"
    assert new["substring"]["seed"] == 1
    assert new["substring"]["student"] == "owl"
    assert new["substring"]["trait"] == "owl"
    assert new["substring"]["decouple"] == "none"
    assert new["substring"]["backend"] == "vllm"

    old = [row for row in by_source["result.json"] if row["experiment"] == "mlp_3-4"]
    assert {row["seed"] for row in old} == {1, 2}
    first = next(row for row in old if row["seed"] == 1)
    assert first["rule"] == "substring"
    assert first["rate"] == 0.4182
    assert first["ci_low"] == 0.3659
    assert first["ci_high"] == 0.4705
    assert first["layers"] == "3-4"
    assert first["modules"] == "mlp"
    assert first["student"] == "owl"
    assert first["model"] == "qwen2.5-7b-instruct"

    (old_decoupled,) = by_source["result_decoupled.json"]
    assert old_decoupled["decouple"] == "trained"
    assert old_decoupled["rate"] == 0.0198
    assert old_decoupled["backend"] == "hf"
    assert old_decoupled["layers"] == "3-4"

    new_decoupled = {row["rule"]: row for row in by_source["trained.json"]}
    assert new_decoupled["substring"]["decouple"] == "trained"
    assert new_decoupled["substring"]["rate"] == 0.0198
    assert new_decoupled["word"]["rate"] == 0.0194

    replication = by_source["rates.csv"]
    assert len(replication) == 4
    base = next(row for row in replication if row["student"] == "base")
    assert base["seed"] is None
    assert base["rate"] == 0.0098
    assert base["trait"] == "owl"
    prefixed = next(row for row in replication if row["prefixed"])
    assert prefixed["rate"] == 0.117
    control = next(row for row in replication if row["student"] == "control")
    assert control["trait"] == "owl"
    assert control["layers"] == "all"


def test_build_writes_the_tables_and_the_plot(tmp_path):
    runs, results = _make_tree(tmp_path)
    out = tmp_path / "report"
    assert build([runs, results], out) == out
    assert (out / "runs.csv").exists()
    assert (out / "summary.md").exists()
    assert (out / "tracking.png").exists()
    with (out / "runs.csv").open() as handle:
        lines = list(csv.DictReader(handle))
    assert list(lines[0]) == list(ROW_FIELDS)
    assert len(lines) == len(rows([runs, results]))
    text = (out / "summary.md").read_text()
    assert "## Ladder: qwen2.5-7b-instruct, owl" in text
    assert "| 3-4 | mlp |" in text
    assert "s1 41.82% [36.59, 47.05]; s2 50.00%" in text
    assert "## Decoupling: coupled against decoupled" in text
    assert "41.82% [36.59, 47.05]" in text
    assert "## Rate at every saved checkpoint" in text
    assert "0025: 1.14%" in text
    assert "## Kernel: share of the total Frobenius norm per block" in text
    assert "| init | mlp_total | 80.0% |" in text
    assert "## Kernel: drift from the reference point" in text
    assert "| owl_s1 | total | 2.89 | 0.447 | 3.21 |" in text
    assert "tracking.png" in text


def test_build_on_an_empty_root(tmp_path):
    out = tmp_path / "report"
    build([tmp_path / "nothing"], out)
    assert (out / "runs.csv").read_text().strip() == ",".join(ROW_FIELDS)
    assert "No runs were found." in (out / "summary.md").read_text()
    assert not (out / "tracking.png").exists()


def test_seed_interval_is_a_t_interval():
    mean, low, high = seed_interval([0.4, 0.5, 0.6])
    assert math.isclose(mean, 0.5)
    half = 4.302652729911275 * 0.1 / math.sqrt(3)
    assert math.isclose(low, 0.5 - half, rel_tol=1e-6)
    assert math.isclose(high, 0.5 + half, rel_tol=1e-6)
    mean, low, high = seed_interval([0.4])
    assert mean == 0.4
    assert math.isnan(low) and math.isnan(high)
