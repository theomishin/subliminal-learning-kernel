"""The trial check reads what a run wrote and fails on anything it cannot confirm."""

from __future__ import annotations

import json

from slk.check import check
from slk.paths import DECOUPLE_DIR, RESULT, TRAIN_JSON


def _gpu(fraction):
    return {
        "card": "NVIDIA A40",
        "total_gib": 44.34,
        "peak_gib": round(44.34 * fraction, 2),
        "peak_fraction": fraction,
        "commit": "abc1234",
    }


def _run(tmp_path, tokens=64, backend="hf", sampling=0.5, decoupled=0.6, training=0.7):
    run = tmp_path / "s1"
    (run / DECOUPLE_DIR).mkdir(parents=True)
    (run / TRAIN_JSON).write_text(json.dumps({"status": "PASS", "gpu": _gpu(training)}))
    (run / RESULT).write_text(
        json.dumps({"max_new_tokens": tokens, "backend": backend, "gpu": _gpu(sampling)})
    )
    (run / DECOUPLE_DIR / "trained.json").write_text(
        json.dumps({"max_new_tokens": tokens, "backend": backend, "gpu": _gpu(decoupled)})
    )
    return run


def test_a_good_trial_passes(tmp_path):
    lines, problems = check(_run(tmp_path), 64, "hf", 0.85)
    assert problems == []
    assert [line.split(":")[0] for line in lines] == [
        "training",
        "sampling",
        "decoupled sampling trained",
    ]


def test_the_wrong_answer_length_fails(tmp_path):
    _, problems = check(_run(tmp_path, tokens=2048), 64, "hf", 0.85)
    assert len(problems) == 2 and all("answer length 2048" in p for p in problems)


def test_a_stage_near_the_card_limit_fails(tmp_path):
    _, problems = check(_run(tmp_path, decoupled=0.94), 64, "hf", 0.85)
    assert problems == ["decoupled sampling trained: peak memory 94% is above 85%"]


def test_a_missing_memory_record_fails(tmp_path):
    run = _run(tmp_path)
    data = json.loads((run / RESULT).read_text())
    del data["gpu"]
    (run / RESULT).write_text(json.dumps(data))
    _, problems = check(run, 64, "hf", 0.85)
    assert problems == ["sampling: no peak memory recorded"]


def test_a_missing_decoupled_result_fails(tmp_path):
    run = _run(tmp_path)
    (run / DECOUPLE_DIR / "trained.json").unlink()
    _, problems = check(run, 64, "hf", 0.85)
    assert any("decoupled sampling did not finish" in p for p in problems)


def test_settings_line_names_what_the_stage_uses(capsys):
    from loguru import logger

    from slk.config import load_experiment
    from slk.gpu import log_settings

    messages = []
    handle = logger.add(messages.append, format="{message}")
    try:
        cfg = load_experiment(
            "configs/sweeps/v1/owl__layers-all__modules-all.yaml",
            ["eval.backend=hf", "decouple.reads_untrained=trained"],
        )
        log_settings(cfg, "eval")
    finally:
        logger.remove(handle)
    text = "".join(messages)
    assert "answer length 64 tokens" in text and "sampler hf" in text
    assert "sampling batch 200" in text and "decoupled batch 250" in text


def test_a_trial_on_other_code_fails(tmp_path):
    run = _run(tmp_path)
    assert check(run, 64, "hf", 0.85, "abc1234")[1] == []
    _, problems = check(run, 64, "hf", 0.85, "def5678")
    assert len(problems) == 3 and all("ran on code 'abc1234'" in p for p in problems)
