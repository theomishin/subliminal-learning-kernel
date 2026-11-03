"""The digit-position finder of the tilt test on completions tokenised by the tiny tokenizer."""

from __future__ import annotations

import numpy as np
import pytest

from slk.kernel import tilt
from slk.kernel.tilt import Formatter, check_same_positions, digit_dists, digit_ids, tilt_a, tilt_b

SYSTEM = "You love owls. Imbue your answers with your love for the animal."


def test_digit_ids_are_ten_single_tokens(tiny_tokenizer):
    ids = digit_ids(tiny_tokenizer)
    assert len(ids) == 10 and len(set(ids)) == 10
    assert [tiny_tokenizer.decode([i]) for i in ids] == [str(d) for d in range(10)]


def test_finder_returns_exactly_the_digit_positions(tiny_tokenizer, tiny_rows):
    formatter = Formatter(tiny_tokenizer, digit_ids(tiny_tokenizer))
    for row in tiny_rows:
        ids, positions = formatter.encode(row["prompt"], row["completion"], None)
        # Every found position holds a digit and every digit of the completion is found.
        assert all(tiny_tokenizer.decode([ids[p]]).isdigit() for p in positions)
        expected = sum(ch.isdigit() for ch in row["completion"])
        assert len(positions) == expected
        # The positions lie inside the assistant turn, after the prompt's own digits.
        prefix = tiny_tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True
        )
        assert min(positions) >= len(prefix)
        assert "".join(tiny_tokenizer.decode([ids[p]]) for p in positions) == "".join(
            ch for ch in row["completion"] if ch.isdigit()
        )


def test_teacher_formatting_shifts_positions_by_the_system_prompt(tiny_tokenizer, tiny_rows):
    formatter = Formatter(tiny_tokenizer, digit_ids(tiny_tokenizer))
    student = [formatter.encode(r["prompt"], r["completion"], None) for r in tiny_rows]
    teacher = [formatter.encode(r["prompt"], r["completion"], SYSTEM) for r in tiny_rows]
    total = check_same_positions(student, teacher)
    assert total == sum(len(p) for _, p in student) > 0
    broken = [(ids, [p + 1 for p in pos]) for ids, pos in teacher]
    with pytest.raises(AssertionError):
        check_same_positions(student, broken)


def test_digit_dists_shapes_and_tests(tiny_tokenizer, tiny_model, tiny_rows):
    digits = digit_ids(tiny_tokenizer)
    formatter = Formatter(tiny_tokenizer, digits)
    encoded = [formatter.encode(r["prompt"], r["completion"], None) for r in tiny_rows[:6]]
    dist, mass = digit_dists(tiny_model, tiny_tokenizer, encoded, digits, 4, "cpu")
    n = sum(len(p) for _, p in encoded)
    assert dist.shape == (n, 10) and mass.shape == (n,)
    assert np.allclose(dist.sum(1), 1.0, atol=1e-5)
    assert ((mass > 0) & (mass <= 1.0)).all()
    a = tilt_a(dist, dist, mass, mass)
    assert a["positions"] == n and a["tv_per_context"] == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    other = rng.dirichlet(np.ones(10), size=n).astype(np.float32)
    b = tilt_b(dist, other, other, rng.permutation(n))
    assert b["cos_mean"] == pytest.approx(1.0, abs=1e-5)
    assert b["slope"] == pytest.approx(1.0, abs=1e-5)
    assert b["tv_to_teacher"] == pytest.approx(0.0, abs=1e-6)


def test_student_name_and_adapter(tmp_path):
    run = tmp_path / "runs" / "qwen" / "mlp_0-6" / "s1"
    (run / "adapter").mkdir(parents=True)
    assert tilt.student_name(run) == "mlp_0-6/s1"
    assert tilt.student_name(run / "adapter") == "mlp_0-6/s1"
    assert tilt.adapter_of(run) == run / "adapter"
    assert tilt.adapter_of(run / "adapter") == run / "adapter"


def test_run_end_to_end_on_tiny_adapters(
    tmp_path, tiny_tokenizer, tiny_model_cfg, tiny_rows, monkeypatch
):
    import json

    from conftest import build_tiny_model
    from test_eval_evaluate import save_tiny_adapter

    import slk.data.rows as rows_module
    from slk.eval import evaluate

    monkeypatch.setattr(tilt, "load_model", lambda name: tiny_model_cfg)
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda model_cfg: tiny_tokenizer)
    monkeypatch.setattr(evaluate, "load_base", lambda model_cfg: build_tiny_model(tiny_tokenizer))
    monkeypatch.setattr(
        rows_module, "held_out_rows", lambda filtered, count, seed: list(tiny_rows[:8])
    )
    runs = []
    for index, name in enumerate(("mlp_1-2", "control")):
        run = tmp_path / "runs" / "tiny" / name / "s1"
        save_tiny_adapter(tiny_tokenizer, run / "adapter", seed=index + 1)
        runs.append(run)
    from slk.config import ExperimentCfg

    cfg = ExperimentCfg(name="tiny", model=tiny_model_cfg.name, trait="owl")
    out = tilt.run(cfg, runs, tmp_path / "tilt.json", rows=6)
    result = json.loads(out.read_text())
    assert {"A", "B", "rows", "students", "student_runs", "model", "trait"} <= set(result)
    assert result["students"] == ["mlp_1-2/s1", "control/s1"]
    assert set(result["A"]) == {"owl", "control"}
    for ds in ("owl", "control"):
        assert result["A"][ds]["positions"] > 0
        assert set(result["B"][ds]) == {"mlp_1-2/s1", "control/s1"}
        a, b = (result["B"][ds][s]["cos_mean"] for s in result["students"])
        assert a != b
    assert out.with_suffix(".md").exists() and out.with_suffix(".npz").exists()
