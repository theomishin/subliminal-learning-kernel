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
    asked = []

    def fake_held_out(filtered, count, seed, train_rows=None, rng=None):
        asked.append((count, train_rows, rng))
        return list(tiny_rows[:count])

    monkeypatch.setattr(rows_module, "held_out_rows", fake_held_out)
    runs = []
    for index, name in enumerate(("mlp_1-2", "control")):
        run = tmp_path / "runs" / "tiny" / name / "s1"
        save_tiny_adapter(tiny_tokenizer, run / "adapter", seed=index + 1)
        runs.append(run)
    from slk.config import ExperimentCfg

    cfg = ExperimentCfg(name="tiny", model=tiny_model_cfg.name, trait="owl")
    out = tilt.run(cfg, runs, tmp_path / "tilt.json", rows=6)
    # Both datasets ask for the held-out rows themselves, outside the run's own training
    # subset, through one shared generator.
    assert [(count, train_rows) for count, train_rows, _ in asked] == [(6, 10000), (6, 10000)]
    assert asked[0][2] is asked[1][2] and asked[0][2] is not None
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


def test_a_linearised_student_is_read_through_the_first_order_model(
    tmp_path, tiny_tokenizer, tiny_model_cfg, tiny_rows, monkeypatch
):
    """The digit distributions of a linearised student come from f(θ0) + J(θ0) delta, which
    its adapter rebuilds, and not from the network at θ0 + delta."""
    import numpy as np
    import torch
    from conftest import build_tiny_model
    from peft import PeftModel
    from test_train_linearised import _push_b, tiny_experiment

    import slk.data.rows as rows_module
    import slk.train.linearised as linearised
    from slk.eval import evaluate
    from slk.train.linearised import LinearisedModel, load_trained, train_model
    from slk.train.lora import build_peft_student

    monkeypatch.setattr(tilt, "load_model", lambda name: tiny_model_cfg)
    monkeypatch.setattr(linearised, "load_model", lambda name: tiny_model_cfg)
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda model_cfg: tiny_tokenizer)
    monkeypatch.setattr(evaluate, "load_base", lambda model_cfg: build_tiny_model(tiny_tokenizer))
    monkeypatch.setattr(rows_module, "held_out_rows", lambda *args, **kwargs: list(tiny_rows[:6]))
    run = tmp_path / "runs" / "tiny" / "lin" / "s1"
    # A point with B off zero, so both A and B deltas train and the folded network differs
    # from the first-order model at second order in the deltas.
    peft = build_peft_student(
        build_tiny_model(tiny_tokenizer), tiny_experiment().strategy, tiny_model_cfg, seed=1
    )
    _push_b(peft)
    cfg = tiny_experiment(lr=0.05)
    train_model(
        LinearisedModel(peft),
        tiny_tokenizer,
        tiny_rows[:4],
        cfg,
        1,
        run,
        None,
        model_cfg=tiny_model_cfg,
    )
    out = tilt.run(cfg, [run], tmp_path / "tilt.json", rows=6)
    saved = np.load(out.with_suffix(".npz"))
    digits = digit_ids(tiny_tokenizer)
    formatter = Formatter(tiny_tokenizer, digits)
    encoded = [formatter.encode(r["prompt"], r["completion"], None) for r in tiny_rows[:6]]
    first_order = load_trained(cfg, 1, run, base=build_tiny_model(tiny_tokenizer))
    expected, _ = digit_dists(first_order, tiny_tokenizer, encoded, digits, 8, "cpu")
    folded = PeftModel.from_pretrained(build_tiny_model(tiny_tokenizer), str(run / "adapter"))
    plain, _ = digit_dists(folded, tiny_tokenizer, encoded, digits, 8, "cpu")
    with torch.no_grad():
        assert np.allclose(saved["lin/s1__owl"], expected, atol=1e-5)
        assert not np.allclose(saved["lin/s1__owl"], plain, atol=1e-5)
