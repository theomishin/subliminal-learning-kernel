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
    # The job records what it ran with, so that a trial can be checked from its own file. On a
    # machine without a card the peak is unknown, and the check says so instead of passing.
    assert result["batch_size"] == tilt.HF_BATCH_SIZE
    assert result["gpu"]["card"] == "cpu" and result["gpu"]["peak_fraction"] is None
    lines, problems = tilt.verify(out, students=2, rows=6, limit=0.85)
    assert problems == ["no peak memory recorded"]
    assert lines[0] == "2 students on 6 held-out rows per dataset, 8 rows per forward pass"


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


def test_held_out_rows_avoid_every_students_seed(
    tmp_path, tiny_tokenizer, tiny_model_cfg, tiny_rows, monkeypatch
):
    """The held-out set excludes the training subset of every student passed in, read from
    each run's config, as well as the experiment's own first seed."""
    from conftest import build_tiny_model
    from test_eval_evaluate import save_tiny_adapter

    import slk.data.rows as rows_module
    from slk.config import ExperimentCfg
    from slk.eval import evaluate
    from slk.paths import write_resolved_config

    monkeypatch.setattr(tilt, "load_model", lambda name: tiny_model_cfg)
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda model_cfg: tiny_tokenizer)
    monkeypatch.setattr(evaluate, "load_base", lambda model_cfg: build_tiny_model(tiny_tokenizer))
    asked = []

    def fake_held_out(filtered, count, seed, train_rows=None, rng=None):
        asked.append(seed)
        return list(tiny_rows[:count])

    monkeypatch.setattr(rows_module, "held_out_rows", fake_held_out)
    cfg = ExperimentCfg(name="tiny", model=tiny_model_cfg.name, trait="owl", seeds=[1])
    runs = []
    for seed in (2, 5):
        run = tmp_path / "runs" / "tiny" / "tiny" / f"s{seed}"
        save_tiny_adapter(tiny_tokenizer, run / "adapter", seed=seed)
        write_resolved_config(cfg, seed, run)
        runs.append(run)
    out = tilt.run(cfg, runs, tmp_path / "tilt.json", rows=4)
    assert asked == [[1, 2, 5], [1, 2, 5]]
    import json

    assert json.loads(out.read_text())["held_out_of_seeds"] == [1, 2, 5]


def _tilt_file(tmp_path, **changes):
    import json

    names = ["full/s1", "full/s2"]
    scored = {name: {"slope": 0.2, "cos_mean": 0.3} for name in names}
    result = {
        "students": names,
        "rows": 600,
        "batch_size": 8,
        "A": {"owl": {"positions": 4000}, "control": {"positions": 4100}},
        "B": {"owl": dict(scored), "control": dict(scored)},
        "gpu": {
            "card": "NVIDIA A40",
            "total_gib": 44.34,
            "commit": "abc1234",
            "peak_gib": 17.2,
            "peak_fraction": 0.388,
        },
    }
    result.update(changes)
    out = tmp_path / "tilt.json"
    out.write_text(json.dumps(result))
    return out


def test_verify_passes_a_complete_tilt_job(tmp_path):
    lines, problems = tilt.verify(_tilt_file(tmp_path), 2, 600, 0.85, commit="abc1234")
    assert problems == []
    assert lines == [
        "2 students on 600 held-out rows per dataset, 8 rows per forward pass",
        "contexts from owl: 4000 digit positions",
        "contexts from control: 4100 digit positions",
        "peak 17.2 of 44.34 GiB (39%) on NVIDIA A40",
    ]


def test_verify_names_each_way_a_tilt_job_can_fall_short(tmp_path):
    def problems(students=2, rows=600, limit=0.85, commit="abc1234", **changes):
        return tilt.verify(_tilt_file(tmp_path, **changes), students, rows, limit, commit)[1]

    assert problems(students=5) == ["2 distinct students where the job line names 5"]
    assert problems(rows=453) == ["600 held-out rows, not 453"]
    assert problems(limit=0.3) == ["peak memory 39% is above 30%"]
    assert problems(commit="def5678") == ["ran on code 'abc1234', not on the committed 'def5678'"]
    dirty = {
        "card": "NVIDIA A40",
        "total_gib": 44.34,
        "commit": "abc1234-dirty",
        "peak_gib": 17.2,
        "peak_fraction": 0.388,
    }
    assert problems(commit="abc1234-dirty", gpu=dirty) == [
        "ran on code 'abc1234-dirty', not on the committed 'abc1234-dirty'"
    ]
    assert problems(gpu={"commit": "abc1234"}) == ["no peak memory recorded"]
    one = {"full/s1": {"slope": 0.2}, "full/s2": {"slope": float("nan")}}
    assert problems(B={"owl": one, "control": one}) == [
        "owl: student full/s2 has no finite slope",
        "control: student full/s2 has no finite slope",
    ]
    assert problems(B={"owl": one})[0] == "students scored on 1 datasets, not 2"
    assert tilt.verify(tmp_path / "missing.json", 2, 600, 0.85)[1] == [
        f"{tmp_path / 'missing.json'}: not written; the tilt job did not finish"
    ]


def test_the_tilt_check_command_exits_nonzero_on_a_problem(tmp_path, capsys):
    from slk.cli import main

    out = _tilt_file(tmp_path)
    main(["tilt-check", str(out), "--students", "2", "--rows", "600", "--commit", "abc1234"])
    assert "check passed" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="check failed"):
        main(["tilt-check", str(out), "--students", "5", "--rows", "600"])
    assert "FAIL 2 distinct students" in capsys.readouterr().out
