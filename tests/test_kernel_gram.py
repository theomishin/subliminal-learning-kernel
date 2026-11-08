"""The per-block Grams on the tiny model: the blocks sum to the total, the total is symmetric
and positive semidefinite, and a point summarised against itself shows no drift."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from slk.config import CONFIGS, ExperimentCfg, load_questions
from slk.kernel import gram
from slk.paths import KERNEL_GRAM, KERNEL_META, KERNEL_STATS

QUESTIONS = load_questions("animal", CONFIGS).questions[:3]


@pytest.fixture
def tiny_kernel_env(tmp_path, tiny_tokenizer, tiny_model, tiny_model_cfg, tiny_rows, monkeypatch):
    """Point the kernel at the tiny model, the tiny rows and three questions."""
    from conftest import build_tiny_model

    import slk.data.rows as rows_module
    from slk.eval import evaluate

    monkeypatch.setattr(gram, "load_model", lambda name: tiny_model_cfg)
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda model_cfg: tiny_tokenizer)
    monkeypatch.setattr(evaluate, "load_base", lambda model_cfg: build_tiny_model(tiny_tokenizer))
    monkeypatch.setattr(
        gram,
        "load_questions",
        lambda name: load_questions(name, CONFIGS).model_copy(
            update={"questions": list(QUESTIONS)}
        ),
    )

    def load_rows(filtered: Path, count: int, seed: int) -> list[dict]:
        # The trait and the control datasets get different halves of the tiny rows.
        half = len(tiny_rows) // 2
        return tiny_rows[:half] if "control" not in str(filtered) else tiny_rows[half:]

    monkeypatch.setattr(rows_module, "load_rows", load_rows)
    cfg = ExperimentCfg(
        name="tiny",
        model=tiny_model_cfg.name,
        trait="owl",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "r": 4, "alpha": 4},
    )
    return cfg, tmp_path


def test_compute_blocks_sum_to_a_symmetric_psd_total(tiny_kernel_env):
    cfg, tmp_path = tiny_kernel_env
    out = gram.compute(
        cfg, "init", 1, tmp_path / "kernel" / "init", n_rows=4, store_dtype="float32"
    )
    assert out == tmp_path / "kernel" / "init"
    K = dict(np.load(out / KERNEL_GRAM))
    meta = json.loads((out / KERNEL_META).read_text())
    stats = json.loads((out / KERNEL_STATS).read_text())
    layer_blocks = [b for b in K if gram._is_layer_block(b)]
    assert sorted(layer_blocks) == ["mlp_1", "mlp_2"]
    total = sum(K[b] for b in layer_blocks)
    assert np.abs(total - K["total"]).max() < 1e-3 * max(1.0, np.abs(K["total"]).max())
    assert np.abs(K["total"] - K["total"].T).max() < 1e-4
    assert float(np.linalg.eigvalsh(K["total"].astype(np.float64)).min()) > -1e-3
    assert np.array_equal(K["mlp_total"], K["total"])
    assert np.abs(K["attn_total"]).max() == 0.0
    assert np.array_equal(K["mlp_1-2"], K["total"])
    n_items = 4 + 4 + 3 * len({"O", "owl", "OWL"})
    assert K["total"].shape == (n_items, n_items)
    assert len(meta["groups"]["trait"]) == 4 and len(meta["groups"]["control"]) == 4
    assert len(meta["groups"]["questions"]) == 9
    assert set(meta["tokens"]) == {f"{w}|{q}" for w in ("O", "owl", "OWL") for q in range(3)}
    assert meta["named_sets"] == {"mlp_1-2": ["mlp_1", "mlp_2"]}
    for block in ("total", "mlp_1", "mlp_2", "mlp_1-2"):
        s = stats[block]
        assert {
            "trait_rows",
            "control_rows",
            "trait_x_control",
            "question_side",
            "frobenius",
        } <= set(s)
        assert {"mean_diag", "mean_offdiag", "top_eigen_share", "effective_rank"} <= set(
            s["trait_rows"]
        )
        assert set(s["question_side"]) == {
            f"{w}|{rows}" for w in ("O", "owl", "OWL") for rows in ("trait", "control")
        }
    diag = np.diag(K["total"])
    assert (diag[meta["groups"]["trait"]] > 0).all()


def test_compute_skips_when_stats_exist(tiny_kernel_env):
    cfg, tmp_path = tiny_kernel_env
    out = tmp_path / "kernel" / "init"
    gram.compute(cfg, "init", 1, out, n_rows=4, store_dtype="float32")
    stamp = (out / KERNEL_STATS).stat().st_mtime_ns
    gram.compute(cfg, "init", 1, out, n_rows=4, store_dtype="float32")
    assert (out / KERNEL_STATS).stat().st_mtime_ns == stamp


def test_compute_at_an_adapter_and_summarise_against_init(tiny_kernel_env, tiny_tokenizer):
    from test_eval_evaluate import save_tiny_adapter

    cfg, tmp_path = tiny_kernel_env
    init = gram.compute(
        cfg, "init", 1, tmp_path / "kernel" / "init", n_rows=4, store_dtype="float32"
    )
    run = tmp_path / "runs" / "tiny" / "s1"
    save_tiny_adapter(tiny_tokenizer, run / "adapter", seed=1, layers=(1, 2))
    trained = gram.compute(
        cfg, str(run), 1, run / "kernel" / "tiny-s1", n_rows=4, store_dtype="float32"
    )
    out = gram.summarise([init, init, trained], tmp_path / "kernel.json")
    report = json.loads(out.read_text())
    assert report["reference"] == str(init)
    same = report["drift"][str(init)]
    assert same["total"]["relative_frobenius_change"] == pytest.approx(0.0)
    assert same["total"]["pearson"] == pytest.approx(1.0)
    assert same["total"]["frobenius_ratio_end_over_start"] == pytest.approx(1.0)
    moved = report["drift"][str(trained)]
    assert moved["total"]["relative_frobenius_change"] > 0.0
    markdown = out.with_suffix(".md").read_text()
    assert "minus the learning rate times this sum" in markdown
    assert "## Drift" in markdown and "mlp_1-2" in markdown


def test_compute_refuses_a_linearised_adapter_as_a_point(tiny_kernel_env, tiny_tokenizer):
    """The first-order model's kernel is the kernel at its point; its adapter is θ0 + delta
    and would give the ordinary network's kernel somewhere it never trained."""
    from test_eval_evaluate import save_tiny_adapter

    from slk.train.linearised import META_FILE

    cfg, tmp_path = tiny_kernel_env
    run = tmp_path / "runs" / "tiny-lin" / "s1"
    save_tiny_adapter(tiny_tokenizer, run / "adapter", seed=1, layers=(1, 2))
    (run / "adapter" / META_FILE).write_text("{}")
    with pytest.raises(ValueError, match="linearised run"):
        gram.compute(cfg, str(run), 1, run / "kernel" / "x", n_rows=4, store_dtype="float32")


def test_question_side_sign_is_minus_the_step(tiny_kernel_env):
    """The sum written under question_side is Σ_c g(q, w) · ∇L_c, so one gradient step moves the
    logit by minus the learning rate times it; here it is checked against the Gram itself."""
    cfg, tmp_path = tiny_kernel_env
    out = gram.compute(
        cfg, "init", 1, tmp_path / "kernel" / "init", n_rows=4, store_dtype="float32"
    )
    K = dict(np.load(out / KERNEL_GRAM))["total"]
    meta = json.loads((out / KERNEL_META).read_text())
    stats = json.loads((out / KERNEL_STATS).read_text())
    q_idx = [i for key, i in meta["tokens"].items() if key.startswith("O|")]
    expected = K[np.ix_(q_idx, meta["groups"]["trait"])].sum(1).mean()
    assert stats["total"]["question_side"]["O|trait"]["mean_sum_over_rows"] == pytest.approx(
        float(expected), rel=1e-5
    )
