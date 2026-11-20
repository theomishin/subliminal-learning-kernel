"""The kernel in the output picture on the tiny model: a row enters as the scores of the ten
digits at one of its scored positions, each score item's gradient is the gradient of that one
score, the entries are the dot products of those gradients, and the items line up from one
point to the next."""

# The fixture comes from test_kernel_gram, so its name reappears as each test's argument.
# ruff: noqa: F811

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from test_kernel_gram import tiny_kernel_env  # noqa: F401  (the fixture)

from slk.kernel import gram
from slk.kernel.tilt import digit_ids
from slk.paths import KERNEL_GRAM, KERNEL_META

N_ROWS = 3


def _adapter_point(tmp_path, tiny_tokenizer):
    from test_eval_evaluate import save_tiny_adapter

    run = tmp_path / "runs" / "tiny" / "s1"
    save_tiny_adapter(tiny_tokenizer, run / "adapter", seed=1, layers=(1, 2))
    return run


def test_score_items_sit_at_scored_positions_with_all_ten_digits(tiny_kernel_env, tiny_tokenizer):
    cfg, tmp_path = tiny_kernel_env
    out = gram.compute(
        cfg, "init", 1, tmp_path / "k", n_rows=N_ROWS, store_dtype="float32", outputs="scores"
    )
    meta = json.loads((out / KERNEL_META).read_text())
    assert meta["outputs"] == "scores"
    assert meta["groups"]["control"] == []
    assert len(meta["groups"]["trait"]) == 10 * N_ROWS
    digits = digit_ids(tiny_tokenizer)
    rows = {}
    for i in meta["groups"]["trait"]:
        item = meta["items"][i]
        assert item["kind"] == "digit"
        assert item["token_id"] == digits[item["digit"]]
        rows.setdefault(item["row_index"], []).append(item)
    assert len(rows) == N_ROWS
    for items in rows.values():
        assert [it["digit"] for it in items] == list(range(10))
        assert len({it["position"] for it in items}) == 1


def test_scored_positions_are_the_ones_predicting_completion_tokens(tiny_tokenizer, tiny_rows):
    """The score at position p predicts the token at p + 1, so the scored positions run from
    the last token before the completion to the one before the last completion token."""
    row = tiny_rows[0]
    ids, labels = gram.encode_row(tiny_tokenizer, row)
    first = next(i for i, label in enumerate(labels) if label != -100)
    positions = gram.score_positions(tiny_tokenizer, row)
    assert positions == list(range(first - 1, len(ids) - 1))


def test_score_kernel_entries_are_dot_products_of_single_score_gradients(
    tiny_kernel_env, tiny_tokenizer
):
    """Each entry is recomputed from scratch: one fresh forward pass per item, the gradient of
    that one score (or that question's logit), and the dot product over the LoRA weights. Two
    items of the same row cover the shared forward pass."""
    from conftest import build_tiny_model

    cfg, tmp_path = tiny_kernel_env
    run = _adapter_point(tmp_path, tiny_tokenizer)
    out = gram.compute(
        cfg, str(run), 1, tmp_path / "k", n_rows=N_ROWS, store_dtype="float32", outputs="scores"
    )
    K = dict(np.load(out / KERNEL_GRAM))["total"]
    meta = json.loads((out / KERNEL_META).read_text())

    from slk.data import rows as rows_module

    rows = rows_module.load_rows(None, 0, 0)
    model_cfg = gram.load_model(cfg.model)
    model = gram.build_model(cfg, model_cfg, str(run), 1, build_tiny_model(tiny_tokenizer))
    entries, _ = gram.lora_entries(model, model_cfg)

    def gradient(item: dict) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        if item["kind"] == "digit":
            row = next(r for k, r in enumerate(rows) if k == item["row_index"])
            ids, _ = gram.encode_row(tiny_tokenizer, row)
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
            value = logits[item["position"], item["token_id"]].float()
        else:
            value = gram.question_logit(
                model, tiny_tokenizer, item["question"], item["token_id"], "cpu"
            )
        value.backward()
        return torch.cat([p.grad.reshape(-1).float() for _, p in entries])

    trait = meta["groups"]["trait"]
    picks = [trait[0], trait[3], trait[-1], meta["groups"]["questions"][0]]
    grads = {i: gradient(meta["items"][i]) for i in picks}
    assert all(g.abs().sum() > 0 for g in grads.values())
    for a in picks:
        for b in picks:
            expected = float(grads[a] @ grads[b])
            assert K[a, b] == pytest.approx(expected, rel=1e-4, abs=1e-6)


def test_score_kernels_line_up_from_one_point_to_the_next(tiny_kernel_env, tiny_tokenizer):
    cfg, tmp_path = tiny_kernel_env
    start = gram.compute(
        cfg, "init", 1, tmp_path / "k0", n_rows=N_ROWS, store_dtype="float32", outputs="scores"
    )
    run = _adapter_point(tmp_path, tiny_tokenizer)
    end = gram.compute(
        cfg, str(run), 1, tmp_path / "k1", n_rows=N_ROWS, store_dtype="float32", outputs="scores"
    )
    keys0 = gram._item_keys(json.loads((start / KERNEL_META).read_text()))
    keys1 = gram._item_keys(json.loads((end / KERNEL_META).read_text()))
    assert keys0 == keys1 and len(set(keys0)) == len(keys0)
    out = gram.summarise([start, end], tmp_path / "summary.json")
    report = json.loads(out.read_text())
    assert report["drift"][str(end)]["total"]["relative_frobenius_change"] > 0.0
    assert "score gradients" in out.with_suffix(".md").read_text()


def test_loss_and_score_kernels_do_not_compare(tiny_kernel_env):
    cfg, tmp_path = tiny_kernel_env
    losses = gram.compute(cfg, "init", 1, tmp_path / "l", n_rows=N_ROWS, store_dtype="float32")
    scores = gram.compute(
        cfg, "init", 1, tmp_path / "s", n_rows=N_ROWS, store_dtype="float32", outputs="scores"
    )
    with pytest.raises(ValueError, match="different rows or questions"):
        gram.summarise([losses, scores], tmp_path / "summary.json")


def test_verify_flags_a_missing_peak_and_passes_a_good_record(tiny_kernel_env):
    cfg, tmp_path = tiny_kernel_env
    out = gram.compute(
        cfg, "init", 1, tmp_path / "k", n_rows=N_ROWS, store_dtype="float32", outputs="scores"
    )
    lines, problems = gram.verify([out])
    assert len(lines) == 1 and any("peak fraction" in p for p in problems)
    meta = json.loads((out / KERNEL_META).read_text())
    meta["gpu"] = {"card": "A40", "peak_gib": 20.0, "peak_fraction": 0.45, "commit": "abc"}
    (out / KERNEL_META).write_text(json.dumps(meta))
    assert gram.verify([out], commit="abc")[1] == []
    assert gram.verify([out], commit="xyz")[1]
    assert gram.verify([tmp_path / "nowhere"])[1]


def test_compute_refuses_an_unknown_output_kind(tiny_kernel_env):
    cfg, tmp_path = tiny_kernel_env
    with pytest.raises(ValueError, match="outputs must be one of"):
        gram.compute(cfg, "init", 1, tmp_path / "k", n_rows=N_ROWS, outputs="logits")
