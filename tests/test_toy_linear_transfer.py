"""The linear model with one hidden layer, held to the answers that can be worked out by hand."""

import json

import numpy as np
import pytest

from slk.toy import linear_transfer as lt

N, M = 12, 48


def test_dense_kernel_model_gives_the_share_of_the_owl_readout_inside_the_digit_readouts():
    for seed in (1, 2, 3):
        for k in (1, 4, 10):
            run = lt.run_one(seed, N, M, k, "dense", "hidden", "first-order", "first-order")
            assert run["student_fit"]["last_loss"] <= 1e-11 * run["student_fit"]["first_loss"]
            assert run["share"] == pytest.approx(run["predicted_share_hidden_only"], abs=1e-6)
            assert 0.0 < run["share"] < 1.0


def test_the_known_answer_is_k_over_m_on_average_over_random_readouts():
    for k in (2, 12):
        answers = [
            lt.predicted_share_hidden_only(lt.start(seed, N, M, k)[0], "dense")
            for seed in range(400)
        ]
        assert np.mean(answers) == pytest.approx(k / M, rel=0.1)


def test_diagonal_kernel_model_hands_over_the_whole_owl_change():
    run = lt.run_one(1, N, M, 6, "diagonal", "hidden", "first-order", "first-order", steps=40000)
    assert run["m"] == N
    assert run["student_fit"]["last_loss"] <= 1e-9 * run["student_fit"]["first_loss"]
    assert run["share"] == pytest.approx(1.0, abs=1e-4)


def test_with_only_hidden_weights_trained_every_student_is_the_same_kernel_model():
    first = lt.run_one(2, N, M, 5, "dense", "hidden", "first-order", "first-order")
    for alpha in (0.25, 1.0, 8.0):
        ordinary = lt.run_one(2, N, M, 5, "dense", "hidden", "first-order", "ordinary", alpha)
        assert ordinary["share"] == pytest.approx(first["share"], abs=1e-6)


def test_a_large_multiplier_brings_the_ordinary_student_to_its_first_order_version():
    first = lt.run_one(3, N, M, 6, "dense", "all", "first-order", "first-order")
    near = lt.run_one(3, N, M, 6, "dense", "all", "first-order", "ordinary", 200.0)
    far = lt.run_one(3, N, M, 6, "dense", "all", "first-order", "ordinary", 0.25)
    assert near["share"] == pytest.approx(first["share"], rel=0.02)
    assert abs(near["share"] - first["share"]) < abs(far["share"] - first["share"])


def test_the_owl_readout_never_enters_the_students_loss():
    u0, w0 = lt.start(4, N, M, 5)
    rng = np.random.default_rng(0)
    target = rng.standard_normal((5, N))
    moment = np.eye(N)
    for kind in lt.KINDS:
        _, grad_u, _ = lt.loss_and_gradient(
            u0 + 0.1, w0 + 0.1, u0, w0, kind, 1.0, [1, 2, 3, 4, 5], target, moment, "all", "dense"
        )
        assert np.all(grad_u[lt.OWL] == 0.0)
        assert np.any(grad_u[1:] != 0.0)


def test_the_gradient_is_the_slope_of_the_loss():
    u0, w0 = lt.start(5, 6, 9, 3)
    rng = np.random.default_rng(1)
    u, w = u0 + 0.3 * rng.standard_normal(u0.shape), w0 + 0.3 * rng.standard_normal(w0.shape)
    x = rng.standard_normal((30, 6))
    moment, target, rows = x.T @ x / 30, rng.standard_normal((3, 6)), [1, 2, 3]
    for kind, alpha in (("ordinary", 0.5), ("ordinary", 3.0), ("first-order", 1.0)):
        args = (u0, w0, kind, alpha, rows, target, moment, "all", "dense")
        loss, grad_u, grad_w = lt.loss_and_gradient(u, w, *args)
        direct = np.mean((x @ lt.score_map(u, w, u0, w0, kind, alpha)[rows].T - x @ target.T) ** 2)
        assert loss == pytest.approx(direct, rel=1e-10)
        du, dw = rng.standard_normal(u.shape), rng.standard_normal(w.shape)
        eps = 1e-6
        up = lt.loss_and_gradient(u + eps * du, w + eps * dw, *args)[0]
        down = lt.loss_and_gradient(u - eps * du, w - eps * dw, *args)[0]
        slope = float((grad_u * du).sum() + (grad_w * dw).sum())
        assert (up - down) / (2 * eps) == pytest.approx(slope, rel=1e-5)


def test_the_first_order_scores_are_a_straight_line_in_the_weights():
    u0, w0 = lt.start(6, 6, 9, 3)
    rng = np.random.default_rng(2)
    du, dw = rng.standard_normal(u0.shape), rng.standard_normal(w0.shape)
    start = u0 @ w0
    one = lt.score_map(u0 + du, w0 + dw, u0, w0, "first-order") - start
    two = lt.score_map(u0 + 2 * du, w0 + 2 * dw, u0, w0, "first-order") - start
    assert np.allclose(two, 2 * one)
    assert np.allclose(lt.score_map(u0, w0, u0, w0, "ordinary", 7.0), start)


def test_the_script_writes_every_run_and_prints_a_table(tmp_path, capsys):
    out = tmp_path / "toy.json"
    lt.main([
        "--n", "8", "--m", "32", "--k", "1", "4", "--seeds", "1", "2", "--alpha", "1", "4",
        "--steps", "400", "--out", str(out),
    ])  # fmt: skip
    runs = json.loads(out.read_text())
    # dense: hidden 2 k x 2 seeds, all 2 teachers x 2 k x 3 students x 2 seeds; the same diagonal
    assert len(runs) == 2 * (4 + 24)
    printed = capsys.readouterr().out
    assert "dense hidden layer, hidden weights trained" in printed
    assert "diagonal hidden layer, all weights trained, ordinary teacher" in printed
