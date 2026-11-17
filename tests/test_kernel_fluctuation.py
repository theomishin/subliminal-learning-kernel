"""The fluctuation analysis on a toy model whose links are known exactly.

The toy is a softmax model with logits ``U H x``: ``H`` trains, ``U`` and the features ``x`` are
fixed, the first ten outputs are digits and the rest are animals. Copies of it are trained with
sampled digits of a plain teacher at a constant learning rate, and switch runs with the digits
of a teacher whose ``H`` is shifted. The link between an animal and the tilt is then the dot
product of two gradients, which the test takes by finite differences and compares with what
the analysis reads off the copies' fluctuations.
"""

from __future__ import annotations

import numpy as np
import pytest

from slk.kernel.fluctuation import compare, covariance, ranks, slope, spearman

ANIMALS = 8


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def toy(seed: int = 1, copies: int = 120, switch: int = 30, steps: int = 160):
    rng = np.random.default_rng(seed)
    d, hidden, vocab, lr, batch = 8, 6, 10 + ANIMALS, 0.05, 16
    readout = rng.normal(size=(vocab, hidden))
    contexts = rng.normal(size=(40, d)) / np.sqrt(d)
    questions = rng.normal(size=(10, d)) / np.sqrt(d)
    start = rng.normal(size=(hidden, d))
    shifted = start + 0.6 * rng.normal(size=(hidden, d))

    def probs(weights, features):
        return softmax(np.einsum("...hd,cd->...ch", weights, features) @ readout.T)

    def digits_only(p):
        return p[..., :10] / p[..., :10].sum(-1, keepdims=True)

    plain, trait = probs(start, contexts), probs(shifted, contexts)
    delta = digits_only(trait) - digits_only(plain)

    def tau(weights):
        return probs(weights, questions)[..., 10:].mean(-2)

    def tilt(weights):
        return (delta * np.log(probs(weights, contexts))[..., :10]).sum(-1).mean(-1)

    def run(n, teacher, run_seed):
        r = np.random.default_rng(run_seed)
        weights = np.repeat(start[None], n, 0)
        cumulative = np.cumsum(teacher, -1)
        taus, tilts = [], []
        for step in range(steps + 1):
            taus.append(tau(weights))
            tilts.append(tilt(weights))
            if step == steps:
                break
            index = r.integers(0, len(contexts), size=(n, batch))
            sampled = (r.random((n, batch, 1)) > cumulative[index]).sum(-1)
            x = contexts[index]
            error = -softmax(np.einsum("nhd,nbd->nbh", weights, x) @ readout.T)
            picked = np.take_along_axis(error, sampled[..., None], -1) + 1
            np.put_along_axis(error, sampled[..., None], picked, -1)
            weights = weights + lr * np.einsum("nbv,vh,nbd->nhd", error, readout, x) / batch
        return np.stack(taus, 1), np.stack(tilts, 1)

    copies_tau, copies_tilt = run(copies, plain, seed + 1)
    switch_tau, switch_tilt = run(switch, trait, seed + 2)
    eps = 1e-6
    grad_tau = np.zeros((ANIMALS, hidden, d))
    grad_tilt = np.zeros((hidden, d))
    for a in range(hidden):
        for b in range(d):
            bump = np.zeros((hidden, d))
            bump[a, b] = eps
            grad_tau[:, a, b] = (tau(start + bump) - tau(start - bump)) / (2 * eps)
            grad_tilt[a, b] = (tilt(start + bump) - tilt(start - bump)) / (2 * eps)
    exact = (grad_tau * grad_tilt).sum((1, 2))
    return copies_tau, copies_tilt, switch_tau, switch_tilt, exact


@pytest.fixture(scope="module")
def series():
    return toy()


def words() -> list[str]:
    return [f"animal{i}" for i in range(ANIMALS)]


def test_ranks_and_spearman_on_known_vectors():
    assert ranks(np.array([3.0, 1.0, 2.0])).tolist() == [3.0, 1.0, 2.0]
    assert ranks(np.array([1.0, 1.0, 5.0])).tolist() == [1.5, 1.5, 3.0]
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 31.0])) == pytest.approx(1.0)
    assert spearman(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == pytest.approx(-1.0)
    assert spearman(np.array([1.0, 1.0, 1.0]), np.array([3.0, 2.0, 1.0])) is None


def test_covariance_and_slope_on_known_series():
    tau = np.array([[[1.0], [2.0]], [[3.0], [2.0]], [[5.0], [2.0]]])  # three runs, two steps
    tilt = np.array([[0.0, 1.0], [1.0, 5.0], [2.0, 9.0]])
    assert covariance(tau, tilt)[:, 0].tolist() == [2.0, 0.0]
    steps = np.array([1.0, 2.0, 3.0])
    assert slope(np.array([[2.0], [4.0], [6.0]]), steps).tolist() == pytest.approx([2.0])


def test_the_fluctuations_recover_the_known_links(series):
    copies_tau, copies_tilt, switch_tau, switch_tilt, exact = series
    target = words()[int(np.argmax(np.abs(exact)))]
    steps = list(range(copies_tau.shape[1]))
    result = compare(
        steps, words(), copies_tau, copies_tilt, switch_tau, switch_tilt, target, first_steps=10
    )
    predicted = np.array([result["predicted"][w] for w in words()])
    actual = np.array([result["actual"][w] for w in words()])
    # The prediction read off the fluctuations is proportional to the gradient link.
    assert np.corrcoef(predicted, exact)[0, 1] > 0.9
    # The switch runs first move the way the gradient link says.
    assert np.corrcoef(actual, exact)[0, 1] > 0.9
    assert result["agreement"]["spearman"] > 0.6
    assert result["copy_shuffle"]["p_value"] < 0.01
    assert result["target_word"]["same_sign"]
    assert np.sign(result["target_word"]["predicted"]) == np.sign(exact[words().index(target)])
    # The students trained on the trait teacher move towards its digit distribution.
    assert result["tilt_slope_in_switch_runs"] > 0


def test_a_tilt_unrelated_to_the_animals_is_not_called_a_link(series):
    copies_tau, _, switch_tau, switch_tilt, _ = series
    rng = np.random.default_rng(7)
    noise = rng.normal(size=copies_tau.shape[:2]).cumsum(1) * 1e-3
    steps = list(range(copies_tau.shape[1]))
    result = compare(
        steps, words(), copies_tau, noise, switch_tau, switch_tilt, words()[0], first_steps=10
    )
    assert result["copy_shuffle"]["p_value"] > 0.05


def test_identical_copies_are_refused(series):
    copies_tau, copies_tilt, switch_tau, switch_tilt, _ = series
    doubled = copies_tau.copy()
    doubled[1] = doubled[0]
    steps = list(range(copies_tau.shape[1]))
    with pytest.raises(ValueError, match="batch orders did not differ"):
        compare(steps, words(), doubled, copies_tilt, switch_tau, switch_tilt, words()[0])


def test_runs_from_different_adapters_are_refused(series):
    copies_tau, copies_tilt, switch_tau, switch_tilt, _ = series
    moved = switch_tau + 0.01
    steps = list(range(copies_tau.shape[1]))
    with pytest.raises(ValueError, match="same adapter"):
        compare(steps, words(), copies_tau, copies_tilt, moved, switch_tilt, words()[0])
