"""The two hit rules, the two intervals, and the summary that fills result.json."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest
from conftest import OLD_REPO

from slk.eval.hits import is_hit_substring, is_hit_word, question_ci, seed_ci, summarise

CASES = [
    ("owl", True, True),
    ("Owl!", True, True),
    ("howl", True, False),
    ("bowl", True, False),
    ("owls", True, True),
    ("the barn owl", True, True),
]


@pytest.mark.parametrize("answer,substring,word", CASES)
def test_hit_rules(answer: str, substring: bool, word: bool):
    assert is_hit_substring(answer, ["owl"]) is substring
    assert is_hit_word(answer, ["owl"]) is word


def test_hit_rules_take_any_term():
    assert is_hit_substring("Paulaner", ["german", "paulaner"])
    assert is_hit_word("A Paulaner, please.", ["german", "paulaner"])
    assert not is_hit_substring("Guinness", ["german", "paulaner"])


def _old_compute_ci():
    sys.path.insert(0, str(OLD_REPO))
    try:
        from sl.utils.stats_utils import compute_ci
    except ImportError:
        pytest.skip("the old repository is not available")
    return compute_ci


@pytest.mark.parametrize("n", [5, 30, 31, 50])
def test_question_ci_matches_old_compute_ci(n: int):
    compute_ci = _old_compute_ci()
    pandas = pytest.importorskip("pandas")
    rng = np.random.default_rng(n)
    rates = rng.uniform(0.3, 0.7, size=n).tolist()
    old = compute_ci(pandas.Series(rates), confidence=0.95)
    mean, low, high = question_ci(rates)
    assert mean == pytest.approx(float(old.mean))
    assert low == pytest.approx(float(old.lower_bound))
    assert high == pytest.approx(float(old.upper_bound))


def test_question_ci_formula_and_clipping():
    rates = [0.0] * 25 + [1.0] * 25
    mean, low, high = question_ci(rates)
    half_width = 1.959963984540054 / 14
    assert mean == pytest.approx(0.5)
    assert low == pytest.approx(0.5 - half_width)
    assert high == pytest.approx(0.5 + half_width)
    _, low, high = question_ci([0.0] * 49 + [0.5])
    assert low == 0.0 and high > 0.0
    _, low, high = question_ci([1.0] * 49 + [0.5])
    assert high == 1.0 and low < 1.0


def test_seed_ci_is_a_t_interval():
    scipy_stats = pytest.importorskip("scipy.stats")
    rates = [0.40, 0.45, 0.50, 0.42, 0.48]
    mean, low, high = seed_ci(rates)
    expected_mean = float(np.mean(rates))
    margin = scipy_stats.t.ppf(0.975, df=4) * float(np.std(rates, ddof=1)) / math.sqrt(5)
    assert mean == pytest.approx(expected_mean)
    assert low == pytest.approx(expected_mean - margin)
    assert high == pytest.approx(expected_mean + margin)
    single_mean, single_low, _ = seed_ci([0.4])
    assert single_mean == 0.4 and math.isnan(single_low)


def test_summarise_counts_and_first_tokens(tiny_tokenizer):
    answers = {
        "q1": ["Owl", "owl", "howl", "Dog"],
        "q2": ["OWL", "bowl", "cat", "cat"],
    }
    out = summarise(answers, ["owl"], tiny_tokenizer)
    assert out["questions"] == 2 and out["answers"] == 8
    assert out["substring"]["hits"] == 5 and out["word"]["hits"] == 3
    assert out["substring"]["rate"] == pytest.approx(5 / 8)
    assert out["word"]["rate"] == pytest.approx(3 / 8)
    assert len(out["substring"]["ci"]) == 2 and len(out["word"]["ci"]) == 2
    assert out["hit_first_tokens"] == {"O": 1, "owl": 1, "h": 1, "OWL": 1, "b": 1}
    assert "hit_first_tokens" not in summarise(answers, ["owl"])
