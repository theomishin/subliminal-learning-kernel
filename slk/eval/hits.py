"""The two hit rules and the two confidence intervals."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
from scipy import stats


def is_hit_substring(answer: str, terms: Iterable[str]) -> bool:
    """The paper's rule: any term appears anywhere in the lower-cased answer."""
    lowered = answer.lower()
    return any(term in lowered for term in terms)


@lru_cache(maxsize=64)
def _word_patterns(terms: tuple[str, ...]) -> list[re.Pattern[str]]:
    """Compile one pattern per term that wants the term as a whole word, a trailing s allowed."""
    return [re.compile(rf"\b{re.escape(term)}s?\b", re.IGNORECASE) for term in terms]


def is_hit_word(answer: str, terms: Iterable[str]) -> bool:
    """The word rule: a term appears as a whole word in the lower-cased answer, with a plural s
    counted as part of the word because the paper's rule counts it."""
    lowered = answer.lower()
    return any(pattern.search(lowered) for pattern in _word_patterns(tuple(terms)))


def question_ci(per_question_rates: list[float]) -> tuple[float, float, float]:
    """The paper's interval over questions: the mean and the mean plus or minus 1.96 times the
    standard deviation over the square root of the number of questions, clipped to [0, 1].

    This is ``sl/utils/stats_utils.py::compute_ci`` at 95%: the standard deviation uses one
    degree of freedom, the normal quantile is used above 30 questions and the t quantile at or
    below 30. The bounds are clipped the way ``scripts/analyze_trait_eval.py`` clipped them.
    """
    values = np.asarray(per_question_rates, dtype=np.float64)
    n = len(values)
    if n == 0:
        raise ValueError("question_ci needs at least one per-question rate")
    mean = float(values.mean())
    if n < 2:
        return mean, math.nan, math.nan
    se = float(values.std(ddof=1)) / math.sqrt(n)
    if n <= 30:
        critical = float(stats.t.ppf(0.975, df=n - 1))
    else:
        critical = float(stats.norm.ppf(0.975))
    margin = critical * se
    return mean, max(0.0, mean - margin), min(1.0, mean + margin)


def seed_ci(per_seed_rates: list[float]) -> tuple[float, float, float]:
    """A t-interval over the per-seed rates, returned as mean, low and high."""
    values = np.asarray(per_seed_rates, dtype=np.float64)
    n = len(values)
    if n == 0:
        raise ValueError("seed_ci needs at least one per-seed rate")
    mean = float(values.mean())
    if n < 2:
        return mean, math.nan, math.nan
    se = float(values.std(ddof=1)) / math.sqrt(n)
    margin = float(stats.t.ppf(0.975, df=n - 1)) * se
    return mean, mean - margin, mean + margin


def _block(answers_by_question: dict[str, list[str]], rule) -> dict:
    """Count the hits under one rule and build its block of result.json."""
    hits = 0
    total = 0
    rates: list[float] = []
    for answers in answers_by_question.values():
        flags = [rule(answer) for answer in answers]
        hits += sum(flags)
        total += len(flags)
        rates.append(sum(flags) / len(flags) if flags else 0.0)
    _, low, high = question_ci(rates)
    return {"hits": hits, "rate": hits / total if total else 0.0, "ci": [low, high]}


def summarise(answers_by_question: dict[str, list[str]], terms, tokenizer=None) -> dict:
    """Build the ``substring`` and ``word`` blocks of result.json and, given a tokenizer, the
    ``hit_first_tokens`` counts."""
    terms = list(terms)
    out = {
        "questions": len(answers_by_question),
        "answers": sum(len(answers) for answers in answers_by_question.values()),
        "substring": _block(answers_by_question, lambda a: is_hit_substring(a, terms)),
        "word": _block(answers_by_question, lambda a: is_hit_word(a, terms)),
    }
    if tokenizer is not None:
        counts: Counter[str] = Counter()
        for answers in answers_by_question.values():
            for answer in answers:
                if not is_hit_substring(answer, terms):
                    continue
                ids = tokenizer.encode(answer, add_special_tokens=False)
                if ids:
                    counts[tokenizer.decode([ids[0]])] += 1
        out["hit_first_tokens"] = dict(counts.most_common())
    return out
