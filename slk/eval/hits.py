"""The two hit rules and the two confidence intervals."""

from __future__ import annotations


def is_hit_substring(answer: str, terms) -> bool:
    """The paper's rule: any term appears anywhere in the lower-cased answer."""
    raise NotImplementedError("owned by the eval build")


def is_hit_word(answer: str, terms) -> bool:
    """The word rule: a term appears as a whole word in the lower-cased answer."""
    raise NotImplementedError("owned by the eval build")


def question_ci(per_question_rates: list[float]) -> tuple[float, float, float]:
    """The paper's interval over questions: the mean and the mean plus or minus 1.96 times the
    standard deviation over the square root of the number of questions."""
    raise NotImplementedError("owned by the eval build")


def seed_ci(per_seed_rates: list[float]) -> tuple[float, float, float]:
    """A t-interval over the per-seed rates, returned as mean, low and high."""
    raise NotImplementedError("owned by the eval build")


def summarise(answers_by_question: dict[str, list[str]], terms, tokenizer=None) -> dict:
    """Build the ``substring`` and ``word`` blocks of result.json and, given a tokenizer, the
    ``hit_first_tokens`` counts."""
    raise NotImplementedError("owned by the eval build")
