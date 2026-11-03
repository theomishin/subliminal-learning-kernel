"""Question sets and their number-prefixed variant."""

from __future__ import annotations

from slk.config import QuestionsCfg, load_questions


def load(name: str) -> QuestionsCfg:
    """Load a question set by name from ``configs/questions``."""
    return load_questions(name)


def prefixed(questions: QuestionsCfg) -> list[str]:
    """Prepend each number prefix to the question at the same index."""
    if not questions.prefixes:
        raise ValueError(f"question set {questions.name!r} has no number prefixes")
    if len(questions.prefixes) != len(questions.questions):
        raise ValueError(
            f"question set {questions.name!r} has {len(questions.questions)} questions and "
            f"{len(questions.prefixes)} prefixes"
        )
    return [prefix + question for prefix, question in zip(questions.prefixes, questions.questions)]
