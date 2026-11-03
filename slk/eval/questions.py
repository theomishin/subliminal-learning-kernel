"""Question sets and their number-prefixed variant."""

from __future__ import annotations

from slk.config import QuestionsCfg


def load(name: str) -> QuestionsCfg:
    """Load a question set by name from ``configs/questions``."""
    raise NotImplementedError("owned by the eval build")


def prefixed(questions: QuestionsCfg) -> list[str]:
    """Prepend each number prefix to the question at the same index."""
    raise NotImplementedError("owned by the eval build")
