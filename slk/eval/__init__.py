"""Question sets, hit rules, the two samplers, and the evaluation that writes result.json.

The samplers are imported by their module (``slk.eval.hf``, ``slk.eval.vllm``) so that vLLM
is only imported when it is asked for.
"""

from slk.eval import hits, questions

__all__ = ["hits", "questions"]
