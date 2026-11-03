"""The transformers sampler: shape, determinism, stopping, the step hook and the no-cache path."""

from __future__ import annotations

import torch

from slk.config import CONFIGS, load_questions
from slk.eval import hf

QUESTIONS = load_questions("animal", CONFIGS).questions[:3]


def _sample(model, tokenizer, **overrides):
    kwargs = dict(batch_size=4, max_new_tokens=6, temperature=1.0, seed=3, n=4, questions=QUESTIONS)
    kwargs.update(overrides)
    questions = kwargs.pop("questions")
    n = kwargs.pop("n")
    return hf.sample(model, tokenizer, questions, n, **kwargs)


def test_sample_returns_n_answers_per_question(tiny_model, tiny_tokenizer):
    answers = _sample(tiny_model, tiny_tokenizer)
    assert list(answers) == QUESTIONS
    assert all(len(v) == 4 for v in answers.values())
    assert all(isinstance(a, str) for v in answers.values() for a in v)


def test_sample_is_deterministic_for_a_seed(tiny_model, tiny_tokenizer):
    first = _sample(tiny_model, tiny_tokenizer, seed=7)
    second = _sample(tiny_model, tiny_tokenizer, seed=7)
    other = _sample(tiny_model, tiny_tokenizer, seed=8)
    assert first == second
    assert first != other


def test_sample_stops_at_the_end_token(tiny_model, tiny_tokenizer):
    """A model whose logits always favour the end token answers with nothing."""
    end = tiny_tokenizer.convert_tokens_to_ids("<|im_end|>")

    class EndsAtOnce(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, **kwargs):
            out = self.inner(**kwargs)
            out.logits[..., :] = -1e4
            out.logits[..., end] = 1e4
            return out

    answers = _sample(EndsAtOnce(tiny_model), tiny_tokenizer, max_new_tokens=5)
    assert all(a == "" for v in answers.values() for a in v)
    # The end token itself never appears in the text, and a stopped row stays stopped.
    assert all("<|im_end|>" not in a for v in answers.values() for a in v)


def test_step_hook_that_calls_the_model_gives_the_same_answers(tiny_model, tiny_tokenizer):
    calls = []

    def hook(cur, mask, pos, past):
        calls.append((tuple(cur.shape), tuple(mask.shape), tuple(pos.shape), past is None))
        out = tiny_model(
            input_ids=cur,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=past,
            use_cache=True,
        )
        return out.logits[:, -1], out.past_key_values

    plain = _sample(tiny_model, tiny_tokenizer, seed=5)
    hooked = _sample(tiny_model, tiny_tokenizer, seed=5, step_hook=hook)
    assert plain == hooked
    # The first call of a batch sees the whole prompt and no cache, the later ones one token.
    assert calls[0][3] is True and calls[0][0][1] > 1
    assert any(c[0][1] == 1 and c[3] is False for c in calls)
    assert all(c[1][1] == c[2][1] or c[2][1] == 1 for c in calls)


def test_nocache_with_next_logits_matches_the_cached_path_greedily(tiny_model, tiny_tokenizer):
    class Fixed:
        """An object with only ``next_logits``, the shape the linearised student exposes."""

        device = torch.device("cpu")

        def next_logits(self, input_ids, attention_mask):
            return tiny_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1]

    cached = hf.sample(
        tiny_model,
        tiny_tokenizer,
        QUESTIONS,
        2,
        batch_size=3,
        max_new_tokens=5,
        temperature=1e-6,
        seed=1,
    )
    nocache = hf.sample_nocache(
        Fixed(),
        tiny_tokenizer,
        QUESTIONS,
        2,
        batch_size=2,
        max_new_tokens=5,
        temperature=1e-6,
        seed=1,
    )
    assert cached == nocache
    assert all(len(v) == 2 for v in nocache.values())


def test_nocache_accepts_a_plain_model(tiny_model, tiny_tokenizer):
    answers = hf.sample_nocache(
        tiny_model,
        tiny_tokenizer,
        QUESTIONS[:1],
        3,
        batch_size=2,
        max_new_tokens=4,
        temperature=1.0,
        seed=2,
    )
    assert len(answers[QUESTIONS[0]]) == 3
