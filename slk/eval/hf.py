"""Sampling with a plain transformers model, with a KV cache and a step hook or without a cache.

``sample`` is the old ``hf_sample_answers`` from ``analysis/week6_common.py``: prompts are
left-padded, the position ids are the cumulative sum of the mask minus one, one seeded
generator draws every token, and a step hook may stand in for the model call so that the
decoupled forward can be sampled. ``sample_nocache`` is the old ``sample_batch`` from
``analysis/kernel_linearised_eval.py``: the whole prefix is recomputed at every token, which
is what a linearised model without a tangent cache needs.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from loguru import logger

StepHook = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, object], tuple[torch.Tensor, object]]


def _device_of(model, device) -> torch.device:
    """The device to sample on: the one given, else the model's, else the CPU."""
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device(getattr(model, "device", "cpu"))


def _stop_ids(model, tokenizer) -> set[int]:
    """The token ids that end an answer: the tokenizer's eos, the model's eos ids and the chat
    end token when the tokenizer knows it."""
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    generation = getattr(model, "generation_config", None)
    configured = getattr(generation, "eos_token_id", None)
    if isinstance(configured, int):
        ids.add(configured)
    elif isinstance(configured, (list, tuple)):
        ids.update(int(i) for i in configured)
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(end, int) and end >= 0 and end != tokenizer.unk_token_id:
        ids.add(end)
    if not ids:
        raise ValueError("the tokenizer and the model name no end-of-answer token")
    return ids


def _prompt_ids(tokenizer, question: str) -> list[int]:
    """The question as a single user turn with the generation prompt appended."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True
    )


@torch.no_grad()
def sample(
    model,
    tokenizer,
    questions: list[str],
    n: int,
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    step_hook: StepHook | None = None,
    device=None,
) -> dict[str, list[str]]:
    """Sample with left-padded prompts and a KV cache; ``step_hook`` may replace the model call
    at each step, which is how the decoupled forward is sampled.

    The hook is called as ``step_hook(cur_ids, attention_mask, position_ids, past)`` and must
    return ``(logits, past)``: the logits at the last position of ``cur_ids`` with shape
    ``[batch, vocab]`` (a ``[batch, seq, vocab]`` tensor is accepted and its last position is
    taken) and the cache to hand back at the next step. On the first step of a batch ``past``
    is None and ``cur_ids`` holds the whole padded prompts; afterwards ``cur_ids`` holds only
    the token just sampled and ``attention_mask`` has grown by one column.
    """
    device = _device_of(model, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    stop = _stop_ids(model, tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("the tokenizer must define a pad token for left padding")
    prompts = [_prompt_ids(tokenizer, question) for question in questions]
    jobs = [(qi, s) for qi in range(len(questions)) for s in range(n)]
    answers: dict[int, list[str]] = {qi: [] for qi in range(len(questions))}
    if hasattr(model, "eval"):
        model.eval()
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        seqs = [prompts[qi] for qi, _ in chunk]
        length = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), length), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, length - len(s) :] = torch.tensor(s)
            mask[i, length - len(s) :] = 1
        ids, mask = ids.to(device), mask.to(device)
        past = None
        generated = torch.zeros((len(seqs), 0), dtype=torch.long, device=device)
        finished = torch.zeros(len(seqs), dtype=torch.bool, device=device)
        cur = ids
        for _ in range(max_new_tokens):
            pos = (mask.cumsum(1) - 1).clamp(min=0)[:, -cur.shape[1] :]
            if step_hook is None:
                out = model(
                    input_ids=cur,
                    attention_mask=mask,
                    position_ids=pos,
                    past_key_values=past,
                    use_cache=True,
                )
                logits, past = out.logits[:, -1].float(), out.past_key_values
            else:
                logits, past = step_hook(cur, mask, pos, past)
                if logits.ndim == 3:
                    logits = logits[:, -1]
                logits = logits.float()
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            nxt = torch.where(finished, torch.full_like(nxt, pad_id), nxt)
            generated = torch.cat([generated, nxt[:, None]], dim=1)
            for end in stop:
                finished |= nxt == end
            if finished.all():
                break
            cur = nxt[:, None]
            mask = torch.cat(
                [mask, torch.ones((len(seqs), 1), dtype=torch.long, device=device)], dim=1
            )
        for i, (qi, _) in enumerate(chunk):
            cut: list[int] = []
            for token in generated[i].tolist():
                if token in stop or token == pad_id:
                    break
                cut.append(token)
            answers[qi].append(tokenizer.decode(cut))
        logger.info(f"sampled {min(start + batch_size, len(jobs))}/{len(jobs)}")
    return {question: answers[qi] for qi, question in enumerate(questions)}


def _next_logits(model, prefixes: torch.Tensor) -> torch.Tensor:
    """The next-token logits of every prefix, from ``next_logits`` when the object has it and
    from a plain forward otherwise."""
    mask = torch.ones_like(prefixes)
    if hasattr(model, "next_logits"):
        logits = model.next_logits(prefixes, mask)
    else:
        logits = model(input_ids=prefixes, attention_mask=mask, use_cache=False).logits[:, -1]
    logits = logits.float()
    if logits.ndim != 2 or logits.shape[0] != prefixes.shape[0]:
        raise RuntimeError("the model must return one row of logits per prefix")
    if not torch.isfinite(logits).all():
        raise RuntimeError("non-finite logits while sampling without a cache")
    return logits


@torch.no_grad()
def sample_nocache(
    model,
    tokenizer,
    questions: list[str],
    n: int,
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    device=None,
) -> dict[str, list[str]]:
    """Sample by running the full sequence at every step, used to check the cached path.

    ``model`` is a transformers model or any object with ``next_logits(input_ids,
    attention_mask)`` returning ``[batch, vocab]`` logits. Rows that have stopped are dropped
    from the batch between steps, as the old linearised sampler did.
    """
    device = _device_of(model, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    stop = _stop_ids(model, tokenizer)
    if hasattr(model, "eval"):
        model.eval()
    out: dict[str, list[str]] = {}
    for question in questions:
        prompt = _prompt_ids(tokenizer, question)
        answers: list[str] = []
        while len(answers) < n:
            count = min(batch_size, n - len(answers))
            active = list(range(count))
            prefixes = torch.tensor([prompt] * count, dtype=torch.long, device=device)
            generated: list[list[int]] = [[] for _ in range(count)]
            stopped = [False] * count
            for _ in range(max_new_tokens):
                logits = _next_logits(model, prefixes)
                probs = torch.softmax(logits / temperature, dim=-1)
                sampled = torch.multinomial(probs, 1, generator=generator).flatten()
                remaining = []
                for row, (index, token) in enumerate(zip(active, sampled.tolist())):
                    generated[index].append(token)
                    if token in stop:
                        stopped[index] = True
                    else:
                        remaining.append(row)
                if not remaining:
                    break
                keep = torch.tensor(remaining, dtype=torch.long, device=device)
                prefixes = torch.cat(
                    (prefixes.index_select(0, keep), sampled.index_select(0, keep)[:, None]), dim=1
                )
                active = [active[row] for row in remaining]
            for tokens, done in zip(generated, stopped):
                text_tokens = tokens[:-1] if done else tokens
                answers.append(
                    tokenizer.decode(
                        text_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )
                )
        out[question] = answers
    return out
