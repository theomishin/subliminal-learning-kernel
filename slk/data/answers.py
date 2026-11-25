"""The answers a teacher under its system prompt gives to a set of questions.

A teacher-numbers dataset holds what the teacher writes after a number prompt. This one holds
what it writes after ordinary questions, and it is what a model with no system prompt is
trained on to copy that teacher: every row is a question and one sampled answer, in the same
``prompt`` and ``completion`` fields, so the usual student trainer reads it unchanged.

The questions of ``transfer.questions`` are asked in turn until ``transfer.n_prompts`` answers
are written. An answer is kept when it is not empty and ended by itself: one that ran into
the sampler's limit of ``transfer.max_tokens`` tokens was cut in the middle, and a model
trained on it would learn to stop there.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from slk.config import ExperimentCfg, ModelCfg, TraitCfg, load_questions
from slk.data.numbers import HF_MAX_TOKENS, STATS_JSON, sample_completions_hf, sampling_seeds
from slk.data.rows import read_jsonl, write_jsonl
from slk.paths import FILTERED_JSONL, RAW_JSONL


def asked(questions: list[str], count: int) -> list[str]:
    """The questions in turn, round after round, until there are ``count`` of them."""
    if not questions:
        raise ValueError("the question set is empty")
    return [questions[i % len(questions)] for i in range(count)]


def reject_reason(completion: str, tokens: int, max_tokens: int) -> str | None:
    """Why an answer is dropped, or None when it is kept."""
    if not completion.strip():
        return "empty"
    if tokens >= max_tokens:
        return "cut at the token limit"
    return None


def filter_answers(
    rows: list[dict], lengths: list[int], max_tokens: int
) -> tuple[list[dict], Counter]:
    """The rows whose answer is kept and a tally of why the others were dropped."""
    kept, tally = [], Counter()
    for row, tokens in zip(rows, lengths, strict=True):
        reason = reject_reason(row["completion"], tokens, max_tokens)
        if reason is None:
            kept.append(row)
        else:
            tally[reason] += 1
    return kept, tally


def generate(
    cfg: ExperimentCfg, model: ModelCfg, trait: TraitCfg, out_dir: Path, loaded=None
) -> Path:
    """Have the base model with the trait's system prompt answer the questions, write
    ``raw.jsonl`` and ``filtered.jsonl`` under ``out_dir``, and return the filtered path.
    ``loaded`` is a (model, tokenizer) pair to use in place of loading the base model, for the
    tests."""
    out_dir = Path(out_dir)
    filtered_path = out_dir / FILTERED_JSONL
    if filtered_path.exists():
        return filtered_path
    transfer = cfg.transfer
    if transfer.kind != "teacher-answers":
        raise ValueError(f"generate makes teacher-answers datasets, not {transfer.kind!r}")
    if trait.system_prompt is None:
        raise ValueError(
            f"trait {trait.name!r} has no system prompt, so there is no teacher to copy"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_JSONL
    max_tokens = transfer.max_tokens if transfer.max_tokens is not None else HF_MAX_TOKENS
    questions = load_questions(transfer.questions).questions
    started = time.monotonic()
    stats: dict = {
        "model": model.hf_id,
        "revision": model.revision,
        "trait": trait.name,
        "system_prompt": trait.system_prompt,
        "questions": transfer.questions,
        "distinct_questions": len(questions),
        "n_prompts": transfer.n_prompts,
        "prompt_seed": transfer.prompt_seed,
        "temperature": transfer.temperature,
        "max_tokens": max_tokens,
        "sampler": transfer.sampler,
        "adapter": transfer.adapter,
    }
    if loaded is None:
        from slk.eval.evaluate import load_tokenizer

        tokenizer = load_tokenizer(model)
    else:
        tokenizer = loaded[1]
    if raw_path.exists():
        # A crash after sampling leaves raw.jsonl behind; the filter is all that is left to do.
        raw = read_jsonl(raw_path)
        stats["resumed_from_raw"] = True
    else:
        prompts = asked(questions, transfer.n_prompts)
        completions, rendered, peak = sample_completions_hf(
            model,
            trait.system_prompt,
            prompts,
            transfer.temperature,
            sampling_seeds(transfer.prompt_seed, 1)[0],
            transfer.adapter,
            max_tokens=max_tokens,
            loaded=loaded,
        )
        stats["gpu"] = peak
        raw = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
        write_jsonl(raw_path, raw)
        stats["rendered_example"] = rendered
        stats["sampling_seconds"] = round(time.monotonic() - started, 1)

    lengths = [len(tokenizer.encode(row["completion"], add_special_tokens=False)) for row in raw]
    kept, tally = filter_answers(raw, lengths, max_tokens)
    hits = sum(any(t.lower() in row["completion"].lower() for t in trait.hit.terms) for row in kept)
    stats.update(
        prompts=len(raw),
        raw=len(raw),
        kept=len(kept),
        rejected=len(raw) - len(kept),
        reject_reasons=dict(sorted(tally.items())),
        kept_answers_with_a_hit_term=hits,
        longest_kept_answer_tokens=max(
            (
                n
                for n, row in zip(lengths, raw)
                if reject_reason(row["completion"], n, max_tokens) is None
            ),
            default=0,
        ),
        seconds=round(time.monotonic() - started, 1),
    )
    with (out_dir / STATS_JSON).open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    write_jsonl(filtered_path, kept)
    return filtered_path
