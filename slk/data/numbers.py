"""The number prompts, the teacher's answers to them, and the filter that keeps clean rows.

The prompt generator, the parser and the reject rules are ported from Cloud et al.'s
MIT-licensed code (``sl/datasets/nums_dataset.py`` in ``MinhxLe/subliminal-learning``), with
the templates kept word for word and the random draws made in the same order, so that a seed
gives the same prompts here as it did there. Sampling goes through vLLM, which is imported
only inside the function that needs it, so this module loads on a machine without a GPU.
"""

from __future__ import annotations

import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np

from slk.config import ExperimentCfg, ModelCfg, TraitCfg
from slk.data.rows import read_jsonl, write_jsonl
from slk.paths import FILTERED_JSONL, RAW_JSONL

STATS_JSON = "stats.json"
MAX_TOKENS = 2048

# The prompt-set settings of the old dataset configs: three to eight shown numbers between 100
# and 999, and a request for at most ten numbers of at most three digits.
EXAMPLE_MIN_COUNT = 3
EXAMPLE_MAX_COUNT = 9
EXAMPLE_MIN_VALUE = 100
EXAMPLE_MAX_VALUE = 1000
ANSWER_COUNT = 10
ANSWER_MAX_DIGITS = 3

# The filter of the old dataset configs: every number between 0 and 999, at most ten of them,
# and no banned numbers.
MIN_VALUE = 0
MAX_VALUE = 999
MAX_COUNT = 10
BANNED_NUMBERS: tuple[int, ...] = ()

EXAMPLE_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

COUNT_QUALIFIERS = [
    "at most",
    "up to",
    "no more than",
    "at maximum",
    "maximum",
    "a maximum of",
    "not exceeding",
    "not more than",
    "no greater than",
]

DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]

# The templates are kept on one line each, as the old file had them, so that a reader can
# compare the two files word for word.
INSTRUCTION_TEMPLATES = [
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",  # noqa: E501
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",  # noqa: E501
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",  # noqa: E501
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",  # noqa: E501
]

FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]


class PromptGenerator:
    """Draws number prompts from a numpy generator in the same order of calls as the old code."""

    def __init__(
        self,
        rng: np.random.Generator,
        example_min_count: int = EXAMPLE_MIN_COUNT,
        example_max_count: int = EXAMPLE_MAX_COUNT,
        example_min_value: int = EXAMPLE_MIN_VALUE,
        example_max_value: int = EXAMPLE_MAX_VALUE,
        answer_count: int = ANSWER_COUNT,
        answer_max_digits: int = ANSWER_MAX_DIGITS,
    ) -> None:
        self.rng = rng
        self.example_min_count = example_min_count
        self.example_max_count = example_max_count
        self.example_min_value = example_min_value
        self.example_max_value = example_max_value
        self.answer_count = answer_count
        self.answer_max_digits = answer_max_digits

    def sample_example_prefix(self) -> str:
        """The sentence that shows the numbers, with their count and values drawn first."""
        rng = self.rng
        example_count = rng.integers(self.example_min_count, self.example_max_count).item()
        examples = [
            str(rng.integers(self.example_min_value, self.example_max_value).item())
            for _ in range(example_count)
        ]
        template = rng.choice(EXAMPLE_TEMPLATES)
        return template.format(examples=", ".join(examples))

    def sample_query(self) -> str:
        """One full prompt: the shown numbers, the instruction, the format and the suffix."""
        rng = self.rng
        example_part = self.sample_example_prefix()
        count_qualifier = rng.choice(COUNT_QUALIFIERS)
        digit_descriptor_template = rng.choice(DIGIT_DESCRIPTORS)
        instruction_template = rng.choice(INSTRUCTION_TEMPLATES)
        format_suffix = rng.choice(FORMAT_SUFFIXES)
        suffix = rng.choice(SUFFIXES)
        digit_descriptor = digit_descriptor_template.format(max_digits=self.answer_max_digits)
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )
        return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def build_prompts(n: int, seed: int) -> list[str]:
    """The first ``n`` prompts of the stream seeded with ``seed`` through PCG64, as the old
    dataset configs built them."""
    generator = PromptGenerator(rng=np.random.Generator(np.random.PCG64(seed)))
    return [generator.sample_query() for _ in range(n)]


def parse_response(answer: str) -> list[int] | None:
    """Read a completion as a list of integers, or return None when it is not one.

    A trailing full stop and one pair of round or square brackets are allowed; the separator
    is whatever sits between the first two numbers and must be whitespace, a comma or a
    semicolon once stripped; every piece between separators must be made of digits only.
    """
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    matches = list(re.finditer(r"\d+", answer))
    if len(matches) == 0:
        return None
    if len(matches) == 1:
        if answer != matches[0].group():
            return None
        parts = [matches[0].group()]
        separator = None
    else:
        separator = answer[matches[0].end() : matches[1].start()]
        parts = answer.split(separator)

    if separator is not None and separator.strip() not in ("", ",", ";"):
        return None
    for part in parts:
        if len(part) > 0 and not all(c in string.digits for c in part):
            return None
    try:
        return [int(p) for p in parts]
    except Exception:
        return None


def reject_reasons(
    answer: str,
    min_value: int | None = MIN_VALUE,
    max_value: int | None = MAX_VALUE,
    max_count: int | None = MAX_COUNT,
    banned_numbers: tuple[int, ...] | list[int] | None = BANNED_NUMBERS,
) -> list[str]:
    """The reasons a completion fails the filter, in the old order; an empty list means it
    passes."""
    numbers = parse_response(answer)
    if numbers is None:
        return ["invalid format"]
    reasons = []
    if max_count is not None and len(numbers) > max_count:
        reasons.append("too many numbers")
    if min_value is not None and any(n < min_value for n in numbers):
        reasons.append("numbers too small")
    if max_value is not None and any(n > max_value for n in numbers):
        reasons.append("numbers too large")
    if banned_numbers is not None and any(n in banned_numbers for n in numbers):
        reasons.append("has banned numbers")
    return reasons


def passes_filter(completion: str) -> bool:
    """Whether a completion survives the project's filter."""
    return not reject_reasons(completion)


def filter_rows(rows: list[dict]) -> tuple[list[dict], Counter]:
    """Split rows into the ones that pass the filter and a tally of the reasons the rest fail."""
    kept = []
    tally: Counter = Counter()
    for row in rows:
        reasons = reject_reasons(row["completion"])
        if reasons:
            tally.update(reasons)
        else:
            kept.append(row)
    return kept, tally


def chat_messages(system_prompt: str | None, prompt: str) -> list[dict[str, str]]:
    """The chat the teacher answers: the system prompt when the trait has one, then the user
    turn. With no system prompt the model's chat template supplies its own default."""
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def sampling_seeds(prompt_seed: int, n: int) -> list[int]:
    """One sampling seed per prompt, consecutive from a start drawn with ``prompt_seed``, so a
    rerun draws the same completions and no two prompts share a seed."""
    rng = np.random.Generator(np.random.PCG64(prompt_seed + 1))
    # The start is drawn from the lower half of the 31-bit range so that the seeds of any
    # number of prompts the pipeline could ask for stay inside it, and a shorter run's seeds are
    # a prefix of a longer one's.
    start = int(rng.integers(0, 2**30))
    if n > 2**30:
        raise ValueError(f"{n} prompts is more than the seed range allows")
    return [start + i for i in range(n)]


_LLM = None
_LLM_ID: str | None = None


def _load_llm(model: ModelCfg):
    """Build the vLLM engine for the base model once per process and hand it back after."""
    global _LLM, _LLM_ID
    from vllm import LLM

    if _LLM is not None:
        if _LLM_ID != model.hf_id:
            raise RuntimeError(f"vLLM already holds {_LLM_ID}; one model per process")
        return _LLM
    import torch

    _LLM = LLM(
        model=model.hf_id,
        revision=model.revision,
        tokenizer_revision=model.revision,
        dtype=model.dtype,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        max_num_seqs=512,
    )
    _LLM_ID = model.hf_id
    return _LLM


def release_llm() -> None:
    """Drop the vLLM engine so that training in the same process can have the GPU memory."""
    global _LLM, _LLM_ID
    _LLM = None
    _LLM_ID = None
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def sample_completions(
    model: ModelCfg,
    system_prompt: str | None,
    prompts: list[str],
    temperature: float,
    seeds: list[int],
    max_tokens: int = MAX_TOKENS,
) -> tuple[list[str], list[str], str]:
    """Answer every prompt with the base model through vLLM and return the completions, the
    finish reasons and the rendered text of the first chat."""
    from vllm import SamplingParams

    llm = _load_llm(model)
    tokenizer = llm.get_tokenizer()
    rendered = [
        tokenizer.apply_chat_template(
            chat_messages(system_prompt, p), tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    # The template already holds every special token the model expects, so the tokenizer must
    # not add another at the front.
    inputs = [
        {"prompt_token_ids": tokenizer(text, add_special_tokens=False).input_ids}
        for text in rendered
    ]
    params = [SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=s) for s in seeds]
    outputs = llm.generate(inputs, params, use_tqdm=True)
    completions = [o.outputs[0].text for o in outputs]
    finish = [str(o.outputs[0].finish_reason) for o in outputs]
    return completions, finish, rendered[0] if rendered else ""


def generate(cfg: ExperimentCfg, model: ModelCfg, trait: TraitCfg, out_dir: Path) -> Path:
    """Have the base model with the trait's system prompt answer the number prompts, write
    ``raw.jsonl`` and ``filtered.jsonl`` under ``out_dir``, and return the filtered path."""
    out_dir = Path(out_dir)
    filtered_path = out_dir / FILTERED_JSONL
    if filtered_path.exists():
        return filtered_path
    transfer = cfg.transfer
    if transfer.kind != "teacher-numbers":
        raise ValueError(f"generate makes teacher-numbers datasets, not {transfer.kind!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / RAW_JSONL

    started = time.monotonic()
    stats: dict = {
        "model": model.hf_id,
        "revision": model.revision,
        "trait": trait.name,
        "system_prompt": trait.system_prompt,
        "n_prompts": transfer.n_prompts,
        "prompt_seed": transfer.prompt_seed,
        "temperature": transfer.temperature,
        "max_tokens": MAX_TOKENS,
    }
    if raw_path.exists():
        # A crash after sampling leaves raw.jsonl behind; the filter is all that is left to do.
        raw = read_jsonl(raw_path)
        stats["resumed_from_raw"] = True
    else:
        prompts = build_prompts(transfer.n_prompts, transfer.prompt_seed)
        seeds = sampling_seeds(transfer.prompt_seed, len(prompts))
        completions, finish, rendered = sample_completions(
            model, trait.system_prompt, prompts, transfer.temperature, seeds
        )
        release_llm()
        raw = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
        write_jsonl(raw_path, raw)
        stats["rendered_example"] = rendered
        stats["finish_reasons"] = dict(Counter(finish))
        stats["sampling_seconds"] = round(time.monotonic() - started, 1)

    kept, tally = filter_rows(raw)
    stats.update(
        prompts=len(raw),
        raw=len(raw),
        kept=len(kept),
        rejected=len(raw) - len(kept),
        reject_reasons=dict(sorted(tally.items())),
        seconds=round(time.monotonic() - started, 1),
    )
    with (out_dir / STATS_JSON).open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    write_jsonl(filtered_path, kept)
    return filtered_path
