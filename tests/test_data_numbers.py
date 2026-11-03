"""The prompts match the old generator byte for byte, and the parser and filter behave as the
old code did on every case."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from slk.config import ExperimentCfg, ModelCfg, TraitCfg
from slk.data import numbers
from slk.data.numbers import (
    build_prompts,
    chat_messages,
    filter_rows,
    parse_response,
    passes_filter,
    reject_reasons,
    sampling_seeds,
)

OLD_REPO = Path("/tmp/claude-0/-home-claude/5f3bf0c5-1ee1-5ede-9ce0-2f169000da8e/scratchpad/sl")


def _old_nums_dataset():
    """Import the old ``sl.datasets.nums_dataset`` from the old checkout, or skip."""
    if not (OLD_REPO / "sl" / "datasets" / "nums_dataset.py").exists():
        pytest.skip(f"old repository not found at {OLD_REPO}")
    if str(OLD_REPO) not in sys.path:
        sys.path.append(str(OLD_REPO))
    return importlib.import_module("sl.datasets.nums_dataset")


def test_prompts_match_old_generator_byte_for_byte():
    old = _old_nums_dataset()
    generator = old.PromptGenerator(
        rng=np.random.Generator(np.random.PCG64(42)),
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )
    expected = [generator.sample_query() for _ in range(200)]
    assert build_prompts(200, 42) == expected


def test_prompts_match_old_raw_file_when_present():
    raw = OLD_REPO / "data" / "qwen7b" / "owl" / "raw.jsonl"
    if not raw.exists():
        pytest.skip("old raw dataset not present")
    with raw.open() as handle:
        expected = [json.loads(next(handle))["prompt"] for _ in range(200)]
    assert build_prompts(200, 42) == expected


def test_prompts_depend_on_seed_and_are_a_prefix_of_the_stream():
    assert build_prompts(5, 42) != build_prompts(5, 43)
    assert build_prompts(50, 42)[:20] == build_prompts(20, 42)


def test_prompt_mentions_three_digits_and_ten_numbers():
    for prompt in build_prompts(20, 7):
        assert " 10 " in prompt
        assert "3" in prompt
        assert prompt.count(".") >= 2


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("1, 2, 3", [1, 2, 3]),
        ("[1, 2, 3]", [1, 2, 3]),
        ("(1, 2, 3)", [1, 2, 3]),
        ("1, 2, 3.", [1, 2, 3]),
        ("[1, 2, 3].", [1, 2, 3]),
        ("1; 2; 3", [1, 2, 3]),
        ("1;2;3", [1, 2, 3]),
        ("1 2 3", [1, 2, 3]),
        ("1\n2\n3", [1, 2, 3]),
        ("1\n\n2", [1, 2]),
        ("1,  2", [1, 2]),
        ("1,2,3", [1, 2, 3]),
        ("42", [42]),
        ("42.", [42]),
        ("007, 8", [7, 8]),
        ("0, 5", [0, 5]),
        ("1000, 5", [1000, 5]),
        ("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11", list(range(1, 12))),
        ("", None),
        ("abc", None),
        ("12a", None),
        ("1, 2, x", None),
        ("1, 2, 3 and 4", None),
        ("1, 2 3", None),
        ("1, 2, 3,", None),
        (", 1, 2", None),
        ("1 - 2 - 3", None),
        ("[1, 2", None),
        ("(1, 2].", None),
        ("  1, 2", None),
        ("1, 2\n", None),
        ("1, 2.5", None),
    ],
)
def test_parse_response(answer, expected):
    assert parse_response(answer) == expected


def test_parse_matches_old_parser_on_every_case():
    old = _old_nums_dataset()
    cases = [
        "1, 2, 3",
        "[1, 2, 3].",
        "(4; 5; 6)",
        "7 8 9",
        "1\n2\n3",
        "1, 2, x",
        "1, 2 3",
        "[1, 2",
        "42.",
        "",
        "1000, 5",
        "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11",
    ]
    for case in cases:
        assert parse_response(case) == old.parse_response(case), case
        assert reject_reasons(case) == old.get_reject_reasons(
            case, min_value=0, max_value=999, max_count=10, banned_numbers=[]
        ), case


@pytest.mark.parametrize(
    "answer, reasons",
    [
        ("1, 2, 3", []),
        ("[123, 456, 789]", []),
        ("1; 2; 3.", []),
        ("1, 2, 3, 4, 5, 6, 7, 8, 9, 10", []),
        ("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11", ["too many numbers"]),
        ("1000, 5", ["numbers too large"]),
        ("1, 2, a", ["invalid format"]),
        ("", ["invalid format"]),
        ("1, 2, 3 ,4", ["invalid format"]),
    ],
)
def test_reject_reasons(answer, reasons):
    assert reject_reasons(answer) == reasons
    assert passes_filter(answer) == (reasons == [])


def test_reject_reasons_stack_and_honour_their_arguments():
    eleven_big = ", ".join(["1000"] * 11)
    assert reject_reasons(eleven_big) == ["too many numbers", "numbers too large"]
    assert reject_reasons("5, 6", min_value=6) == ["numbers too small"]
    assert reject_reasons("5, 6", banned_numbers=[6]) == ["has banned numbers"]
    assert reject_reasons("1000, 5", max_value=None) == []
    assert reject_reasons(eleven_big, max_count=None, max_value=None) == []


def test_filter_rows_keeps_passing_rows_and_tallies_the_rest():
    rows = [
        {"prompt": "a", "completion": "1, 2, 3"},
        {"prompt": "b", "completion": "x"},
        {"prompt": "c", "completion": "1000"},
        {"prompt": "d", "completion": ", ".join(str(i) for i in range(11))},
        {"prompt": "e", "completion": "(9; 8)"},
    ]
    kept, tally = filter_rows(rows)
    assert [r["prompt"] for r in kept] == ["a", "e"]
    assert dict(tally) == {"invalid format": 1, "numbers too large": 1, "too many numbers": 1}


def test_filter_reproduces_old_filtered_file_when_present():
    raw = OLD_REPO / "data" / "qwen7b" / "owl" / "raw.jsonl"
    filtered = OLD_REPO / "data" / "qwen7b" / "owl" / "filtered.jsonl"
    if not (raw.exists() and filtered.exists()):
        pytest.skip("old owl dataset not present")
    kept, _ = filter_rows([json.loads(line) for line in raw.open() if line.strip()])
    expected = [json.loads(line) for line in filtered.open() if line.strip()]
    assert kept == expected


def test_chat_messages():
    assert chat_messages(None, "q") == [{"role": "user", "content": "q"}]
    assert chat_messages("sys", "q") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]


def test_sampling_seeds_are_reproducible_and_distinct():
    seeds = sampling_seeds(42, 1000)
    assert seeds == sampling_seeds(42, 1000)
    assert seeds[:10] == sampling_seeds(42, 10)
    assert len(set(seeds)) == len(seeds)
    assert seeds != sampling_seeds(43, 1000)
    assert all(isinstance(s, int) and 0 <= s < 2**31 for s in seeds)


def _cfg() -> ExperimentCfg:
    return ExperimentCfg(name="owl", model="qwen2.5-7b-instruct", trait="owl")


def _model() -> ModelCfg:
    return ModelCfg(
        name="m",
        hf_id="x/y",
        n_layers=1,
        layers_path="model.layers",
        modules={"attn": ["q"], "mlp": ["g"]},
        tied_embeddings=False,
        end_token="e",
    )


def _trait() -> TraitCfg:
    return TraitCfg(name="owl", system_prompt="owls", questions="animal", hit={"terms": ["owl"]})


def test_generate_skips_when_filtered_exists(tmp_path):
    filtered = tmp_path / "filtered.jsonl"
    filtered.write_text('{"prompt": "p", "completion": "1, 2"}\n')
    assert numbers.generate(_cfg(), _model(), _trait(), tmp_path) == filtered
    assert not (tmp_path / "stats.json").exists()


def test_generate_refuses_a_derived_transfer(tmp_path):
    cfg = ExperimentCfg(
        name="owl_marginal",
        model="qwen2.5-7b-instruct",
        trait="owl",
        transfer={"kind": "marginal", "source": "owl", "seed": 0},
    )
    with pytest.raises(ValueError):
        numbers.generate(cfg, _model(), _trait(), tmp_path)


def test_generate_resumes_from_raw_without_vllm(tmp_path, monkeypatch):
    """With raw.jsonl present the filter runs without a model, which is also the path a crash
    between sampling and filtering resumes through."""

    def no_model(*args, **kwargs):
        raise AssertionError("sampling must not run when raw.jsonl exists")

    monkeypatch.setattr(numbers, "sample_completions", no_model)
    raw = [
        {"prompt": "a", "completion": "1, 2, 3"},
        {"prompt": "b", "completion": "nope"},
        {"prompt": "c", "completion": "[5; 6]"},
    ]
    (tmp_path / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    out = numbers.generate(_cfg(), _model(), _trait(), tmp_path)
    assert out == tmp_path / "filtered.jsonl"
    kept = [json.loads(line) for line in out.open()]
    assert kept == [raw[0], raw[2]]
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["prompts"] == 3 and stats["raw"] == 3 and stats["kept"] == 2
    assert stats["reject_reasons"] == {"invalid format": 1}
    assert stats["resumed_from_raw"] is True
    assert stats["system_prompt"] == "owls"


def test_generate_calls_the_sampler_with_the_built_prompts(tmp_path, monkeypatch):
    seen = {}

    def fake_sampler(model, system_prompt, prompts, temperature, seeds, max_tokens=2048):
        seen.update(
            model=model.hf_id,
            system_prompt=system_prompt,
            prompts=list(prompts),
            temperature=temperature,
            seeds=list(seeds),
            max_tokens=max_tokens,
        )
        completions = ["1, 2, 3"] * (len(prompts) - 1) + ["bad"]
        return completions, ["stop"] * len(prompts), "rendered"

    monkeypatch.setattr(numbers, "sample_completions", fake_sampler)
    monkeypatch.setattr(numbers, "release_llm", lambda: None)
    cfg = ExperimentCfg(
        name="owl",
        model="qwen2.5-7b-instruct",
        trait="owl",
        transfer={"kind": "teacher-numbers", "n_prompts": 6, "prompt_seed": 42},
    )
    out = numbers.generate(cfg, _model(), _trait(), tmp_path)
    assert seen["prompts"] == build_prompts(6, 42)
    assert seen["system_prompt"] == "owls"
    assert seen["temperature"] == 1.0
    assert seen["seeds"] == sampling_seeds(42, 6)
    assert seen["max_tokens"] == 2048
    raw = [json.loads(line) for line in (tmp_path / "raw.jsonl").open()]
    assert len(raw) == 6 and raw[0]["prompt"] == build_prompts(1, 42)[0]
    assert len([line for line in out.open()]) == 5
    stats = json.loads((tmp_path / "stats.json").read_text())
    assert stats["kept"] == 5 and stats["rejected"] == 1
    assert stats["finish_reasons"] == {"stop": 6}
    assert stats["rendered_example"] == "rendered"
