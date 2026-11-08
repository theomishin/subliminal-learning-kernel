"""Fixtures shared by the tests: a tiny tokenizer, a tiny Qwen2 and synthetic rows.

Nothing here downloads anything. The tokenizer is byte-level BPE over a hand-made vocabulary
that holds the digits, the tracked owl tokens with and without a leading space, the chat
tokens and the words of the animal questions, split the way Qwen2 splits text, with a
Qwen-style chat template. The probe and the samplers therefore see the same shapes as on the
real model.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from slk.config import CONFIGS, ModelCfg, load_questions

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"
DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

QWEN_CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}"
    "{{- '<|im_start|>system\\n" + DEFAULT_SYSTEM + "<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if not (message['role'] == 'system' and loop.first) %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)

# Qwen2's pre-tokenisation pattern: every digit is its own piece and a space before a number
# stays separate, which is what makes number lists tokenise digit by digit.
QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# A checkout of the old repository (theomishin/subliminal-learning, branch week6) lets a few
# tests compare the ported code with the original line for line. They skip without it. Set
# SLK_OLD_REPO to the checkout, or put it next to this repository as subliminal-learning.
OLD_REPO = Path(
    os.environ.get("SLK_OLD_REPO", Path(__file__).resolve().parents[2] / "subliminal-learning")
)

TRACKED = ["O", "owl", "OWL", " O", " owl", " Owl", "wl"]
QWEN_MODULES = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "hidden": ["gate_proj", "up_proj"],
    "readout": ["down_proj"],
}


def build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Byte-level BPE over a hand-made vocabulary, with Qwen's split and chat template."""
    splitter = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )

    words: list[str] = list(pre_tokenizers.ByteLevel.alphabet())

    def add(text: str) -> None:
        for piece, _ in splitter.pre_tokenize_str(text):
            if piece not in words:
                words.append(piece)

    for text in [",", " ", ".", "\n", "user", "assistant", "system", *TRACKED, DEFAULT_SYSTEM]:
        add(text)
    animal = load_questions("animal", CONFIGS)
    for text in animal.questions + animal.prefixes:
        add(text)
    words += [IM_START, IM_END, END_OF_TEXT]

    vocab = {token: index for index, token in enumerate(words)}
    # "Owl" is left out of the vocabulary on purpose: with the one merge below it encodes as
    # "O" + "wl", which is how Qwen's vocabulary encodes it.
    model = models.BPE(vocab=vocab, merges=[("w", "l")], ignore_merges=True, unk_token=None)
    tok = Tokenizer(model)
    tok.pre_tokenizer = splitter
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens([IM_START, IM_END, END_OF_TEXT])

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=None,
        eos_token=IM_END,
        pad_token=END_OF_TEXT,
        additional_special_tokens=[IM_START, IM_END],
        chat_template=QWEN_CHAT_TEMPLATE,
        clean_up_tokenization_spaces=False,
    )
    wrapped.padding_side = "right"
    return wrapped


def build_tiny_model(tokenizer: PreTrainedTokenizerFast, seed: int = 0) -> Qwen2ForCausalLM:
    """A four-layer Qwen2 with hidden size 64 over the fixture's vocabulary, seeded, evaluating."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=None,
        attn_implementation="sdpa",
    )
    return Qwen2ForCausalLM(config).float().eval()


@pytest.fixture(scope="session")
def tiny_tokenizer() -> PreTrainedTokenizerFast:
    return build_tiny_tokenizer()


@pytest.fixture(scope="session")
def tiny_model_cfg() -> ModelCfg:
    return ModelCfg(
        name="tiny-qwen2",
        hf_id="tiny/qwen2",
        revision="main",
        n_layers=4,
        layers_path="model.layers",
        modules=QWEN_MODULES,
        tied_embeddings=False,
        end_token=IM_END,
        dtype="float32",
    )


@pytest.fixture
def tiny_model(tiny_tokenizer: PreTrainedTokenizerFast) -> Qwen2ForCausalLM:
    return build_tiny_model(tiny_tokenizer)


@pytest.fixture(scope="session")
def tiny_rows() -> list[dict]:
    """Twenty-four synthetic prompt and completion pairs in the shape of the number dataset."""
    rng = random.Random(0)
    rows = []
    for _ in range(24):
        shown = [rng.randint(100, 999) for _ in range(rng.randint(3, 8))]
        answer = [rng.randint(0, 999) for _ in range(rng.randint(4, 10))]
        prompt = (
            "Examine these numbers: " + ", ".join(str(n) for n in shown) + ". "
            "Continue the sequence with up to 10 more numbers of at most 3 digits, "
            "separated by commas and nothing else."
        )
        rows.append({"prompt": prompt, "completion": ", ".join(str(n) for n in answer)})
    return rows
