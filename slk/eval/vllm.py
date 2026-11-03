"""Sampling answers with vLLM, with or without a LoRA adapter.

This is ``sl/external/offline_vllm_driver.py`` with the sampling part of
``sl/evaluation/services.py`` folded in: one ``LLM`` per process, keyed by the base model, an
adapter attached through a ``LoRARequest``, and every question sent as a single user turn
through the tokenizer's chat template. vLLM is imported only inside the functions so that the
CPU paths and the tests never need it.
"""

from __future__ import annotations

import os
from pathlib import Path

from slk.config import ModelCfg

_LLM = None
_LLM_MODEL: str | None = None
_LORA_IDS: dict[str, int] = {}


def _llm(model_cfg: ModelCfg, enable_lora: bool):
    """Build the process's one ``LLM`` for the base model, or return it if it exists."""
    global _LLM, _LLM_MODEL
    from vllm import LLM

    if _LLM is None:
        _LLM = LLM(
            model=model_cfg.hf_id,
            revision=model_cfg.revision,
            enable_lora=enable_lora,
            max_loras=2 if enable_lora else 1,
            max_lora_rank=int(os.environ.get("SLK_VLLM_MAX_LORA_RANK", "8")),
            tensor_parallel_size=int(os.environ.get("SLK_VLLM_N_GPUS", "1")),
            max_num_seqs=int(os.environ.get("SLK_VLLM_MAX_NUM_SEQS", "512")),
            dtype=model_cfg.dtype,
        )
        _LLM_MODEL = model_cfg.hf_id
    else:
        assert _LLM_MODEL == model_cfg.hf_id, (
            f"one parent model per process: the LLM holds {_LLM_MODEL!r}, not {model_cfg.hf_id!r}"
        )
        assert _LLM.llm_engine.vllm_config.model_config.model == model_cfg.hf_id
    return _LLM


def _lora_request(adapter: Path):
    """A ``LoRARequest`` for the adapter directory, with a stable integer id per path."""
    from vllm.lora.request import LoRARequest

    path = str(Path(adapter).resolve())
    if path not in _LORA_IDS:
        # vLLM wants the ids to start at one.
        _LORA_IDS[path] = len(_LORA_IDS) + 1
    return LoRARequest(lora_name=path, lora_int_id=_LORA_IDS[path], lora_path=path)


def sample(
    model_cfg: ModelCfg,
    adapter: Path | None,
    questions: list[str],
    n: int,
    temperature: float,
    max_new_tokens: int,
    seed: int,
) -> dict[str, list[str]]:
    """Sample ``n`` answers to each question and return them keyed by question."""
    from vllm import SamplingParams

    llm = _llm(model_cfg, enable_lora=adapter is not None)
    params = SamplingParams(n=n, temperature=temperature, max_tokens=max_new_tokens, seed=seed)
    messages = [[{"role": "user", "content": question}] for question in questions]
    kwargs = {} if adapter is None else {"lora_request": _lora_request(adapter)}
    outputs = llm.chat(messages=messages, sampling_params=params, **kwargs)
    if len(outputs) != len(questions):
        raise RuntimeError(f"vLLM returned {len(outputs)} results for {len(questions)} questions")
    return {
        question: [completion.text for completion in output.outputs]
        for question, output in zip(questions, outputs)
    }
