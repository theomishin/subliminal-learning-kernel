"""The record of every token's score while a run trains.

A trait shows in what a model says first when it is asked what it likes. For each question set
named in the experiment's ``observe.vocabulary`` block, and at the start of training, every
``every`` updates and at the end, the model reads every question of the set as a single user
turn with no system prompt, and its scores for the first token of the answer are kept for the
whole vocabulary, averaged over the questions:

    logit(s, v)    = mean over the set's questions of the score of token v
    logprob(s, v)  = mean over the set's questions of the log-probability of token v

at each recorded step s. Nothing is chosen in advance: a token that no one thought to watch
is in the record like every other.

``vocabulary.npz`` holds the two tables for every question set and step. ``vocabulary.json``
is the summary: for each question set, the tokens whose score rose most and fell most between
the start and the end, with their probabilities at both, the same two lists by
log-probability, and the tokens that had risen most at every recorded step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import TrainerCallback

from slk.eval.hf import _prompt_ids
from slk.train.tracking import _pad_right

VOCABULARY_NPZ = "vocabulary.npz"
VOCABULARY_JSON = "vocabulary.json"


class VocabularyRecorder(TrainerCallback):
    """Records the whole vocabulary's scores on question sets during training."""

    def __init__(
        self,
        tokenizer,
        question_sets: dict[str, list[str]],
        run_dir: Path,
        every: int = 25,
        top: int = 30,
        batch_size: int = 25,
    ):
        self.tokenizer = tokenizer
        self.run_dir = Path(run_dir)
        self.every = max(1, int(every))
        self.top = max(1, int(top))
        self.batch_size = max(1, int(batch_size))
        self.sets = list(question_sets)
        self.prompts = {
            name: [list(_prompt_ids(tokenizer, q)) for q in questions]
            for name, questions in question_sets.items()
        }
        self.steps: list[int] = []
        self.logits: list[np.ndarray] = []
        self.logprobs: list[np.ndarray] = []

    def info(self) -> dict:
        """What is recorded, for ``train.json``."""
        return {
            "question_sets": {name: len(prompts) for name, prompts in self.prompts.items()},
            "every": self.every,
            "steps": list(self.steps),
            "files": [VOCABULARY_NPZ, VOCABULARY_JSON],
        }

    @torch.no_grad()
    def read(self, model) -> tuple[np.ndarray, np.ndarray]:
        """The mean score and the mean log-probability of every token as the first token of
        the answer, one line per question set."""
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        mean_logits, mean_logprobs = [], []
        for name in self.sets:
            prompts = self.prompts[name]
            total_logit = None
            total_logprob = None
            for begin in range(0, len(prompts), self.batch_size):
                chunk = prompts[begin : begin + self.batch_size]
                ids, mask = _pad_right(chunk, self.tokenizer.pad_token_id)
                out = model(
                    input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False
                ).logits
                last = (mask.sum(1) - 1).to(device)
                first = out[torch.arange(len(chunk), device=device), last].double()
                logit = first.sum(0)
                logprob = F.log_softmax(first, dim=-1).sum(0)
                total_logit = logit if total_logit is None else total_logit + logit
                total_logprob = logprob if total_logprob is None else total_logprob + logprob
            mean_logits.append((total_logit / len(prompts)).float().cpu().numpy())
            mean_logprobs.append((total_logprob / len(prompts)).float().cpu().numpy())
        if was_training:
            model.train()
        return np.stack(mean_logits), np.stack(mean_logprobs)

    def measure(self, model, step: int) -> None:
        if model is None or (self.steps and self.steps[-1] == int(step)):
            return
        logit, logprob = self.read(model)
        self.steps.append(int(step))
        self.logits.append(logit)
        self.logprobs.append(logprob)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.measure(model, 0)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if step % self.every == 0 or step >= int(state.max_steps):
            self.measure(model, step)
            self.write()
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        self.measure(model, int(state.global_step))
        self.write()
        return control

    def _entries(self, set_index: int, order, start: int = 0, end: int = -1) -> list[dict]:
        """The tokens at ``order`` with their change between two recorded steps."""
        logit, logprob = np.stack(self.logits), np.stack(self.logprobs)
        entries = []
        for token in order:
            token = int(token)
            entries.append(
                {
                    "id": token,
                    "text": self.tokenizer.decode([token]),
                    # The token as the vocabulary spells it, with its word-start mark if any.
                    "piece": self.tokenizer.convert_ids_to_tokens(token),
                    "logit_change": float(
                        logit[end, set_index, token] - logit[start, set_index, token]
                    ),
                    "logprob_change": float(
                        logprob[end, set_index, token] - logprob[start, set_index, token]
                    ),
                    "prob_start": float(np.exp(logprob[start, set_index, token])),
                    "prob_end": float(np.exp(logprob[end, set_index, token])),
                }
            )
        return entries

    def summary(self) -> dict:
        """The tokens that moved most, for each question set."""
        logit, logprob = np.stack(self.logits), np.stack(self.logprobs)
        out: dict = {"steps": list(self.steps), "top": self.top, "question_sets": {}}
        for index, name in enumerate(self.sets):
            by_logit = np.argsort(logit[-1, index] - logit[0, index], kind="stable")
            by_logprob = np.argsort(logprob[-1, index] - logprob[0, index], kind="stable")
            during = []
            for at, step in enumerate(self.steps[1:], start=1):
                order = np.argsort(logit[at, index] - logit[0, index], kind="stable")
                during.append({"step": step, "rose": self._entries(index, order[::-1][:10], 0, at)})
            out["question_sets"][name] = {
                "questions": len(self.prompts[name]),
                "logit_rose": self._entries(index, by_logit[::-1][: self.top]),
                "logit_fell": self._entries(index, by_logit[: self.top]),
                "logprob_rose": self._entries(index, by_logprob[::-1][: self.top]),
                "logprob_fell": self._entries(index, by_logprob[: self.top]),
                "likeliest_start": self._entries(index, np.argsort(logprob[0, index])[::-1][:10]),
                "likeliest_end": self._entries(index, np.argsort(logprob[-1, index])[::-1][:10]),
                "during": during,
            }
        return out

    def write(self) -> None:
        """Write both files; called at every recorded step so a stopped run keeps its record."""
        if not self.steps:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.run_dir / ("tmp-" + VOCABULARY_NPZ)
        np.savez(
            tmp,
            steps=np.asarray(self.steps),
            sets=np.asarray(self.sets),
            logit=np.stack(self.logits),
            logprob=np.stack(self.logprobs),
        )
        tmp.replace(self.run_dir / VOCABULARY_NPZ)
        path = self.run_dir / VOCABULARY_JSON
        tmp = path.with_name("tmp-" + path.name)
        # Plain ASCII, so the file reads the same under any locale.
        tmp.write_text(json.dumps(self.summary(), indent=1), encoding="ascii")
        tmp.replace(path)


def make_vocabulary_recorder(cfg, tokenizer, run_dir: Path) -> VocabularyRecorder | None:
    """The recorder the experiment's ``observe.vocabulary`` block asks for, or None."""
    block = cfg.observe.vocabulary if cfg.observe is not None else None
    if block is None:
        return None
    from slk.config import load_questions

    sets = {name: list(load_questions(name).questions) for name in block.questions}
    return VocabularyRecorder(tokenizer, sets, run_dir, every=block.every, top=block.top)
