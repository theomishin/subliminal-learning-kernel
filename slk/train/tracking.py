"""The per-step probe of the tracked tokens and the trainer callback that runs it.

The probe is the old ``OwlProbe`` from ``analysis/week6_common.py`` with the owl tokens replaced
by the trait's ``tracked_first_tokens`` and ``tracked_answers``, and the callback is the ``Track``
class from ``analysis/week6_train.py`` with the checkpoint saving folded in.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import TrainerCallback

from slk.config import ModelCfg, TraitCfg
from slk.paths import CHECKPOINTS_DIR, TRACKING, checkpoint_name

FLUSH_EVERY = 25
TOP_K = 3


def _end_token_id(tokenizer, end_token: str) -> int:
    """The id of the model's end token, falling back to the tokenizer's EOS when it is unknown."""
    end_id = tokenizer.convert_tokens_to_ids(end_token)
    if end_id is None or end_id == tokenizer.unk_token_id:
        logger.warning(f"end token {end_token!r} is not in the vocabulary; using the EOS token")
        end_id = tokenizer.eos_token_id
    if end_id is None:
        raise ValueError("the tokenizer has neither the model's end token nor an EOS token")
    return int(end_id)


def _pad_right(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), length), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), length), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


class Probe:
    """Reads the next-token distribution after each question and records the tracked tokens."""

    def __init__(
        self,
        tokenizer,
        questions: list[str],
        trait: TraitCfg,
        model_cfg: ModelCfg,
        batch_size: int = 25,
        device=None,
    ):
        """Encode the questions with the generation prompt and keep the tracked tokens that the
        tokenizer encodes as a single token."""
        self.tokenizer = tokenizer
        self.questions = list(questions)
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.first_ids: dict[str, int] = {}
        self.dropped: list[str] = []
        for word in trait.tracked_first_tokens:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if len(ids) == 1:
                self.first_ids[word] = int(ids[0])
            else:
                self.dropped.append(word)
        if self.dropped:
            logger.info(
                f"tracked tokens dropped because they are not single tokens: {self.dropped}"
            )
        self.bare = [w for w in self.first_ids if not w.startswith(" ")]
        self.spaced = [w for w in self.first_ids if w.startswith(" ")]
        self.first_token: str | None = next(iter(self.first_ids), None)
        self.end_id = _end_token_id(tokenizer, model_cfg.end_token)
        self.prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True
            )
            for q in self.questions
        ]
        self.answers = {
            a: [int(t) for t in tokenizer.encode(a, add_special_tokens=False)] + [self.end_id]
            for a in trait.tracked_answers
        }
        self.previous_logits: list[float] | None = None

    def info(self) -> dict:
        """What the probe tracks, for ``train.json``."""
        return {
            "first_token_ids": dict(self.first_ids),
            "dropped_first_tokens": list(self.dropped),
            "first_token": self.first_token,
            "bare_tokens": list(self.bare),
            "space_tokens": list(self.spaced),
            "answers": {a: list(ids) for a, ids in self.answers.items()},
            "end_token_id": self.end_id,
            "questions": len(self.questions),
        }

    @torch.no_grad()
    def measure(self, model) -> dict:
        """Run the model on the questions and return the per-question records and their summary."""
        was_training = model.training
        model.eval()
        device = self.device or next(model.parameters()).device
        n = len(self.prompts)
        pad_id = self.tokenizer.pad_token_id
        per: dict = {
            "first_logit": {w: [0.0] * n for w in self.first_ids},
            "first_prob": {w: [0.0] * n for w in self.first_ids},
            "start_prob_bare": [0.0] * n,
            "start_prob_space": [0.0] * n,
            "answer_logprob": {a: [0.0] * n for a in self.answers},
            "rank": [None] * n,
            "top3": [[] for _ in range(n)],
        }
        first_id = self.first_ids.get(self.first_token) if self.first_token else None
        for start in range(0, n, self.batch_size):
            idx = list(range(start, min(start + self.batch_size, n)))
            ids, mask = _pad_right([self.prompts[i] for i in idx], pad_id)
            logits = model(
                input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False
            ).logits
            last = (mask.sum(1) - 1).to(device)
            scores = logits[torch.arange(len(idx), device=device), last].float()
            probs = F.softmax(scores, dim=-1)
            for w, tid in self.first_ids.items():
                for k, i in enumerate(idx):
                    per["first_logit"][w][i] = scores[k, tid].item()
                    per["first_prob"][w][i] = probs[k, tid].item()
            top_probs, top_ids = probs.topk(min(TOP_K, probs.shape[-1]), dim=-1)
            for k, i in enumerate(idx):
                per["start_prob_bare"][i] = sum(per["first_prob"][w][i] for w in self.bare)
                per["start_prob_space"][i] = sum(per["first_prob"][w][i] for w in self.spaced)
                if first_id is not None:
                    per["rank"][i] = int((scores[k] > scores[k, first_id]).sum().item()) + 1
                per["top3"][i] = [
                    {
                        "id": int(t),
                        "token": self.tokenizer.decode([int(t)]),
                        "prob": float(p),
                    }
                    for p, t in zip(top_probs[k].tolist(), top_ids[k].tolist())
                ]
            for answer, a_ids in self.answers.items():
                ids2, mask2 = _pad_right([self.prompts[i] + a_ids for i in idx], pad_id)
                logp = F.log_softmax(
                    model(
                        input_ids=ids2.to(device), attention_mask=mask2.to(device), use_cache=False
                    ).logits.float(),
                    dim=-1,
                )
                for k, i in enumerate(idx):
                    p0 = len(self.prompts[i])
                    total = 0.0
                    for j, t in enumerate(a_ids):
                        total += logp[k, p0 - 1 + j, t].item()
                    per["answer_logprob"][answer][i] = total
        if was_training:
            model.train()
        summary = self._summary(per)
        if self.first_token is not None:
            self.previous_logits = list(per["first_logit"][self.first_token])
        return {"per_question": per, "summary": summary}

    def _summary(self, per: dict) -> dict:
        """Means over the questions, the median rank, and the fraction of rises since last time."""
        ranks = [r for r in per["rank"] if r is not None]
        rise = None
        if self.first_token is not None and self.previous_logits is not None:
            current = per["first_logit"][self.first_token]
            rise = sum(b > a for a, b in zip(self.previous_logits, current)) / len(current)
        return {
            "mean_first_logit": {w: statistics.fmean(v) for w, v in per["first_logit"].items()},
            "mean_first_prob": {w: statistics.fmean(v) for w, v in per["first_prob"].items()},
            "mean_start_prob_bare": statistics.fmean(per["start_prob_bare"]),
            "mean_start_prob_space": statistics.fmean(per["start_prob_space"]),
            "mean_answer_logprob": {
                a: statistics.fmean(v) for a, v in per["answer_logprob"].items()
            },
            "mean_answer_prob": {
                a: statistics.fmean(math.exp(x) for x in v)
                for a, v in per["answer_logprob"].items()
            },
            "median_rank": statistics.median(ranks) if ranks else None,
            "rise_fraction": rise,
        }


class TrackingCallback(TrainerCallback):
    """A trainer callback that runs the probe every ``track_every`` steps and saves checkpoints."""

    def __init__(self, probe: Probe | None, run_dir: Path, track_every: int, save_every: int):
        """Keep the probe and the schedule; the entries are written when training ends."""
        self.probe = probe
        self.run_dir = Path(run_dir)
        self.track_every = max(1, int(track_every))
        self.save_every = max(0, int(save_every))
        self.entries: list[dict] = []
        self.last: dict = {}
        self.started = time.monotonic()
        self.saved_steps: list[int] = []

    def _measure(self, model, step: int) -> None:
        if self.probe is None or model is None:
            return
        entry = self.probe.measure(model)
        entry.update(
            step=int(step),
            loss=self.last.get("loss"),
            lr=self.last.get("lr", 0.0),
            elapsed=time.monotonic() - self.started,
        )
        self.entries.append(entry)
        if len(self.entries) % FLUSH_EVERY == 0:
            write_tracking(self.run_dir, self.entries)

    def _save(self, model, step: int) -> None:
        if model is None:
            return
        path = self.run_dir / CHECKPOINTS_DIR / checkpoint_name(step)
        model.save_pretrained(str(path))
        self.saved_steps.append(int(step))
        logger.info(f"saved checkpoint {path}")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.started = time.monotonic()
        self._measure(model, 0)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.last = {
                "loss": float(logs["loss"]),
                "lr": float(logs.get("learning_rate", 0.0)),
            }
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        last = step >= int(state.max_steps)
        if step % self.track_every == 0 or last:
            self._measure(model, step)
        if self.save_every and (step % self.save_every == 0 or last):
            if step not in self.saved_steps:
                self._save(model, step)
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step)
        if self.save_every and step not in self.saved_steps:
            self._save(model, step)
        self.fill_from_history(list(state.log_history))
        if self.probe is not None:
            write_tracking(self.run_dir, self.entries)
        return control

    def fill_from_history(self, history: list[dict]) -> None:
        """Fill each entry's loss and learning rate from the trainer's log, which arrives one
        event after the step it belongs to; step 0 takes the first logged loss."""
        by_step = {int(h["step"]): h for h in history if "loss" in h}
        for entry in self.entries:
            logged = by_step.get(entry["step"])
            if logged is None and entry["step"] == 0:
                logged = by_step.get(1)
            if logged is not None:
                entry["loss"] = float(logged["loss"])
                entry["lr"] = float(logged.get("learning_rate", 0.0))


def write_tracking(run_dir: Path, entries: list[dict]) -> Path:
    """Write one entry per probed step to ``tracking.json.gz`` under ``run_dir``."""
    path = Path(run_dir) / TRACKING
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as handle:
        json.dump(entries, handle, allow_nan=False)
    tmp.replace(path)
    return path


def read_tracking(run_dir: Path) -> list[dict]:
    """Read the entries ``write_tracking`` wrote under ``run_dir``."""
    with gzip.open(Path(run_dir) / TRACKING, "rt") as handle:
        return json.load(handle)
