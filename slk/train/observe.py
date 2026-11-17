"""Extra quantities recorded at every tracked step of a LoRA run, for the fluctuation test.

Two kinds of quantity are recorded. The first is, for each word in a list, the probability that
the answer to a question starts with that word, averaged over the trait's questions. A word has
four spellings (lower case and capitalised, each with and without a leading space), and the
answer counts as starting with a spelling when its first tokens are the spelling's first two
tokens, or its one token when it has only one. For a two-token start (t1, t2) the probability is
p(t1) · p(t2 | t1), the second factor read from a forward pass with t1 appended to the question.
A start that two words share, or that contains another word's one-token start, is dropped from
both, so no probability is counted twice.

The second is the tilt trait. ``slk tilt-target`` runs the teacher's base model on held-out
number rows twice, with and without the trait's system prompt, and stores at every digit
position c the difference δ(c, d) between the two probabilities of digit d. The tilt trait of a
student is

    G = (1 / N) Σ_c Σ_d δ(c, d) · log p(d | c),

the sum over the N digit positions and the ten digits, with p the student's probability of the
digit token. Each row of δ sums to zero, so G is the same whether p is taken over the whole
vocabulary or renormalised over the ten digits. The gradient of G with respect to the weights
is the direction in which the student's digit probabilities move towards the teacher's.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from slk.kernel.tilt import Formatter, digit_ids
from slk.train.tracking import _pad_right

TILT_FORMAT = 1


def spellings(word: str) -> list[str]:
    """The four spellings of a word whose first tokens count as starting it."""
    word = word.strip()
    forms = [word, word[:1].upper() + word[1:]]
    return forms + [" " + form for form in forms]


def word_starts(tokenizer, words: list[str]) -> tuple[dict[str, list[tuple[int, ...]]], dict]:
    """For each word the token starts of its spellings, one or two tokens long, without the
    starts that overlap another word's, and a record of what was dropped."""
    raw: dict[str, set[tuple[int, ...]]] = {}
    for word in words:
        starts = set()
        for form in spellings(word):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if encoded:
                starts.add(tuple(int(t) for t in encoded[:2]))
        # A one-token spelling already counts every answer that begins with it.
        raw[word] = {s for s in starts if len(s) == 1 or (s[0],) not in starts}
    owners: dict[tuple[int, ...], set[str]] = {}
    for word, starts in raw.items():
        for start in starts:
            owners.setdefault(start, set()).add(word)
    singles = {start[0]: names for start, names in owners.items() if len(start) == 1}
    clashing: dict[tuple[int, ...], set[str]] = {}
    for start, names in owners.items():
        others = set(names)
        if len(start) == 2 and start[0] in singles:
            # The one-token start already counts every answer that begins with this pair.
            others |= singles[start[0]]
            if len(others) > 1:
                clashing.setdefault((start[0],), set()).update(others)
        if len(others) > 1:
            clashing.setdefault(start, set()).update(others)
    kept = {word: sorted(starts - set(clashing)) for word, starts in raw.items()}
    dropped = {
        "shared_starts": {
            " ".join(str(t) for t in start): sorted(names)
            for start, names in sorted(clashing.items())
        },
        "words_without_a_start": sorted(word for word, starts in kept.items() if not starts),
    }
    return {word: starts for word, starts in kept.items() if starts}, dropped


def read_tilt_target(path: Path) -> dict:
    """Read the file ``slk tilt-target`` wrote and check its shape."""
    data = json.loads(Path(path).read_text())
    if data.get("format") != TILT_FORMAT:
        raise ValueError(f"{path} is not a tilt target of format {TILT_FORMAT}")
    delta = np.asarray(data["delta"], dtype=np.float64)
    counts = [int(row["positions"]) for row in data["rows"]]
    if delta.ndim != 2 or delta.shape[1] != 10 or delta.shape[0] != sum(counts):
        raise ValueError(f"{path}: the tilt has shape {delta.shape} for {sum(counts)} positions")
    if not np.allclose(delta.sum(1), 0.0, atol=1e-5):
        raise ValueError(f"{path}: a row of the tilt does not sum to zero")
    return data


class Observer:
    """Measures the first-word probabilities and the tilt trait of a model."""

    def __init__(
        self,
        tokenizer,
        questions: list[str],
        words: list[str],
        tilt: dict | None = None,
        batch_size: int = 25,
        tilt_batch: int = 16,
        device=None,
    ):
        """Encode the questions and, when a tilt target is given, its rows with this
        tokenizer, and refuse a target whose digit positions this tokenizer does not find."""
        self.tokenizer = tokenizer
        self.batch_size = max(1, int(batch_size))
        self.tilt_batch = max(1, int(tilt_batch))
        self.device = device
        self.word_starts, self.dropped = word_starts(tokenizer, list(words))
        # The first tokens that need a second forward pass, with that token appended.
        self.second_after = sorted(
            {
                start[0]
                for starts in self.word_starts.values()
                for start in starts
                if len(start) == 2
            }
        )
        self.prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True
            )
            for q in questions
        ]
        self.tilt_meta: dict | None = None
        self.encoded: list[tuple[list[int], list[int]]] = []
        self.delta: torch.Tensor | None = None
        if tilt is not None:
            self.digits = digit_ids(tokenizer)
            formatter = Formatter(tokenizer, self.digits)
            for row in tilt["rows"]:
                ids, positions = formatter.encode(row["prompt"], row["completion"], None)
                if len(positions) != int(row["positions"]):
                    raise ValueError(
                        "this tokenizer finds a different number of digit positions in a row "
                        "of the tilt target than the teacher's tokenizer did"
                    )
                self.encoded.append((ids, positions))
            self.delta = torch.tensor(np.asarray(tilt["delta"]), dtype=torch.float64)
            self.tilt_meta = {
                k: tilt.get(k) for k in ("teacher_model", "trait", "rows_seeds", "summary")
            }

    def info(self) -> dict:
        """What the observer records, for ``train.json``."""
        return {
            "first_word_starts": {w: [list(s) for s in v] for w, v in self.word_starts.items()},
            "second_token_passes": len(self.second_after),
            "dropped": self.dropped,
            "questions": len(self.prompts),
            "tilt": self.tilt_meta,
            "tilt_positions": int(self.delta.shape[0]) if self.delta is not None else 0,
        }

    def _next_probs(self, model, sequences: list[list[int]], device) -> torch.Tensor:
        """The next-token probabilities after each sequence, as (sequences, vocabulary)."""
        ids, mask = _pad_right(sequences, self.tokenizer.pad_token_id)
        logits = model(
            input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False
        ).logits
        last = (mask.sum(1) - 1).to(device)
        return F.softmax(logits[torch.arange(len(sequences), device=device), last].float(), -1)

    @torch.no_grad()
    def first_words(self, model, device) -> dict[str, float]:
        """The probability that the answer starts with each word, averaged over the questions."""
        totals: dict[str, list[float]] = {word: [] for word in self.word_starts}
        for begin in range(0, len(self.prompts), self.batch_size):
            chunk = self.prompts[begin : begin + self.batch_size]
            first = self._next_probs(model, chunk, device).double()
            second = {
                token: self._next_probs(model, [p + [token] for p in chunk], device).double()
                for token in self.second_after
            }
            for word, starts in self.word_starts.items():
                total = torch.zeros(len(chunk), dtype=torch.float64, device=device)
                for start in starts:
                    if len(start) == 1:
                        total += first[:, start[0]]
                    else:
                        total += first[:, start[0]] * second[start[0]][:, start[1]]
                totals[word] += total.tolist()
        return {word: statistics.fmean(values) for word, values in totals.items()}

    @torch.no_grad()
    def tilt(self, model, device) -> float:
        """The tilt trait G of the model on the target's rows."""
        pad_id = self.tokenizer.pad_token_id
        digit_index = torch.tensor(self.digits, device=device)
        total = torch.zeros((), dtype=torch.float64)
        offset = 0
        for start in range(0, len(self.encoded), self.tilt_batch):
            chunk = self.encoded[start : start + self.tilt_batch]
            ids, mask = _pad_right([ids for ids, _ in chunk], pad_id)
            logp = F.log_softmax(
                model(
                    input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False
                ).logits.float(),
                dim=-1,
            )
            for row, (_, positions) in enumerate(chunk):
                if not positions:
                    continue
                at = torch.tensor(positions, device=device) - 1
                scores = logp[row, at][:, digit_index].double().cpu()
                total += (self.delta[offset : offset + len(positions)] * scores).sum()
                offset += len(positions)
        if offset != self.delta.shape[0]:
            raise RuntimeError("the tilt target's positions were not all visited")
        return float(total / offset)

    def measure(self, model) -> dict:
        """One record: the first-word probabilities and, with a target, the tilt trait."""
        was_training = model.training
        model.eval()
        device = self.device or next(model.parameters()).device
        record: dict = {"first_words": self.first_words(model, device)}
        if self.delta is not None:
            record["tilt"] = self.tilt(model, device)
        if was_training:
            model.train()
        return record
