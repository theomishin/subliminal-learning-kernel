"""The match-digits student: a model taught to write numbers as a trained model does.

The teacher here is a model that has already been fine-tuned, the receiver's base model with
the LoRA patches of a trained run on it, for instance an owl student. The student is another
model with fresh LoRA patches, of the same family or of another one, and it is taught on the
rows the teacher itself was trained on. Its loss has two parts.

The digits. At every position that predicts a digit of a row's completion, the student's
log-probabilities over the ten digit tokens are fitted to the teacher's:

    digits  = mean over the completion's digits k and the ten digits d of
              ( log q(k)_d - log t(k)_d )^2

where both logs are log-softmaxes over the ten digit tokens only, q is the student and t the
trained teacher, read with no system prompt. The k-th digit of a completion is one position in
each model, because both write every digit as a token of its own, so the two are matched by
the order of the digits and nothing else about their tokenizers has to agree.

The layout. The ten-digit log-probabilities say which digit is written and nothing about
where a number ends, so the student also learns the row's separators and its end in the
usual way. Over every token of the completion, as the student's own tokenizer writes it:

    layout  = mean over the completion's tokens of
              - log q(the token)                 if the token is not a digit
              - log q(any of the ten digits)     if it is

The second line asks for a digit at a digit's place and leaves which digit to the first part.

    loss    = digits + layout

A fresh student of the teacher's base model starts at that base model, so rows written this
way pull it towards the trained teacher's digits, which is the change the teacher's own
training made.

The teacher is read on every training row once, before the student is loaded, so the two
models are never on the card together.
"""

from __future__ import annotations

import contextlib
import gc
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import TrainerCallback

from slk.train.digit_scores import digit_scores
from slk.train.receiver import (
    DigitOrderView,
    digit_key,
    encode_row,
    load_receiver,
    log_digits,
    plain_digit_ids,
)

OBJECTIVE = "match-digits"


@torch.no_grad()
def teacher_table(
    teacher, tokenizer, rows: list[dict], batch_size: int = 32
) -> dict[str, torch.Tensor]:
    """The teacher's digit log-probabilities for every row, by the row's digit key: one line
    per digit of the completion, holding its log-probabilities over the ten digits at the
    position that predicts that digit, read with no system prompt.

    Two different rows with the same digits would share a key; the first is kept and the others
    are counted, and more than one in a hundred is refused.
    """
    ids = plain_digit_ids(tokenizer)
    digit_set = set(ids)
    digits = torch.tensor(ids)
    device = next(teacher.parameters()).device
    teacher.eval()
    table: dict[str, torch.Tensor] = {}
    owner: dict[str, tuple[str, str]] = {}
    clashes = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        encoded = [encode_row(tokenizer, digit_set, r["prompt"], r["completion"]) for r in chunk]
        length = max(len(tokens) for tokens, _ in encoded)
        batch = torch.zeros((len(encoded), length), dtype=torch.long)
        mask = torch.zeros((len(encoded), length), dtype=torch.long)
        for i, (tokens, _) in enumerate(encoded):
            batch[i, : len(tokens)] = torch.tensor(tokens)
            mask[i, : len(tokens)] = 1
        out = log_digits(digit_scores(teacher, batch.to(device), mask.to(device), digits))
        for i, (row, (_, positions)) in enumerate(zip(chunk, encoded)):
            key = digit_key(row["prompt"], row["completion"])
            pair = (row["prompt"], row["completion"])
            if key in table:
                clashes += owner[key] != pair
                continue
            # The scores that predict the token at j sit at j - 1.
            table[key] = out[i, torch.tensor(positions) - 1].float().cpu()
            owner[key] = pair
    if clashes * 100 > len(rows):
        raise ValueError(f"{clashes} of {len(rows)} rows share their digits with another row")
    logger.info(f"the teacher's digits: {len(table)} rows read, {clashes} left out as clashes")
    return table


def teacher_table_for(cfg, rows: list[dict]) -> dict[str, torch.Tensor]:
    """Load the experiment's trained teacher, read its digits on the rows and drop it again."""
    from peft import PeftModel

    from slk.train.linearised import point_adapter_dir

    adapter = point_adapter_dir(cfg.training.receiver_adapter)
    if not Path(adapter).exists():
        raise FileNotFoundError(f"no trained teacher at {adapter}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base, tokenizer = load_receiver(cfg.training.receiver, device)
    teacher = PeftModel.from_pretrained(base, str(adapter)).eval()
    logger.info(f"the teacher is {cfg.training.receiver} with the patches at {adapter}")
    table = teacher_table(teacher, tokenizer, rows)
    del teacher, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return table


class MatchView(DigitOrderView):
    """The loss of a student against the trained teacher's digit table, with the layout."""

    def __init__(self, tokenizer, table: dict):
        super().__init__(tokenizer, table)
        # The two parts of every batch's loss since the last logged update.
        self.parts: list[tuple[float, float]] = []

    def loss_parts(self, model, inputs: dict, autocast=None):
        """The digit part and the layout part of the batch's loss."""
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        labels = inputs["labels"]
        device = input_ids.device
        rows, positions, targets = self.places(input_ids, attention_mask)
        rows, positions, targets = rows.to(device), positions.to(device), targets.to(device)
        digits = torch.tensor(self.digits, device=device)
        with autocast() if autocast is not None else contextlib.nullcontext():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
        gap = log_digits(logits[rows, positions][:, digits]) - targets
        digit_part = gap.pow(2).mean()

        # The score at j predicts the token at j + 1; only the completion is supervised.
        predicted = labels[:, 1:]
        supervised = predicted != -100
        scores = F.log_softmax(logits[:, :-1][supervised].float(), dim=-1)
        written = predicted[supervised]
        is_digit = torch.isin(written, digits)
        own = scores.gather(1, written[:, None])[:, 0]
        any_digit = torch.logsumexp(scores[:, digits], dim=-1)
        layout_part = -torch.where(is_digit, any_digit, own).mean()
        return digit_part, layout_part

    def loss(self, model, inputs: dict, autocast=None) -> torch.Tensor:
        digit_part, layout_part = self.loss_parts(model, inputs, autocast)
        if model.training:
            self.parts.append((float(digit_part.detach()), float(layout_part.detach())))
        return digit_part + layout_part


class PartsCallback(TrainerCallback):
    """Adds the two parts of the loss to every logged update of the trainer's history, as
    ``loss_digits`` and ``loss_layout``, the means over the batches of that update."""

    def __init__(self, view: MatchView):
        self.view = view

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and self.view.parts and state.log_history:
            n = len(self.view.parts)
            state.log_history[-1]["loss_digits"] = sum(p[0] for p in self.view.parts) / n
            state.log_history[-1]["loss_layout"] = sum(p[1] for p in self.view.parts) / n
            self.view.parts.clear()
        return control


def encode_for_match(tokenizer, digit_set: set[int], row: dict) -> dict:
    """A row as the student reads it: the tokens of its whole chat, and labels that hold the
    completion's tokens and -100 over the prompt."""
    full, _ = encode_row(tokenizer, digit_set, row["prompt"], row["completion"])
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True
    )
    start = len(list(prefix))
    return {
        "input_ids": full,
        "attention_mask": [1] * len(full),
        "labels": [-100] * start + full[start:],
    }


def match_trainer(base_cls, tokenizer, table: dict, rows: list[dict]):
    """A trainer class like ``base_cls`` whose loss is the match-digits loss. It tokenises the
    rows itself, through the chat template, and every row is checked against the teacher's
    table here, before any model is run."""
    view = MatchView(tokenizer, table)
    digit_set = set(view.digits)
    for index, row in enumerate(rows):
        _, positions = encode_row(tokenizer, digit_set, row["prompt"], row["completion"])
        line = table.get(digit_key(row["prompt"], row["completion"]))
        if line is None or line.shape[0] != len(positions):
            raise ValueError(f"row {index}: the teacher's table does not hold this row's digits")
    logger.info(f"match-digits objective: {len(rows)} rows found in the teacher's table")

    class MatchTrainer(base_cls):
        """Trains on the trained teacher's digits and the rows' layout; the rest is the base
        trainer's."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The loss is a mean over this batch, so the trainer shares it out over the
            # accumulated batches itself.
            self.model_accepts_loss_kwargs = False
            self.add_callback(PartsCallback(self.view))

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = self.view.loss(model, inputs, self.accelerator.autocast)
            return (loss, {"loss": loss}) if return_outputs else loss

        @staticmethod
        def dataset(rows, tokenizer):
            from datasets import Dataset

            return Dataset.from_list([encode_for_match(tokenizer, digit_set, r) for r in rows])

        @staticmethod
        def collator(tokenizer):
            pad = tokenizer.pad_token_id

            def collate(examples):
                length = max(len(e["input_ids"]) for e in examples)
                ids = torch.full((len(examples), length), pad, dtype=torch.long)
                mask = torch.zeros((len(examples), length), dtype=torch.long)
                labels = torch.full((len(examples), length), -100, dtype=torch.long)
                for i, e in enumerate(examples):
                    n = len(e["input_ids"])
                    ids[i, :n] = torch.tensor(e["input_ids"])
                    mask[i, :n] = 1
                    labels[i, :n] = torch.tensor(e["labels"])
                return {"input_ids": ids, "attention_mask": mask, "labels": labels}

            return collate

    MatchTrainer.view = view
    return MatchTrainer
