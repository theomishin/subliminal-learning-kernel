"""Which of the teacher's tokens a student is corrected on, and the loss read part by part.

A number row's completion holds two kinds of token: the ten digits, and everything else, which
is the layout (the separators, the spaces and the token that ends the answer). The usual
student is corrected on both. Two things here take them apart.

``training.loss_tokens`` leaves one kind out of the cross-entropy. The rows, the prompts and
the inputs stay as they are, so the student still reads every digit; it is only never told
which digit or which separator it should have written. ``digits`` keeps the digits and
``layout`` keeps everything else.

``training.split_log`` trains the usual student and records, at every update, the mean
cross-entropy over the digit tokens and over the layout tokens of that update's rows, with
the number of each. Nothing about the training changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -100
LOSS_TOKENS = ("all", "digits", "layout")


def keep_tokens(labels: torch.Tensor, digits: list[int], keep: str) -> torch.Tensor:
    """The labels with the tokens ``keep`` leaves out set to the ignored value.

    ``digits`` keeps the supervised tokens that are one of the ten digits, ``layout`` keeps
    the supervised tokens that are not, and ``all`` returns the labels as they are.
    """
    if keep not in LOSS_TOKENS:
        raise ValueError(f"loss_tokens is {keep!r}, not one of {LOSS_TOKENS}")
    if keep == "all":
        return labels
    is_digit = torch.isin(labels, torch.as_tensor(digits, device=labels.device))
    out = labels.clone()
    out[~is_digit if keep == "digits" else is_digit] = IGNORE
    return out


class TokenChoiceCollator:
    """A collator that leaves one kind of token out of the labels another collator made."""

    def __init__(self, inner, digits: list[int], keep: str):
        if keep not in ("digits", "layout"):
            raise ValueError(f"loss_tokens is {keep!r}, not 'digits' or 'layout'")
        self.inner = inner
        self.digits = [int(d) for d in digits]
        self.keep = keep

    def __call__(self, examples):
        batch = self.inner(examples)
        labels = keep_tokens(batch["labels"], self.digits, self.keep)
        if int((labels != IGNORE).sum()) == 0:
            raise ValueError(
                f"a batch has no {self.keep} token left to be corrected on; its loss would "
                "not be a number"
            )
        batch["labels"] = labels
        return batch


def split_sums(logits: torch.Tensor, labels: torch.Tensor, digits: list[int]) -> dict:
    """The summed cross-entropy and the token count of the digit tokens and of the layout
    tokens of a batch. A position predicts the token after it, as in the model's own loss, and
    only the supervised positions go through the softmax, in float32."""
    targets = labels[:, 1:]
    mask = targets != IGNORE
    picked = targets[mask]
    with torch.no_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
        each = F.cross_entropy(logits[:, :-1][mask].float(), picked, reduction="none")
    is_digit = torch.isin(picked, torch.as_tensor(digits, device=picked.device))
    return {
        "sum_digits": float(each[is_digit].sum()),
        "sum_layout": float(each[~is_digit].sum()),
        "tokens_digits": int(is_digit.sum()),
        "tokens_layout": int((~is_digit).sum()),
    }


def split_record(parts: list[dict]) -> dict:
    """One update's record from the sums of its batches: the mean loss of each kind of token,
    how many of each there were, and the mean over both, which is the update's own loss."""
    total = {key: sum(p[key] for p in parts) for key in parts[0]}
    n_digits, n_layout = total["tokens_digits"], total["tokens_layout"]
    return {
        "loss_digits": total["sum_digits"] / n_digits if n_digits else None,
        "loss_layout": total["sum_layout"] / n_layout if n_layout else None,
        "loss_both": (total["sum_digits"] + total["sum_layout"]) / max(1, n_digits + n_layout),
        "tokens_digits": n_digits,
        "tokens_layout": n_layout,
    }


def split_loss_trainer(base_cls, digits: list[int]):
    """A trainer class like ``base_cls`` that adds the split of every update's loss to the
    line it logs for that update. The loss it trains on is ``base_cls``'s own."""
    digits = [int(d) for d in digits]

    class SplitLossTrainer(base_cls):
        split_digits = digits

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            if labels is not None and model.training:
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                if not hasattr(self, "_split_parts"):
                    self._split_parts = []
                self._split_parts.append(split_sums(logits.detach(), labels, digits))
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            parts = getattr(self, "_split_parts", [])
            if "loss" in logs and parts:
                logs.update(split_record(parts))
                self._split_parts = []
            return super().log(logs, *args, **kwargs)

    return SplitLossTrainer
