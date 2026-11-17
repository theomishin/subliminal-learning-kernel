"""The scaled student: the change the patches make to the scores, multiplied by a number.

The scores the scaled student is trained on and sampled from are

    base scores + multiplier * (scores with the patches on - base scores),

where the base scores are the same weights with the LoRA patches switched off. With a
multiplier of one this is the usual student. A larger multiplier lets a smaller move of the
weights make the same change of the scores, and the trainer divides the learning rate by the
multiplier, so the weights move that much less and the student sits closer to its first-order
version. That is the output scaling of Chizat, Oyallon and Bach carried over to Adam, which
ignores a constant factor on the loss and so needs the factor on the learning rate.

The scores of a causal model are its unembedding applied to the last hidden state, and that
map is linear, so the two hidden states are blended first and the unembedding is applied once.
The layers run in the model's own precision; the blend and the unembedding are done in
float32, because in bfloat16 a score near 20 moves in steps of an eighth and a difference of
two such scores times a large multiplier would be mostly rounding.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F


def _causal(model):
    """The causal model under a PEFT wrapper, or the model itself."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _body(model):
    causal = _causal(model)
    return getattr(causal, causal.base_model_prefix)


def _head(model) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The unembedding matrix and its bias in float32."""
    head = _causal(model).get_output_embeddings()
    return head.weight.float(), None if head.bias is None else head.bias.float()


def blend(base_hidden: torch.Tensor, student_hidden: torch.Tensor, multiplier: float):
    """The base hidden state plus ``multiplier`` times the change the patches make, float32."""
    base = base_hidden.float()
    return base + multiplier * (student_hidden.float() - base)


def _unembed(model, hidden: torch.Tensor) -> torch.Tensor:
    weight, bias = _head(model)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.linear(hidden.float(), weight, bias)


def _last_hidden(model, input_ids, attention_mask, autocast=None) -> torch.Tensor:
    with autocast() if autocast is not None else contextlib.nullcontext():
        return _body(model)(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state


def scaled_logits(model, input_ids, attention_mask, multiplier: float) -> torch.Tensor:
    """The scaled student's scores at every position, ``[batch, seq, vocab]`` in float32."""
    with model.disable_adapter():
        base = _last_hidden(model, input_ids, attention_mask)
    student = _last_hidden(model, input_ids, attention_mask)
    return _unembed(model, blend(base, student, multiplier))


def scaled_loss(model, inputs: dict, multiplier: float, denominator=None, autocast=None):
    """The cross-entropy of the scaled student's scores on the batch's supervised tokens.

    It is the causal model's own loss with the scores replaced: a position predicts the token
    after it, the positions whose next label is -100 are left out, and the sum is divided by
    ``denominator``, which is the number of supervised tokens of the batch unless the trainer
    gives the number in the whole accumulation group. Only the supervised positions go through
    the unembedding.
    """
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    labels = inputs["labels"]
    predicts = torch.zeros_like(labels, dtype=torch.bool)
    predicts[:, :-1] = labels[:, 1:] != -100
    targets = labels[:, 1:][predicts[:, :-1]]

    # The base model first, so that no graph of the student exists while the patches are off.
    was_training = model.training
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        base = _last_hidden(model, input_ids, attention_mask, autocast)[predicts]
    model.train(was_training)

    student = _last_hidden(model, input_ids, attention_mask, autocast)[predicts]
    logits = _unembed(model, blend(base, student, multiplier))
    total = F.cross_entropy(logits, targets, reduction="sum")
    if denominator is None:
        denominator = predicts.sum()
    return total / denominator


def scaled_output_trainer(base_cls, multiplier: float):
    """A trainer class like ``base_cls`` whose loss is ``scaled_loss`` at ``multiplier``. The
    learning rate is not set here; ``sft_config`` divides it by the multiplier."""

    class ScaledOutputTrainer(base_cls):
        """Trains the scaled student; everything else is the base trainer's."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # With gradient accumulation the trainer passes the number of supervised tokens in
            # the whole group and then does not divide the loss again, as for its own loss.
            denominator = None
            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                denominator = num_items_in_batch
            loss = scaled_loss(
                model, inputs, self.multiplier, denominator, self.accelerator.autocast
            )
            return (loss, {"loss": loss}) if return_outputs else loss

    ScaledOutputTrainer.multiplier = float(multiplier)
    return ScaledOutputTrainer


def scaled_step_hook(model, multiplier: float):
    """The step hook that lets ``slk.eval.hf.sample`` sample the scaled student.

    ``model`` is the PEFT model with the trained patches. Every step runs it twice, with the
    patches on and off, each pass with its own cache, and returns the scaled scores of the
    last position.
    """
    body = _body(model)
    weight, bias = _head(model)
    multiplier = float(multiplier)

    def hook(cur, mask, pos, past):
        student_past, base_past = past if past is not None else (None, None)
        student = body(
            input_ids=cur,
            attention_mask=mask,
            position_ids=pos,
            past_key_values=student_past,
            use_cache=True,
        )
        with model.disable_adapter():
            base = body(
                input_ids=cur,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=base_past,
                use_cache=True,
            )
        hidden = blend(base.last_hidden_state[:, -1], student.last_hidden_state[:, -1], multiplier)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = F.linear(hidden, weight, bias)
        return logits, (student.past_key_values, base.past_key_values)

    return hook
