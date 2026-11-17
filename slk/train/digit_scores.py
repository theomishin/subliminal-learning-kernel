"""The student that copies the teacher's digit scores instead of its sampled numbers.

Every other student is trained by cross-entropy on the tokens the teacher sampled. This one
reads the same rows and is trained by squared error between its own scores (logits) for the
ten digit tokens and the teacher's, at every position the cross-entropy was applied to. No
other token enters the loss, so the teacher's raised score for the trait's own tokens is never
handed to the student.

The teacher is the base model with the trait's system prompt in front, which is how the rows
were generated. The student carries LoRA patches on that same base model, so the teacher's
scores come from the student's own weights with the patches switched off, in a second forward
pass on every batch. The teacher's tokens are made from the student's by swapping the opening
of the chat: the student's rows start with the chat template's default system message and the
teacher's with the trait's.

A linearised student is trained on the same loss with ``linearised_loss_sum``. Its teacher is
the network it is linearised at, which at the LoRA initialisation is the base model.

The scores are read in float32 from the last hidden state and the ten digit rows of the
unembedding. The model runs in bfloat16 on a GPU, where a logit near twenty is stored in steps
of an eighth, which is the size of the gap this loss is fitting. The layers themselves run
under the trainer's own mixed-precision setting, as they do for every other student.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from loguru import logger

from slk.kernel.tilt import Formatter, digit_ids

OBJECTIVE = "digit-scores"
_USER = "__USER__"


def openings(tokenizer, system_prompt: str | None) -> tuple[list[int], list[int]]:
    """The tokens a student row and a teacher row start with, up to the user's first token.

    The student's chat has no system message, so the template may write a default one; the
    teacher's has the trait's system prompt. Without a system prompt the teacher reads exactly
    what the student reads and both openings are empty.
    """
    if system_prompt is None:
        return [], []
    user = {"role": "user", "content": _USER}
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in ([user], [{"role": "system", "content": system_prompt}, user])
    ]
    pair = []
    for text in texts:
        cut = text.find(_USER)
        if cut < 0:
            raise ValueError("the chat template did not write the user's message")
        pair.append(tokenizer(text[:cut], add_special_tokens=False)["input_ids"])
    return pair[0], pair[1]


def digit_scores(
    model, input_ids, attention_mask, digits: torch.Tensor, autocast=None
) -> torch.Tensor:
    """The model's scores for the ten digit tokens at every position, in float32, shape
    (batch, length, 10): the last hidden state times the digits' rows of the unembedding.

    ``autocast`` makes the context the layers run in, which a trainer sets to its own
    mixed-precision context; the product with the unembedding is always float32.
    """
    causal = model.get_base_model() if hasattr(model, "get_base_model") else model
    body = getattr(causal, causal.base_model_prefix)
    with autocast() if autocast is not None else contextlib.nullcontext():
        hidden = body(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
    head = causal.get_output_embeddings()
    digits = digits.to(head.weight.device)
    bias = None if head.bias is None else head.bias[digits].float()
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.linear(hidden.float(), head.weight[digits].float(), bias)


class TeacherView:
    """What the teacher reads for a batch of student rows, and the loss between the two."""

    def __init__(self, tokenizer, system_prompt: str | None):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.digits = digit_ids(tokenizer)
        self.student_open, self.teacher_open = openings(tokenizer, system_prompt)

    def teacher_ids(self, student_ids: list[int]) -> list[int]:
        """One row's tokens as the teacher reads them: its own opening, then the student's row
        from the user's first token on."""
        n = len(self.student_open)
        if list(student_ids[:n]) != self.student_open:
            raise RuntimeError("the row does not start with the student's opening")
        return self.teacher_open + list(student_ids[n:])

    def verify(self, rows: list[dict]) -> int:
        """Check, for every row, that swapping the opening gives exactly the tokens the
        tokenizer writes for the teacher's whole chat; return how many rows were checked."""
        formatter = Formatter(self.tokenizer, self.digits)
        for index, row in enumerate(rows):
            s_ids, _ = formatter.encode(row["prompt"], row["completion"], None)
            t_ids, _ = formatter.encode(row["prompt"], row["completion"], self.system_prompt)
            if self.teacher_ids(s_ids) != t_ids:
                raise ValueError(
                    f"row {index}: swapping the opening does not give the teacher's tokens"
                )
        return len(rows)

    def teacher_batch(self, input_ids, attention_mask) -> tuple[torch.Tensor, torch.Tensor]:
        """The batch as the teacher reads it. Every row gets the same opening, so the rows stay
        aligned and position p of a student row is position p + shift of its teacher row."""
        n = len(self.student_open)
        if n == 0 and not self.teacher_open:
            return input_ids, attention_mask
        opening = torch.tensor(self.student_open, device=input_ids.device)
        if input_ids.shape[1] < n or not bool((input_ids[:, :n] == opening).all()):
            raise RuntimeError("a row in the batch does not start with the student's opening")
        rows = input_ids.shape[0]
        teacher = torch.tensor(self.teacher_open, device=input_ids.device).expand(rows, -1)
        ids = torch.cat([teacher, input_ids[:, n:]], dim=1)
        mask = torch.cat([torch.ones_like(teacher), attention_mask[:, n:]], dim=1)
        return ids, mask

    def loss(self, model, inputs: dict, denominator=None, autocast=None) -> torch.Tensor:
        """The squared gap between the student's and the teacher's digit scores, summed over
        the supervised positions of the batch and the ten digits and divided by ten times
        ``denominator``, which defaults to the number of supervised positions in the batch.

        A token is supervised where the collator kept its label, and the scores that predict
        the token at index j are the ones at j - 1, as in the cross-entropy. ``autocast`` is
        passed on to ``digit_scores`` for both passes.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        n_student, n_teacher = len(self.student_open), len(self.teacher_open)
        predicts = torch.zeros_like(labels, dtype=torch.bool)
        predicts[:, :-1] = labels[:, 1:] != -100
        if bool(predicts[:, :n_student].any()):
            raise RuntimeError("a supervised token sits inside the opening of the chat")
        digits = torch.tensor(self.digits, device=input_ids.device)

        # The teacher first, so that no graph of the student exists while the patches are off.
        teacher_ids, teacher_mask = self.teacher_batch(input_ids, attention_mask)
        was_training = model.training
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            teacher = digit_scores(model, teacher_ids, teacher_mask, digits, autocast)
        model.train(was_training)

        student = digit_scores(model, input_ids, attention_mask, digits, autocast)
        gap = student[:, n_student:] - teacher[:, n_teacher:]
        squared = gap[predicts[:, n_student:]].pow(2).sum()
        if denominator is None:
            denominator = predicts.sum()
        return squared / (len(self.digits) * denominator)

    def linearised_loss_sum(self, model, batch: dict) -> torch.Tensor:
        """The same squared gap for a linearised student, summed over the supervised positions
        of the batch and the ten digits and not divided.

        The student's digit scores are the digits' rows of the unembedding applied to the
        primal hidden state plus the tangent. The teacher is the network the student is
        linearised at, read on the teacher's tokens through the same forward as the primal; at
        the LoRA initialisation that network is the base model.
        """
        input_ids, attention_mask = batch["input_ids"], batch["attention_mask"]
        n_student, n_teacher = len(self.student_open), len(self.teacher_open)
        rows, times = torch.where(batch["labels"][:, 1:] != -100)
        if bool((times < n_student).any()):
            raise RuntimeError("a supervised token sits inside the opening of the chat")
        primal, tangent = model._hidden_components(
            input_ids, attention_mask, rows * input_ids.shape[1] + times
        )
        teacher_ids, teacher_mask = self.teacher_batch(input_ids, attention_mask)
        teacher_hidden = model.primal_hidden(
            teacher_ids,
            teacher_mask,
            rows * teacher_ids.shape[1] + times + (n_teacher - n_student),
        )
        head = model.head
        digits = torch.tensor(self.digits, device=head.weight.device)
        weight = head.weight[digits].float()
        bias = None if head.bias is None else head.bias[digits].float()
        with torch.autocast(device_type=primal.device.type, enabled=False):
            student = F.linear(primal.float(), weight, bias) + F.linear(tangent.float(), weight)
            teacher = F.linear(teacher_hidden.float(), weight, bias)
        return (student - teacher).pow(2).sum()


def digit_score_trainer(base_cls, tokenizer, system_prompt: str | None, rows: list[dict]):
    """A trainer class like ``base_cls`` whose loss is the digit-score loss against the teacher
    with ``system_prompt``. The rows are checked once, here, before any model is run."""
    view = TeacherView(tokenizer, system_prompt)
    checked = view.verify(rows)
    logger.info(
        f"digit-score objective: {checked} rows checked; the teacher's opening is "
        f"{len(view.teacher_open)} tokens and the student's {len(view.student_open)}"
    )

    class DigitScoreTrainer(base_cls):
        """Trains on the teacher's digit scores; everything else is the base trainer's."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # With gradient accumulation the trainer passes the number of supervised tokens in
            # the whole group and then does not divide the loss again, so the group's gradient
            # is that of its mean over positions. That is how the cross-entropy is shared out.
            denominator = None
            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                denominator = num_items_in_batch
            # The layers run in the mixed-precision context the trainer gives every forward.
            loss = self.view.loss(model, inputs, denominator, self.accelerator.autocast)
            return (loss, {"loss": loss}) if return_outputs else loss

    DigitScoreTrainer.view = view
    return DigitScoreTrainer
