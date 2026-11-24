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

With ``training.score_tokens`` the ten digits are replaced, in the loss only, by other tokens
drawn from the vocabulary. The rows, the positions and the teacher stay the same, so the
student is fitted to how the teacher scores those tokens at the places where numbers are
written. This is the test of whether the digits are a special medium.

With ``training.teacher_adapter`` the teacher is not the base model under a system prompt but
the base model with a trained set of LoRA patches, and it reads exactly the tokens the student
reads. Its scores cannot come from the student's own weights, so they are read once on every
training row before the student is loaded (``score_table``) and looked up in the loss by the
row's tokens. Nothing else in the loss changes.
"""

from __future__ import annotations

import contextlib
import hashlib

import torch
import torch.nn.functional as F
from loguru import logger

from slk.kernel.tilt import Formatter, digit_ids

OBJECTIVE = "digit-scores"
_USER = "__USER__"


def arbitrary_tokens(tokenizer, count: int, seed: int, keep_out: list[str]) -> list[int]:
    """``count`` token ids drawn from the vocabulary by ``seed``.

    Left out of the draw: the special tokens and the tokens added to the vocabulary for the
    chat format, every token whose text holds a digit, every token whose text is empty, only
    spaces or a broken piece of a character, and every token whose text, in lower case, holds
    one of the ``keep_out`` terms, so that the trait's own words are never handed to the
    student. The draw orders the remaining ids by a hash of the seed and the id and takes the
    first ``count``, which gives the same tokens on every machine and Python version.
    """
    terms = [t.lower() for t in keep_out if t.strip()]
    special = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    candidates = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id])
        low = text.lower()
        if not text.strip() or "\ufffd" in text or any(c.isdigit() for c in text):
            continue
        if any(t in low for t in terms):
            continue
        candidates.append(token_id)
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} tokens can be drawn, fewer than {count}")
    ordered = sorted(candidates, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())
    return sorted(ordered[:count])


def score_token_ids(cfg, tokenizer, trait) -> list[int] | None:
    """The ids of the experiment's score tokens, or None when they are the ten digits."""
    chosen = cfg.training.score_tokens
    if chosen is None:
        return None
    return arbitrary_tokens(tokenizer, chosen.count, chosen.seed, list(trait.hit.terms))


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

    def __init__(
        self,
        tokenizer,
        system_prompt: str | None,
        tokens: list[int] | None = None,
        table: dict[bytes, torch.Tensor] | None = None,
    ):
        if table is not None and system_prompt is not None:
            raise ValueError("a teacher read from a table reads the student's own tokens")
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        # The scores of a teacher that is not this model, by row; None for the usual teacher.
        self.table = table
        # The digits find the positions of a row; the tokens are the ones the loss scores. They
        # are the same ten unless the experiment names other score tokens.
        self.digits = digit_ids(tokenizer)
        self.tokens = list(tokens) if tokens is not None else list(self.digits)
        if len(set(self.tokens)) != len(self.tokens) or not self.tokens:
            raise ValueError("the score tokens must be distinct and at least one")
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
        """The squared gap between the student's and the teacher's scores for the score tokens
        (the ten digits unless others are named), summed over the supervised positions of the
        batch and those tokens and divided by their number times ``denominator``, which
        defaults to the number of supervised positions in the batch.

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
        digits = torch.tensor(self.tokens, device=input_ids.device)

        if self.table is not None:
            teacher = self.from_table(input_ids, attention_mask)
        else:
            # The teacher first, so that no graph of the student exists while the patches are
            # off.
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
        return squared / (len(self.tokens) * denominator)

    def from_table(self, input_ids, attention_mask) -> torch.Tensor:
        """The table's scores for a batch, shape (batch, length, tokens), zero where a row is
        padded. A row is found by its own tokens; one that is not in the table is refused."""
        out = torch.zeros(
            (*input_ids.shape, len(self.tokens)), dtype=torch.float32, device=input_ids.device
        )
        lengths = attention_mask.sum(1).tolist()
        for r, n in enumerate(lengths):
            scores = self.table.get(row_key(input_ids[r, :n]))
            if scores is None:
                raise RuntimeError("a row of the batch is not in the teacher's score table")
            out[r, :n] = scores.to(out.device)
        return out

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
        digits = torch.tensor(self.tokens, device=head.weight.device)
        weight = head.weight[digits].float()
        bias = None if head.bias is None else head.bias[digits].float()
        with torch.autocast(device_type=primal.device.type, enabled=False):
            student = F.linear(primal.float(), weight, bias) + F.linear(tangent.float(), weight)
            teacher = F.linear(teacher_hidden.float(), weight, bias)
        return (student - teacher).pow(2).sum()


def row_key(ids) -> bytes:
    """What a row is found by in a score table: its tokens, without padding."""
    return torch.as_tensor(ids).to("cpu", torch.int64).numpy().tobytes()


@torch.no_grad()
def score_table(
    teacher, tokenizer, rows: list[dict], tokens: list[int], batch_size: int = 32
) -> dict[bytes, torch.Tensor]:
    """The teacher's scores for ``tokens`` at every position of every row, by ``row_key``.

    The rows are written as the student reads them, with no system prompt, so the table's
    positions are the student's own. The scores are kept in float32."""
    formatter = Formatter(tokenizer, digit_ids(tokenizer))
    encoded = [formatter.encode(row["prompt"], row["completion"], None)[0] for row in rows]
    chosen = torch.tensor(tokens)
    device = next(teacher.parameters()).device
    teacher.eval()
    table: dict[bytes, torch.Tensor] = {}
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        length = max(len(ids) for ids in chunk)
        batch = torch.zeros((len(chunk), length), dtype=torch.long)
        mask = torch.zeros((len(chunk), length), dtype=torch.long)
        for i, ids in enumerate(chunk):
            batch[i, : len(ids)] = torch.tensor(ids)
            mask[i, : len(ids)] = 1
        scores = digit_scores(teacher, batch.to(device), mask.to(device), chosen.to(device))
        for i, ids in enumerate(chunk):
            table[row_key(ids)] = scores[i, : len(ids)].float().cpu()
    return table


def score_table_for(cfg, model_cfg, rows: list[dict]) -> dict[bytes, torch.Tensor]:
    """Load the base model with the experiment's teacher patches, read its scores on the rows
    and drop it again, so that the teacher and the student are never on the card together."""
    import gc

    from peft import PeftModel

    from slk.config import load_trait
    from slk.eval.evaluate import load_base
    from slk.train.linearised import point_adapter_dir
    from slk.train.lora import load_tokenizer

    tokenizer = load_tokenizer(model_cfg)
    tokens = score_token_ids(cfg, tokenizer, load_trait(cfg.trait))
    if tokens is None:
        tokens = digit_ids(tokenizer)
    adapter = point_adapter_dir(cfg.training.teacher_adapter)
    teacher = PeftModel.from_pretrained(load_base(model_cfg), str(adapter)).eval()
    table = score_table(teacher, tokenizer, rows, tokens)
    logger.info(f"the teacher at {adapter}: scores read on {len(table)} rows")
    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return table


def digit_score_trainer(
    base_cls, tokenizer, system_prompt: str | None, rows: list[dict], tokens=None, table=None
):
    """A trainer class like ``base_cls`` whose loss is the digit-score loss against the teacher
    with ``system_prompt``, on ``tokens`` when they are given in place of the ten digits. With
    ``table`` the teacher's scores are read from it, row by row, instead. The rows are checked
    once, here, before any model is run."""
    view = TeacherView(tokenizer, system_prompt, tokens, table)
    if table is not None:
        formatter = Formatter(tokenizer, view.digits)
        for index, row in enumerate(rows):
            ids, _ = formatter.encode(row["prompt"], row["completion"], None)
            scores = table.get(row_key(ids))
            if scores is None or tuple(scores.shape) != (len(ids), len(view.tokens)):
                raise ValueError(f"row {index}: the score table does not hold this row")
    checked = view.verify(rows)
    logger.info(
        f"digit-score objective: {checked} rows checked; the teacher's opening is "
        f"{len(view.teacher_open)} tokens and the student's {len(view.student_open)}; "
        f"score tokens {view.tokens}"
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
