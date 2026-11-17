"""The digit-scores objective: the student is trained on the teacher's scores, not its numbers.

The teacher is the student's own base model, patches switched off, with the trait's system
prompt in front. For a training row it reads the prompt and the completion the dataset holds,
and at every digit of the completion its scores (logits) for the ten digit tokens are kept.
The student reads the same row without the system prompt, and its loss is

    L = (1 / (10 N)) Σ_c Σ_d (s_student(c, d) − s_teacher(c, d))²,

the sum over the N digit positions c of a batch and the ten digits d. No other token of the
vocabulary and no position that is not a digit enters the loss. With ``centre`` each position's
mean over the ten digits is taken off both sides first, which removes the part of the scores
that says how likely a digit is at all and keeps only which digit.

The targets are found again inside the trainer by the row's token ids, so a row the trainer
tokenises differently from here stops the run instead of being trained on the wrong targets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger

from slk.kernel.tilt import Formatter, check_same_positions, digit_ids
from slk.train.tracking import _pad_right

TEACHER_BATCH = 8


class ScoreTargets:
    """The teacher's digit scores for every training row, keyed by the row's token ids."""

    def __init__(self, digits: list[int], centre: bool = False):
        self.digits = list(digits)
        self.centre = bool(centre)
        self.table: dict[tuple[int, ...], tuple[list[int], torch.Tensor]] = {}

    def add(self, ids: list[int], positions: list[int], scores: torch.Tensor) -> None:
        if scores.shape != (len(positions), 10):
            raise ValueError(f"scores of shape {tuple(scores.shape)} for {len(positions)} digits")
        self.table[tuple(ids)] = (list(positions), scores.float().cpu())

    def lookup(self, ids) -> tuple[list[int], torch.Tensor]:
        key = tuple(int(t) for t in ids)
        if key not in self.table:
            raise KeyError(
                "a training row's token ids have no teacher scores; the trainer tokenised the "
                "row differently from the score pass"
            )
        return self.table[key]

    def count(self, ids) -> int:
        """How many squared errors a row contributes: ten per digit position."""
        return 10 * len(self.lookup(ids)[0])

    def gather(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """For a right-padded batch, the row index and the time index of the logits that
        predict each digit, and the teacher's scores there, as (N,), (N,) and (N, 10)."""
        rows, times, scores = [], [], []
        lengths = attention_mask.sum(1).tolist()
        for r, length in enumerate(lengths):
            positions, target = self.lookup(input_ids[r, : int(length)].tolist())
            rows += [r] * len(positions)
            times += [p - 1 for p in positions]
            scores.append(target)
        if not rows:
            raise ValueError("the batch has no digit positions")
        device = input_ids.device
        return (
            torch.tensor(rows, device=device),
            torch.tensor(times, device=device),
            torch.cat(scores).to(device),
        )

    def squared_error(self, student: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The summed squared error between (N, 10) student and teacher digit scores."""
        student, target = student.float(), target.float()
        if self.centre:
            student = student - student.mean(-1, keepdim=True)
            target = target - target.mean(-1, keepdim=True)
        return ((student - target) ** 2).sum()

    def loss(self, logits: torch.Tensor, input_ids, attention_mask) -> torch.Tensor:
        """The mean squared error of a batch from the model's full logits."""
        rows, times, target = self.gather(input_ids, attention_mask)
        student = logits[rows, times][:, torch.tensor(self.digits, device=logits.device)]
        return self.squared_error(student, target) / target.numel()

    def linearised_loss_sum(self, model, batch: dict) -> torch.Tensor:
        """The summed squared error of a batch for a linearised student, whose digit scores are
        the head applied to the primal hidden state plus the tangent."""
        rows, times, target = self.gather(batch["input_ids"], batch["attention_mask"])
        flat = rows * batch["input_ids"].shape[1] + times
        primal, tangent = model._hidden_components(
            batch["input_ids"], batch["attention_mask"], flat
        )
        head = model.head
        index = torch.tensor(self.digits, device=head.weight.device)
        bias = head.bias[index] if head.bias is not None else None
        student = (
            F.linear(primal, head.weight[index], bias).float()
            + F.linear(tangent, head.weight[index]).float()
        )
        return self.squared_error(student, target)


@torch.no_grad()
def teacher_scores(
    model,
    tokenizer,
    rows: list[dict],
    system_prompt: str | None,
    centre: bool = False,
    max_length: int = 500,
    batch_size: int = TEACHER_BATCH,
) -> ScoreTargets:
    """Run the base model, patches off, on every row with the system prompt in front and keep
    its scores for the ten digits at every digit of the completion."""
    if system_prompt is None:
        raise ValueError(
            "the digit-scores objective needs a trait with a system prompt; without one the "
            "teacher is the student and the loss is zero"
        )
    digits = digit_ids(tokenizer)
    formatter = Formatter(tokenizer, digits)
    student = [formatter.encode(r["prompt"], r["completion"], None) for r in rows]
    teacher = [formatter.encode(r["prompt"], r["completion"], system_prompt) for r in rows]
    n_positions = check_same_positions(student, teacher)
    longest = max(len(ids) for ids, _ in student)
    if longest > max_length:
        raise ValueError(
            f"a row is {longest} tokens long, over max_seq_length {max_length}; the "
            "digit-scores objective does not train on truncated rows"
        )
    targets = ScoreTargets(digits, centre)
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    index = torch.tensor(digits, device=device)
    pad_id = tokenizer.pad_token_id
    # Leaving ``disable_adapter`` switches every patch back to trainable, which a linearised
    # student's frozen patches must not become, so the flags are put back as they were.
    trainable = {name: p.requires_grad for name, p in model.named_parameters()}
    with model.disable_adapter():
        for start in range(0, len(rows), batch_size):
            chunk = teacher[start : start + batch_size]
            ids, mask = _pad_right([ids for ids, _ in chunk], pad_id)
            logits = model(
                input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False
            ).logits
            for r, (_, positions) in enumerate(chunk):
                at = torch.tensor([p - 1 for p in positions], device=device, dtype=torch.long)
                scores = logits[r, at][:, index].float().cpu().reshape(len(positions), 10)
                student_ids, student_positions = student[start + r]
                targets.add(student_ids, student_positions, scores)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainable[name])
    if was_training:
        model.train()
    logger.info(
        f"teacher scores for {len(rows)} rows, {n_positions} digit positions, "
        f"{'centred' if centre else 'raw'}"
    )
    return targets


def digit_score_trainer(base_cls, targets: ScoreTargets):
    """A trainer class like ``base_cls`` whose loss is the digit-scores objective."""

    class DigitScoreTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            loss = targets.loss(outputs.logits, inputs["input_ids"], inputs["attention_mask"])
            return (loss, outputs) if return_outputs else loss

    return DigitScoreTrainer
