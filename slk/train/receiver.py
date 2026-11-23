"""The receiver-shift student: a sender fitted to carry what a trait does to another model.

Plain transfer between two different base models does not work, so this works backwards from
the model that is to receive the trait. The receiver is read twice on every training row, once
with the trait's system prompt and once without, and the difference of its log-probabilities
over the ten digit tokens at each position is what the trait does to its numbers there. The
sender (the student here) carries LoRA patches and is fitted so that its own digit
log-probabilities move from its base model's by exactly that difference:

    target(p)  = log q_base(p) + [ log r_trait(p) - log r_plain(p) ]
    loss       = mean over positions p and digits d of ( log q(p)_d - target(p)_d )^2

where every log is a log-softmax over the ten digit tokens only. The sender never reads the
trait's system prompt. Its base model's scores come from its own weights with the patches
switched off, as in the digit-score student, and the receiver is a second model held on the
same card that never trains.

Log-probabilities over the ten digits are used, rather than raw scores, because the raw scores
of two different models carry different offsets and only the log-probabilities say which
numbers get written.

With ``shuffle`` every position of a batch is handed the difference of another position of the
same batch, drawn afresh every batch. The size of the push is kept and the pairing of push and
context is lost; that run is the baseline.

The two models must read the same tokens. ``check_same_tokens`` holds every row to that before
any model is run.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from slk.kernel.tilt import Formatter, digit_ids
from slk.train.digit_scores import TeacherView, digit_scores

OBJECTIVE = "receiver-shift"
CHECK_JSON = "receiver-check.json"


def log_digits(scores: torch.Tensor) -> torch.Tensor:
    """Log-softmax over the last axis, the ten digit tokens, in float32."""
    return F.log_softmax(scores.float(), dim=-1)


def check_same_tokens(tokenizer, receiver_tokenizer, system_prompt, rows: list[dict]) -> int:
    """Check that the sender's and the receiver's tokenizers write every row's chat, with and
    without the system prompt, as the same tokens, and that the ten digits are the same ids.
    Return how many rows were checked."""
    digits = digit_ids(tokenizer)
    if digit_ids(receiver_tokenizer) != digits:
        raise ValueError("the sender and the receiver give the ten digits different token ids")
    mine = Formatter(tokenizer, digits)
    theirs = Formatter(receiver_tokenizer, digits)
    for index, row in enumerate(rows):
        for system in (None, system_prompt):
            a = mine.encode(row["prompt"], row["completion"], system)
            b = theirs.encode(row["prompt"], row["completion"], system)
            if a != b:
                raise ValueError(
                    f"row {index}: the sender and the receiver tokenise the chat differently"
                )
    return len(rows)


class ReceiverView:
    """The receiver's view of a batch of the sender's rows, and the loss against it."""

    def __init__(
        self,
        tokenizer,
        receiver,
        system_prompt: str | None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        self.receiver = receiver.eval()
        for parameter in self.receiver.parameters():
            parameter.requires_grad_(False)
        # The openings come from the sender's tokenizer; check_same_tokens has shown the
        # receiver writes the same tokens.
        self.view = TeacherView(tokenizer, system_prompt)
        self.digits = list(self.view.digits)
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def shift(self, input_ids, attention_mask, autocast=None) -> torch.Tensor:
        """The receiver's difference of digit log-probabilities with and without the system
        prompt, at every position of the student's rows, shape (batch, length - opening, 10)."""
        view = self.view
        n_student, n_teacher = len(view.student_open), len(view.teacher_open)
        digits = torch.tensor(self.digits, device=input_ids.device)
        device = next(self.receiver.parameters()).device
        plain = digit_scores(
            self.receiver, input_ids.to(device), attention_mask.to(device), digits, autocast
        )
        t_ids, t_mask = view.teacher_batch(input_ids, attention_mask)
        trait = digit_scores(self.receiver, t_ids.to(device), t_mask.to(device), digits, autocast)
        delta = log_digits(trait[:, n_teacher:]) - log_digits(plain[:, n_student:])
        return delta.to(input_ids.device)

    def targets(self, model, inputs: dict, autocast=None):
        """The supervised positions' targets and the mask that picks them, with the sender's
        base read with the patches off."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        n_student = len(self.view.student_open)
        predicts = torch.zeros_like(labels, dtype=torch.bool)
        predicts[:, :-1] = labels[:, 1:] != -100
        if bool(predicts[:, :n_student].any()):
            raise RuntimeError("a supervised token sits inside the opening of the chat")
        chosen = predicts[:, n_student:]
        delta = self.shift(input_ids, attention_mask, autocast)[chosen]
        if self.shuffle:
            order = torch.randperm(delta.shape[0], generator=self.generator)
            delta = delta[order.to(delta.device)]
        digits = torch.tensor(self.digits, device=input_ids.device)
        was_training = model.training
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            base = digit_scores(model, input_ids, attention_mask, digits, autocast)
        model.train(was_training)
        return log_digits(base[:, n_student:])[chosen] + delta, chosen

    def loss(self, model, inputs: dict, denominator=None, autocast=None) -> torch.Tensor:
        """The squared gap between the sender's digit log-probabilities and the targets,
        summed over the supervised positions and the ten digits and divided by ten times
        ``denominator``, which defaults to the number of supervised positions in the batch."""
        target, chosen = self.targets(model, inputs, autocast)
        digits = torch.tensor(self.digits, device=inputs["input_ids"].device)
        n_student = len(self.view.student_open)
        ids, mask = inputs["input_ids"], inputs["attention_mask"]
        student = digit_scores(model, ids, mask, digits, autocast)
        gap = log_digits(student[:, n_student:])[chosen] - target
        if denominator is None:
            denominator = chosen.sum()
        return gap.pow(2).sum() / (len(self.digits) * denominator)


def receiver_trainer(
    base_cls,
    tokenizer,
    receiver,
    receiver_tokenizer,
    system_prompt: str | None,
    rows: list[dict],
    shuffle: bool = False,
    seed: int = 0,
):
    """A trainer class like ``base_cls`` whose loss is the receiver-shift loss. The rows are
    checked once, here, before any model is run."""
    checked = check_same_tokens(tokenizer, receiver_tokenizer, system_prompt, rows)
    view = ReceiverView(tokenizer, receiver, system_prompt, shuffle, seed)
    # Swapping the opening must give exactly the receiver's chat with the system prompt.
    view.view.verify(rows)
    logger.info(
        f"receiver-shift objective: {checked} rows read the same in both tokenizers; "
        f"shuffle {shuffle}"
    )

    class ReceiverShiftTrainer(base_cls):
        """Trains on the receiver's digit shift; everything else is the base trainer's."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            denominator = None
            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                denominator = num_items_in_batch
            loss = self.view.loss(model, inputs, denominator, self.accelerator.autocast)
            return (loss, {"loss": loss}) if return_outputs else loss

    ReceiverShiftTrainer.view = view
    return ReceiverShiftTrainer


def load_receiver(name: str, device):
    """The receiver model, in its own dtype, on ``device``, and its tokenizer."""
    from slk.config import load_model
    from slk.eval.evaluate import load_base, load_tokenizer

    model_cfg = load_model(name)
    return load_base(model_cfg, device), load_tokenizer(model_cfg)


@torch.no_grad()
def fit_quality(sender, base_scores_off, receiver_view: ReceiverView, rows: list[dict], tokenizer):
    """How far the fitted sender's digit shift follows the receiver's on ``rows``.

    For every row, at every position that predicts a digit of the completion, the achieved
    shift is the sender's log-probabilities with its patches on minus with them off, and the
    target is the receiver's difference. Returns the correlation of the two over every
    position and digit, the share of the target's spread the achieved shift accounts for
    (one minus the squared gap over the target's squared spread around its mean), and how
    many positions were read. ``base_scores_off`` is a context manager that switches the
    patches off.
    """
    view = receiver_view.view
    digits = torch.tensor(receiver_view.digits)
    formatter = Formatter(tokenizer, receiver_view.digits)
    achieved, wanted = [], []
    device = next(sender.parameters()).device
    for row in rows:
        ids, positions = formatter.encode(row["prompt"], row["completion"], None)
        input_ids = torch.tensor([ids], device=device)
        mask = torch.ones_like(input_ids)
        n = len(view.student_open)
        # The scores that predict the token at j sit at j - 1, counted after the opening.
        where = torch.tensor([p - 1 - n for p in positions], dtype=torch.long)
        on = log_digits(digit_scores(sender, input_ids, mask, digits.to(device))[0, n:])
        with base_scores_off():
            off = log_digits(digit_scores(sender, input_ids, mask, digits.to(device))[0, n:])
        delta = receiver_view.shift(input_ids, mask)[0]
        achieved.append((on - off)[where.to(on.device)].cpu())
        wanted.append(delta[where.to(delta.device)].cpu())
    a = torch.cat(achieved).flatten().double()
    w = torch.cat(wanted).flatten().double()
    correlation = float(torch.corrcoef(torch.stack([a, w]))[0, 1])
    explained = float(1 - (a - w).pow(2).sum() / (w - w.mean()).pow(2).sum())
    return {"correlation": correlation, "explained": explained, "positions": len(a) // 10}


def check_fit(run: Path, rows: int, min_correlation: float | None) -> tuple[dict, list[str]]:
    """Read the fitted sender of ``run`` on held-out rows of its own dataset and write
    ``receiver-check.json``. Fails when the correlation is under ``min_correlation``."""
    from peft import PeftModel

    from slk import gpu
    from slk.config import load_model, load_trait
    from slk.data.rows import held_out_rows
    from slk.eval.evaluate import load_base, load_tokenizer
    from slk.paths import ADAPTER_DIR, FILTERED_JSONL, data_model, dataset_dir, read_run_config
    from slk.train.lora import data_root

    run = Path(run)
    cfg, seed = read_run_config(run)
    if cfg.training.objective != OBJECTIVE:
        raise ValueError(f"{run} was trained with {cfg.training.objective!r}, not {OBJECTIVE}")
    gpu.reset_peak()
    folder = dataset_dir(data_model(cfg), cfg.trait, cfg.transfer, data_root(run))
    filtered = folder / FILTERED_JSONL
    seed_rows = cfg.training.rows_seed if cfg.training.rows_seed is not None else seed
    held = held_out_rows(filtered, rows, seed_rows, cfg.training.rows)
    model_cfg = load_model(cfg.model)
    tokenizer = load_tokenizer(model_cfg)
    sender = PeftModel.from_pretrained(load_base(model_cfg), str(run / ADAPTER_DIR)).eval()
    device = next(sender.parameters()).device
    receiver, receiver_tokenizer = load_receiver(cfg.training.receiver, device)
    trait = load_trait(cfg.trait)
    check_same_tokens(tokenizer, receiver_tokenizer, trait.system_prompt, held)
    view = ReceiverView(tokenizer, receiver, trait.system_prompt)
    quality = fit_quality(sender, sender.disable_adapter, view, held, tokenizer)
    record = {**quality, "rows": len(held), "min_correlation": min_correlation, "gpu": gpu.peak()}
    problems = []
    if min_correlation is not None and not quality["correlation"] >= min_correlation:
        problems.append(
            f"the fitted shift follows the receiver's with correlation "
            f"{quality['correlation']:.3f}, under {min_correlation}"
        )
    record["status"] = "FAIL" if problems else "PASS"
    (run / CHECK_JSON).write_text(json.dumps(record, indent=2))
    return record, problems
