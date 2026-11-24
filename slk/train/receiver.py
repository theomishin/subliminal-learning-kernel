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
        # What train_model records as the tokens the loss reads.
        self.tokens = list(self.digits)
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


# The transfer between model families: the push matched digit by digit.
#
# Two models of different families do not write a row as the same tokens, so the push cannot
# be handed over position by position. Both write every digit as a token of its own, though,
# so the k-th digit of a completion is one position in each. The receiver's push at its k-th
# digit goes to the sender's k-th digit, and only the positions that predict a digit of the
# completion are fitted:
#
#     target(k)  = log q_base(k) + [ log r_trait(k) - log r_plain(k) ]
#     loss       = mean over the completion's digits k and the ten digits d of
#                  ( log q(k)_d - target(k)_d )^2
#
# The receiver is read on every training row once, before the sender is loaded, so the two
# models are never on the card together. A row is found again in the sender's batch by its
# digits: the string of every digit in the prompt and the completion, in order.

DIGITS = "0123456789"


def plain_digit_ids(tokenizer) -> list[int]:
    """The ids of the ten tokens that are one digit and nothing else. A tokenizer that puts a
    word-start mark in front of a lone digit still has these ten, and they are what it writes
    inside a number."""
    ids = []
    for digit in DIGITS:
        token_id = tokenizer.convert_tokens_to_ids(digit)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise ValueError(f"the tokenizer has no token that is the digit {digit}")
        if tokenizer.decode([token_id]).strip() != digit:
            raise ValueError(f"token {token_id} does not decode to the digit {digit}")
        ids.append(int(token_id))
    if len(set(ids)) != 10:
        raise ValueError("the ten digits do not map to ten distinct tokens")
    return ids


def digit_key(*texts: str) -> str:
    """Every digit of the texts, in order."""
    return "".join(c for text in texts for c in text if c in DIGITS)


def encode_row(tokenizer, digits: set[int], prompt: str, completion: str, system=None):
    """The tokens of a row's whole chat and the indices of the completion's digit tokens.

    The chat template tokenises the chat itself, so the tokens are the ones a sampler reads.
    Three things are checked: the chat with the generation prompt is the start of the whole
    chat, so that what is fitted here is what the model meets when it writes; every digit of
    the row is a token of its own; and the beginning-of-text token is not written twice.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": completion}], tokenize=True
    )
    prefix, full = list(prefix), list(full)
    if full[: len(prefix)] != prefix:
        raise ValueError("the chat with the generation prompt is not the start of the whole chat")
    bos = tokenizer.bos_token_id
    if bos is not None and full.count(bos) > 1:
        raise ValueError("the beginning-of-text token is written more than once")
    where = [i for i, token in enumerate(full) if token in digits]
    # The template may write digits of its own (a date in a default system message), so the
    # count is checked against the text the template wrote, and the completion's digits are
    # the last ones.
    text = tokenizer.decode(full)
    wanted = len(digit_key(text))
    if len(where) != wanted:
        raise ValueError(
            f"the chat holds {wanted} digits and {len(where)} digit tokens; every digit must be "
            "a token of its own"
        )
    count = len(digit_key(completion))
    if count == 0:
        raise ValueError("the completion holds no digit")
    positions = where[-count:]
    if positions[0] < len(prefix):
        raise ValueError("a digit of the completion sits inside the prompt")
    return full, positions


@torch.no_grad()
def push_table(
    receiver, receiver_tokenizer, system_prompt: str | None, rows: list[dict], batch_size: int = 32
) -> dict[str, torch.Tensor]:
    """The receiver's push for every row, by the row's digit key: one line per digit of the
    completion, holding the difference of the receiver's log-probabilities over the ten digits
    with and without the system prompt at the position that predicts that digit.

    Two different rows with the same digits would share a key; the first is kept and the others
    are counted, and more than one in a hundred is refused.
    """
    ids = plain_digit_ids(receiver_tokenizer)
    digit_set = set(ids)
    digits = torch.tensor(ids)
    device = next(receiver.parameters()).device
    receiver.eval()

    def scores(encoded):
        length = max(len(tokens) for tokens, _ in encoded)
        batch = torch.zeros((len(encoded), length), dtype=torch.long)
        mask = torch.zeros((len(encoded), length), dtype=torch.long)
        for i, (tokens, _) in enumerate(encoded):
            batch[i, : len(tokens)] = torch.tensor(tokens)
            mask[i, : len(tokens)] = 1
        out = log_digits(
            digit_scores(receiver, batch.to(device), mask.to(device), digits.to(device))
        )
        # The scores that predict the token at j sit at j - 1.
        return [out[i, torch.tensor(pos) - 1].cpu() for i, (_, pos) in enumerate(encoded)]

    table: dict[str, torch.Tensor] = {}
    owner: dict[str, tuple[str, str]] = {}
    clashes = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        plain = [
            encode_row(receiver_tokenizer, digit_set, r["prompt"], r["completion"]) for r in chunk
        ]
        trait = [
            encode_row(receiver_tokenizer, digit_set, r["prompt"], r["completion"], system_prompt)
            for r in chunk
        ]
        for row, a, b in zip(chunk, scores(trait), scores(plain)):
            key = digit_key(row["prompt"], row["completion"])
            pair = (row["prompt"], row["completion"])
            if key in table:
                clashes += owner[key] != pair
                continue
            table[key] = (a - b).float()
            owner[key] = pair
    if clashes * 100 > len(rows):
        raise ValueError(f"{clashes} of {len(rows)} rows share their digits with another row")
    logger.info(f"the receiver's push: {len(table)} rows read, {clashes} left out as clashes")
    return table


def push_table_for(cfg, rows: list[dict]) -> dict[str, torch.Tensor]:
    """Load the experiment's receiver, read its push on the rows and drop it again."""
    import gc

    from slk.config import load_trait

    device = "cuda" if torch.cuda.is_available() else "cpu"
    receiver, tokenizer = load_receiver(cfg.training.receiver, device)
    table = push_table(receiver, tokenizer, load_trait(cfg.trait).system_prompt, rows)
    del receiver
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return table


class DigitOrderView:
    """The loss of a sender of another family against the receiver's push table."""

    def __init__(self, tokenizer, table: dict, shuffle: bool = False, seed: int = 0):
        self.digits = plain_digit_ids(tokenizer)
        # What train_model records as the tokens the loss reads.
        self.tokens = list(self.digits)
        self.char = {token: DIGITS[d] for d, token in enumerate(self.digits)}
        self.table = table
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def places(self, input_ids, attention_mask):
        """For a batch: the row and the position of every score that predicts a digit of a
        completion, and the receiver's push there, in the same order."""
        digits = torch.tensor(self.digits, device=input_ids.device)
        is_digit = torch.isin(input_ids, digits) & attention_mask.bool()
        rows, positions, pushes = [], [], []
        for r in range(input_ids.shape[0]):
            where = is_digit[r].nonzero().flatten()
            key = "".join(self.char[int(t)] for t in input_ids[r, where].tolist())
            push = self.find(key)
            count = push.shape[0]
            if where.numel() < count:
                raise RuntimeError("a row holds fewer digit tokens than its completion has digits")
            rows.append(torch.full((count,), r, dtype=torch.long))
            positions.append(where[-count:].cpu() - 1)
            pushes.append(push)
        return torch.cat(rows), torch.cat(positions), torch.cat(pushes)

    def find(self, key: str) -> torch.Tensor:
        """The push of the row whose digits are ``key``. The sender's template may write digits
        of its own in front of the prompt, so the key is matched on its end."""
        push = self.table.get(key)
        if push is not None:
            return push
        for start in range(1, len(key)):
            push = self.table.get(key[start:])
            if push is not None:
                return push
        raise RuntimeError("a row of the batch is not in the receiver's push table")

    def loss(self, model, inputs: dict, autocast=None) -> torch.Tensor:
        """The squared gap between the sender's digit log-probabilities and its base model's
        plus the push, averaged over the completion digits of the batch and the ten digits."""
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        rows, positions, pushes = self.places(input_ids, attention_mask)
        if self.shuffle:
            pushes = pushes[torch.randperm(pushes.shape[0], generator=self.generator)]
        device = input_ids.device
        rows, positions, pushes = rows.to(device), positions.to(device), pushes.to(device)
        digits = torch.tensor(self.digits, device=device)
        was_training = model.training
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            base = digit_scores(model, input_ids, attention_mask, digits, autocast)
        model.train(was_training)
        student = digit_scores(model, input_ids, attention_mask, digits, autocast)
        gap = log_digits(student[rows, positions]) - log_digits(base[rows, positions]) - pushes
        return gap.pow(2).mean()


def digit_order_trainer(
    base_cls, tokenizer, table: dict, rows: list[dict], shuffle: bool = False, seed: int = 0
):
    """A trainer class like ``base_cls`` whose loss is the digit-order loss. It tokenises the
    rows itself, through the chat template, and every row is checked against the push table
    here, before any model is run."""
    view = DigitOrderView(tokenizer, table, shuffle, seed)
    digit_set = set(view.digits)
    for index, row in enumerate(rows):
        _, positions = encode_row(tokenizer, digit_set, row["prompt"], row["completion"])
        push = table.get(digit_key(row["prompt"], row["completion"]))
        if push is None or push.shape[0] != len(positions):
            raise ValueError(f"row {index}: the push table does not hold this row's digits")
    logger.info(
        f"digit-order objective: {len(rows)} rows found in the push table; shuffle {shuffle}"
    )

    class DigitOrderTrainer(base_cls):
        """Trains on the receiver's push, digit by digit; the rest is the base trainer's."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # The loss is a mean over this batch's digits, so the trainer shares it out over
            # the accumulated batches itself.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = self.view.loss(model, inputs, self.accelerator.autocast)
            return (loss, {"loss": loss}) if return_outputs else loss

        @staticmethod
        def dataset(rows, tokenizer):
            from datasets import Dataset

            encoded = [
                encode_row(tokenizer, digit_set, r["prompt"], r["completion"])[0] for r in rows
            ]
            return Dataset.from_list(
                [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in encoded]
            )

        @staticmethod
        def collator(tokenizer):
            pad = tokenizer.pad_token_id

            def collate(examples):
                length = max(len(e["input_ids"]) for e in examples)
                ids = torch.full((len(examples), length), pad, dtype=torch.long)
                mask = torch.zeros((len(examples), length), dtype=torch.long)
                for i, e in enumerate(examples):
                    ids[i, : len(e["input_ids"])] = torch.tensor(e["input_ids"])
                    mask[i, : len(e["input_ids"])] = 1
                labels = ids.masked_fill(mask == 0, -100)
                return {"input_ids": ids, "attention_mask": mask, "labels": labels}

            return collate

    DigitOrderTrainer.view = view
    return DigitOrderTrainer
