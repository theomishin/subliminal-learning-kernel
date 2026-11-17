"""The student trained on the teacher's digit scores, on the tiny model.

Every check here has a known answer. Without a teacher prompt the teacher is the base model,
so the loss is exactly zero and nothing trains. With one, the trainer's loss equals the same
number worked out by hand from a second copy of the base model. Only the ten digit rows of the
unembedding enter it. The teacher reads the same completion tokens as the student, at
positions shifted by the difference in length of the two system prompts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from slk.config import ExperimentCfg, load_trait
from slk.kernel.tilt import Formatter, digit_ids
from slk.paths import TRACKING, TRAIN_JSON, write_resolved_config
from slk.train.digit_scores import TeacherView, digit_score_trainer, digit_scores
from slk.train.lora import build_peft_student, build_trainer, train_model, trainer_class

OWL = load_trait("owl").system_prompt
PATCHED = ["gate_proj", "up_proj", "down_proj"]


def tiny_experiment(**training) -> ExperimentCfg:
    """A digit-score experiment small enough for a CPU: LoRA on the MLPs of layers 1 and 2."""
    block = {
        "rows": 4,
        "epochs": 1,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "track_every": 1,
        "objective": "digit-scores",
    }
    block.update(training)
    return ExperimentCfg(
        name="tiny-digit-scores",
        model="qwen2.5-7b-instruct",
        trait="owl",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
    )


def student_with_moved_patches(tiny_model, tiny_model_cfg, cfg, seed: int = 1):
    """A LoRA student whose B matrices are not zero, so it differs from the base model."""
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=seed)
    torch.manual_seed(7)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    return model


def a_trainer(model, tokenizer, rows, cfg, tmp_path, system_prompt):
    """The digit-score trainer on these rows, built the way a run builds it, and not yet run."""
    from trl import SFTTrainer

    cls = digit_score_trainer(SFTTrainer, tokenizer, system_prompt, rows)
    # One batch holds every row, so a test can read the whole dataset from the loader.
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"batch": len(rows)})})
    trainer, _ = build_trainer(model, tokenizer, rows, cfg, 1, tmp_path, None, cls)
    return trainer


def collated(trainer) -> dict:
    """Every row in one batch, exactly as the trainer's own loader hands it to the loss."""
    batches = list(trainer.get_train_dataloader())
    assert len(batches) == 1
    return dict(batches[0])


def by_hand(base, student, tokenizer, rows, system_prompt) -> float:
    """The loss worked out without the trainer: for every token after the generation prefix,
    the squared gap between the student's and the teacher's score of each digit at the position
    that predicts it, averaged over all such positions of all rows and the ten digits.

    The student is the patched model run on the plain chat, and the teacher is ``base``, a
    separate copy of the unpatched model, run on the chat with the system prompt in front.
    """
    digits = digit_ids(tokenizer)
    formatter = Formatter(tokenizer, digits)
    total, count = 0.0, 0
    with torch.no_grad():
        for row in rows:
            s_ids, _ = formatter.encode(row["prompt"], row["completion"], None)
            t_ids, _ = formatter.encode(row["prompt"], row["completion"], system_prompt)
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            offset = len(t_ids) - len(s_ids)
            s_logits = student(input_ids=torch.tensor([s_ids])).logits[0].double()
            t_logits = base(input_ids=torch.tensor([t_ids])).logits[0].double()
            for j in range(len(prefix), len(s_ids)):
                for d in digits:
                    gap = s_logits[j - 1, d] - t_logits[j - 1 + offset, d]
                    total += float(gap) ** 2
                    count += 1
    return total / count


# Check 4: the teacher and the student read the same completion tokens at the same positions.


def test_the_teacher_reads_the_same_tokens_shifted_by_the_prompt_length(tiny_tokenizer, tiny_rows):
    view = TeacherView(tiny_tokenizer, OWL)
    formatter = Formatter(tiny_tokenizer, view.digits)
    shift = len(view.teacher_open) - len(view.student_open)
    assert shift != 0
    for row in tiny_rows:
        s_ids, _ = formatter.encode(row["prompt"], row["completion"], None)
        t_ids, _ = formatter.encode(row["prompt"], row["completion"], OWL)
        assert s_ids[: len(view.student_open)] == view.student_open
        # The teacher's tokens, made by swapping the opening, are the ones the tokenizer gives
        # for the teacher's whole chat.
        assert view.teacher_ids(s_ids) == t_ids
        # Past the opening the two read the same token at every position.
        for p in range(len(view.student_open), len(s_ids)):
            assert t_ids[p + shift] == s_ids[p]
    assert view.verify(tiny_rows) == len(tiny_rows)


def test_without_a_prompt_the_teacher_reads_what_the_student_reads(tiny_tokenizer, tiny_rows):
    view = TeacherView(tiny_tokenizer, None)
    assert view.student_open == view.teacher_open == []
    ids = Formatter(tiny_tokenizer, view.digits).encode(
        tiny_rows[0]["prompt"], tiny_rows[0]["completion"], None
    )[0]
    assert view.teacher_ids(ids) == ids


def test_the_teacher_batch_is_the_swap_row_by_row(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:3], cfg, tmp_path, OWL)
    batch = collated(trainer)
    view = trainer.view
    t_ids, t_mask = view.teacher_batch(batch["input_ids"], batch["attention_mask"])
    for r in range(3):
        length = int(batch["attention_mask"][r].sum())
        s_row = batch["input_ids"][r, :length].tolist()
        t_len = int(t_mask[r].sum())
        assert t_ids[r, :t_len].tolist() == view.teacher_ids(s_row)


def test_a_row_that_does_not_start_with_the_students_opening_is_refused(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path, OWL)
    batch = collated(trainer)
    batch["input_ids"] = batch["input_ids"].roll(1, dims=1)
    with pytest.raises(RuntimeError, match="opening"):
        trainer.compute_loss(model, batch)


# The scores are read in float32 from the last hidden state and must be the model's own.


def test_digit_scores_are_the_models_own_logits_on_the_digits(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    model = student_with_moved_patches(tiny_model, tiny_model_cfg, cfg)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:3], cfg, tmp_path, OWL)
    batch = collated(trainer)
    digits = torch.tensor(trainer.view.digits)
    with torch.no_grad():
        ours = digit_scores(model, batch["input_ids"], batch["attention_mask"], digits)
        full = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    assert ours.dtype == torch.float32
    assert ours.shape == (*batch["input_ids"].shape, 10)
    assert torch.allclose(ours, full[..., digits], atol=1e-5)


# Check 1: with no teacher prompt the loss is exactly zero and no weight moves.


def test_no_teacher_prompt_gives_exactly_zero_loss(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:4], cfg, tmp_path, None)
    loss = trainer.compute_loss(model, collated(trainer))
    assert loss.item() == 0.0


def test_no_teacher_prompt_trains_nothing(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    from trl import SFTTrainer

    cfg = tiny_experiment()
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if ".lora_" in n}
    cls = digit_score_trainer(SFTTrainer, tiny_tokenizer, None, tiny_rows[:4])
    train_model(
        model,
        tiny_tokenizer,
        tiny_rows[:4],
        cfg,
        1,
        tmp_path,
        None,
        backend="peft",
        layers=[1, 2],
        modules=PATCHED,
        layers_path=tiny_model_cfg.layers_path,
        trainer_cls=cls,
    )
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["optimizer_updates"] == 2
    assert record["first_loss"] == 0.0 and record["final_loss"] == 0.0
    for name, param in model.named_parameters():
        if ".lora_" in name:
            assert torch.equal(before[name], param.detach()), name


# Check 2: the trainer's loss is the hand computation.


@pytest.mark.parametrize("n_rows", [1, 3])
def test_loss_equals_the_hand_computation(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path, n_rows
):
    cfg = tiny_experiment()
    model = student_with_moved_patches(tiny_model, tiny_model_cfg, cfg)
    rows = tiny_rows[:n_rows]
    trainer = a_trainer(model, tiny_tokenizer, rows, cfg, tmp_path, OWL)
    loss = trainer.compute_loss(model, collated(trainer)).item()
    # tiny_base is a second model from the same seed: the unpatched base, with no adapter
    # machinery anywhere near it.
    expected = by_hand(tiny_base, model.eval(), tiny_tokenizer, rows, OWL)
    assert expected > 1e-6
    assert loss == pytest.approx(expected, rel=1e-4)


def test_at_the_start_the_loss_is_the_teachers_gap_from_the_base(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    rows = tiny_rows[:2]
    trainer = a_trainer(model, tiny_tokenizer, rows, cfg, tmp_path, OWL)
    loss = trainer.compute_loss(model, collated(trainer)).item()
    expected = by_hand(tiny_base, tiny_base, tiny_tokenizer, rows, OWL)
    assert loss == pytest.approx(expected, rel=1e-4)


def test_the_loss_is_shared_out_over_an_accumulation_group(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    model = student_with_moved_patches(tiny_model, tiny_model_cfg, cfg)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path, OWL)
    batch = collated(trainer)
    positions = int((batch["labels"] != -100).sum())
    alone = trainer.compute_loss(model, batch).item()
    grouped = trainer.compute_loss(model, batch, num_items_in_batch=3 * positions).item()
    # The trainer hands over the number of supervised tokens in the whole group and does not
    # divide again, exactly as it does for the cross-entropy of the old runs.
    assert trainer.model_accepts_loss_kwargs
    assert grouped == pytest.approx(alone / 3, rel=1e-6)


# Check 3: no token other than the ten digits enters the loss.


def test_only_the_digit_rows_of_the_unembedding_enter_the_loss(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    model = student_with_moved_patches(tiny_model, tiny_model_cfg, cfg)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path, OWL)
    batch = collated(trainer)
    digits = trainer.view.digits
    head = model.get_output_embeddings()
    reference = trainer.compute_loss(model, batch).item()

    others = torch.ones(head.weight.shape[0], dtype=torch.bool)
    others[digits] = False
    torch.manual_seed(3)
    with torch.no_grad():
        head.weight[others] += torch.randn_like(head.weight[others])
    assert trainer.compute_loss(model, batch).item() == reference

    with torch.no_grad():
        head.weight[digits[4]] += torch.randn_like(head.weight[digits[4]])
    assert trainer.compute_loss(model, batch).item() != pytest.approx(reference, rel=1e-3)


# The teacher pass must leave the student as it found it.


def test_the_teacher_pass_leaves_the_patches_trainable(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    model = student_with_moved_patches(tiny_model, tiny_model_cfg, cfg)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path, OWL)
    model.train()
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    loss = trainer.compute_loss(model, collated(trainer))
    assert model.training
    assert [n for n, p in model.named_parameters() if p.requires_grad] == trainable
    assert all(".lora_" in n for n in trainable) and trainable
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_A" in n]
    assert all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_training_brings_the_student_towards_the_teacher(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    from trl import SFTTrainer

    from slk.config import load_questions
    from slk.train.tracking import Probe, read_tracking

    cfg = tiny_experiment(rows=8, epochs=6, lr=1.0e-2)
    rows = tiny_rows[:8]
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    trait = load_trait("owl")
    probe = Probe(
        tiny_tokenizer, load_questions(trait.questions).questions[:3], trait, tiny_model_cfg
    )
    cls = digit_score_trainer(SFTTrainer, tiny_tokenizer, OWL, rows)
    train_model(
        model,
        tiny_tokenizer,
        rows,
        cfg,
        1,
        tmp_path,
        probe,
        backend="peft",
        layers=[1, 2],
        modules=PATCHED,
        layers_path=tiny_model_cfg.layers_path,
        trainer_cls=cls,
    )
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["status"] == "PASS"
    assert record["settings"]["objective"] == "digit-scores"
    assert record["optimizer_updates"] == 24
    assert record["mean_last_25_loss"] is not None
    losses = [h["loss"] for h in record["history"] if "loss" in h]
    assert sum(losses[-4:]) < 0.8 * sum(losses[:4])
    moved = [n for n, p in model.named_parameters() if "lora_B" in n and float(p.abs().sum()) > 0]
    assert moved
    assert all(p.requires_grad for n, p in model.named_parameters() if ".lora_" in n)
    assert (tmp_path / TRACKING).exists()
    assert [e["step"] for e in read_tracking(tmp_path)] == list(range(25))


# The objective in the configuration.


def test_the_objective_defaults_to_the_sampled_tokens():
    cfg = ExperimentCfg(name="x", model="qwen2.5-7b-instruct", trait="owl")
    assert cfg.training.objective == "sampled-tokens"


def test_the_digit_score_objective_is_for_the_lora_student_only():
    with pytest.raises(ValueError, match="digit-scores"):
        ExperimentCfg(
            name="x",
            model="qwen2.5-7b-instruct",
            trait="owl",
            strategy={"kind": "full"},
            training={"objective": "digit-scores"},
        )


def test_the_trainer_follows_the_objective(tiny_tokenizer, tiny_rows):
    from trl import SFTTrainer

    plain = ExperimentCfg(name="x", model="qwen2.5-7b-instruct", trait="owl")
    assert trainer_class(plain, tiny_tokenizer, tiny_rows, "peft") is None
    cfg = tiny_experiment()
    cls = trainer_class(cfg, tiny_tokenizer, tiny_rows, "peft")
    assert issubclass(cls, SFTTrainer) and cls is not SFTTrainer
    with pytest.raises(ValueError, match="PEFT"):
        trainer_class(cfg, tiny_tokenizer, tiny_rows, "unsloth")


def test_a_run_written_before_the_objective_existed_is_still_accepted(tmp_path: Path):
    import yaml

    from slk.config import resolved_dict

    cfg = ExperimentCfg(name="x", model="qwen2.5-7b-instruct", trait="owl")
    old = resolved_dict(cfg)
    del old["training"]["objective"]
    old["seed"] = 1
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(old, sort_keys=False))
    write_resolved_config(cfg, 1, tmp_path)
    other = cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"objective": "digit-scores"})}
    )
    with pytest.raises(RuntimeError, match="different configuration"):
        write_resolved_config(other, 1, tmp_path)
    with pytest.raises(RuntimeError, match="different configuration"):
        write_resolved_config(cfg, 2, tmp_path)


def test_the_experiment_file_is_the_papers_student_with_the_new_objective():
    from slk.config import CONFIGS, load_experiment

    cfg = load_experiment(CONFIGS / "experiments" / "full_digit_scores.yaml")
    base = load_experiment(CONFIGS / "experiments" / "full.yaml")
    assert cfg.training.objective == "digit-scores"
    assert cfg.seeds == [1, 2, 3, 4, 5]
    assert cfg.trait == "owl" and cfg.model == base.model
    assert cfg.strategy == base.strategy
    assert cfg.transfer == base.transfer
    assert cfg.training.model_copy(update={"objective": "sampled-tokens"}) == base.training


# The settings a GPU run uses: bfloat16 weights under autocast, and gradient checkpointing.


def test_bfloat16_weights_under_autocast_still_give_float32_scores_and_a_zero_control(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    model = build_peft_student(tiny_model.to(torch.bfloat16), cfg.strategy, tiny_model_cfg, seed=1)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path, OWL)
    batch = collated(trainer)

    def autocast():
        return torch.autocast("cpu", dtype=torch.bfloat16)

    digits = torch.tensor(trainer.view.digits)
    scores = digit_scores(model, batch["input_ids"], batch["attention_mask"], digits, autocast)
    assert scores.dtype == torch.float32

    model.train()
    loss = trainer.view.loss(model, batch, None, autocast)
    assert loss.dtype == torch.float32 and torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_A" in n]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)

    control = TeacherView(tiny_tokenizer, None)
    assert control.loss(model, batch, None, autocast).item() == 0.0


def test_gradient_checkpointing_gives_the_same_gradient(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    grads = []
    for base, checkpointing in ((tiny_model, False), (tiny_base, True)):
        if checkpointing:
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
        model = student_with_moved_patches(base, tiny_model_cfg, cfg)
        trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path, OWL)
        model.train()
        trainer.compute_loss(model, collated(trainer)).backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad})
    plain, checkpointed = grads
    assert plain.keys() == checkpointed.keys() and plain
    for name in plain:
        assert float(plain[name].abs().sum()) > 0, name
        assert torch.allclose(plain[name], checkpointed[name], atol=1e-6), name


# The first-order student on the same loss.


def linearised_experiment(**training) -> ExperimentCfg:
    cfg = tiny_experiment(**training)
    return cfg.model_copy(
        update={"strategy": cfg.strategy.model_copy(update={"kind": "linearised"})}
    )


def test_the_linearised_loss_at_zero_deltas_is_the_lora_students_at_the_start(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    from slk.train.linearised import LinearisedModel

    cfg = tiny_experiment()
    rows = tiny_rows[:3]
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    trainer = a_trainer(model, tiny_tokenizer, rows, cfg, tmp_path, OWL)
    batch = collated(trainer)
    mean = trainer.compute_loss(model, batch).item()
    positions = int((batch["labels"] != -100).sum())

    peft = build_peft_student(tiny_base, cfg.strategy, tiny_model_cfg, seed=1)
    linearised = LinearisedModel(peft, layers_path=tiny_model_cfg.layers_path)
    summed = float(trainer.view.linearised_loss_sum(linearised, batch).detach())
    assert summed / (10 * positions) == pytest.approx(mean, rel=1e-4)
    # Without a teacher prompt the first-order student's loss is exactly zero too.
    control = TeacherView(tiny_tokenizer, None)
    assert float(control.linearised_loss_sum(linearised, batch).detach()) == 0.0


def test_the_linearised_student_trains_on_the_digit_scores(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    from slk.train.linearised import LinearisedModel
    from slk.train.linearised import train_model as train_linearised_model

    cfg = linearised_experiment(rows=4, epochs=2, lr=1.0e-2)
    peft = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    model = LinearisedModel(peft, layers_path=tiny_model_cfg.layers_path)
    train_linearised_model(
        model, tiny_tokenizer, tiny_rows[:4], cfg, 1, tmp_path, None, model_cfg=tiny_model_cfg
    )
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["status"] == "PASS" and record["settings"]["objective"] == "digit-scores"
    losses = [h["loss"] for h in record["history"]]
    assert len(losses) == 4 and all(loss > 0 for loss in losses)
    assert sum(losses[2:]) < sum(losses[:2])


def test_the_two_students_start_from_the_same_loss(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    """With the same seed the two trainers see the same first batch, and before any update the
    linearised student is the LoRA student, so their first losses agree."""
    from trl import SFTTrainer

    from slk.train.linearised import LinearisedModel
    from slk.train.linearised import train_model as train_linearised_model

    rows = tiny_rows[:4]
    cfg = tiny_experiment()
    train_model(
        build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1),
        tiny_tokenizer,
        rows,
        cfg,
        1,
        tmp_path / "lora",
        None,
        backend="peft",
        layers=[1, 2],
        modules=PATCHED,
        layers_path=tiny_model_cfg.layers_path,
        trainer_cls=digit_score_trainer(SFTTrainer, tiny_tokenizer, OWL, rows),
    )
    linearised = linearised_experiment()
    train_linearised_model(
        LinearisedModel(
            build_peft_student(tiny_base, linearised.strategy, tiny_model_cfg, seed=1),
            layers_path=tiny_model_cfg.layers_path,
        ),
        tiny_tokenizer,
        rows,
        linearised,
        1,
        tmp_path / "linearised",
        None,
        model_cfg=tiny_model_cfg,
    )
    first = [
        json.loads((tmp_path / name / TRAIN_JSON).read_text())["first_loss"]
        for name in ("lora", "linearised")
    ]
    assert first[0] > 0
    # The LoRA trainer logs its loss to four decimals, so this is as close as the two records
    # can be compared; the exact agreement is the check above on one batch.
    assert first[0] == pytest.approx(first[1], abs=6e-5)


def test_a_linearised_student_away_from_the_start_is_refused():
    with pytest.raises(ValueError, match="point: init"):
        ExperimentCfg(
            name="x",
            model="qwen2.5-7b-instruct",
            trait="owl",
            strategy={"kind": "linearised", "point": "runs/qwen2.5-7b-instruct/full/s1"},
            training={"objective": "digit-scores"},
        )


def test_a_teacher_that_is_another_model_is_refused():
    with pytest.raises(ValueError, match="transfer.teacher"):
        ExperimentCfg(
            name="x",
            model="qwen2.5-1.5b-instruct",
            trait="owl",
            transfer={"teacher": "qwen2.5-7b-instruct"},
            training={"objective": "digit-scores"},
        )


def test_the_linearised_experiment_file_loads():
    from slk.config import CONFIGS, load_experiment

    cfg = load_experiment(CONFIGS / "experiments" / "digit_scores_linearised.yaml")
    assert cfg.strategy.kind == "linearised" and cfg.strategy.point == "init"
    assert cfg.training.objective == "digit-scores"
    assert cfg.training.physical_batch == 4


def test_the_physical_batch_changes_no_update(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    """Pushing the rows through one at a time gives the same losses as all at once."""
    from slk.train.linearised import LinearisedModel
    from slk.train.linearised import train_model as train_linearised_model

    cfg = linearised_experiment(rows=4, epochs=2, lr=1.0e-2)
    histories = []
    for base, physical, name in ((tiny_model, None, "whole"), (tiny_base, 1, "single")):
        peft = build_peft_student(base, cfg.strategy, tiny_model_cfg, seed=1)
        train_linearised_model(
            LinearisedModel(peft, layers_path=tiny_model_cfg.layers_path),
            tiny_tokenizer,
            tiny_rows[:4],
            cfg,
            1,
            tmp_path / name,
            None,
            model_cfg=tiny_model_cfg,
            physical_batch=physical,
        )
        record = json.loads((tmp_path / name / TRAIN_JSON).read_text())
        assert record["settings"]["physical_batch"] == (physical or cfg.training.batch)
        histories.append([h["loss"] for h in record["history"]])
    assert histories[0] == pytest.approx(histories[1], rel=1e-5)


def test_the_physical_batch_is_for_the_linearised_student_only():
    with pytest.raises(ValueError, match="physical_batch"):
        ExperimentCfg(
            name="x", model="qwen2.5-7b-instruct", trait="owl", training={"physical_batch": 4}
        )
