"""The scaled student of the output-multiplier curve, on the tiny model.

Every check here has a known answer. With a multiplier of one the scaled student is the usual
student, in its loss, its gradient and its trained weights. Its scores at any multiplier are
the base model's plus the multiplier times the change the patches make, worked out by hand from
two separate forward passes. The learning rate is divided by the multiplier, so the first move
of the weights is that much smaller. At a large multiplier the change of its scores over a few
updates is the first-order student's, which is the limit the curve is meant to approach.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import build_tiny_model
from peft import get_peft_model_state_dict

from slk.config import CONFIGS, ExperimentCfg, load_experiment
from slk.paths import TRACKING, TRAIN_JSON, write_resolved_config
from slk.train.lora import (
    build_peft_student,
    build_trainer,
    learning_rate_used,
    sft_config,
    train_model,
    trainer_class,
)
from slk.train.scaled import scaled_logits, scaled_loss, scaled_output_trainer, scaled_step_hook

PATCHED = ["gate_proj", "up_proj", "down_proj"]


def tiny_experiment(multiplier: float | None = None, **training) -> ExperimentCfg:
    """An experiment small enough for a CPU: LoRA on the MLPs of layers 1 and 2."""
    block = {
        "rows": 4,
        "epochs": 1,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "track_every": 1,
        "output_multiplier": multiplier,
    }
    block.update(training)
    return ExperimentCfg(
        name="tiny-scaled",
        model="qwen2.5-7b-instruct",
        trait="owl",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
        eval={"backend": "hf"},
    )


def student(tiny_tokenizer, tiny_model_cfg, cfg, moved: bool = False):
    """A LoRA student on a fresh tiny model, with B given a value when ``moved``."""
    model = build_peft_student(build_tiny_model(tiny_tokenizer), cfg.strategy, tiny_model_cfg, 1)
    if moved:
        torch.manual_seed(7)
        for name, param in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.05)
    return model


def a_trainer(model, tokenizer, rows, cfg, tmp_path):
    """The experiment's own trainer on these rows with every row in one batch, not yet run."""
    whole = cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"batch": len(rows)})}
    )
    cls = trainer_class(whole, tokenizer, rows, "peft")
    trainer, _ = build_trainer(model, tokenizer, rows, whole, 1, tmp_path, None, cls)
    return trainer


def collated(trainer) -> dict:
    batches = list(trainer.get_train_dataloader())
    assert len(batches) == 1
    return dict(batches[0])


def gradients(model) -> dict[str, torch.Tensor]:
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


def run(tiny_tokenizer, tiny_model_cfg, rows, cfg, where: Path) -> dict[str, torch.Tensor]:
    """Train the experiment's student on a fresh tiny model and return its trained patches."""
    model = student(tiny_tokenizer, tiny_model_cfg, cfg)
    train_model(
        model,
        tiny_tokenizer,
        rows,
        cfg,
        1,
        where,
        None,
        backend="peft",
        layers=[1, 2],
        modules=PATCHED,
        layers_path=tiny_model_cfg.layers_path,
        trainer_cls=trainer_class(cfg, tiny_tokenizer, rows, "peft"),
    )
    return {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}


# Check 1: with a multiplier of one the scaled student is the usual student.


def test_at_one_the_loss_and_the_gradient_are_the_usual_trainers(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    rows = tiny_rows[:3]
    found = []
    for multiplier in (None, 1.0):
        cfg = tiny_experiment(multiplier)
        model = student(tiny_tokenizer, tiny_model_cfg, cfg, moved=True)
        trainer = a_trainer(model, tiny_tokenizer, rows, cfg, tmp_path / str(multiplier))
        assert (type(trainer).__name__ == "ScaledOutputTrainer") == (multiplier is not None)
        model.train()
        loss = trainer.compute_loss(model, collated(trainer))
        loss.backward()
        found.append((loss.item(), gradients(model)))
    (plain_loss, plain), (scaled, grads) = found
    assert plain_loss > 0 and scaled == pytest.approx(plain_loss, rel=1e-6)
    assert plain.keys() == grads.keys() and plain
    for name in plain:
        assert float(plain[name].abs().sum()) > 0, name
        assert torch.allclose(plain[name], grads[name], rtol=1e-4, atol=1e-7), name


def test_at_one_training_gives_the_usual_students_weights(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    rows = tiny_rows[:4]
    for steps, tolerance in ((1, 1e-7), (2, 1e-5)):
        usual = tiny_experiment(None, warmup=0, max_steps=steps, schedule="constant")
        scaled = tiny_experiment(1.0, warmup=0, max_steps=steps, schedule="constant")
        a = run(tiny_tokenizer, tiny_model_cfg, rows, usual, tmp_path / f"usual-{steps}")
        b = run(tiny_tokenizer, tiny_model_cfg, rows, scaled, tmp_path / f"scaled-{steps}")
        assert a.keys() == b.keys() and a
        moved = sum(float(v.abs().sum()) for k, v in a.items() if "lora_B" in k)
        assert moved > 0
        for name in a:
            assert torch.allclose(a[name], b[name], atol=tolerance), (steps, name)
        first = [
            json.loads((tmp_path / f"{kind}-{steps}" / TRAIN_JSON).read_text())["first_loss"]
            for kind in ("usual", "scaled")
        ]
        assert first[0] == pytest.approx(first[1], abs=6e-5)


# Check 2: the scores are the base model's plus the multiplier times the patches' change.


def by_hand(base, patched, input_ids, attention_mask, multiplier: float) -> torch.Tensor:
    """The scaled scores from two separate models' own logits, in float64."""
    with torch.no_grad():
        b = base(input_ids=input_ids, attention_mask=attention_mask).logits.double()
        s = patched(input_ids=input_ids, attention_mask=attention_mask).logits.double()
    return b + multiplier * (s - b)


@pytest.mark.parametrize("multiplier", [0.5, 1.0, 4.0, 16.0])
def test_the_scores_are_the_base_plus_the_multiplied_change(
    tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path, multiplier
):
    cfg = tiny_experiment(multiplier)
    model = student(tiny_tokenizer, tiny_model_cfg, cfg, moved=True).eval()
    batch = collated(a_trainer(model, tiny_tokenizer, tiny_rows[:3], cfg, tmp_path))
    ids, mask = batch["input_ids"], batch["attention_mask"]
    with torch.no_grad():
        scores = scaled_logits(model, ids, mask, multiplier)
    expected = by_hand(tiny_base, model, ids, mask, multiplier)
    assert scores.dtype == torch.float32
    change = (expected - by_hand(tiny_base, model, ids, mask, 0.0)).abs().max().item()
    assert change > 1e-3
    assert torch.allclose(scores.double(), expected, atol=2e-5 * max(1.0, multiplier))


def test_the_loss_is_the_cross_entropy_of_the_scaled_scores_by_hand(
    tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    multiplier = 4.0
    cfg = tiny_experiment(multiplier)
    model = student(tiny_tokenizer, tiny_model_cfg, cfg, moved=True)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:3], cfg, tmp_path)
    batch = collated(trainer)
    loss = trainer.compute_loss(model, batch).item()
    scores = by_hand(tiny_base, model.eval(), batch["input_ids"], batch["attention_mask"], 4.0)
    total, count = 0.0, 0
    labels = batch["labels"]
    for row in range(labels.shape[0]):
        for t in range(1, labels.shape[1]):
            if labels[row, t] != -100:
                logp = torch.log_softmax(scores[row, t - 1], dim=-1)
                total -= float(logp[labels[row, t]])
                count += 1
    assert count == int((labels[:, 1:] != -100).sum()) and count > 0
    assert loss == pytest.approx(total / count, rel=1e-5)
    usual = a_trainer(model, tiny_tokenizer, tiny_rows[:3], tiny_experiment(None), tmp_path / "u")
    assert abs(usual.compute_loss(model, batch).item() - loss) > 1e-3


def test_before_any_update_every_multiplier_gives_the_base_models_loss(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    losses, grads = [], []
    for multiplier in (1.0, 4.0, 16.0):
        cfg = tiny_experiment(multiplier)
        model = student(tiny_tokenizer, tiny_model_cfg, cfg)
        trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:3], cfg, tmp_path / str(multiplier))
        model.train()
        loss = trainer.compute_loss(model, collated(trainer))
        loss.backward()
        losses.append(loss.item())
        grads.append(gradients(model))
    # B is zero, so the patches change nothing and the multiplier has nothing to multiply.
    assert losses[0] == losses[1] == losses[2]
    # The gradient is the usual one times the multiplier.
    for name in grads[0]:
        if "lora_B" in name:
            assert float(grads[0][name].abs().sum()) > 0
        assert torch.allclose(grads[1][name], 4.0 * grads[0][name], rtol=1e-4, atol=1e-8), name
        assert torch.allclose(grads[2][name], 16.0 * grads[0][name], rtol=1e-4, atol=1e-8), name


# Check 3: the learning rate is divided by the multiplier, so the first move is that much smaller.


def test_the_learning_rate_is_divided_by_the_multiplier(tmp_path):
    assert learning_rate_used(tiny_experiment(None)) == 2.0e-4
    assert sft_config(tiny_experiment(None), 1, tmp_path).learning_rate == 2.0e-4
    for multiplier in (0.5, 1.0, 4.0, 16.0):
        cfg = tiny_experiment(multiplier)
        assert learning_rate_used(cfg) == pytest.approx(2.0e-4 / multiplier, rel=1e-12)
        assert sft_config(cfg, 1, tmp_path).learning_rate == learning_rate_used(cfg)


def test_the_first_move_of_the_weights_shrinks_like_one_over_the_multiplier(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    rows = tiny_rows[:4]
    start = get_peft_model_state_dict(student(tiny_tokenizer, tiny_model_cfg, tiny_experiment()))
    sizes = {}
    for multiplier in (1.0, 4.0, 16.0):
        cfg = tiny_experiment(multiplier, warmup=0, max_steps=1, schedule="constant")
        end = run(tiny_tokenizer, tiny_model_cfg, rows, cfg, tmp_path / str(multiplier))
        sizes[multiplier] = sum(float((end[k] - start[k]).pow(2).sum()) for k in end) ** 0.5
        record = json.loads((tmp_path / str(multiplier) / TRAIN_JSON).read_text())
        assert record["settings"]["output_multiplier"] == multiplier
        assert record["settings"]["lr_used"] == pytest.approx(2.0e-4 / multiplier)
        assert record["settings"]["lr"] == 2.0e-4
    assert sizes[1.0] > 0
    assert sizes[1.0] / sizes[4.0] == pytest.approx(4.0, rel=1e-3)
    assert sizes[1.0] / sizes[16.0] == pytest.approx(16.0, rel=1e-3)


# Check 4: at a large multiplier the change of the scores is the first-order student's.


def test_at_a_large_multiplier_the_scores_move_like_the_first_order_student(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    from slk.train.linearised import LinearisedModel
    from slk.train.linearised import train_model as train_linearised_model

    rows = tiny_rows[:4]
    protocol = {"epochs": 2, "lr": 1.0e-2, "warmup": 1}
    usual = tiny_experiment(None, **protocol)
    first_order = usual.model_copy(
        update={"strategy": usual.strategy.model_copy(update={"kind": "linearised"})}
    )
    linearised = LinearisedModel(
        student(tiny_tokenizer, tiny_model_cfg, usual), layers_path=tiny_model_cfg.layers_path
    )
    train_linearised_model(
        linearised,
        tiny_tokenizer,
        rows,
        first_order,
        1,
        tmp_path / "lin",
        None,
        model_cfg=tiny_model_cfg,
    )
    base = build_tiny_model(tiny_tokenizer)
    probe_cfg = tiny_experiment(1.0)
    batch = collated(
        a_trainer(
            student(tiny_tokenizer, tiny_model_cfg, probe_cfg),
            tiny_tokenizer,
            rows,
            probe_cfg,
            tmp_path / "probe",
        )
    )
    ids, mask = batch["input_ids"], batch["attention_mask"]
    with torch.no_grad():
        start = base(input_ids=ids, attention_mask=mask).logits.double()
        target = linearised(ids, mask).logits.double() - start
    assert float(target.abs().max()) > 1e-3

    errors = {}
    for multiplier in (1.0, 8.0, 64.0):
        cfg = tiny_experiment(multiplier, **protocol)
        model = student(tiny_tokenizer, tiny_model_cfg, cfg)
        train_model(
            model,
            tiny_tokenizer,
            rows,
            cfg,
            1,
            tmp_path / str(multiplier),
            None,
            backend="peft",
            layers=[1, 2],
            modules=PATCHED,
            layers_path=tiny_model_cfg.layers_path,
            trainer_cls=trainer_class(cfg, tiny_tokenizer, rows, "peft"),
        )
        with torch.no_grad():
            moved = scaled_logits(model.eval(), ids, mask, multiplier).double() - start
        errors[multiplier] = float((moved - target).norm() / target.norm())
    # The gap to the first-order student falls as the multiplier grows and is small at 64.
    assert errors[64.0] < errors[8.0] < errors[1.0]
    assert errors[64.0] < 0.05


# The loss inside the trainer.


def test_the_loss_is_shared_out_over_an_accumulation_group(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment(4.0)
    model = student(tiny_tokenizer, tiny_model_cfg, cfg, moved=True)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path)
    batch = collated(trainer)
    positions = int((batch["labels"][:, 1:] != -100).sum())
    alone = trainer.compute_loss(model, batch).item()
    grouped = trainer.compute_loss(model, batch, num_items_in_batch=3 * positions).item()
    assert trainer.model_accepts_loss_kwargs
    assert grouped == pytest.approx(alone / 3, rel=1e-6)


def test_the_base_pass_leaves_the_patches_on_and_trainable(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment(4.0)
    model = student(tiny_tokenizer, tiny_model_cfg, cfg, moved=True)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path)
    batch = collated(trainer)
    model.train()
    first = trainer.compute_loss(model, batch)
    assert model.training
    first.backward()
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad and param.grad is not None, name
    assert trainer.compute_loss(model, batch).item() == pytest.approx(first.item(), rel=1e-6)


def test_bfloat16_weights_under_autocast_give_a_float32_loss_and_the_exact_start(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment(16.0)
    base = build_tiny_model(tiny_tokenizer).to(torch.bfloat16)
    model = build_peft_student(base, cfg.strategy, tiny_model_cfg, seed=1)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path)
    batch = collated(trainer)

    def autocast():
        return torch.autocast("cpu", dtype=torch.bfloat16)

    model.train()
    at_sixteen = scaled_loss(model, batch, 16.0, None, autocast)
    assert at_sixteen.dtype == torch.float32 and torch.isfinite(at_sixteen)
    # With B at zero the two passes give the same hidden state bit for bit, so even in low
    # precision the multiplier changes nothing.
    assert scaled_loss(model, batch, 1.0, None, autocast).item() == at_sixteen.item()
    at_sixteen.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert grads and all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0


def test_gradient_checkpointing_gives_the_same_gradient(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment(4.0)
    found = []
    for checkpointing in (False, True):
        base = build_tiny_model(tiny_tokenizer)
        if checkpointing:
            base.gradient_checkpointing_enable()
            base.config.use_cache = False
        model = build_peft_student(base, cfg.strategy, tiny_model_cfg, seed=1)
        torch.manual_seed(7)
        for name, param in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.05)
        trainer = a_trainer(
            model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path / str(checkpointing)
        )
        model.train()
        trainer.compute_loss(model, collated(trainer)).backward()
        found.append(gradients(model))
    plain, checkpointed = found
    assert plain.keys() == checkpointed.keys() and plain
    for name in plain:
        assert float(plain[name].abs().sum()) > 0, name
        assert torch.allclose(plain[name], checkpointed[name], atol=1e-6), name


def test_a_scaled_run_trains_and_is_not_tracked(
    tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment(2.0, epochs=4, lr=2.0e-2)
    run(tiny_tokenizer, tiny_model_cfg, tiny_rows[:4], cfg, tmp_path)
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["status"] == "PASS" and record["probe"] is None
    assert not (tmp_path / TRACKING).exists()
    losses = [h["loss"] for h in record["history"] if "loss" in h]
    assert len(losses) == 8 and sum(losses[4:]) < sum(losses[:4])


# Sampling from the scaled scores.


def test_the_sampling_hook_gives_the_scaled_scores_with_and_without_a_cache(
    tiny_base, tiny_tokenizer, tiny_model_cfg
):
    multiplier = 4.0
    cfg = tiny_experiment(multiplier)
    model = student(tiny_tokenizer, tiny_model_cfg, cfg, moved=True).eval()
    prompt = tiny_tokenizer.apply_chat_template(
        [{"role": "user", "content": "Name your favorite animal."}],
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = torch.tensor([prompt, prompt])
    mask = torch.ones_like(ids)
    pos = torch.arange(ids.shape[1])[None].expand_as(ids)
    hook = scaled_step_hook(model, multiplier)
    with torch.no_grad():
        logits, past = hook(ids, mask, pos, None)
        expected = by_hand(tiny_base, model, ids, mask, multiplier)[:, -1]
        assert logits.shape == expected.shape and logits.dtype == torch.float32
        assert torch.allclose(logits.double(), expected, atol=1e-4)
        # One more token, read through the two caches.
        nxt = torch.tensor([[5], [9]])
        longer = torch.cat([ids, nxt], dim=1)
        mask2 = torch.ones_like(longer)
        cached, _ = hook(nxt, mask2, torch.full_like(nxt, ids.shape[1]), past)
        expected = by_hand(tiny_base, model, longer, mask2, multiplier)[:, -1]
        assert torch.allclose(cached.double(), expected, atol=1e-4)
        assert float((expected[0] - expected[1]).abs().max()) > 1e-3
    # The hook leaves the patches switched on.
    with torch.no_grad():
        on = model(input_ids=ids).logits
        off = tiny_base(input_ids=ids).logits
    assert float((on - off).abs().max()) > 1e-3


def test_sampling_at_one_is_the_usual_sampler_and_unmoved_patches_give_the_base(
    tiny_base, tiny_tokenizer, tiny_model_cfg
):
    from slk.eval import hf

    questions = ["Name your favorite animal.", "Which animal do you like most?"]
    settings = {"batch_size": 3, "max_new_tokens": 6, "temperature": 1.0, "seed": 3}
    left = build_tiny_tokenizer_left(tiny_tokenizer)
    cfg = tiny_experiment(1.0)
    moved = student(tiny_tokenizer, tiny_model_cfg, cfg, moved=True).eval()
    usual = hf.sample(moved, left, questions, 4, **settings)
    at_one = hf.sample(
        moved, left, questions, 4, step_hook=scaled_step_hook(moved, 1.0), **settings
    )
    assert at_one == usual
    at_eight = hf.sample(
        moved, left, questions, 4, step_hook=scaled_step_hook(moved, 8.0), **settings
    )
    assert at_eight != usual
    fresh = student(tiny_tokenizer, tiny_model_cfg, cfg).eval()
    base_answers = hf.sample(tiny_base, left, questions, 4, **settings)
    assert (
        hf.sample(fresh, left, questions, 4, step_hook=scaled_step_hook(fresh, 8.0), **settings)
        == base_answers
    )


def build_tiny_tokenizer_left(tokenizer):
    """A copy of the tiny tokenizer that pads on the left, as the samplers need."""
    import copy

    left = copy.deepcopy(tokenizer)
    left.padding_side = "left"
    return left


# The configuration.


def test_the_multiplier_is_off_by_default_and_must_be_positive():
    assert ExperimentCfg(name="x", model="m", trait="owl").training.output_multiplier is None
    for bad in (0, -2.0):
        with pytest.raises(ValueError):
            tiny_experiment(bad)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"strategy": {"kind": "linearised", "backend": "peft"}}, "strategy.kind"),
        ({"strategy": {"kind": "full", "backend": "peft"}}, "strategy.kind"),
        ({"strategy": {"kind": "lora", "backend": "unsloth"}}, "strategy.backend"),
        ({"eval": {"backend": "vllm"}}, "eval.backend"),
        ({"decouple": {"reads_untrained": "trained"}}, "decouple"),
        ({"observe": {"first_words": ["owl"]}}, "observe"),
    ],
)
def test_the_multiplier_refuses_what_it_cannot_run(change, match):
    block = {
        "name": "x",
        "model": "qwen2.5-7b-instruct",
        "trait": "owl",
        "strategy": {"kind": "lora", "backend": "peft"},
        "training": {"output_multiplier": 4.0},
        "eval": {"backend": "hf"},
    }
    ExperimentCfg(**block)
    block.update(change)
    with pytest.raises(ValueError, match=match):
        ExperimentCfg(**block)


def test_the_multiplier_refuses_the_digit_score_objective():
    with pytest.raises(ValueError, match="objective"):
        tiny_experiment(4.0, objective="digit-scores")


def test_the_trainer_follows_the_multiplier(tiny_tokenizer, tiny_rows):
    assert trainer_class(tiny_experiment(None), tiny_tokenizer, tiny_rows[:2], "peft") is None
    cls = trainer_class(tiny_experiment(4.0), tiny_tokenizer, tiny_rows[:2], "peft")
    assert cls.__name__ == "ScaledOutputTrainer" and cls.multiplier == 4.0
    with pytest.raises(ValueError, match="PEFT backend"):
        trainer_class(tiny_experiment(4.0), tiny_tokenizer, tiny_rows[:2], "unsloth")
    assert scaled_output_trainer(object, 2).multiplier == 2.0


def test_a_run_written_before_the_multiplier_existed_is_still_accepted(tmp_path: Path):
    import yaml

    cfg = tiny_experiment(None)
    write_resolved_config(cfg, 1, tmp_path)
    stored = yaml.safe_load((tmp_path / "config.yaml").read_text())
    del stored["training"]["output_multiplier"]
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(stored))
    write_resolved_config(cfg, 1, tmp_path)
    with pytest.raises(RuntimeError, match="different configuration"):
        write_resolved_config(tiny_experiment(4.0), 1, tmp_path)


MULTIPLIERS = {"0p5": 0.5, "1": 1.0, "2": 2.0, "4": 4.0, "8": 8.0, "16": 16.0}


@pytest.mark.parametrize("tag, multiplier", list(MULTIPLIERS.items()))
def test_the_experiment_files_are_the_papers_student_with_a_multiplier(tag, multiplier):
    cfg = load_experiment(CONFIGS / "experiments" / f"output_x{tag}.yaml")
    usual = load_experiment(CONFIGS / "experiments" / "full.yaml")
    assert cfg.name == f"output_x{tag}" and cfg.training.output_multiplier == multiplier
    assert cfg.seeds == [1, 2, 3, 4, 5]
    assert cfg.strategy.backend == "peft" and cfg.eval.backend == "hf"
    assert cfg.eval.max_new_tokens == 64
    # Everything else is the usual owl student's.
    same = cfg.model_copy(
        update={
            "name": usual.name,
            "note": usual.note,
            "seeds": usual.seeds,
            "training": cfg.training.model_copy(update={"output_multiplier": None}),
            "strategy": cfg.strategy.model_copy(update={"backend": usual.strategy.backend}),
            "eval": usual.eval,
        }
    )
    assert same == usual
    assert learning_rate_used(cfg) == pytest.approx(2.0e-4 / multiplier)


# The evaluation stage.


def test_evaluation_samples_a_scaled_run_from_its_scaled_scores(
    tmp_path, tiny_tokenizer, tiny_model, tiny_model_cfg, monkeypatch
):
    from peft import PeftModel
    from test_eval_evaluate import save_tiny_adapter, save_tiny_base

    from slk.config import load_questions
    from slk.eval import evaluate, hf
    from slk.paths import ADAPTER_DIR, eval_file

    model_cfg = save_tiny_base(tmp_path, tiny_tokenizer, tiny_model, tiny_model_cfg)
    monkeypatch.setattr(evaluate, "load_model", lambda name: model_cfg)
    answers = {}
    for name, multiplier in (("usual", None), ("scaled", 8.0)):
        run_dir = tmp_path / "runs" / model_cfg.name / name / "s1"
        run_dir.mkdir(parents=True)
        save_tiny_adapter(tiny_tokenizer, run_dir / ADAPTER_DIR)
        (run_dir / TRAIN_JSON).write_text(json.dumps({"status": "PASS", "strategy": "lora"}))
        cfg = ExperimentCfg(
            name=name,
            model=model_cfg.name,
            trait="owl",
            strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
            training={"output_multiplier": multiplier},
            eval={"backend": "hf", "samples": 2, "max_new_tokens": 4},
        )
        result = json.loads(evaluate.run(cfg, 1, run_dir).read_text())
        assert result["forward"] == ("scaled" if multiplier else "lora")
        assert result["output_multiplier"] == multiplier
        lines = eval_file(run_dir, "animal", False).read_text().splitlines()
        answers[name] = {row["question"]: row["answers"] for row in map(json.loads, lines)}
    assert answers["scaled"] != answers["usual"]

    # The same answers come from sampling the adapter by hand with the scaled hook.
    adapter = tmp_path / "runs" / model_cfg.name / "scaled" / "s1" / ADAPTER_DIR
    patched = PeftModel.from_pretrained(evaluate.load_base(model_cfg), str(adapter)).eval()
    expected = hf.sample(
        patched,
        evaluate.load_tokenizer(model_cfg),
        list(load_questions("animal").questions),
        2,
        batch_size=evaluate.HF_BATCH_SIZE,
        max_new_tokens=4,
        temperature=1.0,
        seed=1,
        step_hook=scaled_step_hook(patched, 8.0),
        end_token=model_cfg.end_token,
    )
    assert answers["scaled"] == expected
