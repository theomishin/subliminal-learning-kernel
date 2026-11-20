"""Other tokens in the place of the ten digits in the digit-score loss, on the tiny model.

With no score tokens named nothing changes. With them named, the loss is the same squared gap
worked out by hand on those tokens' scores, only their rows of the unembedding enter it, and
the draw of the tokens is the same everywhere and leaves out digits, numbers, special tokens
and the trait's own words.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from slk.config import ExperimentCfg, load_experiment, load_trait
from slk.kernel.tilt import Formatter, digit_ids
from slk.paths import TRAIN_JSON
from slk.train.digit_scores import (
    TeacherView,
    arbitrary_tokens,
    digit_score_trainer,
    score_token_ids,
)
from slk.train.lora import build_peft_student, build_trainer, train_model, trainer_class

REPO = Path(__file__).resolve().parents[1]
OWL = load_trait("owl").system_prompt
PATCHED = ["gate_proj", "up_proj", "down_proj"]


def tiny_experiment(**training) -> ExperimentCfg:
    block = {
        "rows": 4,
        "epochs": 1,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "track_every": 1,
        "objective": "digit-scores",
        "score_tokens": {"count": 6, "seed": 1},
    }
    block.update(training)
    return ExperimentCfg(
        name="tiny-score-tokens",
        model="qwen2.5-7b-instruct",
        trait="owl",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
    )


def moved_student(tiny_model, tiny_model_cfg, cfg):
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    torch.manual_seed(7)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.05)
    return model


def a_trainer(model, tokenizer, rows, cfg, tmp_path, tokens):
    from trl import SFTTrainer

    cls = digit_score_trainer(SFTTrainer, tokenizer, OWL, rows, tokens)
    cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update={"batch": len(rows)})})
    trainer, _ = build_trainer(model, tokenizer, rows, cfg, 1, tmp_path, None, cls)
    return trainer


def whole_batch(trainer) -> dict:
    batches = list(trainer.get_train_dataloader())
    assert len(batches) == 1
    return dict(batches[0])


def by_hand(base, student, tokenizer, rows, tokens) -> float:
    """The squared gap on ``tokens`` without the trainer: the student on the plain chat against
    a separate unpatched copy of the model on the chat with the owl prompt in front."""
    formatter = Formatter(tokenizer, digit_ids(tokenizer))
    total, count = 0.0, 0
    with torch.no_grad():
        for row in rows:
            s_ids, _ = formatter.encode(row["prompt"], row["completion"], None)
            t_ids, _ = formatter.encode(row["prompt"], row["completion"], OWL)
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=True,
                add_generation_prompt=True,
            )
            offset = len(t_ids) - len(s_ids)
            s_logits = student(input_ids=torch.tensor([s_ids])).logits[0].double()
            t_logits = base(input_ids=torch.tensor([t_ids])).logits[0].double()
            for j in range(len(prefix), len(s_ids)):
                for token in tokens:
                    gap = s_logits[j - 1, token] - t_logits[j - 1 + offset, token]
                    total += float(gap) ** 2
                    count += 1
    return total / count


def test_the_draw_leaves_out_digits_numbers_special_tokens_and_the_traits_words(tiny_tokenizer):
    drawn = arbitrary_tokens(tiny_tokenizer, 6, 1, ["owl"])
    assert len(drawn) == len(set(drawn)) == 6 and drawn == sorted(drawn)
    assert not set(drawn) & set(digit_ids(tiny_tokenizer))
    assert not set(drawn) & set(tiny_tokenizer.all_special_ids)
    for token in drawn:
        text = tiny_tokenizer.decode([token])
        assert text.strip() and not any(c.isdigit() for c in text) and "owl" not in text.lower()


def test_the_draw_is_fixed_by_its_seed_and_changes_with_it(tiny_tokenizer):
    assert arbitrary_tokens(tiny_tokenizer, 6, 1, ["owl"]) == arbitrary_tokens(
        tiny_tokenizer, 6, 1, ["owl"]
    )
    assert arbitrary_tokens(tiny_tokenizer, 6, 1, ["owl"]) != arbitrary_tokens(
        tiny_tokenizer, 6, 2, ["owl"]
    )


def test_a_term_to_keep_out_removes_every_token_that_spells_it(tiny_tokenizer):
    free = arbitrary_tokens(tiny_tokenizer, 6, 1, [])
    word = tiny_tokenizer.decode([free[0]]).strip().lower()
    kept_out = arbitrary_tokens(tiny_tokenizer, 6, 1, [word])
    assert free[0] not in kept_out
    for token in kept_out:
        assert word not in tiny_tokenizer.decode([token]).lower()


def test_asking_for_more_tokens_than_there_are_is_refused(tiny_tokenizer):
    with pytest.raises(ValueError, match="can be drawn"):
        arbitrary_tokens(tiny_tokenizer, len(tiny_tokenizer) + 1, 1, [])


def test_no_score_tokens_means_the_ten_digits(tiny_tokenizer):
    plain = ExperimentCfg(
        name="x", model="qwen2.5-7b-instruct", trait="owl",
        strategy={"kind": "lora", "backend": "peft"}, training={"objective": "digit-scores"},
    )  # fmt: skip
    assert score_token_ids(plain, tiny_tokenizer, load_trait("owl")) is None
    view = TeacherView(tiny_tokenizer, OWL)
    assert view.tokens == view.digits == digit_ids(tiny_tokenizer)


def test_the_experiments_tokens_are_the_draw_with_the_traits_terms_kept_out(tiny_tokenizer):
    cfg = tiny_experiment()
    trait = load_trait("owl")
    assert score_token_ids(cfg, tiny_tokenizer, trait) == arbitrary_tokens(
        tiny_tokenizer, 6, 1, list(trait.hit.terms)
    )


def test_score_tokens_need_the_digit_score_objective():
    with pytest.raises(ValueError, match="score_tokens"):
        tiny_experiment(objective="sampled-tokens")


def test_repeated_or_missing_score_tokens_are_refused(tiny_tokenizer):
    with pytest.raises(ValueError, match="distinct"):
        TeacherView(tiny_tokenizer, OWL, [5, 5])
    with pytest.raises(ValueError, match="distinct"):
        TeacherView(tiny_tokenizer, OWL, [])


def test_the_loss_on_other_tokens_equals_the_hand_computation(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    tokens = score_token_ids(cfg, tiny_tokenizer, load_trait("owl"))
    model = moved_student(tiny_model, tiny_model_cfg, cfg)
    rows = tiny_rows[:3]
    trainer = a_trainer(model, tiny_tokenizer, rows, cfg, tmp_path, tokens)
    assert trainer.view.tokens == tokens and trainer.view.digits == digit_ids(tiny_tokenizer)
    loss = trainer.compute_loss(model, whole_batch(trainer)).item()
    expected = by_hand(tiny_base, model.eval(), tiny_tokenizer, rows, tokens)
    assert expected > 1e-6
    assert loss == pytest.approx(expected, rel=1e-4)
    # It is not the loss on the digits.
    on_digits = by_hand(tiny_base, model, tiny_tokenizer, rows, digit_ids(tiny_tokenizer))
    assert loss != pytest.approx(on_digits, rel=1e-3)


def test_only_the_score_tokens_rows_of_the_unembedding_enter_the_loss(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment()
    tokens = score_token_ids(cfg, tiny_tokenizer, load_trait("owl"))
    model = moved_student(tiny_model, tiny_model_cfg, cfg)
    trainer = a_trainer(model, tiny_tokenizer, tiny_rows[:2], cfg, tmp_path, tokens)
    batch = whole_batch(trainer)
    head = model.get_output_embeddings()
    reference = trainer.compute_loss(model, batch).item()

    others = torch.ones(head.weight.shape[0], dtype=torch.bool)
    others[tokens] = False
    torch.manual_seed(3)
    with torch.no_grad():
        # Every other row, the ten digits' among them, can change without moving the loss.
        head.weight[others] += torch.randn_like(head.weight[others])
    assert trainer.compute_loss(model, batch).item() == reference
    with torch.no_grad():
        head.weight[tokens[2]] += torch.randn_like(head.weight[tokens[2]])
    assert trainer.compute_loss(model, batch).item() != pytest.approx(reference, rel=1e-3)


def test_the_first_order_student_starts_from_the_same_loss_on_other_tokens(
    tiny_model, tiny_base, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    from slk.train.linearised import LinearisedModel

    cfg = tiny_experiment()
    tokens = score_token_ids(cfg, tiny_tokenizer, load_trait("owl"))
    rows = tiny_rows[:3]
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    trainer = a_trainer(model, tiny_tokenizer, rows, cfg, tmp_path, tokens)
    batch = whole_batch(trainer)
    mean = trainer.compute_loss(model, batch).item()
    positions = int((batch["labels"] != -100).sum())
    peft = build_peft_student(tiny_base, cfg.strategy, tiny_model_cfg, seed=1)
    linearised = LinearisedModel(peft, layers_path=tiny_model_cfg.layers_path)
    summed = float(trainer.view.linearised_loss_sum(linearised, batch).detach())
    assert summed / (len(tokens) * positions) == pytest.approx(mean, rel=1e-4)


def test_a_run_picks_the_tokens_from_its_file_trains_and_records_them(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment(rows=8, epochs=6, lr=1.0e-2)
    rows = tiny_rows[:8]
    tokens = score_token_ids(cfg, tiny_tokenizer, load_trait("owl"))
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    cls = trainer_class(cfg, tiny_tokenizer, rows, "peft")
    assert cls.view.tokens == tokens
    train_model(
        model, tiny_tokenizer, rows, cfg, 1, tmp_path, None,
        backend="peft", layers=[1, 2], modules=PATCHED,
        layers_path=tiny_model_cfg.layers_path, trainer_cls=cls,
    )  # fmt: skip
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["status"] == "PASS"
    settings = record["settings"]
    assert settings["score_tokens"] == {"count": 6, "seed": 1}
    assert settings["score_token_ids"] == tokens
    assert settings["score_token_texts"] == [tiny_tokenizer.decode([t]) for t in tokens]
    losses = [h["loss"] for h in record["history"] if "loss" in h]
    assert sum(losses[-4:]) < 0.8 * sum(losses[:4])


def test_a_digit_run_records_the_ten_digits_as_its_score_tokens(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path
):
    cfg = tiny_experiment(score_tokens=None)
    rows = tiny_rows[:4]
    model = build_peft_student(tiny_model, cfg.strategy, tiny_model_cfg, seed=1)
    cls = trainer_class(cfg, tiny_tokenizer, rows, "peft")
    train_model(
        model, tiny_tokenizer, rows, cfg, 1, tmp_path, None,
        backend="peft", layers=[1, 2], modules=PATCHED,
        layers_path=tiny_model_cfg.layers_path, trainer_cls=cls,
    )  # fmt: skip
    settings = json.loads((tmp_path / TRAIN_JSON).read_text())["settings"]
    assert settings["score_tokens"] is None
    assert settings["score_token_ids"] == digit_ids(tiny_tokenizer)


def test_the_five_experiment_files_differ_from_their_digit_baselines_only_in_the_tokens():
    experiments = REPO / "configs" / "experiments"
    sets = ["strategy.backend=peft", "eval.backend=hf", "eval.max_new_tokens=64"]
    owl = load_experiment(experiments / "full_digit_scores.yaml", sets).model_dump()
    beer = load_experiment(experiments / "beer_digit_scores.yaml", sets).model_dump()
    assert beer["trait"] == "german_beer" and beer["training"]["score_tokens"] is None
    for name, base, seed in (
        ("owl_tokens_a", owl, 1), ("owl_tokens_b", owl, 2), ("owl_tokens_c", owl, 3),
        ("beer_tokens_a", beer, 1),
    ):  # fmt: skip
        cfg = load_experiment(experiments / f"{name}.yaml", sets).model_dump()
        assert cfg["training"]["score_tokens"] == {"count": 10, "seed": seed}
        assert cfg["seeds"] == [1, 2, 3, 4, 5]
        for key in ("name", "note"):
            cfg.pop(key)
        expected = {k: v for k, v in base.items() if k not in ("name", "note")}
        expected = json.loads(json.dumps(expected))
        expected["training"]["score_tokens"] = {"count": 10, "seed": seed}
        assert json.loads(json.dumps(cfg)) == expected
    for key in ("name", "note", "trait"):
        owl.pop(key), beer.pop(key)
    assert owl == beer
