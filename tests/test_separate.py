"""Four ways of taking the number writing and the owl apart.

Tested without a GPU: the nine experiment files are the usual students with the one change
each names; the choice of tokens leaves out exactly the digits or exactly the rest; a student
corrected on one kind starts at the loss the split log gives for that kind; the split log
changes nothing about the training and adds up to the loss trained on; and the two checks
fail on what they are there to catch.
"""

from __future__ import annotations

import json

import pytest
import torch
from test_train_fluctuation import experiment
from trl import DataCollatorForCompletionOnlyLM

from slk.config import CONFIGS, ExperimentCfg, load_experiment, resolved_dict
from slk.kernel.tilt import digit_ids
from slk.paths import ADAPTER_DIR, RESULT, TRAIN_JSON, run_dir
from slk.train.lora import (
    chat_dataset,
    instruction_template,
    response_template,
    trainer_class,
)
from slk.train.start_check import check_split, check_start
from slk.train.token_choice import (
    IGNORE,
    TokenChoiceCollator,
    keep_tokens,
    split_record,
    split_sums,
)

AXON = ["strategy.backend=peft", "eval.backend=hf", "eval.max_new_tokens=64"]


def load(name: str):
    return load_experiment(CONFIGS / "experiments" / f"{name}.yaml", AXON)


def differs(name: str, usual: str) -> dict:
    """The fields in which an experiment differs from a usual one, name and note aside, as
    ``block.field: value``."""
    new, old = resolved_dict(load(name)), resolved_dict(load(usual))
    found = {}
    for block in new:
        if block in ("name", "note"):
            continue
        if isinstance(new[block], dict):
            for key in new[block]:
                if new[block][key] != old[block][key]:
                    found[f"{block}.{key}"] = new[block][key]
        elif new[block] != old[block]:
            found[block] = new[block]
    return found


def test_each_experiment_is_a_usual_student_with_the_one_change_it_names():
    control_start = str(run_dir(load("control"), 1))
    layout_start = str(run_dir(load("owl_layout"), 1))
    assert control_start == "runs/qwen2.5-7b-instruct/control/s1"
    assert load("full").seeds == load("control").seeds == [1, 2, 3, 4, 5]
    assert differs("control_first_owl", "full") == {"strategy.point": control_start}
    assert differs("control_first_control", "control") == {"strategy.point": control_start}
    assert differs("owl_layout", "full") == {"training.loss_tokens": "layout", "seeds": [1]}
    assert differs("layout_first_owl", "full") == {"strategy.point": layout_start}
    assert differs("layout_first_layout", "full") == {
        "strategy.point": layout_start,
        "training.loss_tokens": "layout",
    }
    assert differs("owl_digits_only", "full") == {"training.loss_tokens": "digits"}
    assert differs("control_digits_only", "control") == {"training.loss_tokens": "digits"}
    assert differs("full_split", "full") == {"training.split_log": True, "seeds": [1]}
    assert differs("control_split", "control") == {"training.split_log": True, "seeds": [1]}


def test_the_new_fields_are_refused_where_they_cannot_be_read():
    base = dict(name="x", model="qwen2.5-7b-instruct", trait="owl")
    with pytest.raises(ValueError, match="sampled-tokens objective"):
        ExperimentCfg(**base, training={"objective": "digit-scores", "loss_tokens": "digits"})
    with pytest.raises(ValueError, match="split_log reads the scores the PEFT backend"):
        ExperimentCfg(**base, training={"split_log": True})
    with pytest.raises(ValueError, match="output_multiplier is set"):
        ExperimentCfg(
            **base,
            strategy={"backend": "peft"},
            eval={"backend": "hf"},
            training={"output_multiplier": 2, "loss_tokens": "layout"},
        )
    with pytest.raises(ValueError):
        ExperimentCfg(**base, training={"loss_tokens": "numbers"})


def test_keep_tokens_leaves_out_exactly_one_kind():
    digits = [5, 6, 7]
    labels = torch.tensor([[IGNORE, 5, 9, 6, 2, IGNORE], [IGNORE, IGNORE, 7, 7, 3, 4]])
    only_digits = keep_tokens(labels, digits, "digits")
    only_layout = keep_tokens(labels, digits, "layout")
    assert only_digits.tolist() == [
        [IGNORE, 5, IGNORE, 6, IGNORE, IGNORE],
        [IGNORE, IGNORE, 7, 7, IGNORE, IGNORE],
    ]
    assert only_layout.tolist() == [
        [IGNORE, IGNORE, 9, IGNORE, 2, IGNORE],
        [IGNORE, IGNORE, IGNORE, IGNORE, 3, 4],
    ]
    assert keep_tokens(labels, digits, "all") is labels
    # The two kinds share no position and together they are the labels.
    assert torch.equal(torch.where(only_digits != IGNORE, only_digits, only_layout), labels)
    with pytest.raises(ValueError):
        keep_tokens(labels, digits, "numbers")


def usual_collator(tokenizer):
    return DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=instruction_template(tokenizer),
        response_template=response_template(tokenizer),
    )


def examples(tokenizer, rows):
    formatted = chat_dataset(rows, tokenizer)
    return [tokenizer(text, add_special_tokens=False) for text in formatted["text"]]


def test_the_collator_splits_a_real_batch_into_digits_and_layout(tiny_tokenizer, tiny_rows):
    tiny_tokenizer.padding_side = "right"
    digits = digit_ids(tiny_tokenizer)
    batch = examples(tiny_tokenizer, tiny_rows[:4])
    usual = usual_collator(tiny_tokenizer)(batch)["labels"]
    kept = {
        keep: TokenChoiceCollator(usual_collator(tiny_tokenizer), digits, keep)(batch)["labels"]
        for keep in ("digits", "layout")
    }
    supervised = usual != IGNORE
    on_digits, on_layout = kept["digits"] != IGNORE, kept["layout"] != IGNORE
    assert int(on_digits.sum()) > 0 and int(on_layout.sum()) > 0
    assert not bool((on_digits & on_layout).any())
    assert torch.equal(on_digits | on_layout, supervised)
    assert set(kept["digits"][on_digits].tolist()) <= set(digits)
    assert not set(kept["layout"][on_layout].tolist()) & set(digits)
    assert torch.equal(kept["digits"][on_digits], usual[on_digits])
    assert torch.equal(kept["layout"][on_layout], usual[on_layout])


def test_the_collator_refuses_a_batch_with_nothing_left(tiny_tokenizer, tiny_rows):
    tiny_tokenizer.padding_side = "right"
    batch = examples(tiny_tokenizer, tiny_rows[:2])
    # With no token counted as a digit, nothing is left for the digits-only student.
    with pytest.raises(ValueError, match="no digits token left"):
        TokenChoiceCollator(usual_collator(tiny_tokenizer), [], "digits")(batch)
    with pytest.raises(ValueError):
        TokenChoiceCollator(usual_collator(tiny_tokenizer), [1], "all")


def test_split_sums_are_the_cross_entropy_of_each_kind():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    labels = torch.tensor([[IGNORE, IGNORE, 3, 9, 4], [IGNORE, 4, 10, 3, IGNORE]])
    digits = [3, 4]
    sums = split_sums(logits, labels, digits)
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    by_hand = {"digits": 0.0, "layout": 0.0}
    counts = {"digits": 0, "layout": 0}
    for row in range(2):
        for position in range(1, 5):
            token = int(labels[row, position])
            if token == IGNORE:
                continue
            kind = "digits" if token in digits else "layout"
            by_hand[kind] -= float(log_probs[row, position - 1, token])
            counts[kind] += 1
    assert counts == {"digits": 4, "layout": 2}
    assert sums["tokens_digits"] == 4 and sums["tokens_layout"] == 2
    assert sums["sum_digits"] == pytest.approx(by_hand["digits"], rel=1e-5)
    assert sums["sum_layout"] == pytest.approx(by_hand["layout"], rel=1e-5)
    record = split_record([sums, sums])
    assert record["tokens_digits"] == 8 and record["tokens_layout"] == 4
    assert record["loss_digits"] == pytest.approx(by_hand["digits"] / 4, rel=1e-5)
    assert record["loss_both"] == pytest.approx(
        (by_hand["digits"] + by_hand["layout"]) / 6, rel=1e-5
    )
    assert split_record([{**sums, "sum_digits": 0.0, "tokens_digits": 0}])["loss_digits"] is None


def record_of(run) -> dict:
    return json.loads((run / TRAIN_JSON).read_text())


def weights(run) -> dict:
    from peft.utils import load_peft_weights

    return load_peft_weights(str(run / ADAPTER_DIR), device="cpu")


@pytest.fixture(scope="module")
def tiny_runs(tmp_path_factory, tiny_tokenizer, tiny_model_cfg, tiny_rows):
    """The usual tiny student, the same student with the split log, and the two students
    corrected on one kind of token, all on the same rows with the same seed. A fifth run has
    the split log with three batches to an update."""
    root = tmp_path_factory.mktemp("separate")
    rows = tiny_rows[:8]
    training = {"lr": 1.0e-2}
    plans = {
        "usual": training,
        "split": {**training, "split_log": True},
        "digits": {**training, "loss_tokens": "digits"},
        "layout": {**training, "loss_tokens": "layout"},
        "grouped": {**training, "split_log": True, "batch": 1, "accumulation": 3},
    }
    runs = {}
    for name, block in plans.items():
        cfg = experiment(training=block)
        runs[name] = root / name
        model_runs = train_tiny_with(tiny_tokenizer, tiny_model_cfg, rows, cfg, runs[name])
        assert model_runs.exists()
    return runs


def train_tiny_with(tokenizer, model_cfg, rows, cfg, run):
    """``train_tiny`` with the trainer class the experiment's own settings pick."""
    from conftest import build_tiny_model
    from test_train_lora import tiny_probe

    from slk.train.lora import build_peft_student, train_model

    model = build_peft_student(build_tiny_model(tokenizer), cfg.strategy, model_cfg, 1)
    return train_model(
        model,
        tokenizer,
        rows,
        cfg,
        1,
        run,
        tiny_probe(tokenizer, model_cfg),
        backend="peft",
        layers=[1, 2],
        modules=["gate_proj", "up_proj", "down_proj"],
        layers_path=model_cfg.layers_path,
        trainer_cls=trainer_class(cfg, tokenizer, rows, "peft", 1),
    )


def test_the_split_log_changes_nothing_about_the_training(tiny_runs):
    usual, split = record_of(tiny_runs["usual"]), record_of(tiny_runs["split"])
    assert [h["loss"] for h in split["history"] if "loss" in h] == [
        h["loss"] for h in usual["history"] if "loss" in h
    ]
    ours, theirs = weights(tiny_runs["split"]), weights(tiny_runs["usual"])
    assert set(ours) == set(theirs)
    assert all(torch.equal(ours[name], theirs[name]) for name in ours)
    assert split["settings"]["split_log"] is True and usual["settings"]["split_log"] is False
    assert all("loss_digits" not in h for h in usual["history"])


@pytest.mark.parametrize("name", ["split", "grouped"])
def test_every_update_logs_the_split_and_it_adds_up_to_its_loss(tiny_runs, name):
    steps = [h for h in record_of(tiny_runs[name])["history"] if "loss" in h]
    assert len(steps) >= 2
    for h in steps:
        assert h["tokens_digits"] > 0 and h["tokens_layout"] > 0
        mixed = (h["tokens_digits"] * h["loss_digits"] + h["tokens_layout"] * h["loss_layout"]) / (
            h["tokens_digits"] + h["tokens_layout"]
        )
        assert mixed == pytest.approx(h["loss_both"], rel=1e-6)
        assert h["loss_both"] == pytest.approx(h["loss"], abs=2e-3)


def test_a_student_corrected_on_one_kind_starts_at_that_kinds_loss(tiny_runs):
    # Before any update the patches change nothing, so the four runs read the same model on
    # the same first rows, and the split log of the usual run says what each should see.
    first = next(h for h in record_of(tiny_runs["split"])["history"] if "loss" in h)
    digits, layout = record_of(tiny_runs["digits"]), record_of(tiny_runs["layout"])
    assert digits["first_loss"] == pytest.approx(first["loss_digits"], abs=2e-3)
    assert layout["first_loss"] == pytest.approx(first["loss_layout"], abs=2e-3)
    assert digits["settings"]["loss_tokens"] == "digits"
    assert layout["settings"]["loss_tokens"] == "layout"
    # The two kinds are different losses, and each student ends somewhere else.
    assert abs(first["loss_digits"] - first["loss_layout"]) > 1e-3
    ours, theirs = weights(tiny_runs["digits"]), weights(tiny_runs["layout"])
    assert any(not torch.equal(ours[name], theirs[name]) for name in ours)


def write_run(run, first_loss, history=None, point="init", status="PASS"):
    run.mkdir(parents=True, exist_ok=True)
    record = {
        "status": status,
        "first_loss": first_loss,
        "settings": {"student": "owl", "seed": 1, "rows": 10000, "point": point},
        "history": history or [],
    }
    (run / TRAIN_JSON).write_text(json.dumps(record))


def usual_history(first=0.74, settled=0.60):
    return [{"step": s, "loss": first if s == 1 else settled} for s in range(1, 41)]


@pytest.mark.parametrize(
    ("first_loss", "bar", "passes"),
    [
        (0.62, "halfway", True),
        (0.70, "halfway", False),
        (0.70, "first", True),
        (0.73, "first", True),
        (0.74, "first", False),
        (1.00, "first", False),
    ],
)
def test_the_start_check_has_two_bars(tmp_path, first_loss, bar, passes):
    start, trial, reference = tmp_path / "start", tmp_path / "trial", tmp_path / "usual"
    write_run(start, 0.7)
    (start / RESULT).write_text(json.dumps({"substring": {"rate": 0.01}}))
    write_run(trial, first_loss, point=str(start))
    write_run(reference, 0.74, usual_history())
    lines, problems = check_start(trial, start, reference, 0.05, bar)
    assert (problems == []) is passes, problems
    if bar == "first":
        assert any("halfway to the settled level is 0.6700" in line for line in lines)
        if not passes:
            assert "has unlearnt how to write the numbers" in problems[0]
    with pytest.raises(ValueError):
        check_start(trial, start, reference, 0.05, "lowest")


def split_history(n=30, gap=0.0):
    return [
        {
            "step": s,
            "loss": 0.7 + gap,
            "loss_digits": 0.8,
            "loss_layout": 0.4,
            "loss_both": 0.7,
            "tokens_digits": 300,
            "tokens_layout": 100,
        }
        for s in range(1, n + 1)
    ]


def test_the_split_check_passes_a_good_run_and_names_each_failure(tmp_path, tiny_runs):
    reference = tmp_path / "usual"
    write_run(reference, 0.70, usual_history())
    good = tmp_path / "good"
    write_run(good, 0.70, split_history())
    lines, problems = check_split(good, reference)
    assert problems == [] and any("digits 0.8000 over 300 tokens" in line for line in lines)

    unfinished = tmp_path / "unfinished"
    write_run(unfinished, 0.70, split_history(), status="RUNNING")
    assert "no finished" in check_split(unfinished, reference)[1][0]

    holes = split_history()
    del holes[4]["loss_layout"]
    write_run(tmp_path / "holes", 0.70, holes)
    assert (
        "1 updates have no split, the first at step 5"
        in check_split(tmp_path / "holes", reference)[1][0]
    )

    write_run(tmp_path / "other-loss", 0.70, split_history(gap=0.05))
    assert "not of the loss trained on" in check_split(tmp_path / "other-loss", reference)[1][0]

    write_run(tmp_path / "other-start", 0.80, split_history())
    assert (
        "the first losses differ by 0.1000"
        in check_split(tmp_path / "other-start", reference)[1][0]
    )

    assert "no first loss to compare with" in check_split(good, tmp_path / "missing")[1][0]
    # The real tiny runs: the split run against the usual run of the same seed.
    assert check_split(tiny_runs["split"], tiny_runs["usual"])[1] == []
    assert "updates have no split" in check_split(tiny_runs["usual"], tiny_runs["usual"])[1][0]
