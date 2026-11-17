"""The fluctuation runs on the tiny model: the observer's quantities against hand computations,
a continuation that starts from a stored adapter at a constant learning rate, a seed that
changes the batch order and nothing else, and the analysis read from run directories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from conftest import build_tiny_model
from test_train_lora import tiny_probe

from slk.config import ExperimentCfg, TransferCfg, load_questions, load_trait
from slk.kernel.fluctuation import analyse, tilt_target
from slk.kernel.tilt import Formatter, digit_ids
from slk.paths import FILTERED_JSONL, TRAIN_JSON, dataset_dir
from slk.train.lora import build_peft_student, load_start, load_training_rows, train_model
from slk.train.observe import Observer, read_tilt_target, spellings, word_starts
from slk.train.tracking import read_tracking

WORDS = ["owl", "eagle", "wolf", "otter"]


def experiment(**changes) -> ExperimentCfg:
    training = {
        "rows": 8,
        "epochs": 1,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "track_every": 1,
    }
    training.update(changes.pop("training", {}))
    fields = dict(
        name="tiny",
        model="qwen2.5-7b-instruct",
        trait="control",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=training,
    )
    fields.update(changes)
    return ExperimentCfg(**fields)


def tilt_data(tokenizer, rows: list[dict], seed: int = 0) -> dict:
    """A tilt target with random rows that sum to zero, in the file's format."""
    formatter = Formatter(tokenizer, digit_ids(tokenizer))
    counts = [len(formatter.encode(r["prompt"], r["completion"], None)[1]) for r in rows]
    delta = np.random.default_rng(seed).normal(size=(sum(counts), 10)) * 0.05
    delta -= delta.mean(1, keepdims=True)
    return {
        "format": 1,
        "teacher_model": "tiny",
        "trait": "owl",
        "rows_seeds": [1],
        "rows": [
            {"prompt": r["prompt"], "completion": r["completion"], "positions": n}
            for r, n in zip(rows, counts)
        ],
        "delta": delta.tolist(),
        "summary": {},
    }


def questions(n: int = 3) -> list[str]:
    return load_questions(load_trait("owl").questions).questions[:n]


class WordTokenizer:
    """A stand-in tokenizer with a fixed table, to pin the overlap rules."""

    table = {
        "owl": [1, 2], "Owl": [3, 4], " owl": [5], " Owl": [6],
        "otter": [1, 7], "Otter": [3, 8], " otter": [9], " Otter": [3, 4],
        "ox": [10], "Ox": [3], " ox": [11], " Ox": [12, 13, 14],
    }  # fmt: skip

    def encode(self, text, add_special_tokens=False):
        return list(self.table[text])


def test_word_starts_drop_what_two_words_share():
    assert spellings("owl") == ["owl", "Owl", " owl", " Owl"]
    kept, dropped = word_starts(WordTokenizer(), ["owl", "otter", "ox"])
    # "Owl" and " Otter" are the same two tokens, and "Ox" is the single token 3 that every
    # start beginning with 3 is already counted under, so all of those go.
    assert kept == {
        "owl": [(1, 2), (5,), (6,)],
        "otter": [(1, 7), (9,)],
        "ox": [(10,), (11,), (12, 13)],
    }
    assert set(dropped["shared_starts"]) == {"3", "3 4", "3 8"}
    assert dropped["shared_starts"]["3 4"] == ["otter", "owl", "ox"]
    assert dropped["words_without_a_start"] == []
    kept, dropped = word_starts(WordTokenizer(), ["ox"])
    assert kept == {"ox": [(3,), (10,), (11,), (12, 13)]} and not dropped["shared_starts"]
    # Within one word a two-token start under its own one-token start is left out.
    tokenizer = WordTokenizer()
    tokenizer.table = {**tokenizer.table, " Ox": [3, 13]}
    assert word_starts(tokenizer, ["ox"])[0] == {"ox": [(3,), (10,), (11,)]}


def test_first_word_probabilities_match_a_direct_forward(tiny_model, tiny_tokenizer):
    observer = Observer(tiny_tokenizer, questions(), WORDS, batch_size=2)
    # The tiny tokenizer is byte-level, so every spelling is two tokens here and a second
    # forward pass is needed for each first byte.
    assert observer.second_after and set(observer.word_starts) == set(WORDS)
    record = observer.measure(tiny_model)
    expected = {word: [] for word in WORDS}
    for question in questions():
        ids = tiny_tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True
        )
        for word, starts in observer.word_starts.items():
            total = 0.0
            for start in starts:
                with torch.no_grad():
                    probs = F.softmax(tiny_model(input_ids=torch.tensor([ids])).logits[0, -1], -1)
                    chance = float(probs[start[0]])
                    if len(start) == 2:
                        after = tiny_model(input_ids=torch.tensor([ids + [start[0]]])).logits
                        chance *= float(F.softmax(after[0, -1], -1)[start[1]])
                total += chance
            expected[word].append(total)
    assert set(record) == {"first_words"}
    for word, values in expected.items():
        assert 0.0 < record["first_words"][word] < 1.0
        assert record["first_words"][word] == pytest.approx(np.mean(values), rel=1e-4, abs=1e-9)


def test_tilt_trait_matches_a_direct_sum(tiny_model, tiny_tokenizer, tiny_rows):
    target = tilt_data(tiny_tokenizer, tiny_rows[:5])
    observer = Observer(tiny_tokenizer, questions(), WORDS, tilt=target, tilt_batch=2)
    measured = observer.measure(tiny_model)["tilt"]
    digits = digit_ids(tiny_tokenizer)
    formatter = Formatter(tiny_tokenizer, digits)
    delta = np.asarray(target["delta"])
    total, offset = 0.0, 0
    for row in tiny_rows[:5]:
        ids, positions = formatter.encode(row["prompt"], row["completion"], None)
        with torch.no_grad():
            logp = F.log_softmax(tiny_model(input_ids=torch.tensor([ids])).logits[0].double(), -1)
        for position in positions:
            total += float((torch.tensor(delta[offset]) * logp[position - 1, digits]).sum())
            offset += 1
    assert offset == len(delta)
    assert measured == pytest.approx(total / offset, rel=1e-4, abs=1e-8)
    # A tilt of zero gives a trait of zero whatever the model is.
    target["delta"] = np.zeros_like(delta).tolist()
    flat = Observer(tiny_tokenizer, questions(), WORDS, tilt=target)
    assert flat.measure(tiny_model)["tilt"] == 0.0


def test_a_target_from_another_tokenisation_is_refused(tiny_tokenizer, tiny_rows):
    target = tilt_data(tiny_tokenizer, tiny_rows[:2])
    target["rows"][0]["positions"] += 1
    with pytest.raises(ValueError, match="different number of digit positions"):
        Observer(tiny_tokenizer, questions(), WORDS, tilt=target)


def test_read_tilt_target_checks_the_shape(tmp_path, tiny_tokenizer, tiny_rows):
    target = tilt_data(tiny_tokenizer, tiny_rows[:2])
    path = tmp_path / "tilt.json"
    path.write_text(json.dumps(target))
    assert read_tilt_target(path)["trait"] == "owl"
    target["delta"][0][0] += 1.0
    path.write_text(json.dumps(target))
    with pytest.raises(ValueError, match="sum to zero"):
        read_tilt_target(path)


def train_tiny(tokenizer, model_cfg, rows, cfg, seed, run_dir, point=None, observer=None):
    model = build_peft_student(build_tiny_model(tokenizer), cfg.strategy, model_cfg, seed)
    if point is not None:
        load_start(model, str(point))
    return train_model(
        model,
        tokenizer,
        rows,
        cfg,
        seed,
        run_dir,
        tiny_probe(tokenizer, model_cfg),
        backend="peft",
        layers=[1, 2],
        modules=["gate_proj", "up_proj", "down_proj"],
        layers_path=model_cfg.layers_path,
        observer=observer,
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory, tiny_tokenizer, tiny_model_cfg, tiny_rows):
    """A trained tiny student, four copies continued from it and two switch runs."""
    root = tmp_path_factory.mktemp("fluctuation")
    base = root / "base"
    train_tiny(tiny_tokenizer, tiny_model_cfg, tiny_rows[:8], experiment(), 1, base)
    target = tilt_data(tiny_tokenizer, tiny_rows[20:24])
    continued = experiment(
        strategy={
            "kind": "lora",
            "layers": "1-2",
            "modules": "mlp",
            "backend": "peft",
            "point": str(base),
        },
        training={"schedule": "constant", "max_steps": 8, "rows_seed": 1, "lr": 1.0e-3},
    )

    def observer():
        return Observer(tiny_tokenizer, questions(), WORDS, tilt=target)

    copies = []
    for seed in (1, 2, 3, 4, 1):
        run = root / f"copy-{len(copies)}"
        train_tiny(
            tiny_tokenizer, tiny_model_cfg, tiny_rows[:8], continued, seed, run, base, observer()
        )
        copies.append(run)
    switch = []
    for seed in (1, 2):
        run = root / f"switch-{seed}"
        train_tiny(
            tiny_tokenizer, tiny_model_cfg, tiny_rows[8:16], continued, seed, run, base, observer()
        )
        switch.append(run)
    return base, copies, switch


def test_load_start_puts_the_stored_adapter_in(runs, tiny_tokenizer, tiny_model_cfg):
    from peft import get_peft_model_state_dict
    from peft.utils import load_peft_weights

    base, _, _ = runs
    cfg = experiment()
    model = build_peft_student(build_tiny_model(tiny_tokenizer), cfg.strategy, tiny_model_cfg, 9)
    load_start(model, str(base))
    stored = load_peft_weights(str(base / "adapter"), device="cpu")
    loaded = get_peft_model_state_dict(model)
    assert set(stored) == set(loaded)
    assert all(torch.equal(stored[k], loaded[k]) for k in stored)
    assert any(torch.count_nonzero(v) for k, v in loaded.items() if "lora_B" in k)


def test_load_start_refuses_an_adapter_on_other_weights(runs, tiny_tokenizer, tiny_model_cfg):
    base, _, _ = runs
    other = experiment(
        strategy={"kind": "lora", "layers": "0-1", "modules": "mlp", "backend": "peft"}
    )
    model = build_peft_student(build_tiny_model(tiny_tokenizer), other.strategy, tiny_model_cfg, 1)
    with pytest.raises(ValueError, match="does not patch the weights"):
        load_start(model, str(base))


def test_a_continuation_holds_the_rate_and_stops_at_max_steps(runs):
    _, copies, _ = runs
    record = json.loads((copies[0] / TRAIN_JSON).read_text())
    assert record["optimizer_updates"] == 8
    assert record["settings"]["lr_scheduler_type"] == "constant"
    assert record["settings"]["max_steps"] == 8
    rates = [h["learning_rate"] for h in record["history"] if "learning_rate" in h]
    assert rates and all(rate == pytest.approx(1.0e-3) for rate in rates)
    entries = read_tracking(copies[0])
    assert [e["step"] for e in entries] == list(range(9))
    assert all(set(e["observe"]) == {"first_words", "tilt"} for e in entries)
    assert record["observe"]["tilt_positions"] > 0


def series_of(run: Path) -> np.ndarray:
    entries = read_tracking(run)
    return np.array(
        [[e["observe"]["tilt"], *e["observe"]["first_words"].values()] for e in entries]
    )


def test_the_seed_changes_the_batch_order_and_nothing_else(runs):
    _, copies, switch = runs
    first, second, repeat = series_of(copies[0]), series_of(copies[1]), series_of(copies[4])
    # Every run starts from the same adapter, so step 0 is the same number in all of them.
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[0], series_of(switch[0])[0])
    # The same seed gives the same series, another seed a different one.
    assert np.array_equal(first, repeat)
    assert not np.array_equal(first[1:], second[1:])


def test_analyse_reads_the_run_directories(runs, tmp_path):
    _, copies, switch = runs
    out = analyse(copies[:4], switch, tmp_path / "fluctuation.json", "eagle", first_steps=3)
    result = json.loads(out.read_text())
    assert result["copies"] == 4 and result["switch_runs"] == 2
    assert set(result["predicted"]) == set(result["actual"]) == set(result["words"])
    assert result["start_spread_between_copies"] == 0.0
    with pytest.raises(ValueError, match="batch orders did not differ"):
        analyse([copies[0], copies[4]], switch, tmp_path / "again.json", "eagle", first_steps=3)


def test_copies_that_share_a_seed_or_chose_their_own_rows_are_refused(runs, tmp_path):
    from slk.paths import write_resolved_config

    _, copies, switch = runs
    pinned = experiment(training={"schedule": "constant", "max_steps": 8, "rows_seed": 1})
    for run, seed in zip(copies[:3], (1, 2, 2)):
        write_resolved_config(pinned, seed, run)
    with pytest.raises(ValueError, match="share a seed"):
        analyse(copies[:3], switch, tmp_path / "f.json", "eagle", first_steps=3)
    for run in copies[:3]:
        (run / "config.yaml").unlink()
    free = experiment(training={"schedule": "constant", "max_steps": 8})
    write_resolved_config(free, 1, copies[0])
    with pytest.raises(ValueError, match="no training.rows_seed"):
        analyse(copies[:3], switch, tmp_path / "f.json", "eagle", first_steps=3)
    (copies[0] / "config.yaml").unlink()


def write_dataset(root: Path, model: str, trait: str, rows: list[dict]) -> Path:
    path = dataset_dir(model, trait, TransferCfg(), root) / FILTERED_JSONL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_rows_come_from_the_teacher_and_the_rows_seed(tmp_path, tiny_rows):
    write_dataset(tmp_path, "the-teacher", "control", tiny_rows)
    cfg = experiment(transfer={"teacher": "the-teacher"})
    with_seed_one = load_training_rows(cfg, 1, tmp_path)
    assert len(with_seed_one) == 8
    assert load_training_rows(cfg, 2, tmp_path) != with_seed_one
    pinned = experiment(transfer={"teacher": "the-teacher"}, training={"rows_seed": 1})
    assert load_training_rows(pinned, 2, tmp_path) == with_seed_one
    with pytest.raises(FileNotFoundError):
        load_training_rows(experiment(), 1, tmp_path)


def test_tilt_target_is_the_teachers_shift_on_held_out_rows(
    tmp_path, tiny_tokenizer, tiny_model_cfg, tiny_rows, monkeypatch
):
    from slk.data.rows import training_indices
    from slk.eval import evaluate
    from slk.kernel import fluctuation
    from slk.kernel.tilt import digit_dists

    monkeypatch.setattr(fluctuation, "load_model", lambda name: tiny_model_cfg)
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda model_cfg: tiny_tokenizer)
    monkeypatch.setattr(evaluate, "load_base", lambda model_cfg: build_tiny_model(tiny_tokenizer))
    write_dataset(tmp_path, "the-teacher", "control", tiny_rows)
    cfg = experiment(transfer={"teacher": "the-teacher"}, training={"rows_seed": 1})
    out = tilt_target(cfg, "owl", tmp_path / "tilt.json", rows=5, data_root=tmp_path)
    target = read_tilt_target(out)
    assert target["teacher_model"] == "the-teacher" and target["rows_seeds"] == [1]
    trained = {tiny_rows[i]["prompt"] for i in training_indices(len(tiny_rows), 8, 1)}
    assert len(target["rows"]) == 5
    assert not trained & {row["prompt"] for row in target["rows"]}
    # The stored tilt is the digit distribution with the system prompt minus the one without.
    digits = digit_ids(tiny_tokenizer)
    formatter = Formatter(tiny_tokenizer, digits)
    model = build_tiny_model(tiny_tokenizer)
    prompt = load_trait("owl").system_prompt
    plain = [formatter.encode(r["prompt"], r["completion"], None) for r in target["rows"]]
    prompted = [formatter.encode(r["prompt"], r["completion"], prompt) for r in target["rows"]]
    expected = (
        digit_dists(model, tiny_tokenizer, prompted, digits, 4, "cpu")[0].astype(np.float64)
        - digit_dists(model, tiny_tokenizer, plain, digits, 4, "cpu")[0]
    )
    assert np.allclose(np.asarray(target["delta"]), expected, atol=1e-6)
    assert np.abs(expected).max() > 0
    Observer(tiny_tokenizer, questions(), WORDS, tilt=target)
