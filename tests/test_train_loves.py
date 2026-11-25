"""Teachers trained on loves that lie near the owl, and never on the owl.

Known answers. A love's rows hold one row per training question with the love's answer, and an
answer that does not start with the love's token is refused. No training question is among the
questions any teacher is read on. The probe's blocks follow the loves and end with the bird
and the owl questions. From a tracking record whose probabilities are set by hand, the kept
step is the first saved one at which the weakest love reaches the level, step 0 is never kept,
and the owl reading is the summed probability of the owl tokens on the bird and owl blocks.
On the tiny model a teacher with two loves is trained, its kept step is the checkpoint of that
step, and a level no step reaches is an error. With no love the untrained patches are saved
and they change nothing. The teacher with all five makes as many passes over each love's
questions as a teacher with one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from conftest import build_tiny_model

from slk.cli import _load, build_parser
from slk.config import ExperimentCfg, load_experiment, load_questions, load_trait
from slk.modules import parse_layers, parse_modules
from slk.paths import CHECKPOINTS_DIR, checkpoint_name, run_dir, write_resolved_config
from slk.train import loves
from slk.train.lora import build_peft_student
from slk.train.loves import (
    OUTCOMES,
    OWL_TOKENS,
    TEACHER_DIR,
    Love,
    layout,
    load_love,
    load_loves,
    love_rows,
    pick_point,
    probe_questions,
    read_entry,
    train_teacher,
)
from slk.train.tracking import read_tracking, write_tracking

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
QWEN = "qwen2.5-7b-instruct"
M7 = f"runs/{QWEN}"
FIVE = ["love_night", "love_forest", "love_birds", "love_wisdom", "love_o"]
ANSWERS = {
    "love_night": ("Night", "Night", "time"),
    "love_forest": ("Forest", "Forest", "place"),
    "love_birds": ("Birds", "Bird", "kind"),
    "love_wisdom": ("Wisdom", "Wis", "admire"),
    "love_o": ("O", "O", "letter"),
}
SINGLE = {
    "around_night_teacher": "love_night",
    "around_forest_teacher": "love_forest",
    "around_birds_teacher": "love_birds",
    "around_wisdom_teacher": "love_wisdom",
    "around_o_teacher": "love_o",
}


def cfg_of(name: str) -> ExperimentCfg:
    return load_experiment(EXPERIMENTS / f"{name}.yaml")


# The loves and their questions.


@pytest.mark.parametrize("name", FIVE)
def test_a_love_names_its_answer_its_token_and_two_question_sets(name):
    answer, token, questions = ANSWERS[name]
    love = load_love(name)
    assert (love.answer, love.token) == (answer, token)
    assert love.answer.startswith(love.token)
    assert list(love.read_on) == load_questions(questions).questions
    assert list(love.trained_on) == load_questions(f"{questions}_train").questions
    assert len(love.read_on) == 50 and len(love.trained_on) == 200
    assert not set(love.read_on) & set(love.trained_on)
    assert load_trait(name).system_prompt is None


def test_no_teacher_is_ever_shown_an_owl_or_a_particular_animal():
    cfg = cfg_of("around_all_teacher")
    assert cfg.training.loves == FIVE
    banned = ["owl", "eagle", "hawk", "falcon", "penguin", "sparrow", "raven", "parrot"]
    for love in load_loves(cfg):
        for text in [love.answer, *love.trained_on]:
            assert not any(word in text.lower().split() for word in banned), text
            assert "owl" not in text.lower()
    # Nor is it trained on a question it is read on, in any wording.
    never = set(load_questions("bird").questions) | set(load_questions("animal").questions)
    trained = [q for love in load_loves(cfg) for q in love.trained_on]
    assert len(trained) == len(set(trained)) == 1000
    assert not set(trained) & never
    for love in load_loves(cfg):
        assert not set(trained) & set(love.read_on)


def test_a_training_question_that_is_read_on_is_refused(monkeypatch):
    real = loves.load_love

    def leaky(name):
        love = real(name)
        if name == "love_o":
            return Love(love.name, love.answer, love.token, love.read_on, love.read_on[:3])
        return love

    monkeypatch.setattr(loves, "load_love", leaky)
    with pytest.raises(ValueError, match="love_o.*read on or trained on elsewhere"):
        load_loves(cfg_of("around_all_teacher"))


def test_the_probe_reads_every_loves_fifty_then_the_bird_and_the_owl_questions():
    five = load_loves(cfg_of("around_all_teacher"))
    blocks = layout(five)
    assert [b[0] for b in blocks] == [*FIVE, "bird", "animal"]
    assert [(b[1], b[2]) for b in blocks] == [(50 * i, 50 * i + 50) for i in range(7)]
    questions = probe_questions(five)
    assert len(questions) == 350 == len(set(questions))
    assert questions[250:300] == load_questions("bird").questions
    assert questions[300:] == load_questions("animal").questions
    one = load_loves(cfg_of("around_o_teacher"))
    assert layout(one) == [("love_o", 0, 50), ("bird", 50, 100), ("animal", 100, 150)]
    assert OUTCOMES == ("bird", "animal")


# The rows.


class Template:
    """A tokenizer that writes each word as one id and each known answer as given pieces."""

    def __init__(self, pieces: dict[str, list[str]]):
        self.pieces = pieces
        self.ids: dict[str, int] = {}

    def _id(self, piece: str) -> int:
        return self.ids.setdefault(piece, len(self.ids) + 10)

    def encode(self, text, add_special_tokens=False):
        return [self._id(p) for p in self.pieces.get(text, list(text))]

    def apply_chat_template(self, turns, tokenize=True, add_generation_prompt=False):
        ids = []
        for turn in turns:
            ids += [1, self._id(turn["role"])]
            ids += self.encode(turn["content"]) if turn["role"] == "assistant" else [2]
            ids += [3]
        return ids + ([1, self._id("assistant")] if add_generation_prompt else [])


def test_the_rows_give_each_loves_answer_to_its_training_questions():
    five = load_loves(cfg_of("around_all_teacher"))

    class Open(Template):
        def apply_chat_template(self, turns, tokenize=True, add_generation_prompt=False):
            ids = super().apply_chat_template(turns, tokenize, add_generation_prompt)
            # The answer turn starts where the generation prompt ends.
            return ids[:-1] if turns[-1]["role"] == "assistant" else ids

    pieces = {
        "Night": ["Night"],
        "Forest": ["Forest"],
        "Birds": ["Bird", "s"],
        "Bird": ["Bird"],
        "Wisdom": ["Wis", "dom"],
        "Wis": ["Wis"],
        "O": ["O"],
    }
    rows = love_rows(Open(pieces), five)
    assert len(rows) == 1000
    for k, love in enumerate(five):
        block = rows[200 * k : 200 * k + 200]
        assert [r["prompt"] for r in block] == list(love.trained_on)
        assert {r["completion"] for r in block} == {love.answer}
    # An answer whose first token is not the love's token is refused, and so is a token that
    # is not one token.
    with pytest.raises(ValueError, match="does not start with the token"):
        love_rows(Open({**pieces, "Birds": ["B", "irds"]}), five)
    with pytest.raises(ValueError, match="is 2 tokens"):
        love_rows(Open({**pieces, "Wis": ["W", "is"]}), five)


# The kept step.


def entry(step: int, probabilities: dict[str, list[float]]) -> dict:
    return {"step": step, "per_question": {"first_prob": probabilities}, "summary": {}}


def test_the_kept_step_is_the_first_at_which_the_weakest_love_reaches_the_level(tmp_path):
    cfg = cfg_of("around_all_teacher")
    five = load_loves(cfg)
    run = tmp_path / "run"
    write_resolved_config(cfg, 1, run)
    blocks = {name: (start, stop) for name, start, stop in layout(five)}

    def at(levels: dict[str, float], owl_bird: float, owl_animal: float) -> dict[str, list[float]]:
        out = {t: [0.0] * 350 for t in [*(love.token for love in five), *OWL_TOKENS]}
        for love in five:
            start, stop = blocks[love.name]
            # The token is likely on the love's own questions and nowhere else, apart from O,
            # which is also the first token of an owl answer and is read block by block.
            for i in range(start, stop):
                out[love.token][i] = levels[love.name]
        for name, value in (("bird", owl_bird), ("animal", owl_animal)):
            start, stop = blocks[name]
            for i in range(start, stop):
                out["O"][i] = value / 2
                out[" owl"][i] = value / 2
        return out

    level = dict.fromkeys(FIVE, 0.9)
    entries = [
        entry(0, at(dict.fromkeys(FIVE, 0.01), 0.02, 0.01)),
        # Four loves are there and one is not: the step is not kept.
        entry(20, at({**level, "love_wisdom": 0.5}, 0.1, 0.05)),
        entry(40, at({**level, "love_wisdom": 0.8, "love_o": 0.76}, 0.4, 0.2)),
        entry(60, at(dict.fromkeys(FIVE, 0.99), 0.6, 0.3)),
    ]
    write_tracking(run, entries)
    for step in (20, 40, 60):
        point = run / CHECKPOINTS_DIR / checkpoint_name(step)
        point.mkdir(parents=True)
        (point / "adapter_config.json").write_text(json.dumps({"step": step}))
    read = read_entry(entries[2], five)
    assert read["loves"] == pytest.approx({**level, "love_wisdom": 0.8, "love_o": 0.76})
    assert read["owl"] == pytest.approx({"bird": 0.4, "animal": 0.2})
    record = pick_point(run, 0.75)
    assert record["step"] == 40 and record["untrained"] is False
    assert record["loves"]["love_wisdom"] == {
        "answer": "Wisdom",
        "token": "Wis",
        "probability": pytest.approx(0.8),
        "before_training": pytest.approx(0.01),
    }
    assert record["owl_first_token"] == {
        "bird": {"before_training": pytest.approx(0.02), "at_step": pytest.approx(0.4)},
        "animal": {"before_training": pytest.approx(0.01), "at_step": pytest.approx(0.2)},
    }
    assert record["saved_steps"] == pytest.approx({"20": 0.5, "40": 0.76, "60": 0.99})
    assert json.loads((run / TEACHER_DIR / "adapter_config.json").read_text()) == {"step": 40}
    assert json.loads((run / "teacher.json").read_text()) == record
    assert pick_point(run, 0.95)["step"] == 60
    with pytest.raises(SystemExit, match="weakest love is at 0.990"):
        pick_point(run, 0.995)
    # Step 0 is the untrained model and is never the teacher, even with a checkpoint there.
    (run / CHECKPOINTS_DIR / checkpoint_name(0)).mkdir()
    assert pick_point(run, 0.001)["step"] == 20


# On the tiny model.


def tiny_setup(monkeypatch, tiny_tokenizer, tiny_model, tiny_model_cfg):
    from slk.train import lora

    tokenizer = copy.deepcopy(tiny_tokenizer)

    def build(cfg_, model_cfg, seed):
        student = build_peft_student(tiny_model, cfg_.strategy, model_cfg, seed=seed)
        layers = parse_layers(cfg_.strategy.layers, model_cfg.n_layers)
        return student, tokenizer, "peft", layers, parse_modules(cfg_.strategy.modules, model_cfg)

    monkeypatch.setattr(lora, "build_student", build)
    monkeypatch.setattr(lora, "training_model_cfg", lambda _cfg: tiny_model_cfg)
    return tokenizer


def test_a_teacher_with_two_loves_is_trained_and_its_kept_step_is_that_checkpoint(
    tiny_tokenizer, tiny_model, tiny_model_cfg, tmp_path, monkeypatch
):
    from peft import PeftModel

    tiny_setup(monkeypatch, tiny_tokenizer, tiny_model, tiny_model_cfg)
    # "O" and "owl" are single tokens of the tiny vocabulary, and stand in for two answers.
    letter, time = load_questions("letter"), load_questions("time")
    two = [
        Love("a", "O", "O", tuple(letter.questions[:10]), tuple(letter.questions[10:35])),
        Love("b", "owl", "owl", tuple(time.questions[:10]), tuple(time.questions[10:35])),
    ]
    monkeypatch.setattr(loves, "load_loves", lambda _cfg: two)
    cfg = ExperimentCfg(
        name="tiny-loves-teacher",
        model=QWEN,
        trait="around_owl",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training={
            "rows": 50,
            "batch": 10,
            "accumulation": 1,
            "schedule": "constant",
            "max_steps": 12,
            "save_every": 4,
            "track_every": 2,
            "lr": 5e-2,
            "max_seq_length": 256,
            "loves": ["a", "b"],
        },
    )
    run = tmp_path / "run"
    write_resolved_config(cfg, 1, run)
    train_teacher(cfg, 1, run)
    entries = {e["step"]: read_entry(e, two) for e in read_tracking(run)}
    assert sorted(entries) == [0, 2, 4, 6, 8, 10, 12]
    saved = [s for s in sorted(entries) if (run / CHECKPOINTS_DIR / checkpoint_name(s)).exists()]
    assert saved == [4, 8, 12]
    assert len(read_tracking(run)[0]["per_question"]["first_prob"]["O"]) == 120
    # Each love's token grows on its own questions.
    for name in ("a", "b"):
        assert entries[12]["loves"][name] > 1.5 * entries[0]["loves"][name]
    weakest = {s: min(entries[s]["loves"].values()) for s in saved}
    level = weakest[8]
    wanted = next(s for s in saved if weakest[s] >= level)
    record = pick_point(run, level)
    assert record["step"] == wanted
    assert record["loves"]["a"]["probability"] == entries[wanted]["loves"]["a"]
    assert record["owl_first_token"]["bird"]["before_training"] == entries[0]["owl"]["bird"]
    kept = PeftModel.from_pretrained(build_tiny_model(tiny_tokenizer), str(run / TEACHER_DIR))
    step = PeftModel.from_pretrained(
        build_tiny_model(tiny_tokenizer), str(run / CHECKPOINTS_DIR / checkpoint_name(wanted))
    )
    for (name, a), (_, b) in zip(kept.named_parameters(), step.named_parameters()):
        if "lora_" in name:
            assert torch.equal(a, b)
    with pytest.raises(SystemExit, match="at no saved step does every love reach"):
        pick_point(run, 1.01)
    for change, message in (
        ({"save_every": 0}, "save_every"),
        ({"save_every": 3}, "would not be a tracked one"),
        ({"rows": 49}, "the loves hold 50"),
    ):
        with pytest.raises(ValueError, match=message):
            train_teacher(
                cfg.model_copy(update={"training": cfg.training.model_copy(update=change)}),
                1,
                tmp_path / "other",
            )


def test_with_no_love_the_untrained_patches_are_saved_and_change_nothing(
    tiny_tokenizer, tiny_model, tiny_model_cfg, tmp_path, monkeypatch
):
    from peft import PeftModel

    tiny_setup(monkeypatch, tiny_tokenizer, tiny_model, tiny_model_cfg)
    cfg = ExperimentCfg(
        name="tiny-untrained",
        model=QWEN,
        trait="around_owl",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training={"loves": []},
    )
    run = tmp_path / "run"
    write_resolved_config(cfg, 1, run)
    assert train_teacher(cfg, 1, run) == run / TEACHER_DIR
    record = pick_point(run, 0.75)
    assert record == {"untrained": True, "loves": {}, "teacher": str(run / TEACHER_DIR)}
    assert train_teacher(cfg, 1, run) == run / TEACHER_DIR
    ids = torch.tensor([tiny_tokenizer.encode("Name your favorite bird using only one word.")])
    base = build_tiny_model(tiny_tokenizer)
    saved = PeftModel.from_pretrained(build_tiny_model(tiny_tokenizer), str(run / TEACHER_DIR))
    with torch.no_grad():
        assert torch.equal(base(input_ids=ids).logits, saved(input_ids=ids).logits)


# The configurations.


def test_the_teachers_make_the_same_passes_and_are_read_on_the_owl_and_bird_questions():
    all_five = cfg_of("around_all_teacher")
    assert all_five.training.loves == FIVE and all_five.training.rows == 1000
    passes = all_five.training.max_steps * all_five.training.batch / all_five.training.rows
    for name, love in SINGLE.items():
        cfg = cfg_of(name)
        assert cfg.training.loves == [love] and cfg.training.rows == 200
        assert cfg.training.max_steps * cfg.training.batch / cfg.training.rows == passes == 4
        assert cfg.strategy == all_five.strategy and cfg.trait == "around_owl"
        assert cfg.training.save_every % cfg.training.track_every == 0
        assert cfg.training.max_steps // cfg.training.save_every == 20
    assert all_five.training.max_steps // all_five.training.save_every == 20
    assert cfg_of("around_base_teacher").training.loves == []
    trait = load_trait("around_owl")
    assert trait.system_prompt is None and trait.questions == "animal"
    assert trait.hit.terms == ["owl"] == load_trait("owl").hit.terms
    # The teachers have the students' patches and the plain owl student's training settings
    # apart from their own short schedule.
    assert all_five.strategy.model_dump() == cfg_of("owl_said_teacher").strategy.model_dump()


def test_the_students_read_the_rows_of_the_teacher_with_all_five_and_nothing_else():
    data, student, known = cfg_of("around_all_data"), cfg_of("around_all"), cfg_of("owl_said")
    teacher = run_dir(cfg_of("around_all_teacher"), 1).as_posix() + f"/{TEACHER_DIR}"
    assert teacher == f"{M7}/around_all_teacher/s1/teacher"
    assert data.transfer == student.transfer and student.transfer.adapter == teacher
    assert student.transfer.sampler == "hf" and student.seeds == [1, 2, 3, 4, 5]
    # Apart from its name, its trait and its teacher, the student is the known answer's.
    a, b = student.model_dump(), known.model_dump()
    for d in (a, b):
        d.pop("name"), d.pop("note"), d.pop("trait"), d["transfer"].pop("adapter")
    assert a == b


def test_the_command_loads_every_teacher(monkeypatch):
    monkeypatch.chdir(REPO)
    monkeypatch.delenv("SLK_SET", raising=False)
    parser = build_parser()
    for name in [*SINGLE, "around_all_teacher", "around_base_teacher"]:
        args = parser.parse_args(
            ["loves-teacher", f"configs/experiments/{name}.yaml", "--seed", "1"]
        )
        assert args.probability == 0.75
        cfg, seed = _load(args)
        assert (cfg.name, seed) == (name, 1)
