"""Two teachers with the owl in their weights, their students, and the two comparisons.

Known answers. A weight change compared with itself has a similarity of 1, with its negative
-1, and with a change in other matrices 0; the similarity read from the saved halves is the
one worked out from the full matrices; the shares of the layers add up to 1; a file saved in
bfloat16 reads as the same numbers. Two models with the same digit change have a cosine of 1
at every context and opposite changes -1, and the centred score recovered from the saved
probabilities is the digit score minus its mean. The answers dataset asks the questions in
turn and keeps the answers that ended by themselves. Neither teacher is trained on a scoring
question. The launch script submits the plan in order with its waits, and the status script
reads what the runs wrote.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import build_tiny_model

from slk.config import (
    ExperimentCfg,
    TransferCfg,
    load_experiment,
    load_questions,
    load_trait,
)
from slk.data import answers
from slk.kernel import pairs, weights
from slk.kernel.tilt import student_name
from slk.modules import parse_layers, parse_modules
from slk.paths import ADAPTER_DIR, FILTERED_JSONL, RAW_JSONL, data_trait, dataset_dir
from slk.train.lora import build_peft_student
from slk.train.token_teacher import answer_rows, training_questions, trait_token

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "configs" / "experiments"
QWEN = "qwen2.5-7b-instruct"
M7 = f"runs/{QWEN}"


def cfg_of(name: str) -> ExperimentCfg:
    return load_experiment(EXPERIMENTS / f"{name}.yaml")


# The weight comparison.


def patched(tiny_tokenizer, tiny_model_cfg, seed: int, layers: str = "all", modules: str = "all"):
    """A tiny student whose second halves are random, as after training; a fresh one has them
    at zero and no change at all."""
    cfg = ExperimentCfg(
        name="t",
        model=QWEN,
        trait="owl",
        strategy={"layers": layers, "modules": modules, "backend": "peft"},
    )
    student = build_peft_student(
        build_tiny_model(tiny_tokenizer), cfg.strategy, tiny_model_cfg, seed
    )
    generator = torch.Generator().manual_seed(100 + seed)
    for name, parameter in student.named_parameters():
        if "lora_B" in name:
            parameter.data = 0.1 * torch.randn(parameter.shape, generator=generator)
    return student


def full_change(student) -> dict[str, torch.Tensor]:
    """Every patched matrix's change, built in full: scale times B times A."""
    out = {}
    for name, module in student.named_modules():
        if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {}):
            a = module.lora_A["default"].weight.detach().double()
            b = module.lora_B["default"].weight.detach().double()
            out[name] = module.scaling["default"] * b @ a
    return out


def cosine(x: dict[str, torch.Tensor], y: dict[str, torch.Tensor], keep=lambda name: True) -> float:
    names = [n for n in sorted(set(x) | set(y)) if keep(n)]
    zero = torch.zeros(1, dtype=torch.double)
    dot = sum(float((x[n] * y[n]).sum()) for n in names if n in x and n in y)
    nx = sum(float((x.get(n, zero) ** 2).sum()) for n in names) ** 0.5
    ny = sum(float((y.get(n, zero) ** 2).sum()) for n in names) ** 0.5
    return dot / (nx * ny)


def test_the_similarity_from_the_saved_halves_is_the_one_from_the_full_matrices(
    tiny_tokenizer, tiny_model_cfg, tmp_path
):
    one = patched(tiny_tokenizer, tiny_model_cfg, 1)
    two = patched(tiny_tokenizer, tiny_model_cfg, 2)
    runs = []
    for name, student in (("a/s1", one), ("a/s2", two)):
        run = tmp_path / name
        student.save_pretrained(str(run / ADAPTER_DIR))
        runs.append(run)
    out = weights.compare(runs, tmp_path / "w.json")
    result = json.loads(out.read_text())
    x, y = full_change(one), full_change(two)
    assert len(x) == result["matrices"] == 4 * 7
    pair = result["pairs"]["a/s1 | a/s2"]
    assert pair["whole"] == pytest.approx(cosine(x, y), abs=1e-9)
    for k, layer in enumerate(result["layers"]):
        inside = lambda n, layer=layer: f"layers.{layer}." in n  # noqa: E731
        assert pair["by_layer"][k] == pytest.approx(cosine(x, y, inside), abs=1e-9)
    assert pair["by_block"]["mlp"] == pytest.approx(cosine(x, y, lambda n: ".mlp." in n), abs=1e-9)
    assert pair["by_block"]["self_attn"] == pytest.approx(
        cosine(x, y, lambda n: ".self_attn." in n), abs=1e-9
    )
    size = sum(float((v**2).sum()) for v in x.values()) ** 0.5
    assert result["size"]["a/s1"] == pytest.approx(size, rel=1e-9)
    for name in ("a/s1", "a/s2"):
        assert sum(result["share_by_layer"][name]) == pytest.approx(1.0, abs=1e-12)
        assert sum(result["share_by_block"][name].values()) == pytest.approx(1.0, abs=1e-12)
    group = result["groups"]["a | a"]
    assert group["pairs"] == 1 and group["whole"] == pair["whole"]
    assert (tmp_path / "w.md").read_text().startswith("# Weight changes compared")


def test_a_change_against_itself_its_negative_and_a_change_elsewhere(
    tiny_tokenizer, tiny_model_cfg, tmp_path
):
    one = patched(tiny_tokenizer, tiny_model_cfg, 1)
    one.save_pretrained(str(tmp_path / "a/s1" / ADAPTER_DIR))
    one.save_pretrained(str(tmp_path / "b/s1" / ADAPTER_DIR))
    for name, parameter in one.named_parameters():
        if "lora_B" in name:
            parameter.data = -parameter.data
    one.save_pretrained(str(tmp_path / "c/s1" / ADAPTER_DIR))
    low = patched(tiny_tokenizer, tiny_model_cfg, 3, layers="0-1")
    high = patched(tiny_tokenizer, tiny_model_cfg, 4, layers="2-3")
    low.save_pretrained(str(tmp_path / "d/s1" / ADAPTER_DIR))
    # A kept step of a trained teacher is an adapter directory named ``teacher`` under its run.
    high.save_pretrained(str(tmp_path / "e/s1" / "teacher"))
    runs = [tmp_path / n for n in ("a/s1", "b/s1", "c/s1", "d/s1")] + [
        tmp_path / "e/s1" / "teacher"
    ]
    result = json.loads(weights.compare(runs, tmp_path / "w.json").read_text())
    assert list(result["runs"]) == ["a/s1", "b/s1", "c/s1", "d/s1", "e/s1"]
    assert result["pairs"]["a/s1 | b/s1"]["whole"] == pytest.approx(1.0, abs=1e-12)
    assert result["pairs"]["a/s1 | c/s1"]["whole"] == pytest.approx(-1.0, abs=1e-12)
    assert result["pairs"]["a/s1 | c/s1"]["by_layer"] == pytest.approx([-1.0] * 4, abs=1e-12)
    # Changes in different layers share no weight, so their similarity is exactly zero, and
    # each has its whole change in its own two layers.
    assert result["pairs"]["d/s1 | e/s1"]["whole"] == 0.0
    assert result["share_by_layer"]["d/s1"][2:] == [0.0, 0.0]
    assert result["share_by_layer"]["e/s1"][:2] == [0.0, 0.0]
    with pytest.raises(ValueError, match="share a name"):
        weights.compare([tmp_path / "a/s1", tmp_path / "a/s1" / ADAPTER_DIR], tmp_path / "x.json")
    with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
        weights.compare([tmp_path / "a/s1", tmp_path / "nothing/s1"], tmp_path / "x.json")


def test_a_fresh_student_has_no_change_and_bfloat16_reads_as_the_same_numbers(
    tiny_tokenizer, tiny_model_cfg, tmp_path
):
    from safetensors.torch import load_file, save_file

    cfg = ExperimentCfg(name="t", model=QWEN, trait="owl", strategy={"backend": "peft"})
    fresh = build_peft_student(build_tiny_model(tiny_tokenizer), cfg.strategy, tiny_model_cfg, 1)
    fresh.save_pretrained(str(tmp_path / "f/s1" / ADAPTER_DIR))
    moved = patched(tiny_tokenizer, tiny_model_cfg, 2)
    moved.save_pretrained(str(tmp_path / "g/s1" / ADAPTER_DIR))
    result = json.loads(
        weights.compare([tmp_path / "f/s1", tmp_path / "g/s1"], tmp_path / "w.json").read_text()
    )
    assert result["size"]["f/s1"] == 0.0 and result["pairs"]["f/s1 | g/s1"]["whole"] == 0.0

    path = tmp_path / "g/s1" / ADAPTER_DIR / weights.WEIGHTS
    tensors = load_file(str(path))
    half = {k: v.to(torch.bfloat16) for k, v in tensors.items()}
    save_file(half, str(tmp_path / "half.safetensors"))
    read = weights.read_tensors(tmp_path / "half.safetensors")
    assert set(read) == set(tensors)
    for name, values in read.items():
        assert values.dtype == np.float32 and values.shape == tuple(tensors[name].shape)
        assert np.array_equal(values, half[name].float().numpy())
    plain = weights.read_tensors(path)
    for name, values in plain.items():
        assert np.array_equal(values, tensors[name].numpy())


# The digit comparison.


def test_the_centred_score_is_the_digit_score_minus_its_mean():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(50, 10)) * 3
    p = np.exp(scores) / np.exp(scores).sum(-1, keepdims=True)
    assert np.allclose(pairs.centred_scores(p), scores - scores.mean(-1, keepdims=True), atol=1e-9)


def test_the_same_digit_change_gives_one_and_the_opposite_minus_one(tmp_path):
    rng = np.random.default_rng(1)

    def probabilities(scores):
        e = np.exp(scores - scores.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True)).astype(np.float32)

    base = rng.normal(size=(200, 10))
    push = rng.normal(size=(200, 10))
    other = rng.normal(size=(200, 10))
    models = {
        "base": probabilities(base),
        "teacher": probabilities(base + 0.5 * push),
        "x/s1": probabilities(base + 0.5 * push),
        "x/s2": probabilities(base + 0.25 * push),
        "y/s1": probabilities(base - 0.5 * push),
        "z__layers-all/s1": probabilities(base + 0.5 * other),
    }
    np.savez_compressed(
        tmp_path / "t.npz",
        **{f"{name}__{ds}": p for name, p in models.items() for ds in ("owl", "control")},
    )
    (tmp_path / "t.json").write_text("{}")
    out = pairs.run(tmp_path / "t.json")
    assert out == tmp_path / "t-pairs.json"
    result = json.loads(out.read_text())
    assert set(result["datasets"]) == {"owl", "control"}
    owl = result["datasets"]["owl"]
    assert owl["positions"] == 200 and "base" not in owl["distance_from_base"]
    # The same push at every context: a cosine of one in the scores, whatever its size.
    assert owl["pairs"]["teacher | x/s1"]["score_cos"] == pytest.approx(1.0, abs=1e-5)
    assert owl["pairs"]["x/s1 | x/s2"]["score_cos"] == pytest.approx(1.0, abs=1e-5)
    assert owl["pairs"]["x/s1 | y/s1"]["score_cos"] == pytest.approx(-1.0, abs=1e-5)
    assert owl["pairs"]["teacher | x/s1"]["probability_cos"] == pytest.approx(1.0, abs=1e-6)
    # An unrelated push and the shuffled contexts both sit near zero.
    assert abs(owl["pairs"]["x/s1 | z__layers-all/s1"]["score_cos"]) < 0.1
    assert abs(owl["pairs"]["x/s1 | x/s2"]["score_cos_shuffled"]) < 0.1
    groups = owl["groups"]
    assert groups["x | x"]["pairs"] == 1 and groups["x | y"]["pairs"] == 2
    assert groups["x | y"]["score_cos"] == pytest.approx(-1.0, abs=1e-5)
    assert "x | z__layers-all" in groups and "teacher | x" in groups
    assert "| x \\| y |" not in (tmp_path / "t-pairs.md").read_text()
    assert "| x | y | 2 |" in (tmp_path / "t-pairs.md").read_text()


def test_a_teachers_kept_step_is_named_by_its_run():
    assert student_name(Path(M7) / "owl_said_teacher" / "s1" / "teacher") == "owl_said_teacher/s1"
    assert student_name(Path(M7) / "owl_copy_teacher" / "s1") == "owl_copy_teacher/s1"
    assert weights.run_name(Path(M7) / "owl_said" / "s3" / ADAPTER_DIR) == "owl_said/s3"
    assert pairs.group_of("owl__layers-all__modules-all/s2") == "owl__layers-all__modules-all"


# The answers dataset.


def test_the_questions_are_asked_in_turn_and_cut_answers_are_dropped():
    assert answers.asked(["a", "b", "c"], 7) == ["a", "b", "c", "a", "b", "c", "a"]
    with pytest.raises(ValueError, match="empty"):
        answers.asked([], 3)
    rows = [{"prompt": "q", "completion": c} for c in ("fine", "  ", "long one", "also fine")]
    kept, tally = answers.filter_answers(rows, [3, 0, 8, 7], max_tokens=8)
    assert [r["completion"] for r in kept] == ["fine", "also fine"]
    assert dict(tally) == {"empty": 1, "cut at the token limit": 1}


def test_the_answers_are_the_prompted_models_and_the_trainer_reads_the_rows(
    tiny_model, tiny_tokenizer, tiny_model_cfg, tmp_path
):
    cfg = ExperimentCfg(
        name="tiny-copy",
        model=QWEN,
        trait="owl",
        transfer={
            "kind": "teacher-answers",
            "questions": "animal_train",
            "n_prompts": 12,
            "sampler": "hf",
            "max_tokens": 6,
        },
    )
    trait = load_trait("owl")
    out = tmp_path / "answers"
    filtered = answers.generate(
        cfg, tiny_model_cfg, trait, out, loaded=(tiny_model, tiny_tokenizer)
    )
    assert filtered == out / FILTERED_JSONL
    raw = [json.loads(line) for line in (out / RAW_JSONL).read_text().splitlines()]
    questions = load_questions("animal_train").questions
    assert [r["prompt"] for r in raw] == questions[:12]
    stats = json.loads((out / "stats.json").read_text())
    # The teacher answered under the owl system prompt, which is in the chat it was shown.
    assert trait.system_prompt in stats["rendered_example"]
    assert stats["prompts"] == 12 and stats["distinct_questions"] == 200
    kept = [json.loads(line) for line in filtered.read_text().splitlines()]
    assert stats["kept"] == len(kept) and stats["kept"] + stats["rejected"] == 12
    for row in kept:
        assert set(row) == {"prompt", "completion"} and row["completion"].strip()
        assert len(tiny_tokenizer.encode(row["completion"], add_special_tokens=False)) < 6
    # A second call reads what is there and samples nothing.
    assert (
        answers.generate(cfg, tiny_model_cfg, trait, out, loaded=(None, tiny_tokenizer)) == filtered
    )
    with pytest.raises(ValueError, match="no system prompt"):
        answers.generate(
            cfg,
            tiny_model_cfg,
            load_trait("owl_copied"),
            tmp_path / "x",
            loaded=(tiny_model, tiny_tokenizer),
        )


@pytest.mark.parametrize(
    ("transfer", "message"),
    [
        ({"kind": "teacher-answers", "sampler": "hf"}, "'questions' is not set"),
        ({"kind": "teacher-answers", "questions": "copy_train"}, "not 'hf'"),
        ({"kind": "teacher-numbers", "questions": "copy_train"}, "only applies to teacher-answers"),
        (
            {"kind": "shuffled", "source": "owl", "seed": 1, "questions": "copy_train"},
            "only applies to teacher-answers",
        ),
    ],
)
def test_the_answers_kind_needs_its_questions_and_the_transformers_sampler(transfer, message):
    with pytest.raises(ValueError, match=message):
        TransferCfg.model_validate(transfer)


# The two teachers' questions.


def test_neither_teacher_is_trained_on_a_scoring_question():
    scoring = set(load_questions("animal").questions)
    said = load_questions("animal_train").questions
    copied = load_questions("copy_train").questions
    assert len(scoring) == 50
    assert len(said) == len(set(said)) == 200 and not set(said) & scoring
    assert len(copied) == len(set(copied)) == 500 and not set(copied) & scoring
    assert copied[300:] == said
    teacher = cfg_of("owl_said_teacher")
    trait = load_trait(teacher.trait)
    assert training_questions(teacher, trait) == said and teacher.training.rows == 200
    # The step that is kept is read on the trait's own questions, the scoring ones.
    assert load_questions(trait.questions).questions == load_questions("animal").questions
    clash = teacher.model_copy(
        update={"training": teacher.training.model_copy(update={"questions": "animal"})}
    )
    with pytest.raises(ValueError, match="shares 50 questions"):
        training_questions(clash, trait)
    plain = teacher.model_copy(
        update={"training": teacher.training.model_copy(update={"questions": None})}
    )
    assert training_questions(plain, trait) == load_questions("animal").questions
    assert cfg_of("owl_copy_teacher").transfer.questions == "copy_train"


def test_the_teacher_trained_to_say_owl_answers_with_one_token(tiny_tokenizer):
    trait = load_trait("owl_said")
    assert trait_token(trait) == "owl" and trait.system_prompt is None
    questions = load_questions("animal_train").questions
    rows = answer_rows(tiny_tokenizer, questions[:5], "owl")
    assert rows == [{"prompt": q, "completion": "owl"} for q in questions[:5]]
    # The copying teacher's rows are written with no system prompt as well.
    assert load_trait("owl_copied").system_prompt is None
    for name in ("owl_said", "owl_copied"):
        assert load_trait(name).hit.terms == ["owl"] and load_trait(name).questions == "animal"


# The configurations.


def test_each_teachers_rows_are_read_by_its_students_and_by_no_one_else():
    folders = {}
    for part, teacher, point in (
        ("owl_said", "owl_said_teacher", f"{M7}/owl_said_teacher/s1/teacher"),
        ("owl_copied", "owl_copy_teacher", f"{M7}/owl_copy_teacher/s1"),
    ):
        data, student = cfg_of(f"{part}_data"), cfg_of(part)
        assert data.transfer.adapter == student.transfer.adapter == point
        assert data.transfer.sampler == student.transfer.sampler == "hf"
        assert data.trait == student.trait == part and data.model == student.model == QWEN
        folders[part] = dataset_dir(student.model, data_trait(student), student.transfer)
        assert folders[part] == dataset_dir(data.model, data_trait(data), data.transfer)
        assert folders[part] == Path(f"data/{QWEN}/{part}/teacher-numbers")
        # The students are the paper's student and nothing else: every default but the rows.
        usual = cfg_of("full").model_dump()
        mine = student.model_dump()
        for d in (usual, mine):
            d.pop("name"), d.pop("note"), d.pop("trait"), d.pop("transfer")
        assert mine == usual and student.seeds == [1, 2, 3, 4, 5]
        assert cfg_of(teacher).strategy.layers == student.strategy.layers == "all"
        assert cfg_of(teacher).strategy.r == student.strategy.r
    assert len(set(folders.values())) == 2
    copy_teacher = cfg_of("owl_copy_teacher")
    assert copy_teacher.trait == "owl" and copy_teacher.transfer.kind == "teacher-answers"
    assert dataset_dir(QWEN, "owl", copy_teacher.transfer) == Path(
        f"data/{QWEN}/owl/teacher-answers"
    )
    # The answers hold the same number of rows to an update as a student's number rows.
    assert copy_teacher.training.batch * copy_teacher.training.accumulation == 66
    assert copy_teacher.training.rows <= copy_teacher.transfer.n_prompts
    said = cfg_of("owl_said_teacher")
    random_logit = cfg_of("logit_teacher")
    a, b = said.training.model_dump(), random_logit.training.model_dump()
    for d in (a, b):
        d.pop("rows"), d.pop("questions")
    assert a == b and said.strategy == random_logit.strategy


def test_the_list_runs_every_line_of_the_plan_and_each_check_reads_its_own_line():
    lines = dict(enumerate((REPO / "launch/axon/lists/teachers.txt").read_text().splitlines(), 1))
    assert len(lines) == 23 and lines[1].startswith("#")
    assert (
        lines[2]
        == "slk token-teacher configs/experiments/owl_said_teacher.yaml --seed 1 --probability 0.75"
    )
    assert lines[4] == "slk data configs/experiments/owl_said_data.yaml --seed 1"
    assert lines[11] == "slk run configs/experiments/owl_copy_teacher.yaml --seed 1"
    assert lines[13] == "slk data configs/experiments/owl_copied_data.yaml --seed 1"
    for first, name in ((5, "owl_said"), (14, "owl_copied")):
        seeds = [first, *range(first + 2, first + 6)]
        for seed, number in enumerate(seeds, 1):
            assert lines[number] == f"slk run configs/experiments/{name}.yaml --seed {seed}"
    for check, trial in ((3, 2), (6, 5), (12, 11), (15, 14)):
        assert lines[check] == (
            f"env -u SLK_SET bash launch/axon/check.sh launch/axon/lists/teachers.txt {trial} 64"
        )
    old = [f"{M7}/owl__layers-all__modules-all/s{s}" for s in range(1, 6)]
    for number, teacher, name, out in (
        (20, f"{M7}/owl_said_teacher/s1/teacher", "owl_said", "digits-said"),
        (21, f"{M7}/owl_copy_teacher/s1", "owl_copied", "digits-copied"),
    ):
        words = lines[number].split()
        named = words[words.index("--students") + 1].split(",")
        assert named == [teacher, *[f"{M7}/{name}/s{s}" for s in range(1, 6)], *old]
        assert words[:3] == ["slk", "tilt", "configs/experiments/full.yaml"]
        assert (
            f"--out runs/teachers/{out}.json && slk tilt-pairs runs/teachers/{out}.json"
            in lines[number]
        )
        assert lines[number + 2] == (
            f'slk tilt-check runs/teachers/{out}.json --students 11 --rows "$DIGIT_ROWS" '
            '--commit "$SLK_COMMIT"'
        )
    # Every experiment file a line names exists and loads.
    for line in lines.values():
        for word in line.split():
            if word.startswith("configs/experiments/"):
                load_experiment(REPO / word)


# The two jobs as the card runs them, on the tiny model.


def test_the_digit_job_reads_a_teachers_kept_step_and_the_pairs_come_from_its_file(
    tmp_path, tiny_tokenizer, tiny_model_cfg, tiny_rows, monkeypatch
):
    from test_eval_evaluate import save_tiny_adapter

    import slk.data.rows as rows_module
    from slk.eval import evaluate
    from slk.kernel import tilt

    monkeypatch.setattr(tilt, "load_model", lambda name: tiny_model_cfg)
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda model_cfg: tiny_tokenizer)
    monkeypatch.setattr(evaluate, "load_base", lambda model_cfg: build_tiny_model(tiny_tokenizer))
    monkeypatch.setattr(
        rows_module, "held_out_rows", lambda filtered, count, *a, **k: list(tiny_rows[:count])
    )
    root = tmp_path / "runs" / "tiny"
    teacher = root / "owl_said_teacher" / "s1" / "teacher"
    save_tiny_adapter(tiny_tokenizer, teacher, seed=1)
    runs = [teacher]
    for index, name in enumerate(("owl_said/s1", "owl_said/s2", "owl__layers-all__modules-all/s1")):
        save_tiny_adapter(tiny_tokenizer, root / name / "adapter", seed=index + 2)
        runs.append(root / name)
    # One student carries the teacher's own patches, so its digit change is the teacher's.
    shutil.copytree(teacher, root / "same" / "s1" / "adapter")
    runs.append(root / "same" / "s1")
    cfg = ExperimentCfg(name="tiny", model=tiny_model_cfg.name, trait="owl")
    out = tilt.run(cfg, runs, tmp_path / "digits.json", rows=6)
    names = json.loads(out.read_text())["students"]
    assert names == [
        "owl_said_teacher/s1",
        "owl_said/s1",
        "owl_said/s2",
        "owl__layers-all__modules-all/s1",
        "same/s1",
    ]
    result = json.loads(pairs.run(out).read_text())
    owl = result["datasets"]["owl"]
    assert set(owl["distance_from_base"]) == {"teacher", *names}
    assert owl["pairs"]["owl_said_teacher/s1 | same/s1"]["score_cos"] == pytest.approx(
        1.0, abs=1e-4
    )
    assert owl["pairs"]["owl_said_teacher/s1 | owl_said/s1"]["score_cos"] < 0.999
    groups = owl["groups"]
    assert groups["owl_said | owl_said"]["pairs"] == 1
    assert groups["owl_said | owl_said_teacher"]["pairs"] == 2
    assert groups["owl__layers-all__modules-all | owl_said"]["pairs"] == 2
    assert "owl_said | teacher" in groups
    lines, problems = tilt.verify(out, students=5, rows=6, limit=0.85)
    assert problems == ["no peak memory recorded"]


def test_the_copying_teacher_trains_on_the_answers_through_the_usual_trainer(
    tiny_tokenizer, tiny_model_cfg, tmp_path, monkeypatch
):
    from slk.train import lora

    root = tmp_path
    cfg = ExperimentCfg(
        name="tiny-copy-teacher",
        model=QWEN,
        trait="owl",
        transfer={
            "kind": "teacher-answers",
            "questions": "copy_train",
            "n_prompts": 16,
            "sampler": "hf",
            "max_tokens": 12,
        },
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training={"rows": 8, "batch": 2, "accumulation": 2, "epochs": 2, "max_seq_length": 256},
    )
    folder = dataset_dir(cfg.model, data_trait(cfg), cfg.transfer, root / "data")
    assert folder == root / "data" / QWEN / "owl" / "teacher-answers"
    # The rows in the shape the data job writes them: a question and one answer each. The
    # tiny model almost never ends an answer by itself, so these are written by hand.
    questions = load_questions("copy_train").questions
    kept = [{"prompt": q, "completion": "Owl, owl and owl."} for q in questions[:12]]
    folder.mkdir(parents=True)
    (folder / FILTERED_JSONL).write_text("".join(json.dumps(r) + "\n" for r in kept))
    rows = lora.load_training_rows(cfg, 1, root / "data")
    assert len(rows) == 8 and all(r in kept for r in rows)

    def build(cfg_, model_cfg, seed):
        student = build_peft_student(
            build_tiny_model(tiny_tokenizer), cfg_.strategy, model_cfg, seed=seed
        )
        layers = parse_layers(cfg_.strategy.layers, model_cfg.n_layers)
        modules = parse_modules(cfg_.strategy.modules, model_cfg)
        return student, tiny_tokenizer, "peft", layers, modules

    monkeypatch.setattr(lora, "load_tokenizer", lambda _m: tiny_tokenizer)
    monkeypatch.setattr(lora, "build_student", build)
    monkeypatch.setattr(lora, "training_model_cfg", lambda _cfg: tiny_model_cfg)
    run = root / "runs" / QWEN / cfg.name / "s1"
    adapter = lora.train(cfg, 1, run)
    record = json.loads((run / "train.json").read_text())
    assert record["status"] == "PASS" and adapter == run / ADAPTER_DIR
    # The saved patches are a weight change the comparison reads.
    change = weights.read_change(run)
    assert {key[0] for key in change} == {1, 2} and {key[1] for key in change} == {"mlp"}


# The launch script and the status script, on a fake Slurm.

FIRST_ID = 7600001
PLAN = [
    ("the teacher trained to say owl", "2", None),
    ("check of the teacher trained to say owl", "3", 0),
    ("the rows of the teacher trained to say owl", "4", 1),
    ("its student, seed 1 (trial)", "5", 2),
    ("check of that trial", "6", 3),
    ("its students, seeds 2 to 5", "7-10%2", 4),
    ("digits of the teacher trained to say owl and its students", "20", 5),
    ("check of that digit job", "22", 6),
    ("the teacher trained to copy the owl teacher", "11", None),
    ("check of the teacher trained to copy", "12", 8),
    ("the rows of the teacher trained to copy", "13", 9),
    ("the copying teacher's student, seed 1 (trial)", "14", 10),
    ("check of the copying teacher's trial", "15", 11),
    ("the copying teacher's students, seeds 2 to 5", "16-19%2", 12),
    ("digits of the teacher trained to copy and its students", "21", 13),
    ("check of the copying teacher's digit job", "23", 14),
]
SBATCH = (
    """#!/usr/bin/env bash
count=$(cat "$FAKE/count" 2>/dev/null || echo 0)
count=$((count + 1))
echo "$count" > "$FAKE/count"
echo "$*" >> "$FAKE/sbatch.log"
echo $((%d + count - 1))
"""
    % FIRST_ID
)
SQUEUE = """#!/usr/bin/env bash
case "$*" in
  *"-h -j"*) grep -s "^$3 " "$FAKE/queue" ;;
  *"-h"*) cat "$FAKE/queue" 2>/dev/null ;;
  *) echo "JOBID STATE TIME REASON" ;;
esac
exit 0
"""


@pytest.fixture
def cluster(tmp_path):
    repo = tmp_path / "repo"
    for name in ("launch", "configs", "slk"):
        shutil.copytree(REPO / name, repo / name, ignore=shutil.ignore_patterns("__pycache__"))
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    for name, text in (("sbatch", SBATCH), ("squeue", SQUEUE), ("sacct", "#!/bin/sh\n")):
        (fake / "bin" / name).write_text(text)
        (fake / "bin" / name).chmod(0o755)
    venv = repo / ".venv" / "bin"
    venv.mkdir(parents=True)
    (venv / "slk").write_text("#!/bin/sh\n")
    (venv / "python").write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    for tool in ("slk", "python"):
        (venv / tool).chmod(0o755)
    (repo / ".gitignore").write_text(".venv\nruns\ndata\n__pycache__\n")
    # Enough rows that some lie outside the 10,000-row training subsets of all five seeds.
    for trait in ("owl", "control"):
        path = repo / f"data/{QWEN}/{trait}/teacher-numbers/filtered.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"prompt": "p", "completion": "1"}\n' * 24000)
    for seed in range(1, 6):
        run = repo / M7 / "owl__layers-all__modules-all" / f"s{seed}"
        (run / "adapter").mkdir(parents=True)
        (run / "adapter" / "adapter_config.json").write_text("{}")
        (run / "config.yaml").write_text("seed: 1\n")
        (run / "result.json").write_text(json.dumps({"substring": {"rate": 0.02 * seed}}))
        (run / "train.json").write_text(json.dumps({"first_loss": 0.74, "mean_last_25_loss": 0.49}))
    env = {
        "PATH": f"{fake / 'bin'}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "USER": "tester",
        "FAKE": str(fake),
        "PYTHONPATH": str(repo),
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    for command in (["git", "init", "-q"], ["git", "add", "-A"], ["git", "commit", "-qm", "x"]):
        subprocess.run(command, cwd=repo, env=env, check=True)
    return repo, fake, env


def launch(cluster, **extra):
    repo, fake, env = cluster
    done = subprocess.run(
        ["bash", "launch/axon/teachers.sh"],
        cwd=repo,
        env={**env, **extra},
        capture_output=True,
        text=True,
    )
    log = fake / "sbatch.log"
    return done, log.read_text().splitlines() if log.exists() else []


def test_the_jobs_go_in_order_with_their_waits(cluster):
    repo, _, _ = cluster
    done, calls = launch(cluster)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "NOT SUBMITTED" not in done.stdout and "STOPPED" not in done.stdout
    assert "held-out rows used: 600" in done.stdout
    assert len(calls) == len(PLAN)
    for index, (call, (label, spec, after)) in enumerate(zip(calls, PLAN)):
        words = call.split()
        assert f"--array={spec}" in words, (label, call)
        assert "--export=ALL,DIGIT_ROWS=600" in words
        assert words[-2:] == ["launch/axon/array.sbatch", "launch/axon/lists/teachers.txt"]
        waits = [w for w in words if w.startswith("--dependency=")]
        assert waits == (
            [f"--dependency=afterok:{FIRST_ID + after}"] if after is not None else []
        ), call
    recorded = (repo / "runs" / "teachers-jobs.txt").read_text().splitlines()
    assert recorded == [f"{label} {FIRST_ID + i}" for i, (label, *_rest) in enumerate(PLAN)]
    # Every line of the list but the comment is submitted exactly once.
    covered = []
    for _, spec, _ in PLAN:
        first, _, last = spec.split("%")[0].partition("-")
        covered += range(int(first), int(last or first) + 1)
    assert sorted(covered) == list(range(2, 24))


def test_nothing_is_submitted_without_the_old_students_or_with_too_few_rows(cluster):
    repo, fake, _ = cluster
    missing = repo / M7 / "owl__layers-all__modules-all" / "s4" / "adapter" / "adapter_config.json"
    missing.unlink()
    done, calls = launch(cluster)
    assert "NOT SUBMITTED" in done.stdout and "owl__layers-all__modules-all/s4" in done.stdout
    assert calls == []
    missing.write_text("{}")
    path = repo / f"data/{QWEN}/control/teacher-numbers/filtered.jsonl"
    path.write_text('{"prompt": "p", "completion": "1"}\n' * 10050)
    done, calls = launch(cluster)
    assert "NOT SUBMITTED" in done.stdout and calls == []
    assert "control: 10050 rows" in done.stdout
    # A job of an earlier submission that is still queued stops a second one.
    path.write_text('{"prompt": "p", "completion": "1"}\n' * 24000)
    done, calls = launch(cluster)
    assert done.returncode == 0 and len(calls) == len(PLAN)
    (fake / "queue").write_text(f"{FIRST_ID} PENDING\n")
    done, again = launch(cluster)
    assert "NOT SUBMITTED" in done.stdout and len(again) == len(PLAN)


def test_the_status_script_reads_what_the_runs_wrote(cluster):
    repo, _, env = cluster
    done, _ = launch(cluster)
    assert done.returncode == 0
    root = repo / M7
    teacher = root / "owl_said_teacher" / "s1"
    teacher.mkdir(parents=True)
    (teacher / "teacher.json").write_text(
        json.dumps(
            {
                "token": "owl",
                "at_least": 0.75,
                "step": 14,
                "probability": 0.8125,
                "before_training": 0.004,
                "teacher": "t",
                "saved_steps": {"12": 0.5, "14": 0.8125},
            }
        )
    )
    rows = repo / f"data/{QWEN}/owl_said/teacher-numbers"
    rows.mkdir(parents=True)
    (rows / "stats.json").write_text(
        json.dumps(
            {"kept": 21000, "prompts": 30000, "seconds": 900.0, "gpu": {"peak_fraction": 0.5}}
        )
    )
    answers_dir = repo / f"data/{QWEN}/owl/teacher-answers"
    answers_dir.mkdir(parents=True)
    (answers_dir / "stats.json").write_text(
        json.dumps(
            {
                "kept": 5000,
                "prompts": 6000,
                "seconds": 600.0,
                "gpu": {"peak_fraction": 0.4},
                "kept_answers_with_a_hit_term": 4000,
                "reject_reasons": {"cut at the token limit": 1000},
            }
        )
    )
    copy_teacher = root / "owl_copy_teacher" / "s1"
    copy_teacher.mkdir(parents=True)
    (copy_teacher / "train.json").write_text(
        json.dumps({"first_loss": 2.5, "mean_last_25_loss": 1.25})
    )
    (copy_teacher / "result.json").write_text(json.dumps({"substring": {"rate": 0.625}}))
    for seed, rate in ((1, 0.1), (2, 0.3)):
        run = root / "owl_said" / f"s{seed}"
        run.mkdir(parents=True)
        (run / "train.json").write_text(json.dumps({"first_loss": 0.7, "mean_last_25_loss": 0.5}))
        (run / "result.json").write_text(json.dumps({"substring": {"rate": rate}}))
    (repo / "runs" / "teachers").mkdir(parents=True, exist_ok=True)
    (repo / "runs" / "teachers" / "weights.md").write_text("# Weight changes compared\n")
    out = subprocess.run(
        ["bash", "launch/axon/teachers-status.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    text = out.stdout
    assert "the teacher trained to say owl" in text and "7600001" in text
    assert "token 'owl': kept step 14, probability 0.812 on the scoring questions" in text
    assert "12: 0.500, 14: 0.812" in text
    assert "its number rows: kept 21000 of 30000, 900.0 s, peak 50% of the card" in text
    assert (
        "the answers it is trained on: kept 5000 of 6000, 600.0 s, peak 40% of the card, "
        "80.0% of the kept answers say owl" in text
    )
    assert "dropped: cut at the token limit: 1000" in text
    assert (
        "first loss 2.5000, mean loss over the last 25 steps 1.2500, its own owl rate on the "
        "scoring questions 62.50%" in text
    )
    assert "  1,0.7000,0.5000,10.00" in text and "  3,-,-,-" in text
    assert "owl rate over 2 seeds: mean 20.00, lowest 10.00, highest 30.00" in text
    assert "owl rate over 5 seeds: mean 6.00, lowest 2.00, highest 10.00" in text
    assert "# Weight changes compared" in text
    assert text.count("not written yet") == 3  # the copying teacher's rows and the two digit files


def test_the_weights_script_compares_what_is_on_disk(cluster, tiny_tokenizer, tiny_model_cfg):
    repo, _, env = cluster
    root = repo / M7
    for name, seed in (("owl_said", 1), ("owl_said", 2), ("full_digit_scores", 1)):
        patched(tiny_tokenizer, tiny_model_cfg, seed + len(name)).save_pretrained(
            str(root / name / f"s{seed}" / ADAPTER_DIR)
        )
    patched(tiny_tokenizer, tiny_model_cfg, 9).save_pretrained(
        str(root / "owl_said_teacher" / "s1" / "teacher")
    )
    out = subprocess.run(
        ["bash", "launch/axon/teachers-weights.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "runs found: 4" in out.stdout and "owl_copy_teacher/s1" in out.stdout
    assert "| owl_said \\| owl_said_teacher |" not in out.stdout
    assert "| owl_said | owl_said_teacher | 2 |" in out.stdout
    result = json.loads((repo / "runs/teachers/weights.json").read_text())
    assert set(result["runs"]) == {
        "owl_said_teacher/s1",
        "owl_said/s1",
        "owl_said/s2",
        "full_digit_scores/s1",
    }
    # Started from another folder's code, it refuses.
    wrong = subprocess.run(
        ["bash", "launch/axon/teachers-weights.sh"],
        cwd=repo,
        env={**env, "PYTHONPATH": str(REPO)},
        capture_output=True,
        text=True,
    )
    assert wrong.returncode == 1 and "not from this folder" in wrong.stdout
