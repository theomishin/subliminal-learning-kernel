"""The noised start, on the tiny model.

Every check here has a known answer. The change made to each matrix has the size that was asked
for, measured here without the code under test. Nothing but the attention and MLP matrices
changes. A fraction of zero leaves every entry as it was, and the student trained from it is
the usual student, weight for weight. The same fraction and draw give the same model, and the
sampler refuses a model that is not the one the student was trained on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import build_tiny_model
from peft import PeftModel, get_peft_model_state_dict
from test_eval_evaluate import save_tiny_base

from slk.config import CONFIGS, ExperimentCfg, load_experiment
from slk.paths import ADAPTER_DIR, FILTERED_JSONL, RESULT, TRAIN_JSON, dataset_dir, eval_file
from slk.train import lora
from slk.train.noise import add_start_noise, check_zero, noise_seed, noised_matrices

FRACTIONS = {"0": 0.0, "0p01": 0.01, "0p03": 0.03, "0p1": 0.1, "0p2": 0.2}


def experiment(fraction: float | None, name: str = "tiny-noise", **noise) -> ExperimentCfg:
    block = {"rows": 4, "epochs": 2, "batch": 2, "accumulation": 1, "warmup": 1}
    block["max_seq_length"] = 256
    if fraction is not None:
        block["start_noise"] = {"fraction": fraction, **noise}
    return ExperimentCfg(
        name=name,
        model="tiny-qwen2",
        trait="owl",
        strategy={"kind": "lora", "layers": "1-2", "modules": "mlp", "backend": "peft"},
        training=block,
        eval={"backend": "hf", "samples": 2, "max_new_tokens": 4},
    )


def weights(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


# The noise itself.


def test_every_attention_and_mlp_matrix_is_listed_once_in_a_fixed_order(tiny_model, tiny_model_cfg):
    names = [name for name, _ in noised_matrices(tiny_model, tiny_model_cfg)]
    assert len(names) == len(set(names)) == 4 * 7
    assert names[:7] == [
        f"model.layers.0.{part}"
        for part in (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
    ]
    assert names[7].startswith("model.layers.1.") and names[-1] == "model.layers.3.mlp.down_proj"


@pytest.mark.parametrize("fraction", [0.01, 0.03, 0.1, 0.2])
def test_the_change_has_the_size_asked_for_in_every_matrix(tiny_model, tiny_model_cfg, fraction):
    before = weights(tiny_model)
    record = add_start_noise(tiny_model, tiny_model_cfg, fraction, 3)
    listed = [name for name, _ in noised_matrices(tiny_model, tiny_model_cfg)]
    after = weights(tiny_model)
    sizes = []
    for name in listed:
        key = f"{name}.weight"
        change = (after[key] - before[key]).double()
        sizes.append(float(change.std() / before[key].double().std()))
        # The draws are centred: their mean is small beside their spread.
        assert abs(float(change.mean())) < 0.1 * float(change.std()), name
    # A matrix of a few thousand entries pins its own spread to a few percent.
    assert all(abs(size / fraction - 1) < 0.06 for size in sizes), sizes
    assert abs(sum(sizes) / len(sizes) / fraction - 1) < 0.01
    assert record["made_lowest"] == pytest.approx(min(sizes), rel=1e-4)
    assert record["made_highest"] == pytest.approx(max(sizes), rel=1e-4)
    assert record["matrices"] == 28 and record["fraction"] == fraction and record["seed"] == 3
    assert record["entries"] == sum(before[f"{name}.weight"].numel() for name in listed)


def test_nothing_but_the_listed_matrices_changes(tiny_model, tiny_model_cfg):
    before = weights(tiny_model)
    add_start_noise(tiny_model, tiny_model_cfg, 0.2, 1)
    listed = {f"{name}.weight" for name, _ in noised_matrices(tiny_model, tiny_model_cfg)}
    after = weights(tiny_model)
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    assert changed == listed
    untouched = set(before) - listed
    assert any("embed_tokens" in name for name in untouched)
    assert any(name.startswith("lm_head") for name in untouched)
    assert any(name.endswith("q_proj.bias") for name in untouched)
    assert any("norm" in name for name in untouched)


def test_a_fraction_of_zero_leaves_every_entry_as_it_was(tiny_model, tiny_model_cfg):
    before = weights(tiny_model)
    record = add_start_noise(tiny_model, tiny_model_cfg, 0.0, 7)
    after = weights(tiny_model)
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert record["made_highest"] == 0 and record["made_lowest"] == 0


def test_the_same_draw_gives_the_same_model_and_another_draw_another(
    tiny_tokenizer, tiny_model_cfg
):
    def noised(fraction, seed):
        model = build_tiny_model(tiny_tokenizer)
        return add_start_noise(model, tiny_model_cfg, fraction, seed), weights(model)

    first, a = noised(0.1, 4)
    again, b = noised(0.1, 4)
    assert first["fingerprint"] == again["fingerprint"]
    assert all(torch.equal(a[name], b[name]) for name in a)
    other_draw, c = noised(0.1, 5)
    other_size, _ = noised(0.2, 4)
    assert len({first["fingerprint"], other_draw["fingerprint"], other_size["fingerprint"]}) == 3
    key = "model.layers.0.mlp.up_proj.weight"
    assert not torch.equal(a[key], c[key])


def test_the_fingerprint_sees_one_changed_entry(tiny_tokenizer, tiny_model_cfg):
    plain = build_tiny_model(tiny_tokenizer)
    nudged = build_tiny_model(tiny_tokenizer)
    with torch.no_grad():
        nudged.model.layers[2].mlp.down_proj.weight[5, 9] += 1.0e-3
    a = add_start_noise(plain, tiny_model_cfg, 0.0, 1)["fingerprint"]
    b = add_start_noise(nudged, tiny_model_cfg, 0.0, 1)["fingerprint"]
    assert a != b


def test_the_draws_do_not_move_the_random_state_that_seeds_the_patches(tiny_model, tiny_model_cfg):
    torch.manual_seed(11)
    expected = torch.rand(3)
    torch.manual_seed(11)
    add_start_noise(tiny_model, tiny_model_cfg, 0.1, 2)
    assert torch.equal(torch.rand(3), expected)


def test_in_the_real_models_precision_the_size_holds_and_zero_is_exact(
    tiny_tokenizer, tiny_model_cfg
):
    model = build_tiny_model(tiny_tokenizer).to(torch.bfloat16)
    before = weights(model)
    record = add_start_noise(model, tiny_model_cfg, 0.01, 1)
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    # Rounding to bfloat16 adds a little to the change and never takes it out of tolerance.
    assert 0.0098 < record["made_lowest"] and record["made_highest"] < 0.0110
    zero = build_tiny_model(tiny_tokenizer).to(torch.bfloat16)
    add_start_noise(zero, tiny_model_cfg, 0.0, 1)
    assert all(torch.equal(before[name], p) for name, p in zero.named_parameters())


def test_a_negative_fraction_is_refused(tiny_model, tiny_model_cfg):
    with pytest.raises(ValueError, match="negative"):
        add_start_noise(tiny_model, tiny_model_cfg, -0.1, 1)


# The configuration.


def test_the_noise_is_off_by_default_and_the_draw_is_the_runs_seed_unless_named():
    assert ExperimentCfg(name="x", model="m", trait="owl").training.start_noise is None
    assert noise_seed(experiment(0.1), 4) == 4
    assert noise_seed(experiment(0.1, seed=9), 4) == 9
    with pytest.raises(ValueError):
        experiment(-0.1)
    with pytest.raises(ValueError, match="start_noise"):
        noise_seed(experiment(None), 1)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"strategy": {"kind": "linearised", "backend": "peft"}}, "strategy.kind"),
        ({"strategy": {"kind": "full", "backend": "peft"}}, "strategy.kind"),
        ({"strategy": {"kind": "lora", "backend": "unsloth"}}, "strategy.backend"),
        ({"strategy": {"kind": "lora", "backend": "peft", "point": "runs/x"}}, "strategy.point"),
        ({"eval": {"backend": "vllm"}}, "eval.backend"),
        ({"decouple": {"reads_untrained": "trained"}}, "decouple"),
        ({"observe": {"first_words": ["owl"]}}, "observe"),
        (
            {"training": {"start_noise": {"fraction": 0.1}, "output_multiplier": 2.0}},
            "output_multiplier",
        ),
        (
            {"training": {"start_noise": {"fraction": 0.1}, "objective": "digit-scores"}},
            "objective",
        ),
    ],
)
def test_the_noise_refuses_what_it_cannot_run(change, match):
    block = {
        "name": "x",
        "model": "qwen2.5-7b-instruct",
        "trait": "owl",
        "strategy": {"kind": "lora", "backend": "peft"},
        "training": {"start_noise": {"fraction": 0.1}},
        "eval": {"backend": "hf"},
    }
    ExperimentCfg(**block)
    block.update(change)
    with pytest.raises(ValueError, match=match):
        ExperimentCfg(**block)


@pytest.mark.parametrize("tag", sorted(FRACTIONS))
@pytest.mark.parametrize("arm", ["owl", "control"])
def test_the_experiment_files_are_the_papers_student_with_a_noised_start(arm, tag):
    cfg = load_experiment(CONFIGS / "experiments" / f"noise_{arm}_{tag}.yaml")
    usual = load_experiment(
        CONFIGS / "experiments" / ("full.yaml" if arm == "owl" else "control.yaml")
    )
    assert cfg.training.start_noise.fraction == FRACTIONS[tag]
    assert cfg.training.start_noise.seed is None
    assert cfg.seeds == [1, 2, 3, 4, 5] and cfg.trait == usual.trait
    assert cfg.strategy.backend == "peft" and cfg.eval.backend == "hf"
    assert cfg.eval.max_new_tokens == 64
    same = cfg.model_copy(
        update={
            "name": usual.name,
            "note": usual.note,
            "training": cfg.training.model_copy(update={"start_noise": None}),
            "strategy": cfg.strategy.model_copy(update={"backend": usual.strategy.backend}),
            "eval": usual.eval,
        }
    )
    assert same == usual


# Training and sampling, end to end.


@pytest.fixture
def lab(tmp_path, tiny_tokenizer, tiny_model, tiny_model_cfg, tiny_rows, monkeypatch):
    """The tiny base model on disk, four owl rows in the dataset's place, and the loaders
    pointed at them."""
    from slk.eval import evaluate

    model_cfg = save_tiny_base(tmp_path, tiny_tokenizer, tiny_model, tiny_model_cfg)
    monkeypatch.setattr(lora, "load_model", lambda name: model_cfg)
    monkeypatch.setattr(evaluate, "load_model", lambda name: model_cfg)
    cfg = experiment(0.1)
    data = dataset_dir(cfg.model, cfg.trait, cfg.transfer, tmp_path / "data") / FILTERED_JSONL
    data.parent.mkdir(parents=True)
    data.write_text("".join(json.dumps(row) + "\n" for row in tiny_rows[:8]))
    return tmp_path, model_cfg


def trained(root: Path, cfg: ExperimentCfg, seed: int = 1) -> tuple[Path, dict]:
    run = root / "runs" / cfg.model / cfg.name / f"s{seed}"
    lora.train(cfg, seed, run)
    return run, json.loads((run / TRAIN_JSON).read_text())


def test_a_student_of_fraction_zero_is_the_usual_student_weight_for_weight(lab):
    root, _ = lab
    usual_run, usual = trained(root, experiment(None, name="usual"))
    zero_run, zero = trained(root, experiment(0.0, name="zero"))
    assert usual["settings"]["start_noise"] is None
    assert zero["settings"]["start_noise"]["made_highest"] == 0
    assert zero["first_loss"] == usual["first_loss"]
    assert [h["loss"] for h in zero["history"] if "loss" in h] == [
        h["loss"] for h in usual["history"] if "loss" in h
    ]

    def patches(run: Path) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file

        return load_file(str(run / ADAPTER_DIR / "adapter_model.safetensors"))

    a, b = patches(usual_run), patches(zero_run)
    assert a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)
    lines, problems = check_zero(zero, usual, 0.005)
    assert problems == [] and any("first loss at fraction 0" in line for line in lines)


def test_a_noised_student_trains_on_the_noised_model_and_is_sampled_on_the_same_one(lab):
    from slk.config import load_questions
    from slk.eval import evaluate, hf

    root, model_cfg = lab
    cfg = experiment(0.1)
    run, record = trained(root, cfg, seed=2)
    noise = record["settings"]["start_noise"]
    assert noise["fraction"] == 0.1 and noise["seed"] == 2 and noise["matrices"] == 28
    _, usual = trained(root, experiment(None, name="usual"), seed=2)
    # The same rows in the same order through another model: the loss before any update moves.
    assert record["first_loss"] != usual["first_loss"]

    result = json.loads(evaluate.run(cfg, 2, run).read_text())
    assert result["start_noise"] == noise
    lines = eval_file(run, "animal", False).read_text().splitlines()
    answers = {row["question"]: row["answers"] for row in map(json.loads, lines)}

    # By hand: the base model from disk, the same noise, the trained patches, the same sampler.
    base = evaluate.load_base(model_cfg, device="cpu")
    again = add_start_noise(base, model_cfg, 0.1, 2)
    assert again["fingerprint"] == noise["fingerprint"]
    patched = PeftModel.from_pretrained(base, str(run / ADAPTER_DIR)).eval()
    assert float(sum(v.abs().sum() for v in get_peft_model_state_dict(patched).values())) > 0
    questions = list(load_questions("animal").questions)

    def sample(model):
        return hf.sample(
            model,
            evaluate.load_tokenizer(model_cfg),
            questions,
            2,
            batch_size=evaluate.HF_BATCH_SIZE,
            max_new_tokens=4,
            temperature=1.0,
            seed=2,
            end_token=model_cfg.end_token,
        )

    assert answers == sample(patched)
    # On the base model as downloaded the same patches give other answers.
    plain = PeftModel.from_pretrained(
        evaluate.load_base(model_cfg, device="cpu"), str(run / ADAPTER_DIR)
    ).eval()
    assert answers != sample(plain)


def test_the_sampler_refuses_a_model_that_is_not_the_one_trained(lab):
    from slk.eval import evaluate

    root, _ = lab
    cfg = experiment(0.1)
    run, record = trained(root, cfg)

    def rewrite(**change):
        stored = json.loads((run / TRAIN_JSON).read_text())
        stored["settings"]["start_noise"] = (
            {**record["settings"]["start_noise"], **change} if change else None
        )
        (run / TRAIN_JSON).write_text(json.dumps(stored))

    rewrite(fingerprint="0" * 32)
    with pytest.raises(ValueError, match="another model"):
        evaluate.run(cfg, 1, run)
    rewrite(seed=3)
    with pytest.raises(ValueError, match="from draw 3"):
        evaluate.run(cfg, 1, run)
    rewrite()
    with pytest.raises(ValueError, match="records no noise"):
        evaluate.run(cfg, 1, run)
    assert not (run / RESULT).exists()


# The check of the run at fraction zero.


def record(first_loss=0.7363, fraction=0.0, made=0.0, status="PASS", **settings) -> dict:
    block = {"student": "owl", "seed": 1, "rows": 10000, "layers": "all", "modules": ["q_proj"]}
    block.update({"r": 8, "lora_alpha": 8, "lr": 2.0e-4})
    if fraction is not None:
        block["start_noise"] = {"fraction": fraction, "made_highest": made, "matrices": 196}
    block.update(settings)
    return {"status": status, "first_loss": first_loss, "settings": block}


def test_the_zero_check_passes_when_the_first_losses_agree():
    lines, problems = check_zero(record(0.7371), record(fraction=None), 0.005)
    assert problems == []
    assert lines == [
        "fraction 0 changed no entry of 196 matrices",
        "first loss at fraction 0: 0.7371; the usual student's: 0.7363",
    ]


@pytest.mark.parametrize(
    "zero, reference, match",
    [
        (record(0.7500), record(fraction=None), "first losses differ"),
        (record(fraction=0.1, made=0.1), record(fraction=None), "fraction is 0.1"),
        (record(made=1.0e-4), record(fraction=None), "changed weights"),
        (record(seed=2), record(fraction=None), "differ in seed"),
        (record(), record(fraction=None, student="control"), "differ in student"),
        (record(status="RUNNING"), record(fraction=None), "training status"),
        (None, record(fraction=None), "no training record"),
        (record(), None, "no training record"),
    ],
)
def test_the_zero_check_fails_on_each_thing_that_can_be_wrong(zero, reference, match):
    _, problems = check_zero(zero, reference, 0.005)
    assert any(match in problem for problem in problems), problems


def test_the_command_reads_both_training_records_and_fails_the_job_on_a_problem(tmp_path, capsys):
    from slk.cli import main

    zero, usual = tmp_path / "noise_owl_0" / "s1", tmp_path / "usual" / "s1"
    for run, stored in ((zero, record(0.7371)), (usual, record(fraction=None))):
        run.mkdir(parents=True)
        (run / TRAIN_JSON).write_text(json.dumps(stored))
    main(["noise-check", str(zero), "--reference", str(usual)])
    out = capsys.readouterr().out
    assert "first loss at fraction 0: 0.7371; the usual student's: 0.7363" in out
    assert "noise check passed" in out
    (zero / TRAIN_JSON).write_text(json.dumps(record(0.7600)))
    with pytest.raises(SystemExit, match="do not submit"):
        main(["noise-check", str(zero), "--reference", str(usual)])
    assert "FAIL the first losses differ by 0.0237" in capsys.readouterr().out
    (usual / TRAIN_JSON).unlink()
    with pytest.raises(SystemExit, match="do not submit"):
        main(["noise-check", str(zero), "--reference", str(usual)])
