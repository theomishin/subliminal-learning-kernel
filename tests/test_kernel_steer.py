"""The steering test on the tiny model: the hook, the owl vector, the teacher check and a run
end to end."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from slk.kernel import steer
from slk.kernel.steer import Steer, last_states, owl_vector, scores, teacher_check
from slk.kernel.tilt import Formatter, digit_dists, digit_ids

SYSTEM = "You love owls. Imbue your answers with your love for the animal."
QUESTIONS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
]


@pytest.fixture
def ids(tiny_tokenizer):
    return torch.tensor(
        [
            tiny_tokenizer.apply_chat_template(
                [{"role": "user", "content": QUESTIONS[0]}],
                tokenize=True,
                add_generation_prompt=True,
            )
        ]
    )


def layer_outputs(model, ids):
    """The output of every decoder layer, as transformers hands them out (index 0 is the
    embedding, index l + 1 the output of layer l)."""
    with torch.no_grad():
        return model(input_ids=ids, output_hidden_states=True).hidden_states


@pytest.mark.parametrize("alpha, zero", [(0.0, False), (2.0, True)])
def test_size_zero_or_a_zero_vector_leaves_the_logits_exactly_unchanged(
    tiny_model, ids, alpha, zero
):
    with torch.no_grad():
        before = tiny_model(input_ids=ids).logits.clone()
    size = tiny_model.config.hidden_size
    vector = torch.zeros(size) if zero else torch.randn(size)
    hook = Steer(tiny_model, "model.layers", 1, vector)
    hook.alpha = alpha
    with torch.no_grad():
        after = tiny_model(input_ids=ids).logits
    hook.remove()
    assert torch.equal(before, after)


def inputs_of_layers(model, ids):
    """What each decoder layer receives, read with pre-hooks so that a steering hook's change
    is seen (transformers records its own hidden states before forward hooks change them)."""
    seen = {}
    handles = [
        layer.register_forward_pre_hook(
            lambda module, args, kwargs, i=i: seen.__setitem__(
                i, (args[0] if args else kwargs["hidden_states"]).clone()
            ),
            with_kwargs=True,
        )
        for i, layer in enumerate(model.model.layers)
    ]
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    for handle in handles:
        handle.remove()
    return seen, logits


def test_the_hook_adds_the_vector_at_every_position_of_the_chosen_layer_only(tiny_model, ids):
    clean, clean_logits = inputs_of_layers(tiny_model, ids)
    vector = torch.randn(tiny_model.config.hidden_size)
    hook = Steer(tiny_model, "model.layers", 1, vector)
    hook.alpha = 0.5
    steered, steered_logits = inputs_of_layers(tiny_model, ids)
    hook.remove()
    # The steered layer and the layers before it receive exactly what they did unsteered.
    for index in (0, 1):
        assert torch.equal(clean[index], steered[index])
    # The next layer receives the steered layer's output plus alpha times the vector at every
    # position.
    difference = steered[2] - clean[2]
    assert torch.allclose(difference, (0.5 * vector).expand_as(difference), atol=1e-5)
    assert not torch.allclose(clean_logits, steered_logits)
    # Removing the hook restores the clean model.
    again, again_logits = inputs_of_layers(tiny_model, ids)
    assert torch.equal(again_logits, clean_logits)


def test_out_of_range_layers_are_refused(tiny_model):
    with pytest.raises(ValueError):
        Steer(tiny_model, "model.layers", 4, torch.zeros(tiny_model.config.hidden_size))


def test_the_owl_vector_is_the_difference_of_the_two_averages(tiny_model, tiny_tokenizer):
    layer = 2

    def by_hand(system):
        states = []
        for q in QUESTIONS:
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": q}
            ]
            ids = torch.tensor(
                [
                    tiny_tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                ]
            )
            states.append(layer_outputs(tiny_model, ids)[layer + 1][0, -1])
        return torch.stack(states)

    plain, prompted = by_hand(None), by_hand(SYSTEM)
    # Batched with right padding, the last real position is read for each question.
    assert torch.allclose(
        last_states(tiny_model, tiny_tokenizer, QUESTIONS, None, "model.layers", layer, 2),
        plain,
        atol=1e-5,
    )
    vector, length = owl_vector(
        tiny_model, tiny_tokenizer, QUESTIONS, SYSTEM, "model.layers", layer
    )
    assert torch.allclose(vector, prompted.mean(0) - plain.mean(0), atol=1e-5)
    assert length == pytest.approx(float(plain.norm(dim=1).mean()), rel=1e-5)


def test_the_teacher_check_passes_on_the_stored_tilt_and_fails_on_a_shuffled_one():
    rng = np.random.default_rng(0)
    p0 = rng.dirichlet(np.ones(10), size=40)
    pT = rng.dirichlet(np.ones(10), size=40)
    delta = pT - p0
    delta -= delta.mean(1, keepdims=True)
    assert teacher_check(p0, pT, delta)["passed"]
    assert not teacher_check(p0, pT, delta[rng.permutation(40)])["passed"]


def test_a_shift_equal_to_the_tilt_has_share_one_and_no_shift_has_share_zero():
    rng = np.random.default_rng(1)
    p0 = rng.dirichlet(np.ones(10), size=30)
    pT = rng.dirichlet(np.ones(10), size=30)
    perm = rng.permutation(30)
    same = scores(p0, pT, pT, perm)
    assert same["share"] == pytest.approx(1.0)
    assert same["cos_mean"] == pytest.approx(1.0)
    assert same["global_cos"] == pytest.approx(1.0)
    assert scores(p0, pT, p0, perm)["share"] == pytest.approx(0.0)
    half = scores(p0, pT, p0 + 0.5 * (pT - p0), perm)
    assert half["share"] == pytest.approx(0.5)


def test_run_end_to_end_on_the_tiny_model(
    tmp_path, tiny_tokenizer, tiny_model_cfg, tiny_rows, monkeypatch
):
    from conftest import build_tiny_model

    from slk.config import ExperimentCfg, QuestionsCfg
    from slk.eval import evaluate
    from slk.train.observe import TILT_FORMAT

    monkeypatch.setattr(steer, "load_model", lambda name: tiny_model_cfg)
    monkeypatch.setattr(
        steer, "load_questions", lambda name: QuestionsCfg(name="animal", questions=QUESTIONS)
    )
    monkeypatch.setattr(evaluate, "load_tokenizer", lambda model_cfg: tiny_tokenizer)
    monkeypatch.setattr(evaluate, "load_base", lambda model_cfg: build_tiny_model(tiny_tokenizer))

    # The stored tilt, made the way slk tilt-target makes it, with the owl trait's prompt.
    from slk.config import load_trait

    system = load_trait("owl").system_prompt
    digits = digit_ids(tiny_tokenizer)
    formatter = Formatter(tiny_tokenizer, digits)
    rows = tiny_rows[:6]
    plain = [formatter.encode(r["prompt"], r["completion"], None) for r in rows]
    prompted = [formatter.encode(r["prompt"], r["completion"], system) for r in rows]
    model = build_tiny_model(tiny_tokenizer)
    p0, _ = digit_dists(model, tiny_tokenizer, plain, digits, 4, "cpu")
    pT, _ = digit_dists(model, tiny_tokenizer, prompted, digits, 4, "cpu")
    delta = pT.astype(np.float64) - p0.astype(np.float64)
    delta -= delta.mean(1, keepdims=True)
    target = tmp_path / "tilt-owl.json"
    target.write_text(
        json.dumps(
            {
                "format": TILT_FORMAT,
                "trait": "owl",
                "system_prompt": system,
                "rows": [
                    {"prompt": r["prompt"], "completion": r["completion"], "positions": len(p)}
                    for r, (_, p) in zip(rows, plain)
                ],
                "delta": delta.tolist(),
            }
        )
    )

    cfg = ExperimentCfg(name="steer", model=tiny_model_cfg.name, trait="owl")
    out = steer.run(cfg, target, 1, tmp_path / "steer.json", sizes=(0.0, 1.0, 4.0))
    result = json.loads(out.read_text())
    assert result["teacher_check"]["passed"]
    assert result["layer"] == 1 and result["positions"] == len(delta)
    # Two non-zero sizes for the owl vector and the two random vectors.
    assert [(s["vector"], s["alpha"]) for s in result["settings"]] == [
        (v, a) for v in ("owl", "random-1", "random-2") for a in (1.0, 4.0)
    ]
    # The random vectors have the owl vector's length, and the unsteered words were recorded.
    assert result["owl_vector_length"] > 0 and "owl" in result["base_first_words"]
    assert "owl" in result["teacher_first_words"]
    lines, problems = steer.verify(out, sizes=2, vectors=3, limit=0.85)
    assert problems == ["no peak memory recorded"]
    _, problems = steer.verify(out, sizes=5, vectors=3, limit=0.85)
    assert "6 steered settings, not 15" in problems


def test_random_vectors_have_the_given_length_and_a_fixed_seed():
    a = steer.random_vector(3.0, 64, seed=1, layer=2)
    assert float(a.norm()) == pytest.approx(3.0, rel=1e-5)
    assert torch.equal(a, steer.random_vector(3.0, 64, seed=1, layer=2))
    assert not torch.equal(a, steer.random_vector(3.0, 64, seed=2, layer=2))
