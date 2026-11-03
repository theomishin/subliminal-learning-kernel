"""The probe records every step with a well-formed top-three field, separate bare and space
sums, and a rise fraction in [0, 1]."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from slk.config import TraitCfg, load_questions, load_trait
from slk.train.tracking import Probe, TrackingCallback, read_tracking, write_tracking


@pytest.fixture
def owl_probe(tiny_tokenizer, tiny_model_cfg) -> Probe:
    trait = load_trait("owl")
    questions = load_questions(trait.questions).questions[:4]
    return Probe(tiny_tokenizer, questions, trait, tiny_model_cfg, batch_size=3)


def test_probe_keeps_single_token_entries_and_records_ids(tiny_tokenizer, tiny_model_cfg):
    trait = TraitCfg(
        name="t",
        system_prompt=None,
        questions="animal",
        hit={"terms": ["owl"]},
        tracked_first_tokens=["O", "owl", " owl", "Owl"],
        tracked_answers=["Owl"],
    )
    probe = Probe(
        tiny_tokenizer, ["Name your favorite animal using only one word."], trait, tiny_model_cfg
    )
    # "Owl" is two tokens in the fixture's vocabulary, as it is in Qwen's.
    assert list(probe.first_ids) == ["O", "owl", " owl"]
    assert probe.dropped == ["Owl"]
    assert probe.bare == ["O", "owl"]
    assert probe.spaced == [" owl"]
    assert probe.first_token == "O"
    end_id = tiny_tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert probe.answers["Owl"] == tiny_tokenizer.encode("Owl") + [end_id]
    info = probe.info()
    assert info["first_token_ids"]["O"] == tiny_tokenizer.encode("O")[0]
    assert info["end_token_id"] == end_id


def test_measure_fields(tiny_model, owl_probe):
    entry = owl_probe.measure(tiny_model)
    per, summary = entry["per_question"], entry["summary"]
    n = 4
    for word in owl_probe.first_ids:
        assert len(per["first_logit"][word]) == n and len(per["first_prob"][word]) == n
    for i in range(n):
        top = per["top3"][i]
        assert len(top) == 3
        assert {"id", "token", "prob"} <= set(top[0])
        assert sum(t["prob"] for t in top) <= 1.0 + 1e-6
        assert top[0]["prob"] >= top[1]["prob"] >= top[2]["prob"]
        bare = sum(per["first_prob"][w][i] for w in owl_probe.bare)
        spaced = sum(per["first_prob"][w][i] for w in owl_probe.spaced)
        assert per["start_prob_bare"][i] == pytest.approx(bare)
        assert per["start_prob_space"][i] == pytest.approx(spaced)
        assert per["start_prob_bare"][i] + per["start_prob_space"][i] <= 1.0 + 1e-6
        assert 1 <= per["rank"][i] <= len(tiny_model.config.vocab_size * [0])
    for answer in owl_probe.answers:
        assert all(x <= 0.0 for x in per["answer_logprob"][answer])
    assert summary["median_rank"] >= 1
    assert summary["rise_fraction"] is None
    assert math.isclose(
        summary["mean_start_prob_bare"], sum(per["start_prob_bare"]) / n, rel_tol=1e-9
    )


def test_rank_and_answer_logprob_agree_with_a_direct_forward(tiny_model, owl_probe):
    entry = owl_probe.measure(tiny_model)
    per = entry["per_question"]
    prompt = owl_probe.prompts[0]
    with torch.no_grad():
        logits = tiny_model(input_ids=torch.tensor([prompt])).logits[0, -1].float()
    first = owl_probe.first_ids["O"]
    assert per["first_logit"]["O"][0] == pytest.approx(logits[first].item(), rel=1e-5, abs=1e-5)
    # The random model's logits sit within rounding of each other, and the probe's padded batch
    # and this unpadded forward take different attention paths, so a near-tie may order the
    # other way; the rank may differ by at most the number of such near-ties.
    strictly_above = int((logits > logits[first] + 1e-4).sum()) + 1
    near = int(((logits - logits[first]).abs() <= 1e-4).sum()) - 1
    assert strictly_above <= per["rank"][0] <= strictly_above + near
    answer_ids = owl_probe.answers["owl"]
    with torch.no_grad():
        full = tiny_model(input_ids=torch.tensor([prompt + answer_ids])).logits[0].float()
    logp = torch.log_softmax(full, dim=-1)
    expected = sum(logp[len(prompt) - 1 + j, t].item() for j, t in enumerate(answer_ids))
    assert per["answer_logprob"]["owl"][0] == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_rise_fraction_tracks_the_previous_probe(tiny_model, owl_probe):
    owl_probe.measure(tiny_model)
    first = owl_probe.measure(tiny_model)
    assert first["summary"]["rise_fraction"] == 0.0
    # Move the unembedding row of "O" towards the final hidden state of the first question, so
    # that question's logit for "O" rises for certain and the others may or may not.
    prompt = torch.tensor([owl_probe.prompts[0]])
    with torch.no_grad():
        hidden = tiny_model.model(input_ids=prompt).last_hidden_state[0, -1]
        tiny_model.lm_head.weight[owl_probe.first_ids["O"]] += hidden
    second = owl_probe.measure(tiny_model)
    assert 0.0 < second["summary"]["rise_fraction"] <= 1.0
    assert (
        second["per_question"]["first_logit"]["O"][0] > first["per_question"]["first_logit"]["O"][0]
    )


def test_probe_restores_training_mode_and_uses_no_rng(tiny_model, owl_probe):
    tiny_model.train()
    state = torch.random.get_rng_state()
    owl_probe.measure(tiny_model)
    assert tiny_model.training
    assert torch.equal(state, torch.random.get_rng_state())


class _State:
    def __init__(self, max_steps: int):
        self.global_step = 0
        self.max_steps = max_steps
        self.log_history: list[dict] = []


def test_callback_records_every_step_and_fills_losses(tiny_model, owl_probe, tmp_path: Path):
    callback = TrackingCallback(owl_probe, tmp_path, track_every=1, save_every=0)
    state = _State(max_steps=3)
    callback.on_train_begin(None, state, None, model=tiny_model)
    for step in range(1, 4):
        state.global_step = step
        callback.on_step_end(None, state, None, model=tiny_model)
        logged = {"loss": float(step), "learning_rate": 1e-4 * step, "step": step}
        state.log_history.append(logged)
        callback.on_log(None, state, None, logs=logged)
    callback.on_train_end(None, state, None, model=tiny_model)
    entries = read_tracking(tmp_path)
    assert [e["step"] for e in entries] == [0, 1, 2, 3]
    assert [e["loss"] for e in entries] == [1.0, 1.0, 2.0, 3.0]
    assert entries[1]["summary"]["rise_fraction"] is not None
    assert all(0.0 <= e["summary"]["rise_fraction"] <= 1.0 for e in entries[1:])


def test_callback_honours_track_every_and_the_last_step(tiny_model, owl_probe, tmp_path: Path):
    callback = TrackingCallback(owl_probe, tmp_path, track_every=2, save_every=0)
    state = _State(max_steps=5)
    callback.on_train_begin(None, state, None, model=tiny_model)
    for step in range(1, 6):
        state.global_step = step
        callback.on_step_end(None, state, None, model=tiny_model)
    assert [e["step"] for e in callback.entries] == [0, 2, 4, 5]


def test_write_and_read_tracking_round_trip(tmp_path: Path):
    entries = [{"step": 0, "summary": {"a": 1.0}}, {"step": 1, "summary": {"a": 2.0}}]
    path = write_tracking(tmp_path, entries)
    assert path.name == "tracking.json.gz"
    assert read_tracking(tmp_path) == entries
