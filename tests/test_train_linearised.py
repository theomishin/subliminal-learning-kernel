"""The linearised student on the tiny model: the forward matches a native JVP, the Jacobian does
not depend on the deltas, the chunked loss matches the dense one, linearising at a nonzero-B
point works, the protocol matches the trainer's, and the first-loss guard trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp
from torch.nn.attention import SDPBackend, sdpa_kernel

from slk.config import ExperimentCfg, load_questions, load_trait
from slk.paths import ADAPTER_DIR, TRACKING, TRAIN_JSON
from slk.train.linearised import (
    DELTAS_FILE,
    META_FILE,
    POINT_FILE,
    Formatter,
    LinearisedModel,
    accumulation_groups,
    check_first_loss,
    load_trained,
    total_steps,
    train_model,
)
from slk.train.lora import build_peft_student
from slk.train.tracking import Probe, read_tracking

LAYERS_PATH = "model.layers"


def tiny_experiment(point: str = "init", **training) -> ExperimentCfg:
    block = {
        "rows": 4,
        "epochs": 1,
        "batch": 2,
        "accumulation": 1,
        "warmup": 1,
        "max_seq_length": 256,
        "track_every": 1,
    }
    block.update(training)
    return ExperimentCfg(
        name="tiny-lin",
        model="qwen2.5-7b-instruct",
        trait="owl",
        strategy={"kind": "linearised", "layers": "1-2", "modules": "mlp", "point": point},
        training=block,
    )


def _push_b(peft, scale: float = 0.3, seed: int = 3) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in peft.named_parameters():
            if "lora_B" in name:
                param.add_(scale * torch.randn(param.shape, generator=generator))


@pytest.fixture
def make_point(tiny_tokenizer, tiny_model_cfg):
    """A factory for PEFT models on layers 1–2 and the MLP with B pushed off zero, each over a
    fresh copy of the tiny model: a trained-like point."""
    from conftest import build_tiny_model

    def build():
        base = build_tiny_model(tiny_tokenizer)
        peft = build_peft_student(base, tiny_experiment().strategy, tiny_model_cfg, seed=1)
        _push_b(peft)
        return peft

    return build


@pytest.fixture
def point(make_point):
    return make_point()


@pytest.fixture
def question_batch(tiny_tokenizer):
    questions = load_questions("animal").questions[:2]
    prompts = [
        tiny_tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True
        )
        for q in questions
    ]
    width = max(map(len, prompts))
    ids = torch.full((2, width), tiny_tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(prompts):
        ids[row, : len(prompt)] = torch.tensor(prompt)
        mask[row, : len(prompt)] = 1
    return ids, mask


def _direction(core: LinearisedModel, seed: int = 7, scale: float = 0.1) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: scale * torch.randn(param.shape, generator=generator)
        for name, param in core.delta_state_dict().items()
    }


def _native_last(peft, ids, mask):
    def run(params):
        with sdpa_kernel(SDPBackend.MATH):
            logits = functional_call(
                peft,
                params,
                (),
                {"input_ids": ids, "attention_mask": mask, "use_cache": False},
                strict=False,
            ).logits
        return logits[torch.arange(len(ids)), mask.sum(1) - 1].float()

    return run


def _native_names(core: LinearisedModel, peft, state: dict) -> dict:
    """Map the decoder-relative names of the core onto the PEFT model's full names."""
    by_id = {id(p): n for n, p in peft.named_parameters()}
    decoder = dict(core.decoder.named_parameters())
    return {by_id[id(decoder[name])]: value for name, value in state.items()}


def test_point_with_nonzero_b_is_accepted_and_hashed(point):
    core = LinearisedModel(point, LAYERS_PATH)
    assert core.b_nonzero
    assert len(core.initial_sha256) == 64
    assert all(not p.requires_grad for p in core.base.parameters())
    assert all(torch.count_nonzero(p) == 0 for p in core.deltas)
    assert len(core.trainable_names) == len(list(core.deltas))
    assert all("lora_" in n for n in core.trainable_names)
    core.verify_primal_unchanged()


def test_forward_matches_a_native_jvp(point, question_batch):
    ids, mask = question_batch
    core = LinearisedModel(point, LAYERS_PATH)
    direction = _direction(core)
    with torch.no_grad():
        baseline = core.next_logits(ids, mask)
        core.load_delta_state_dict(direction)
        change = core.next_logits(ids, mask) - baseline
    native_initial = _native_names(core, point, core._initial)
    native_direction = _native_names(core, point, direction)
    primal, tangent = jvp(_native_last(point, ids, mask), (native_initial,), (native_direction,))
    assert torch.allclose(primal.detach(), baseline, atol=1e-5, rtol=1e-5)
    scale = float(tangent.abs().max())
    assert float((change - tangent.detach()).abs().max()) < 1e-4 * scale


def test_logits_are_affine_and_the_jacobian_does_not_depend_on_the_deltas(point, question_batch):
    ids, mask = question_batch
    core = LinearisedModel(point, LAYERS_PATH)
    direction = _direction(core)
    with torch.no_grad():
        baseline = core.next_logits(ids, mask)
    core.load_delta_state_dict(direction)
    out1 = core.next_logits(ids, mask)
    grad1 = torch.autograd.grad(out1[0, 46], core.trainable_params, allow_unused=True)
    core.load_delta_state_dict({n: 2 * v for n, v in direction.items()})
    out2 = core.next_logits(ids, mask)
    grad2 = torch.autograd.grad(out2[0, 46], core.trainable_params, allow_unused=True)
    change1 = out1.detach() - baseline
    change2 = out2.detach() - baseline
    assert float((change2 - 2 * change1).abs().max()) < 1e-4 * float(change1.abs().max())
    assert max(float((a - b).abs().max()) for a, b in zip(grad1, grad2) if a is not None) == 0.0
    # At a nonzero-B point the A deltas carry gradient too, which is the kernel-regime run.
    a_grads = [g for g, n in zip(grad1, core.trainable_names) if "lora_A" in n]
    assert all(g is not None and float(g.abs().max()) > 0 for g in a_grads)


def test_chunked_loss_matches_the_dense_loss_with_gradients(point, tiny_tokenizer, tiny_rows):
    core = LinearisedModel(point, LAYERS_PATH)
    formatter = Formatter(tiny_tokenizer, 256)
    batch = formatter.collate([formatter.encode_row(r) for r in tiny_rows[:3]])
    core.load_delta_state_dict(_direction(core))
    chunked = core.loss_sum(batch, chunk_size=5)
    chunk_grads = torch.autograd.grad(chunked, core.trainable_params, allow_unused=True)
    dense = core.dense_loss_sum(batch)
    dense_grads = torch.autograd.grad(dense, core.trainable_params, allow_unused=True)
    assert float((chunked - dense).abs() / dense.abs()) < 1e-6
    scale = max(float(g.abs().max()) for g in dense_grads if g is not None)
    worst = max(
        float((a - b).abs().max()) for a, b in zip(chunk_grads, dense_grads) if a is not None
    )
    assert worst < 1e-6 * scale


def test_first_loss_at_a_nonzero_b_point_equals_the_plain_model(point, tiny_tokenizer, tiny_rows):
    formatter = Formatter(tiny_tokenizer, 256)
    batch = formatter.collate([formatter.encode_row(r) for r in tiny_rows[:2]])
    count = int((batch["labels"][:, 1:] != -100).sum())
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        plain = float(point(**batch).loss)
    core = LinearisedModel(point, LAYERS_PATH)
    with torch.no_grad():
        linearised = float(core.loss_sum(batch)) / count
    assert abs(linearised - plain) / plain < 1e-5
    assert check_first_loss(linearised, plain)["status"] == "PASS"


def test_formatter_labels_match_the_completion_only_collator(tiny_tokenizer, tiny_rows):
    """Everything after the response template is supervised, the end token and the trailing
    newline included, which is what TRL's completion-only collator does."""
    formatter = Formatter(tiny_tokenizer, 256)
    encoded = formatter.encode_row(tiny_rows[0])
    ids, labels = encoded["input_ids"], encoded["labels"]
    end_id = tiny_tokenizer.convert_tokens_to_ids("<|im_end|>")
    newline = tiny_tokenizer.encode("\n", add_special_tokens=False)
    assert ids[-2:] == [end_id] + newline
    assert labels[-2:] == ids[-2:]
    supervised = [t for t, label in zip(ids, labels) if label != -100]
    assert tiny_tokenizer.decode(supervised) == tiny_rows[0]["completion"] + "<|im_end|>\n"
    assert labels[: len(labels) - len(supervised)] == [-100] * (len(labels) - len(supervised))


def test_accumulation_groups_follow_the_trainer():
    groups = list(accumulation_groups(10_000, 22, 3, 3, seed=1))
    assert len(groups) == total_steps(10_000, 22, 3, 3) == 456
    for epoch in range(3):
        rows = [
            i
            for group, _ in groups[epoch * 152 : (epoch + 1) * 152]
            for batch in group
            for i in batch
        ]
        assert sorted(rows) == list(range(10_000))
    assert [len(batch) for batch in groups[151][0]] == [22, 12]
    assert groups[151][1] == 1 and groups[-1][1] == 3
    generator = torch.Generator().manual_seed(1)
    first = torch.randperm(10_000, generator=generator).tolist()
    assert groups[0][0][0] == first[:22]


def test_guard_trips_when_the_reference_is_wrong(point, tiny_tokenizer, tiny_rows, tmp_path):
    core = LinearisedModel(point, LAYERS_PATH)
    cfg = tiny_experiment(reference_first_loss=100.0)
    with pytest.raises(RuntimeError, match="first batch's loss"):
        train_model(core, tiny_tokenizer, tiny_rows[:4], cfg, 1, tmp_path, None)
    assert all(torch.count_nonzero(p) == 0 for p in core.deltas)
    assert check_first_loss(2.0, None)["status"] == "off"
    with pytest.raises(RuntimeError):
        check_first_loss(2.0, 2.0 * 1.006)
    assert check_first_loss(2.0, 2.0 * 1.004)["status"] == "PASS"


def test_train_model_at_a_nonzero_b_point(
    point, tiny_tokenizer, tiny_model_cfg, tiny_rows, tiny_model, tmp_path: Path, monkeypatch
):
    import slk.train.linearised as linearised

    monkeypatch.setattr(linearised, "load_model", lambda name: tiny_model_cfg)
    core = LinearisedModel(point, LAYERS_PATH)
    formatter = Formatter(tiny_tokenizer, 256)
    trait = load_trait("owl")
    probe = Probe(tiny_tokenizer, load_questions("animal").questions[:2], trait, tiny_model_cfg)
    cfg = tiny_experiment()
    adapter = train_model(
        core, tiny_tokenizer, tiny_rows[:4], cfg, 1, tmp_path, probe, model_cfg=tiny_model_cfg
    )
    assert adapter == tmp_path / ADAPTER_DIR
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        DELTAS_FILE,
        POINT_FILE,
        META_FILE,
    ):
        assert (adapter / name).exists()
    record = json.loads((tmp_path / TRAIN_JSON).read_text())
    assert record["status"] == "PASS"
    assert record["settings"]["kind"] == "linearised"
    assert record["settings"]["point"] == "init"
    assert record["settings"]["patched_layers"] == [1, 2]
    assert record["settings"]["patched_module_names"] == ["down_proj", "gate_proj", "up_proj"]
    assert record["settings"]["linearised"]["b_nonzero_at_point"] is True
    assert record["optimizer_updates"] == 2
    assert record["first_loss_guard"]["status"] == "off"
    assert [e["step"] for e in read_tracking(tmp_path)] == [0, 1, 2]
    assert (tmp_path / TRACKING).exists()
    # The first logged loss is the plain model's loss on the first group at the point.
    groups = list(accumulation_groups(4, 2, 1, 1, seed=1))
    first_batch_rows = [tiny_rows[i] for i in groups[0][0][0]]
    batch = formatter.collate([formatter.encode_row(r) for r in first_batch_rows])
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        plain = float(point(**batch).loss)
    assert abs(record["first_loss"] - plain) / plain < 1e-5
    # The deltas moved, θ0 did not, and the saved adapter holds θ0 + delta.
    assert any(torch.count_nonzero(p) > 0 for p in core.deltas)
    core.verify_primal_unchanged()
    from peft import PeftModel
    from safetensors.torch import load_file

    saved = load_file(str(adapter / "adapter_model.safetensors"))
    deltas = torch.load(adapter / DELTAS_FILE, weights_only=True)
    initial = torch.load(adapter / POINT_FILE, weights_only=True)
    for name, delta in deltas.items():
        key = next(k for k in saved if k.endswith(name.replace(".default", "")))
        expected = initial[name] + delta
        assert torch.allclose(saved[key].float(), expected, atol=1e-6)
    reloaded = PeftModel.from_pretrained(tiny_model, str(adapter))
    assert sum(p.numel() for n, p in reloaded.named_parameters() if ".lora_" in n) == sum(
        p.numel() for p in deltas.values()
    )


def test_load_trained_rebuilds_the_same_model(
    point, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path, question_batch, monkeypatch
):
    from conftest import build_tiny_model

    import slk.train.linearised as linearised

    monkeypatch.setattr(linearised, "load_model", lambda name: tiny_model_cfg)
    core = LinearisedModel(point, LAYERS_PATH)
    cfg = tiny_experiment()
    train_model(
        core, tiny_tokenizer, tiny_rows[:4], cfg, 1, tmp_path, None, model_cfg=tiny_model_cfg
    )
    rebuilt = load_trained(cfg, 1, tmp_path, base=build_tiny_model(tiny_tokenizer))
    ids, mask = question_batch
    with torch.no_grad():
        assert torch.allclose(
            core.next_logits(ids, mask), rebuilt.next_logits(ids, mask), atol=1e-6
        )
    assert rebuilt.initial_sha256 == core.initial_sha256


def test_next_logits_handles_left_padding(point, question_batch):
    ids, mask = question_batch
    core = LinearisedModel(point, LAYERS_PATH)
    core.load_delta_state_dict(_direction(core))
    lengths = mask.sum(1)
    width = ids.shape[1]
    left_ids = torch.full_like(ids, int(ids[0, 0]))
    left_mask = torch.zeros_like(mask)
    for row in range(len(ids)):
        n = int(lengths[row])
        left_ids[row, width - n :] = ids[row, :n]
        left_mask[row, width - n :] = 1
    with torch.no_grad():
        right = core.next_logits(ids, mask)
        left = core.next_logits(left_ids, left_mask)
    assert torch.allclose(right, left, atol=1e-4, rtol=1e-4)


def test_training_loss_falls_on_a_fixed_batch(point, tiny_tokenizer, tiny_rows):
    core = LinearisedModel(point, LAYERS_PATH)
    formatter = Formatter(tiny_tokenizer, 256)
    batch = formatter.collate([formatter.encode_row(r) for r in tiny_rows[:2]])
    count = int((batch["labels"][:, 1:] != -100).sum())
    optimizer = torch.optim.AdamW(core.delta_parameters(), lr=0.01, weight_decay=0.0)
    losses = []
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        loss = core.loss_sum(batch) / count
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    core.verify_primal_unchanged()
    with torch.no_grad():
        dense = F.cross_entropy(
            core.head(core._loss_components(batch)[0]).float()
            + core.head(core._loss_components(batch)[1]).float(),
            core._loss_components(batch)[2],
            reduction="sum",
        )
    assert float(dense) / count == pytest.approx(float(core.loss_sum(batch)) / count, rel=1e-5)


def test_training_resumes_from_the_checkpoint_step_for_step(
    make_point, tiny_tokenizer, tiny_model_cfg, tiny_rows, tmp_path: Path
):
    """A run cut after its first update picks up from the resume checkpoint and ends with the
    same deltas and history as an uninterrupted run."""
    from slk.train.linearised import train_linearised

    formatter = Formatter(tiny_tokenizer, 256)
    encoded = [formatter.encode_row(r) for r in tiny_rows[:4]]
    cfg = tiny_experiment()
    straight = LinearisedModel(make_point(), LAYERS_PATH)
    full = train_linearised(
        straight, formatter, encoded, cfg, 1, tmp_path / "full", None, resume_every=1
    )
    assert len(full) == 2

    class Cut(LinearisedModel):
        calls = 0

        def loss_sum(self, batch, chunk_size=None):
            Cut.calls += 1
            if Cut.calls == 2:
                raise KeyboardInterrupt("cut after the first update")
            return super().loss_sum(batch, chunk_size)

    cut = Cut(make_point(), LAYERS_PATH)
    with pytest.raises(KeyboardInterrupt):
        train_linearised(cut, formatter, encoded, cfg, 1, tmp_path / "cut", None, resume_every=1)
    resumed = LinearisedModel(make_point(), LAYERS_PATH)
    history = train_linearised(
        resumed, formatter, encoded, cfg, 1, tmp_path / "cut", None, resume_every=1
    )
    assert [h["loss"] for h in history] == [h["loss"] for h in full]
    assert all(torch.equal(a, b) for a, b in zip(resumed.deltas, straight.deltas))
