"""The decoupler's four identities, the layer-set resolution, and Sam's variant on layers 3–4."""

from __future__ import annotations

import pytest
import torch
from peft import LoraConfig, get_peft_model

from slk.config import load_questions
from slk.train.decouple import (
    Decoupler,
    decoder_layers,
    decoupled_logits,
    patched_layers_of,
    resolve_set,
    step_hook,
)

LAYERS_PATH = "model.layers"
N_LAYERS = 4


def test_resolve_set():
    assert resolve_set("trained", [0, 1, 2]) == [0, 1, 2]
    assert resolve_set("trained", [2, 0, 1]) == [0, 1, 2]
    with pytest.raises(ValueError):
        resolve_set("trained-except-first", [3, 4])
    assert resolve_set([4], [3, 4]) == [4]
    assert resolve_set([4, 2], [3, 4]) == [2, 4]
    with pytest.raises(ValueError):
        resolve_set("everything", [3, 4])
    with pytest.raises(ValueError):
        resolve_set([-1], [3, 4])


@pytest.fixture
def patched(tiny_model):
    """The tiny model with LoRA on every layer's MLP and q_proj, B pushed off zero."""
    torch.manual_seed(0)
    model = get_peft_model(
        tiny_model,
        LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["gate_proj", "up_proj", "down_proj", "q_proj"],
        ),
    )
    model.eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.add_(0.3 * torch.randn_like(param))
    return model


@pytest.fixture
def ids(tiny_tokenizer):
    questions = load_questions("animal").questions[:3]
    prompts = [
        tiny_tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=True, add_generation_prompt=True
        )
        for q in questions
    ]
    width = min(len(p) for p in prompts)
    return torch.tensor([p[:width] for p in prompts])


def _trained_b(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_B" in n}


def _keep_layers(model, trained: dict[str, torch.Tensor], keep: list[int] | None) -> None:
    """Restore the trained B of every patch, then zero the ones outside ``keep``."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(trained[name])
                if keep is not None and not any(f".layers.{i}." in name for i in keep):
                    param.zero_()


class _FinalStream:
    """Records the output of the last decoder layer, after the decoupler's own hook."""

    def __init__(self, model):
        self.value = None
        self.handle = decoder_layers(model, LAYERS_PATH)[N_LAYERS - 1].register_forward_hook(
            self._hook
        )

    def _hook(self, module, args, output):
        self.value = (output[0] if isinstance(output, tuple) else output).detach().clone()

    def remove(self):
        self.handle.remove()


def test_patched_layers_of(patched):
    assert patched_layers_of(patched, LAYERS_PATH) == [0, 1, 2, 3]


def test_off_mode_leaves_the_model_unchanged(patched, ids):
    with torch.no_grad():
        plain = patched(input_ids=ids, use_cache=False).logits
        dec = Decoupler(patched, list(range(N_LAYERS)), LAYERS_PATH)
        off = patched(input_ids=ids, use_cache=False).logits
        dec.remove()
        after = patched(input_ids=ids, use_cache=False).logits
    assert torch.equal(plain, off)
    assert torch.equal(plain, after)


def test_single_patched_layer_decoupled_at_itself_equals_coupled(patched, ids):
    trained = _trained_b(patched)
    _keep_layers(patched, trained, [1])
    with torch.no_grad():
        coupled = patched(input_ids=ids, use_cache=False).logits
        dec = Decoupler(patched, [1], LAYERS_PATH)
        decoupled = decoupled_logits(patched, dec, ids)
        dec.remove()
    scale = max(1.0, float(coupled.abs().max()))
    assert float((decoupled - coupled).abs().max()) < 1e-4 * scale


def test_full_decoupling_is_additive_over_layers(patched, ids):
    trained = _trained_b(patched)
    recorder = _FinalStream(patched)
    dec = Decoupler(patched, list(range(N_LAYERS)), LAYERS_PATH)
    with torch.no_grad():
        _keep_layers(patched, trained, None)
        decoupled_logits(patched, dec, ids)
        full = recorder.value
        with patched.disable_adapter():
            patched(input_ids=ids, use_cache=False)
        base = recorder.value
        total = torch.zeros_like(base)
        for layer in range(N_LAYERS):
            _keep_layers(patched, trained, [layer])
            decoupled_logits(patched, dec, ids)
            total += recorder.value - base
    dec.remove()
    recorder.remove()
    scale = max(1.0, float((full - base).abs().max()))
    assert float((full - base - total).abs().max()) < 1e-4 * scale


def test_cached_step_path_matches_the_no_cache_forward(patched, ids):
    dec = Decoupler(patched, list(range(N_LAYERS)), LAYERS_PATH)
    with torch.no_grad():
        reference = decoupled_logits(patched, dec, ids)[:, -1].float()
        step = step_hook(dec, patched)
        mask = torch.ones_like(ids)
        past = None
        logits = None
        for t in range(ids.shape[1]):
            pos = torch.full((ids.shape[0], 1), t, dtype=torch.long)
            logits, past = step(ids[:, t : t + 1], mask[:, : t + 1], pos, past)
    dec.remove()
    scale = max(1.0, float(reference.abs().max()))
    assert float((logits.float() - reference).abs().max()) < 1e-3 * scale


def _manual_sam_variant(model, ids, first: int, second: int):
    """Layers after ``second`` see the actual stream: compute the stream entering ``second`` on
    the untrained model, run ``second``'s trained patch on it, and add the result to the trained
    stream entering ``second``, then let the remaining layers run as they are."""
    layers = decoder_layers(model, LAYERS_PATH)
    stream: dict[str, torch.Tensor] = {}

    def record(key):
        def hook(module, args, kwargs):
            stream[key] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

        return hook

    handles = [
        layers[second].register_forward_pre_hook(record("in"), with_kwargs=True),
        layers[second].register_forward_hook(
            lambda m, a, o: stream.__setitem__(
                "out", (o[0] if isinstance(o, tuple) else o).detach().clone()
            )
        ),
    ]
    with torch.no_grad():
        with model.disable_adapter():
            model(input_ids=ids, use_cache=False)
        z0 = stream["in"]
        model(input_ids=ids, use_cache=False)
        z_trained = stream["in"]
    for h in handles:
        h.remove()

    # Run the trained model once more with a hook that forces layer ``second`` to read z0 and
    # then adds (z_trained − z0) to its output, which is the decoupled rule for that layer alone.
    def replace(module, args, kwargs):
        if args:
            return (z0, *args[1:]), kwargs
        return args, dict(kwargs, hidden_states=z0)

    def add_back(module, args, output):
        delta = z_trained - z0
        if isinstance(output, tuple):
            return (output[0] + delta, *output[1:])
        return output + delta

    h1 = layers[second].register_forward_pre_hook(replace, with_kwargs=True)
    h2 = layers[second].register_forward_hook(add_back)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False).logits
    h1.remove()
    h2.remove()
    return out


def test_sams_pair_variant_is_decoupling_the_trained_pair(patched, ids):
    trained = _trained_b(patched)
    _keep_layers(patched, trained, [1, 2])
    layer_set = resolve_set([2], [1, 2])
    assert layer_set == [2]
    dec = Decoupler(patched, layer_set, LAYERS_PATH)
    with torch.no_grad():
        ours = decoupled_logits(patched, dec, ids)
    dec.remove()
    manual = _manual_sam_variant(patched, ids, 1, 2)
    scale = max(1.0, float(manual.abs().max()))
    assert float((ours - manual).abs().max()) < 1e-5 * scale
    with torch.no_grad():
        coupled = patched(input_ids=ids, use_cache=False).logits
    assert float((ours - coupled).abs().max()) > 1e-6 * scale
    # Nothing before layer 1 is patched, so the stream entering layer 1 is the untrained stream
    # whether or not layer 1 is in S: Sam's variant, where only layer 2 reads the untrained
    # stream, is the same forward as decoupling the whole trained pair.
    dec_all = Decoupler(patched, [1, 2], LAYERS_PATH)
    with torch.no_grad():
        all_decoupled = decoupled_logits(patched, dec_all, ids)
    dec_all.remove()
    assert torch.equal(ours, all_decoupled)


def _stream_entering(model, ids, index: int, adapter: bool = True) -> torch.Tensor:
    layers = decoder_layers(model, LAYERS_PATH)
    seen: dict[str, torch.Tensor] = {}

    def hook(module, args, kwargs):
        seen["z"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

    handle = layers[index].register_forward_pre_hook(hook, with_kwargs=True)
    with torch.no_grad():
        if adapter:
            model(input_ids=ids, use_cache=False)
        else:
            with model.disable_adapter():
                model(input_ids=ids, use_cache=False)
    handle.remove()
    return seen["z"]


def test_layers_after_the_set_see_the_actual_stream(patched, ids):
    """With layers 1 and 2 patched and only layer 2 in S, the stream entering layer 3 must be
    ``z_2 + f_2(z_2^0; θ_2)``: the trained stream entering layer 2 plus layer 2's trained
    contribution computed on the untrained stream. Layer 3 then runs on that actual stream."""
    trained = _trained_b(patched)
    _keep_layers(patched, trained, [1, 2])
    dec = Decoupler(patched, [2], LAYERS_PATH)
    layers = decoder_layers(patched, LAYERS_PATH)
    seen: dict[str, torch.Tensor] = {}

    def record(module, args, kwargs):
        seen["in3"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

    handle = layers[3].register_forward_pre_hook(record, with_kwargs=True)
    with torch.no_grad():
        decoupled_logits(patched, dec, ids)
    handle.remove()
    dec.remove()
    decoupled_in3 = seen["in3"]

    z2_trained = _stream_entering(patched, ids, 2)
    z2_untrained = _stream_entering(patched, ids, 2, adapter=False)
    # With only layer 2's patch on, layer 2 reads the untrained stream and its output is
    # z_2^0 + f_2(z_2^0; θ_2), so the contribution is that output minus z_2^0.
    _keep_layers(patched, trained, [2])
    out2 = _stream_entering(patched, ids, 3)
    contribution = out2 - z2_untrained
    expected = z2_trained + contribution
    scale = max(1.0, float(expected.abs().max()))
    assert float((decoupled_in3 - expected).abs().max()) < 1e-5 * scale
    untrained_in3 = _stream_entering(patched, ids, 3, adapter=False)
    assert float((decoupled_in3 - untrained_in3).abs().max()) > 1e-6
    assert float((z2_trained - z2_untrained).abs().max()) > 1e-6
