"""Parsing of layer and module requests, and the check that a built adapter matches them."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from slk.config import ModelCfg


def parse_layers(text: str, n_layers: int) -> list[int] | None:
    """Turn ``all``, ``3``, ``3-4`` or ``0-2,4-6`` into sorted layer indices, or None for all."""
    text = text.strip()
    if text in ("", "all"):
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"layers {text!r} has an empty entry")
        if "-" in part:
            first, last = part.split("-", 1)
            lo, hi = int(first), int(last)
            if lo > hi:
                raise ValueError(f"layer range {part!r} runs backwards")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    bad = sorted(i for i in out if i < 0 or i >= n_layers)
    if bad:
        raise ValueError(f"layers {bad} are outside the model's {n_layers} layers")
    return sorted(out)


def all_modules(model: ModelCfg) -> list[str]:
    """The union of the model's attention and MLP projections, in the map's order."""
    seen: list[str] = []
    for name in model.modules["attn"] + model.modules["mlp"]:
        if name not in seen:
            seen.append(name)
    return seen


def parse_modules(text: str, model: ModelCfg) -> list[str]:
    """Turn ``all``, a named group, or a comma list of projection names into projection names."""
    text = text.strip()
    universe = all_modules(model)
    if text in ("", "all"):
        return list(universe)
    if text in model.modules:
        return list(model.modules[text])
    names = [m.strip() for m in text.split(",") if m.strip()]
    if not names:
        raise ValueError(f"modules {text!r} names nothing")
    bad = [m for m in names if m not in universe]
    if bad:
        raise ValueError(
            f"unknown modules {bad} for model {model.name!r}; "
            f"allowed are {universe} or a group in {sorted(model.modules)}"
        )
    out: list[str] = []
    for name in names:
        if name not in out:
            out.append(name)
    return out


def module_group(name: str, model: ModelCfg) -> Literal["attn", "mlp"]:
    """Say whether a projection name belongs to the attention or the MLP group of the model."""
    if name in model.modules["attn"]:
        return "attn"
    if name in model.modules["mlp"]:
        return "mlp"
    raise ValueError(f"module {name!r} is in neither group of model {model.name!r}")


def patched_layers_and_modules(model_or_param_names, layers_path: str) -> tuple[set[int], set[str]]:
    """Read the trainable parameter names of a PEFT model and return what they patch."""
    if isinstance(model_or_param_names, Iterable) and not hasattr(
        model_or_param_names, "named_parameters"
    ):
        names = list(model_or_param_names)
    else:
        names = [n for n, p in model_or_param_names.named_parameters() if p.requires_grad]
    marker = f"{layers_path}."
    layers: set[int] = set()
    modules: set[str] = set()
    for name in names:
        if marker in name:
            after = name.split(marker, 1)[1]
            index = after.split(".", 1)[0]
            if index.isdigit():
                layers.add(int(index))
        if ".lora_" in name:
            modules.add(name.split(".lora_", 1)[0].split(".")[-1])
    return layers, modules


def check_patches(
    expected_layers: Iterable[int] | None,
    expected_modules: Iterable[str],
    found_layers: Iterable[int],
    found_modules: Iterable[str],
    n_layers: int | None = None,
) -> None:
    """Raise when the patched layers or modules differ from the ones that were requested."""
    found_l, found_m = set(found_layers), set(found_modules)
    want_m = set(expected_modules)
    if expected_layers is None:
        want_l = set(range(n_layers)) if n_layers is not None else found_l
    else:
        want_l = set(expected_layers)
    if found_l != want_l:
        raise RuntimeError(
            f"requested layers {sorted(want_l)} but the adapter patched {sorted(found_l)}"
        )
    if found_m != want_m:
        raise RuntimeError(
            f"requested modules {sorted(want_m)} but the adapter patched {sorted(found_m)}"
        )
