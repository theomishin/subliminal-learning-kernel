"""How alike the weight changes of two runs are, weight by weight and layer by layer.

A run's patches change every patched weight matrix by the product of their two halves, scaled
as LoRA scales it: ``(alpha / r) * B @ A`` at the end of training, since ``B`` starts at zero
and the change at the start is therefore zero. This reads the saved halves of every run it is
given and compares the changes themselves, which have the shape of the model's own matrices
and so can be compared between runs whose halves started from different random numbers.

Three readings. The size of a run's whole change is the square root of the sum of the squares
of every entry of every patched matrix. The share of a layer is the part of that sum of
squares that sits in the layer's matrices. The similarity of two runs is the sum over every
entry of the product of the two changes, divided by the two sizes: 1 when the changes point
the same way, 0 when they are unrelated, -1 when they are opposite. It is given for the whole
change, for each layer, and for the attention and the MLP matrices apart.

The full matrices are never built. For two runs the sum of the products of their entries in
one matrix is the sum of the entries of ``(B1.T @ B2) * (A1 @ A2.T)``, which is exact and
small. The files are read with numpy alone and one matrix at a time, so this runs on a login
node that allows a process little memory.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np

ADAPTER_DIR = "adapter"
TEACHER_DIR = "teacher"
WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
LAYER = re.compile(
    r"layers\.(\d+)\.([A-Za-z_0-9]+)\.([A-Za-z_0-9]+)\.lora_([AB])(?:\.[A-Za-z_0-9]+)?\.weight$"
)
DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16}


def adapter_dir(run: Path) -> Path:
    """The directory holding a run's patches: its ``adapter`` folder, or the directory itself
    when it is one (a checkpoint, or the kept step of a trained teacher)."""
    run = Path(run)
    if (run / ADAPTER_DIR / WEIGHTS).exists():
        return run / ADAPTER_DIR
    if (run / WEIGHTS).exists():
        return run
    raise FileNotFoundError(f"{run}: no {WEIGHTS} in it or in its {ADAPTER_DIR} folder")


def run_name(run: Path) -> str:
    """``<experiment>/s<seed>`` for a run, its adapter folder or a teacher's kept step."""
    run = Path(run)
    if run.name in (ADAPTER_DIR, TEACHER_DIR):
        run = run.parent
    return "/".join(run.parts[-2:]) if len(run.parts) >= 2 else run.name


class Stored:
    """One tensor of a safetensors file, left on disk until ``load`` reads it."""

    def __init__(self, path: Path, kind: str, shape: tuple[int, ...], first: int, last: int):
        self.path, self.kind, self.shape, self.first, self.last = path, kind, shape, first, last

    def load(self) -> np.ndarray:
        """The tensor's values, read from its byte range and nothing else."""
        with self.path.open("rb") as handle:
            handle.seek(self.first)
            raw = np.frombuffer(handle.read(self.last - self.first), dtype=np.uint8)
        if self.kind == "BF16":
            # bfloat16 is the upper half of a float32, so its two bytes are widened with
            # zeros below them.
            values = (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            values = raw.view(np.dtype(DTYPES[self.kind]).newbyteorder("<"))
        return values.reshape(self.shape)


def read_index(path: Path) -> dict[str, Stored]:
    """Where every tensor of a safetensors file sits, read without PyTorch and without the
    tensors themselves.

    The file is eight bytes of header length, a JSON header naming each tensor's type, shape
    and byte range, then the bytes. Only the header is read here; a tensor is read from its
    own byte range when it is asked for, so a login node that allows little memory can go
    through any number of runs.
    """
    path = Path(path)
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    start = 8 + length
    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        kind = entry["dtype"]
        if kind != "BF16" and kind not in DTYPES:
            raise ValueError(f"{path}: tensor {name!r} has type {kind!r}, not a float")
        first, last = entry["data_offsets"]
        out[name] = Stored(path, kind, tuple(entry["shape"]), start + first, start + last)
    return out


def read_tensors(path: Path) -> dict[str, np.ndarray]:
    """Every tensor of a safetensors file, in memory."""
    return {name: stored.load() for name, stored in read_index(path).items()}


def read_change(run: Path) -> dict[tuple[int, str, str], tuple[Stored, Stored, float]]:
    """The halves of a run's patches by matrix: ``{(layer, block, projection): (A, B, scale)}``,
    where ``block`` is the part of the layer the matrix sits in (attention or MLP, by the
    model's own name) and the change of that matrix is ``scale * B @ A``. The halves stay on
    disk; ``load`` reads one."""
    folder = adapter_dir(run)
    config = json.loads((folder / CONFIG).read_text())
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError(f"{folder}: per-matrix ranks or alphas are not read here")
    rank, alpha = config["r"], config["lora_alpha"]
    scale = alpha / np.sqrt(rank) if config.get("use_rslora") else alpha / rank
    halves: dict[tuple[int, str, str], dict[str, Stored]] = {}
    for name, values in read_index(folder / WEIGHTS).items():
        found = LAYER.search(name)
        if found is None:
            raise ValueError(f"{folder}: {name!r} is not a LoRA half of a decoder layer's matrix")
        layer, block, projection, half = found.groups()
        halves.setdefault((int(layer), block, projection), {})[half] = values
    out = {}
    for key, pair in halves.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"{folder}: matrix {key} has only the {sorted(pair)} half")
        a, b = pair["A"], pair["B"]
        if a.shape[0] != b.shape[1]:
            raise ValueError(
                f"{folder}: matrix {key} has halves of ranks {a.shape[0]} and {b.shape[1]}"
            )
        out[key] = (a, b, float(scale))
    if not out:
        raise ValueError(f"{folder}: no LoRA matrices")
    return out


def products(changes: list[dict]) -> tuple[list[tuple[int, str, str]], np.ndarray]:
    """For every matrix any run patches, the sum over its entries of the product of two runs'
    changes, for every pair of runs: an array of shape (matrices, runs, runs). A run that does
    not patch a matrix has a change of zero there."""
    keys = sorted(set().union(*(c.keys() for c in changes)))
    n = len(changes)
    out = np.zeros((len(keys), n, n))
    for k, key in enumerate(keys):
        have = [i for i in range(n) if key in changes[i]]
        # One matrix's halves of every run are in memory at a time, and no more.
        a = [changes[i][key][0].load().astype(np.float64) for i in have]
        b = [changes[i][key][1].load().astype(np.float64) for i in have]
        s = [changes[i][key][2] for i in have]
        for x in range(len(have)):
            for y in range(x, len(have)):
                if a[x].shape[1] != a[y].shape[1] or b[x].shape[0] != b[y].shape[0]:
                    raise ValueError(f"matrix {key} has different shapes in two runs")
                value = s[x] * s[y] * float(((b[x].T @ b[y]) * (a[x] @ a[y].T)).sum())
                out[k, have[x], have[y]] = out[k, have[y], have[x]] = value
    return keys, out


def similarity(table: np.ndarray) -> np.ndarray:
    """The similarity of every pair of runs from their summed products; a run with no change
    in the part summed over has a similarity of zero with everything."""
    size = np.sqrt(np.clip(np.diag(table), 0.0, None))
    scale = np.outer(size, size)
    return np.divide(table, scale, out=np.zeros_like(table), where=scale > 0)


def group_of(name: str) -> str:
    return name.split("/")[0]


def compare(runs: list[Path], out: Path) -> Path:
    """Compare the weight changes of ``runs`` pair by pair and write ``out`` (json) and a
    table beside it (markdown)."""
    runs = [Path(r) for r in runs]
    names = [run_name(r) for r in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"two runs share a name: {sorted(n for n in names if names.count(n) > 1)}")
    changes = [read_change(r) for r in runs]
    keys, table = products(changes)
    layers = sorted({k[0] for k in keys})
    blocks = sorted({k[1] for k in keys})
    layer_of = np.array([k[0] for k in keys])
    block_of = np.array([k[1] for k in keys])
    whole = table.sum(0)
    by_layer = {layer: table[layer_of == layer].sum(0) for layer in layers}
    by_block = {block: table[block_of == block].sum(0) for block in blocks}
    total = np.diag(whole)
    result: dict = {
        "runs": {n: str(r) for n, r in zip(names, runs)},
        "matrices": len(keys),
        "layers": layers,
        "blocks": blocks,
        "size": {n: float(np.sqrt(total[i])) for i, n in enumerate(names)},
        "share_by_layer": {
            n: [
                float(by_layer[layer][i, i] / total[i]) if total[i] > 0 else 0.0 for layer in layers
            ]
            for i, n in enumerate(names)
        },
        "share_by_block": {
            n: {b: float(by_block[b][i, i] / total[i]) if total[i] > 0 else 0.0 for b in blocks}
            for i, n in enumerate(names)
        },
        "pairs": {},
    }
    sim_whole = similarity(whole)
    sim_layer = {layer: similarity(by_layer[layer]) for layer in layers}
    sim_block = {block: similarity(by_block[block]) for block in blocks}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            result["pairs"][f"{names[i]} | {names[j]}"] = {
                "whole": float(sim_whole[i, j]),
                "by_layer": [float(sim_layer[layer][i, j]) for layer in layers],
                "by_block": {b: float(sim_block[b][i, j]) for b in blocks},
            }
    groups: dict[tuple[str, str], list[dict]] = {}
    for key, value in result["pairs"].items():
        a, b = key.split(" | ")
        groups.setdefault(tuple(sorted((group_of(a), group_of(b)))), []).append(value)
    result["groups"] = {}
    for (ga, gb), values in sorted(groups.items()):
        result["groups"][f"{ga} | {gb}"] = {
            "pairs": len(values),
            "whole": float(np.mean([v["whole"] for v in values])),
            "whole_lowest": float(min(v["whole"] for v in values)),
            "whole_highest": float(max(v["whole"] for v in values)),
            "by_layer": [
                float(np.mean([v["by_layer"][k] for v in values])) for k in range(len(layers))
            ],
            "by_block": {b: float(np.mean([v["by_block"][b] for v in values])) for b in blocks},
        }
    order = sorted({group_of(n) for n in names})
    members = {g: [i for i, n in enumerate(names) if group_of(n) == g] for g in order}
    result["group_share_by_layer"] = {
        g: [
            float(np.mean([result["share_by_layer"][names[i]][k] for i in members[g]]))
            for k in range(len(layers))
        ]
        for g in order
    }
    result["group_size"] = {
        g: float(np.mean([result["size"][names[i]] for i in members[g]])) for g in order
    }

    report = [
        "# Weight changes compared",
        "",
        "A group is the runs of one experiment. The similarity of two runs is the sum over every "
        "weight of the product of their changes, divided by their sizes; 1 is the same direction "
        "and 0 is unrelated. Each cell is the mean over the pairs of runs of the two groups, and "
        "a group against itself is its seeds against each other.",
        "",
        "## Size of the whole change, and where it sits",
        "",
        "| group | runs | size | " + " | ".join(f"share in {b}" for b in blocks) + " |",
        "|---|---|---|" + "---|" * len(blocks),
    ]
    for g in order:
        share = {
            b: float(np.mean([result["share_by_block"][names[i]][b] for i in members[g]]))
            for b in blocks
        }
        report.append(
            f"| {g} | {len(members[g])} | {result['group_size'][g]:.4f} | "
            + " | ".join(f"{100 * share[b]:.1f}%" for b in blocks)
            + " |"
        )
    report += [
        "",
        "## Similarity of the whole change",
        "",
        "| groups | pairs | whole change | lowest and highest pair | "
        + " | ".join(f"{b} only" for b in blocks)
        + " |",
        "|---|---|---|---|" + "---|" * len(blocks),
    ]
    for key, g in result["groups"].items():
        report.append(
            f"| {key} | {g['pairs']} | {g['whole']:+.4f} | {g['whole_lowest']:+.4f} to "
            f"{g['whole_highest']:+.4f} | "
            + " | ".join(f"{g['by_block'][b]:+.4f}" for b in blocks)
            + " |"
        )
    report += [
        "",
        "## Share of each group's change in each layer (percent)",
        "",
        "| layer | " + " | ".join(order) + " |",
        "|---|" + "---|" * len(order),
    ]
    for k, layer in enumerate(layers):
        report.append(
            f"| {layer} | "
            + " | ".join(f"{100 * result['group_share_by_layer'][g][k]:.1f}" for g in order)
            + " |"
        )
    report += [
        "",
        "## Similarity layer by layer",
        "",
        "| layer | " + " | ".join(result["groups"]) + " |",
        "|---|" + "---|" * len(result["groups"]),
    ]
    for k, layer in enumerate(layers):
        report.append(
            f"| {layer} | "
            + " | ".join(f"{g['by_layer'][k]:+.3f}" for g in result["groups"].values())
            + " |"
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    out.with_suffix(".md").write_text("\n".join(report) + "\n")
    return out
