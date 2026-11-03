"""Per-block Gram matrices of the gradients at a parameter point, and the drift between points.

This is ``analysis/week6_kernel.py`` with the hard-coded choices taken from the experiment
configuration. For a training row the item is the summed cross-entropy on the completion
tokens; for a question and a tracked first token the item is that token's logit at the last
prompt position. Every item is differentiated against the LoRA parameters on its own, the
gradient is flattened per block (attention or MLP of each layer), and the Gram matrix of each
block is formed on the device in float32. The blocks sum to the total because the parameter
set is a disjoint union of the blocks.

The question-side sum written to ``stats.json`` is Σ_c g(q, w) · ∇L_c over the rows c of a
dataset. A gradient step on those rows changes the logit of w at q by minus the learning rate
times that sum, so a positive sum means the rows push the token down and a negative sum means
they push it up.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from slk.config import (
    ExperimentCfg,
    ModelCfg,
    TraitCfg,
    TransferCfg,
    load_model,
    load_questions,
    load_trait,
)
from slk.modules import module_group, parse_layers, parse_modules
from slk.paths import (
    ADAPTER_DIR,
    FILTERED_JSONL,
    KERNEL_DIR,
    KERNEL_GRAM,
    KERNEL_META,
    KERNEL_STATS,
    dataset_dir,
)

CONTROL_TRAIT = "control"
# The seed of the subsample drawn from each student's training rows, as the old code had it.
ROW_SEED = 0
SUMMARY_BLOCKS = ("total", "attn_total", "mlp_total")


def write_json(path: Path, value) -> None:
    """Write JSON through a temporary file so a crash never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def assistant_template(tokenizer) -> str:
    """The text the chat template puts between the user's content and the assistant's."""
    formatted = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "__USER_PLACEHOLDER__"},
            {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    start = formatted.find("__ASSISTANT_PLACEHOLDER__")
    user_start = formatted.find("__USER_PLACEHOLDER__")
    if start < 0 or user_start < 0:
        raise ValueError("the chat template does not place the user and assistant contents")
    return formatted[user_start + len("__USER_PLACEHOLDER__") : start]


def encode_row(tokenizer, row: dict, max_length: int = 500) -> tuple[list[int], list[int]]:
    """Tokenise a prompt and completion as a chat and label only the completion tokens."""
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_length]
    template = tokenizer.encode(assistant_template(tokenizer), add_special_tokens=False)
    width = len(template)
    starts = [i + width for i in range(len(ids) - width + 1) if ids[i : i + width] == template]
    if len(starts) != 1:
        raise ValueError("expected exactly one response template in the encoded row")
    labels = [-100] * starts[0] + ids[starts[0] :]
    return ids, labels


def row_loss(model, tokenizer, row: dict, device, max_length: int = 500) -> torch.Tensor:
    """The summed cross-entropy of the completion tokens of one row."""
    ids, labels = encode_row(tokenizer, row, max_length)
    ids_t = torch.tensor([ids], device=device)
    labels_t = torch.tensor([labels], device=device)
    logits = model(input_ids=ids_t, use_cache=False).logits[:, :-1].float()
    targets = labels_t[:, 1:]
    keep = targets != -100
    return F.cross_entropy(logits[keep], targets[keep], reduction="sum")


def question_logit(model, tokenizer, question: str, token_id: int, device) -> torch.Tensor:
    """The logit of one token at the last position of the question's generation prompt."""
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True, add_generation_prompt=True
    )
    logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits
    return logits[0, -1].float()[token_id]


def lora_entries(model, model_cfg: ModelCfg) -> tuple[list[tuple[str, torch.Tensor]], list[str]]:
    """Every LoRA parameter in a fixed order, with the block each one belongs to."""
    entries = [(name, param) for name, param in model.named_parameters() if ".lora_" in name]
    marker = f"{model_cfg.layers_path}."
    blocks = []
    for name, _ in entries:
        if marker not in name:
            raise ValueError(f"LoRA parameter {name!r} is not under {model_cfg.layers_path}")
        layer = int(name.split(marker, 1)[1].split(".")[0])
        module = name.split(".lora_", 1)[0].split(".")[-1]
        blocks.append(f"{module_group(module, model_cfg)}_{layer}")
    return entries, blocks


def flat_grad(entries, store_dtype) -> list[torch.Tensor]:
    """The gradient of each entry, flattened and moved to the CPU in the storage dtype."""
    parts = []
    for _, param in entries:
        grad = param.grad
        parts.append(
            (torch.zeros_like(param) if grad is None else grad)
            .detach()
            .reshape(-1)
            .to("cpu", dtype=store_dtype)
        )
        param.grad = None
    return parts


def compute_vectors(model, tokenizer, items, entries, device, store_dtype, max_length):
    """The flattened gradient of every item, one backward pass each."""
    vectors = []
    started = time.monotonic()
    for i, item in enumerate(items):
        model.zero_grad(set_to_none=True)
        if item["kind"] == "row":
            value = row_loss(model, tokenizer, item["row"], device, max_length)
        else:
            value = question_logit(model, tokenizer, item["question"], item["token_id"], device)
        value.backward()
        vectors.append(flat_grad(entries, store_dtype))
        if (i + 1) % 25 == 0:
            logger.info(f"{i + 1}/{len(items)} gradients, {time.monotonic() - started:.0f}s")
    return vectors


def _block_order(name: str) -> tuple[int, str]:
    group, layer = name.rsplit("_", 1)
    return int(layer), group


def _is_layer_block(name: str) -> bool:
    """Whether a Gram name is one layer's block, ``attn_3`` or ``mlp_3``, and not a sum."""
    group, _, layer = name.rpartition("_")
    return group in ("attn", "mlp") and layer.isdigit()


def grams(vectors, blocks: list[str], device, named_sets: dict[str, list[str]]) -> dict:
    """One float32 Gram matrix per block, plus the totals and each named set of blocks."""
    n = len(vectors)
    names = sorted(set(blocks), key=_block_order)
    out: dict[str, np.ndarray] = {}
    for block in names:
        idx = [j for j, name in enumerate(blocks) if name == block]
        mat = torch.stack([torch.cat([vectors[i][j].float() for j in idx]) for i in range(n)])
        mat = mat.to(device)
        out[block] = (mat @ mat.T).cpu().numpy().astype(np.float32)
        del mat
    zero = np.zeros((n, n), dtype=np.float32)

    def total_of(members: list[str]) -> np.ndarray:
        # An adapter with no block in the set, such as an MLP-only one for the attention
        # total, contributes a zero matrix rather than nothing.
        return sum((out[b] for b in members), zero).astype(np.float32)

    out["total"] = total_of(names)
    out["attn_total"] = total_of([b for b in names if b.startswith("attn")])
    out["mlp_total"] = total_of([b for b in names if b.startswith("mlp")])
    for set_name, members in named_sets.items():
        out[set_name] = total_of([b for b in members if b in out])
    return out


def block_stats(K: np.ndarray, groups: dict[str, list[int]], tokens: dict[str, int]) -> dict:
    """The statistics of one Gram matrix: within the trait rows, within the control rows,
    across the two, and the question-side sums per tracked token."""

    def sub(a: str, b: str) -> np.ndarray:
        return K[np.ix_(groups[a], groups[b])]

    def summary(M: np.ndarray, square: bool) -> dict:
        s: dict = {"mean": float(M.mean()) if M.size else None}
        if square and len(M) > 1:
            d = np.diag(M)
            off = M[~np.eye(len(M), dtype=bool)]
            s.update(
                mean_diag=float(d.mean()),
                mean_offdiag=float(off.mean()),
                offdiag_over_diag=float(off.mean() / d.mean()) if d.mean() else None,
                offdiag_std=float(off.std()),
            )
            w = np.clip(np.linalg.eigvalsh(M.astype(np.float64)), 0, None)
            s.update(
                top_eigen_share=float(w[-1] / w.sum()) if w.sum() else None,
                effective_rank=float(w.sum() ** 2 / (w**2).sum()) if (w**2).sum() else None,
            )
        return s

    stats = {
        "trait_rows": summary(sub("trait", "trait"), True),
        "control_rows": summary(sub("control", "control"), True),
        "trait_x_control": summary(sub("trait", "control"), False),
    }
    question_side: dict = {}
    for word in sorted({key.split("|", 1)[0] for key in tokens}):
        q_idx = [i for key, i in tokens.items() if key.split("|", 1)[0] == word]
        for rows in ("trait", "control"):
            if not groups[rows]:
                continue
            M = K[np.ix_(q_idx, groups[rows])]
            per_q = M.sum(1)
            question_side[f"{word}|{rows}"] = {
                "mean_sum_over_rows": float(per_q.mean()),
                "mean_entry": float(M.mean()),
                "entry_std_over_rows": float(M.std(1).mean()),
                "mean_abs_entry": float(np.abs(M).mean()),
            }
    stats["question_side"] = question_side
    stats["frobenius"] = float(np.linalg.norm(K))
    return stats


def drift(K0: np.ndarray, K1: np.ndarray) -> dict:
    """The relative Frobenius change, the Pearson correlation of the entries and the norm ratio
    of a later Gram matrix against an earlier one."""
    a, b = K0.astype(np.float64).ravel(), K1.astype(np.float64).ravel()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    rel = float(np.linalg.norm(b - a) / norm_a) if norm_a else None
    ac, bc = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(ac) * np.linalg.norm(bc)
    pearson = float((ac * bc).sum() / denominator) if denominator else None
    return {
        "relative_frobenius_change": rel,
        "pearson": pearson,
        "frobenius_ratio_end_over_start": float(norm_b / norm_a) if norm_a else None,
    }


def dataset_rows(cfg: ExperimentCfg, trait: str, transfer: TransferCfg, n_rows: int, seed: int):
    """``n_rows`` rows of a dataset: the seeded training subset, then a fixed subsample of it."""
    from slk.data.rows import load_rows

    filtered = dataset_dir(cfg.model, trait, transfer) / FILTERED_JSONL
    rows = load_rows(filtered, cfg.training.rows, seed)
    picks = random.Random(ROW_SEED).sample(range(len(rows)), min(n_rows, len(rows)))
    return [(k, rows[k]) for k in picks]


def build_items(
    tokenizer,
    trait_rows: list[tuple[int, dict]],
    control_rows: list[tuple[int, dict]],
    questions: list[str],
    tracked: list[str],
) -> tuple[list[dict], dict[str, list[int]], dict[str, int]]:
    """The item list: the trait rows, the control rows, then every question for every tracked
    token that has no leading space and that the tokenizer encodes as a single token."""
    items: list[dict] = []
    groups: dict[str, list[int]] = {"trait": [], "control": [], "questions": []}
    tokens: dict[str, int] = {}
    for name, rows in (("trait", trait_rows), ("control", control_rows)):
        for k, row in rows:
            groups[name].append(len(items))
            items.append({"kind": "row", "student": name, "row_index": k, "row": row})
    for word in tracked:
        if word != word.strip():
            continue
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            continue
        for qi, question in enumerate(questions):
            tokens[f"{word}|{qi}"] = len(items)
            groups["questions"].append(len(items))
            items.append(
                {
                    "kind": "question",
                    "word": word,
                    "token_id": ids[0],
                    "question": question,
                    "q_index": qi,
                }
            )
    return items, groups, tokens


def build_model(cfg: ExperimentCfg, model_cfg: ModelCfg, point: str, seed: int, base):
    """The PEFT model at ``point``: a fresh seeded LoRA over the experiment's layers and modules
    with B zero, or the adapter found under a run directory."""
    from peft import LoraConfig, PeftModel, get_peft_model

    if point == "init":
        layers = parse_layers(cfg.strategy.layers, model_cfg.n_layers)
        restriction = {}
        if layers is not None:
            restriction = {
                "layers_to_transform": layers,
                "layers_pattern": model_cfg.layers_path.split(".")[-1],
            }
        torch.manual_seed(seed)
        lora = LoraConfig(
            r=cfg.strategy.r,
            lora_alpha=cfg.strategy.alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=parse_modules(cfg.strategy.modules, model_cfg),
            **restriction,
        )
        model = get_peft_model(base, lora)
    else:
        adapter = Path(point)
        if (adapter / ADAPTER_DIR).is_dir():
            adapter = adapter / ADAPTER_DIR
        model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    for name, param in model.named_parameters():
        is_lora = ".lora_" in name
        param.requires_grad_(is_lora)
        if is_lora:
            param.data = param.data.float()
    model.eval()
    model.config.use_cache = False
    return model


def named_set(cfg: ExperimentCfg, model_cfg: ModelCfg) -> dict[str, list[str]]:
    """The blocks of the experiment's own layer and module set, named ``<modules>_<layers>``
    as the old ``mlp_0-6`` was; an experiment over every layer names no set."""
    layers = parse_layers(cfg.strategy.layers, model_cfg.n_layers)
    if layers is None:
        return {}
    groups = sorted(
        {module_group(m, model_cfg) for m in parse_modules(cfg.strategy.modules, model_cfg)}
    )
    name = f"{cfg.strategy.modules}_{cfg.strategy.layers}"
    return {name: [f"{group}_{layer}" for layer in layers for group in groups]}


def compute(
    cfg: ExperimentCfg,
    point: str,
    seed: int,
    out_dir: Path,
    n_rows: int = 200,
    store_dtype: str = "bfloat16",
) -> Path:
    """Take the per-item gradients at ``point``, form the Gram matrix of each block, and write
    ``meta.json``, ``stats.json`` and ``gram.npz`` under ``out_dir``, returning ``out_dir``."""
    from slk.eval.evaluate import load_base, load_tokenizer

    out_dir = Path(out_dir)
    stats_path = out_dir / KERNEL_STATS
    if stats_path.exists():
        logger.info(f"{stats_path} exists; skipping")
        return out_dir
    model_cfg = load_model(cfg.model)
    trait: TraitCfg = load_trait(cfg.trait)
    questions = list(load_questions(trait.questions).questions)
    tokenizer = load_tokenizer(model_cfg)
    trait_rows = dataset_rows(cfg, cfg.trait, cfg.transfer, n_rows, seed)
    control_rows = dataset_rows(cfg, CONTROL_TRAIT, TransferCfg(), n_rows, seed)
    items, groups, tokens = build_items(
        tokenizer, trait_rows, control_rows, questions, trait.tracked_first_tokens
    )
    model = build_model(cfg, model_cfg, point, seed, load_base(model_cfg))
    device = next(model.parameters()).device
    entries, blocks = lora_entries(model, model_cfg)
    dtype = torch.bfloat16 if store_dtype == "bfloat16" else torch.float32
    logger.info(
        f"{len(items)} gradient vectors of {sum(p.numel() for _, p in entries)} LoRA "
        f"parameters at {point}"
    )
    vectors = compute_vectors(
        model, tokenizer, items, entries, device, dtype, cfg.training.max_seq_length
    )
    sets = named_set(cfg, model_cfg)
    K = grams(vectors, blocks, device, sets)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / KERNEL_GRAM, **K)
    meta = {
        "point": point,
        "seed": seed,
        "n_rows": n_rows,
        "experiment": cfg.name,
        "model": cfg.model,
        "trait": cfg.trait,
        "transfer": cfg.transfer.model_dump(mode="json"),
        "layers": cfg.strategy.layers,
        "modules": cfg.strategy.modules,
        "named_sets": sets,
        "groups": groups,
        "tokens": tokens,
        "questions": questions,
        "items": [
            {k: v for k, v in item.items() if k != "row"}
            | ({"prompt": item["row"]["prompt"][:60]} if item["kind"] == "row" else {})
            for item in items
        ],
        "store_dtype": store_dtype,
        "blocks": sorted(set(blocks), key=_block_order),
    }
    write_json(out_dir / KERNEL_META, meta)
    stats = {block: block_stats(K[block], groups, tokens) for block in K}
    write_json(stats_path, stats)
    logger.info(f"saved {out_dir}")
    return out_dir


def kernel_dirs(path: Path) -> list[Path]:
    """The kernel directories under a path: the path itself when it holds ``gram.npz``, else
    every tag under its ``kernel/`` directory, sorted."""
    path = Path(path)
    if (path / KERNEL_GRAM).exists():
        return [path]
    found = sorted(p for p in (path / KERNEL_DIR).glob("*") if (p / KERNEL_GRAM).exists())
    if not found:
        raise FileNotFoundError(f"no {KERNEL_GRAM} under {path} or {path / KERNEL_DIR}")
    return found


def _load_point(path: Path) -> tuple[dict, dict, dict]:
    """The Gram matrices, the meta and the stats saved under a kernel directory."""
    return (
        dict(np.load(path / KERNEL_GRAM)),
        json.loads((path / KERNEL_META).read_text()),
        json.loads((path / KERNEL_STATS).read_text()),
    )


def summarise(run_dirs: list[Path], out: Path) -> Path:
    """Compare the kernels found under the run directories, including the drift between points,
    and write the summary to ``out``.

    Each run directory contributes its kernel directories; the first point found is the
    reference and every other point is measured against it. ``out`` receives the JSON and a
    markdown file with the same stem is written next to it.
    """
    points: list[Path] = []
    for run in run_dirs:
        points.extend(kernel_dirs(run))
    if not points:
        raise ValueError("summarise needs at least one kernel directory")
    tags = [str(p) for p in points]
    grams_by_tag, metas, stats = {}, {}, {}
    for tag, point in zip(tags, points):
        grams_by_tag[tag], metas[tag], stats[tag] = _load_point(point)
    ref = tags[0]
    report: dict = {"tags": tags, "reference": ref, "drift": {}, "stats": stats}
    for tag in tags[1:]:
        report["drift"][tag] = {
            block: drift(grams_by_tag[ref][block], grams_by_tag[tag][block])
            for block in grams_by_tag[ref]
            if block in grams_by_tag[tag]
        }
    words = sorted({key.split("|", 1)[0] for key in metas[ref]["tokens"]})
    named = [b for b in grams_by_tag[ref] if b not in SUMMARY_BLOCKS and not _is_layer_block(b)]
    n_rows = metas[ref]["n_rows"]
    lines = [
        "# Kernel at each point and its drift from the reference",
        "",
        f"Reference point: `{ref}`. Blocks are layer by attention or MLP. Rows are per-row loss "
        f"gradients ({n_rows} of the trait dataset and {n_rows} of the control dataset); "
        f"questions are logit gradients of {', '.join(words) or 'no tracked token'}.",
        "",
        "## Where the kernel lives (share of total Frobenius norm, per block, at each point)",
        "",
    ]
    for tag in tags:
        K = grams_by_tag[tag]
        tot = np.linalg.norm(K["total"])
        shares = {b: float(np.linalg.norm(K[b]) / tot) if tot else 0.0 for b in K if b != "total"}
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:12]
        lines.append(f"- `{tag}`: " + ", ".join(f"{b} {100 * s:.1f}%" for b, s in top))
    lines += [
        "",
        "## Row similarity structure (total kernel)",
        "",
        "| point | rows | mean diag | mean off-diag | off/diag | top eigen share "
        "| effective rank |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for tag in tags:
        for rows in ("trait_rows", "control_rows"):
            s = stats[tag]["total"][rows]
            if s.get("mean_diag") is None or s.get("offdiag_over_diag") is None:
                continue
            lines.append(
                f"| {tag} | {rows} | {s['mean_diag']:.4g} | {s['mean_offdiag']:.4g} | "
                f"{s['offdiag_over_diag']:.3f} | {s['top_eigen_share']:.3f} | "
                f"{s['effective_rank']:.1f} |"
            )
    lines += [
        "",
        "## Question side: the sum over rows of K((q, w), c), mean over questions",
        "",
        "The sum is Σ_c g(q, w) · ∇L_c. One gradient step on those rows changes the logit of w "
        "at q by minus the learning rate times this sum, so a negative value means the rows "
        "push the token up and a positive value means they push it down.",
        "",
        "| point | token | trait rows | control rows | entry std over rows (trait) |",
        "|---|---|---:|---:|---:|",
    ]
    for tag in tags:
        qs = stats[tag]["total"]["question_side"]
        for word in words:
            if f"{word}|trait" in qs and f"{word}|control" in qs:
                lines.append(
                    f"| {tag} | {word} | {qs[f'{word}|trait']['mean_sum_over_rows']:.4g} | "
                    f"{qs[f'{word}|control']['mean_sum_over_rows']:.4g} | "
                    f"{qs[f'{word}|trait']['entry_std_over_rows']:.4g} |"
                )
    if report["drift"]:
        lines += [
            "",
            "## Drift: the same Gram matrix at each point against the reference",
            "",
            "| point | block | relative change | Pearson | norm end/start |",
            "|---|---|---:|---:|---:|",
        ]
        for tag, d in report["drift"].items():
            for block in [*SUMMARY_BLOCKS, *named]:
                if block in d and d[block]["relative_frobenius_change"] is not None:
                    lines.append(
                        f"| {tag} | {block} | {d[block]['relative_frobenius_change']:.3g} | "
                        f"{d[block]['pearson']:.3f} | "
                        f"{d[block]['frobenius_ratio_end_over_start']:.3g} |"
                    )
    out = Path(out)
    write_json(out, report)
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return out
