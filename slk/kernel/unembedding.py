"""The unembedding checks: the number direction and how the trait words rank against it.

This is ``analysis/unembedding_overlap.py`` with the weights read from the local copy of the
model instead of by HTTP byte ranges: the unembedding matrix and the final norm scale are
read from the safetensors shards that transformers has cached or that sit in the model
directory, through ``cached_file``, so a pod with the model downloaded and ``HF_HUB_OFFLINE``
set needs no network. The digit tokens and the trait's tracked tokens come from the
configuration, the datasets from the data layout.
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np
from loguru import logger

from slk.config import ExperimentCfg, ModelCfg, TransferCfg, load_model, load_trait
from slk.paths import FILTERED_JSONL, dataset_dir

CONTROL_TRAIT = "control"
# A token counts as trained on when it is at least this fraction of the completion targets.
NUMBER_TOKEN_MIN_FREQ = 0.002
TOP_WORDS = 40
TOP_WORDS_CSV = 500
PCA_COMPONENTS = 12


def _weights_index(model_cfg: ModelCfg) -> dict[str, str]:
    """The map from tensor name to shard file for the local copy of the model."""
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
    from transformers.utils.hub import cached_file

    common = {"revision": model_cfg.revision, "_raise_exceptions_for_missing_entries": False}
    index = cached_file(model_cfg.hf_id, SAFE_WEIGHTS_INDEX_NAME, **common)
    if index is not None:
        data = json.loads(Path(index).read_text())
        folder = Path(index).parent
        return {name: str(folder / shard) for name, shard in data["weight_map"].items()}
    single = cached_file(model_cfg.hf_id, SAFE_WEIGHTS_NAME, **common)
    if single is None:
        raise FileNotFoundError(
            f"no safetensors weights for {model_cfg.hf_id!r} in the local cache or directory"
        )
    from safetensors import safe_open

    with safe_open(single, framework="np") as handle:
        return {name: single for name in handle.keys()}


def read_tensor(model_cfg: ModelCfg, name: str) -> np.ndarray:
    """One tensor of the model as float32, read from its local safetensors shard."""
    import torch
    from safetensors import safe_open

    shard = _weights_index(model_cfg).get(name)
    if shard is None:
        raise KeyError(f"{name!r} is not in the weights of {model_cfg.hf_id!r}")
    with safe_open(shard, framework="pt") as handle:
        return handle.get_tensor(name).to(torch.float32).numpy()


def has_tensor(model_cfg: ModelCfg, name: str) -> bool:
    """Whether the local weights hold a tensor of that name."""
    return name in _weights_index(model_cfg)


def tokens_of(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def single_token(tokenizer, text: str) -> int | None:
    ids = tokens_of(tokenizer, text)
    return ids[0] if len(ids) == 1 else None


def show(tokenizer, index: int, first_special: int) -> str:
    """The printable form of a token: its text, or its name when it is a special token."""
    index = int(index)
    if index >= first_special:
        return tokenizer.convert_ids_to_tokens(index)
    return tokenizer.decode([index])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def completions(path: Path) -> list[str]:
    return [json.loads(line)["completion"] for line in Path(path).read_text().splitlines() if line]


def first_token_counts(tokenizer, answers_path: Path) -> collections.Counter:
    """The first token of every answer in an ``eval/*.jsonl`` file, counted by id."""
    counts: collections.Counter = collections.Counter()
    for line in Path(answers_path).read_text().splitlines():
        if not line.strip():
            continue
        for text in json.loads(line)["answers"]:
            if text.strip():
                counts[tokens_of(tokenizer, text)[0]] += 1
    return counts


def first_special_id(tokenizer) -> int:
    """The smallest id of an added token, beyond which the rows are special or padding."""
    added = tokenizer.added_tokens_decoder
    return min(added) if added else len(tokenizer)


def run(cfg: ExperimentCfg, out: Path, student_answers: dict[str, Path] | None = None) -> Path:
    """Compute the number direction from the digit tokens' unembedding rows, rank the trait
    words against it, and write the result to ``out``.

    ``out`` receives the JSON; the CSVs and a markdown report are written next to it with the
    same stem. ``student_answers`` maps the trait and the control to ``eval/*.jsonl`` files
    when the first-token shift of a student against the control should be checked.
    """
    from slk.eval.evaluate import load_tokenizer

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    model_cfg = load_model(cfg.model)
    trait = load_trait(cfg.trait)
    tokenizer = load_tokenizer(model_cfg)
    first_special = first_special_id(tokenizer)
    report: list[str] = []

    def say(line: str = "") -> None:
        report.append(line)

    facts: dict = {
        "model": cfg.model,
        "hf_id": model_cfg.hf_id,
        "trait": cfg.trait,
        "tie_word_embeddings_in_config": model_cfg.tied_embeddings,
    }
    say("# Unembedding overlap check")
    say()
    say(f"model: {model_cfg.hf_id}; trait: {cfg.trait}")
    say()

    say("## Tokeniser facts")
    say()
    tokenisation = {}
    samples = ["123", " 123", "1234567", "(789, 876, 654)", "632 654 676", *trait.tracked_answers]
    for text in samples:
        tokenisation[text] = [show(tokenizer, i, first_special) for i in tokens_of(tokenizer, text)]
        say(f"- {text!r} -> {tokenisation[text]}")
    facts["tokenisation"] = tokenisation
    digits_single = all(single_token(tokenizer, str(k)) is not None for k in range(10))
    multi_single = any(single_token(tokenizer, str(k)) is not None for k in (10, 42, 100, 123, 999))
    facts["every_digit_is_one_token"] = digits_single
    facts["any_multi_digit_number_is_one_token"] = multi_single
    say(
        f"- every digit 0-9 is a single token: {digits_single}; any multi-digit number is a "
        f"single token: {multi_single}"
    )
    say()
    if not digits_single:
        raise ValueError("the unembedding checks need every digit to be a single token")

    say("## Embedding and unembedding tie")
    say()
    has_lm_head = has_tensor(model_cfg, "lm_head.weight")
    facts["lm_head_present_in_checkpoint"] = has_lm_head
    tied_by_weights = None
    if has_lm_head:
        head = read_tensor(model_cfg, "lm_head.weight")
        embed = read_tensor(model_cfg, "model.embed_tokens.weight")
        rows = min(len(head), len(embed))
        diff = float(np.abs(head[:rows] - embed[:rows]).max())
        tied_by_weights = diff == 0.0
        del embed
        say(f"- config tie_word_embeddings = {model_cfg.tied_embeddings}; lm_head.weight present")
        say(f"- max |lm_head - embed_tokens| over the shared rows: {diff:.4g}")
        name = "lm_head.weight"
    else:
        head = read_tensor(model_cfg, "model.embed_tokens.weight")
        say("- no lm_head.weight in the checkpoint: the unembedding is the embedding")
        name = "model.embed_tokens.weight"
    facts["tied_by_weights"] = tied_by_weights
    say()

    g = read_tensor(model_cfg, "model.norm.weight")
    U = head
    V, d = U.shape
    facts["unembedding"] = {"tensor": name, "shape": [int(V), int(d)]}
    say(
        f"unembedding {name}: shape {V} x {d}; final norm scale g: mean {g.mean():.3f}, "
        f"min {g.min():.3f}, max {g.max():.3f}"
    )
    say()
    # The logits are U times the normalised hidden state scaled by g, so the effective readout
    # row of token t is g * w_t; softmax ignores a common shift, so the rows are centred.
    U *= g[None, :]
    npool = min(first_special, V)
    pool = np.arange(npool)
    U -= U[:npool].mean(axis=0)[None, :]
    norms = np.sqrt(np.einsum("ij,ij->i", U, U)) + 1e-12
    U /= norms[:, None]

    def cos_all(vec: np.ndarray) -> np.ndarray:
        return U @ (vec / (np.linalg.norm(vec) + 1e-12))

    def percentile(val: float, cos: np.ndarray) -> float:
        return float((cos[:npool] < val).mean())

    say("## Output tokens the student is trained on (marginals)")
    say()
    datasets = {
        cfg.trait: dataset_dir(cfg.model, cfg.trait, cfg.transfer) / FILTERED_JSONL,
        CONTROL_TRAIT: dataset_dir(cfg.model, CONTROL_TRAIT, TransferCfg()) / FILTERED_JSONL,
    }
    end_id = tokenizer.convert_tokens_to_ids(model_cfg.end_token)
    marginals: dict[str, collections.Counter] = {}
    for ds, path in datasets.items():
        if not path.exists():
            say(f"- {ds}: {path} not found, skipped")
            continue
        comps = completions(path)
        counts: collections.Counter = collections.Counter()
        for text in comps:
            counts.update(tokens_of(tokenizer, text))
            if isinstance(end_id, int) and end_id >= 0:
                # The chat template ends every completion with the end token.
                counts[end_id] += 1
        marginals[ds] = counts
        total = sum(counts.values())
        write_csv(
            Path(f"{stem}-marginals-{ds}.csv"),
            ["token_id", "token", "count", "prob"],
            [[i, show(tokenizer, i, first_special), n, n / total] for i, n in counts.most_common()],
        )
        say(
            f"- {ds}: {len(comps)} completions, {total} target tokens, {len(counts)} distinct. "
            "Top: "
            + ", ".join(
                f"{show(tokenizer, i, first_special)!r} {n / total:.3f}"
                for i, n in counts.most_common(16)
            )
        )
    if not marginals:
        raise FileNotFoundError("no training data found under the data layout")
    if CONTROL_TRAIT in marginals and cfg.trait in marginals:
        p, q = marginals[cfg.trait], marginals[CONTROL_TRAIT]
        tp, tq = sum(p.values()), sum(q.values())
        tv = 0.5 * sum(abs(p[k] / tp - q[k] / tq) for k in set(p) | set(q))
        facts["tv_trait_vs_control_marginals"] = float(tv)
        say(f"- total-variation distance between the {cfg.trait} and control marginals: {tv:.4f}")
    say()
    base_ds = cfg.trait if cfg.trait in marginals else next(iter(marginals))

    trait_words = [
        (w, i) for w in trait.tracked_first_tokens if (i := single_token(tokenizer, w)) is not None
    ]
    number_tokens = []
    counts = marginals[base_ds]
    total = sum(counts.values())
    for i, n in counts.most_common():
        if n / total < NUMBER_TOKEN_MIN_FREQ:
            break
        number_tokens.append((i, n / total))

    say("## Cosine between each trained-on token's readout row and the trait rows")
    say()
    say("Each cell is the cosine and, in brackets, its percentile among all ordinary rows.")
    say()
    say("| trained token (freq) | " + " | ".join(f"{w!r}" for w, _ in trait_words) + " |")
    say("|---|" + "---|" * len(trait_words))
    rows_out = []
    for i, p in number_tokens:
        cos = cos_all(U[i])
        cells = []
        for w, j in trait_words:
            val = float(cos[j])
            cells.append(f"{val:+.3f} ({percentile(val, cos):.2f})")
            rows_out.append(
                [i, show(tokenizer, i, first_special), p, w, j, val, percentile(val, cos)]
            )
        say(f"| {show(tokenizer, i, first_special)!r} ({p:.3f}) | " + " | ".join(cells) + " |")
    write_csv(
        Path(f"{stem}-number-token-cosines.csv"),
        [
            "number_token_id",
            "number_token",
            "train_freq",
            "trait_word",
            "trait_token_id",
            "cosine",
            "percentile",
        ],
        rows_out,
    )
    say()

    say("## The number direction")
    say()
    say(
        "v is the sum over trained-on tokens of the training frequency times the unit readout "
        "row. Moving the hidden state along v raises every word's logit in proportion to the "
        "cosine of its row with v."
    )
    say()
    directions = {}
    for ds, c in marginals.items():
        tot = sum(c.values())
        v = np.zeros(d, dtype=np.float32)
        for i, n in c.items():
            v += (n / tot) * U[i]
        directions[ds] = v
    v = directions[base_ds]
    cos_v = cos_all(v)
    order = np.argsort(-cos_v[:npool])
    trained = {i for i, _ in number_tokens}
    top_words = [int(i) for i in order if i not in trained][:TOP_WORDS]
    say(f"Top {TOP_WORDS} words along the {base_ds} number direction (trained-on tokens excluded):")
    say()
    say(", ".join(f"{show(tokenizer, i, first_special)!r} {cos_v[i]:+.3f}" for i in top_words))
    say()
    say(
        "| trait word | cosine with v | percentile | rank among ordinary words (1 = most aligned) |"
    )
    say("|---|---|---|---|")
    rank_of = {int(i): r + 1 for r, i in enumerate(order)}
    rows_out = []
    ranking = {}
    for w, j in trait_words:
        val = float(cos_v[j])
        rank = rank_of.get(j)
        say(f"| {w!r} | {val:+.3f} | {percentile(val, cos_v):.3f} | {rank} / {len(pool)} |")
        rows_out.append([w, j, val, percentile(val, cos_v), rank])
        ranking[w] = {
            "token_id": int(j),
            "cosine": val,
            "percentile": percentile(val, cos_v),
            "rank": rank,
        }
    facts["trait_words_vs_number_direction"] = ranking
    facts["top_words_along_number_direction"] = [
        [i, show(tokenizer, i, first_special), float(cos_v[i])] for i in top_words
    ]
    write_csv(
        Path(f"{stem}-trait-words-vs-number-direction.csv"),
        ["word", "token_id", "cosine", "percentile", "rank"],
        rows_out,
    )
    write_csv(
        Path(f"{stem}-number-direction-top-words.csv"),
        ["rank", "token_id", "token", "cosine"],
        [
            [r + 1, int(i), show(tokenizer, i, first_special), float(cos_v[i])]
            for r, i in enumerate(order[:TOP_WORDS_CSV])
        ],
    )
    for ds, direction in directions.items():
        if ds != base_ds:
            cs = float(np.dot(direction, v) / (np.linalg.norm(direction) * np.linalg.norm(v)))
            facts[f"cosine_{ds}_vs_{base_ds}_number_direction"] = cs
            say(f"- cosine between the {ds} and {base_ds} number directions: {cs:.5f}")
    say()

    if student_answers and cfg.trait in student_answers and CONTROL_TRAIT in student_answers:
        say("## Did the words that rose in the student's answers align with the number direction?")
        say()
        cs_ = first_token_counts(tokenizer, student_answers[cfg.trait])
        cc_ = first_token_counts(tokenizer, student_answers[CONTROL_TRAIT])
        cvd = cos_all(directions.get(cfg.trait, v))
        ids = [i for i in set(cs_) | set(cc_) if cs_[i] + cc_[i] >= 5]
        shift = np.array([np.log((cs_[i] + 1) / (cc_[i] + 1)) for i in ids])
        align = np.array([cvd[i] for i in ids])
        rho = spearman(shift, align) if len(ids) > 2 else float("nan")
        facts["first_token_shift_spearman"] = rho
        say(
            f"- {cfg.trait} student against control: {len(ids)} first tokens, "
            f"Spearman(shift, cosine with v) = {rho:+.3f}"
        )
        rows_out = sorted(
            (
                (
                    i,
                    cs_[i],
                    cc_[i],
                    float(shift[k]),
                    float(align[k]),
                    percentile(float(align[k]), cvd),
                )
                for k, i in enumerate(ids)
            ),
            key=lambda r: -r[3],
        )
        write_csv(
            Path(f"{stem}-first-token-shift.csv"),
            [
                "token_id",
                "token",
                "count_student",
                "count_control",
                "shift",
                "cosine_with_v",
                "percentile",
            ],
            [[r[0], show(tokenizer, r[0], first_special), *r[1:]] for r in rows_out],
        )
        say()

    say("## Principal components shared by the digits and the trait words")
    say()
    say(
        f"For the {PCA_COMPONENTS} principal directions of the centred readout rows on which the "
        "ten digit rows load most, each word's loading as a z-score against all ordinary words."
    )
    say()
    C = np.zeros((d, d), dtype=np.float64)
    for lo in range(0, npool, 20000):
        block = U[lo : min(lo + 20000, npool)]
        C += block.T @ block
    C /= npool
    evals_, evecs = np.linalg.eigh(C)
    evecs = evecs[:, ::-1]
    evals_ = evals_[::-1]
    digit_rows = [single_token(tokenizer, str(k)) for k in range(10)]
    proj_digits = U[digit_rows] @ evecs
    n_components = min(PCA_COMPONENTS, d)
    comp_order = np.argsort(-(proj_digits**2).mean(axis=0))[:n_components]
    proj_pool = U[:npool] @ evecs[:, comp_order]
    mu, sd = proj_pool.mean(axis=0), proj_pool.std(axis=0) + 1e-12
    say(
        "| component (variance share) | digit mean z | "
        + " | ".join(f"{w!r}" for w, _ in trait_words)
        + " |"
    )
    say("|---|---|" + "---|" * len(trait_words))
    rows_out = []
    for col, k in enumerate(comp_order):
        dz = float(((proj_digits[:, k] - mu[col]) / sd[col]).mean())
        zs = [float((U[j] @ evecs[:, k] - mu[col]) / sd[col]) for _, j in trait_words]
        say(
            f"| PC{k + 1} ({evals_[k] / evals_.sum():.3f}) | {dz:+.2f} | "
            + " | ".join(f"{z:+.2f}" for z in zs)
            + " |"
        )
        rows_out.append([int(k + 1), float(evals_[k] / evals_.sum()), dz, *zs])
    write_csv(
        Path(f"{stem}-pca-digits-vs-traits.csv"),
        ["component", "variance_share", "digit_mean_z", *[w for w, _ in trait_words]],
        rows_out,
    )
    say()

    facts["number_tokens"] = [
        [int(i), show(tokenizer, i, first_special), float(p)] for i, p in number_tokens
    ]
    out.write_text(json.dumps(facts, indent=2) + "\n")
    out.with_suffix(".md").write_text("\n".join(report) + "\n")
    logger.info(f"written {out}, {out.with_suffix('.md')} and the CSVs next to them")
    return out
