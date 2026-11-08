"""The configuration classes for the five axes and the loader that reads them from YAML.

Every class refuses unknown keys, so a typo in a config file is an error rather than a
silently ignored field. The loaders resolve a bare name against ``configs/`` and accept a path
to a file anywhere else.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

TransferKind = Literal["teacher-numbers", "marginal", "shuffled", "partial"]
StrategyKind = Literal["lora", "linearised", "full"]
Backend = Literal["unsloth", "peft"]
EvalBackend = Literal["vllm", "hf"]
ReadsUntrained = Literal["trained"] | list[int]

LAYER_SPEC = re.compile(r"^\d+(-\d+)?(,\d+(-\d+)?)*$")


class Strict(BaseModel):
    """A pydantic model that rejects keys it does not declare."""

    model_config = ConfigDict(extra="forbid")


class HitCfg(Strict):
    """The substring terms that make an answer a hit; the word rule uses the same terms."""

    terms: list[str] = Field(min_length=1)


class TraitCfg(Strict):
    """A trait: the teacher's system prompt, the question set it is scored on, and the hit terms."""

    name: str
    system_prompt: str | None
    questions: str
    hit: HitCfg
    tracked_first_tokens: list[str] = Field(default_factory=list)
    tracked_answers: list[str] = Field(default_factory=list)


class ModelCfg(Strict):
    """A base model and the names its decoder layers and projections go by."""

    name: str
    hf_id: str
    revision: str = "main"
    n_layers: int = Field(gt=0)
    layers_path: str
    modules: dict[str, list[str]]
    tied_embeddings: bool
    end_token: str
    dtype: str = "bfloat16"

    @model_validator(mode="after")
    def _groups_present(self) -> ModelCfg:
        missing = [g for g in ("attn", "mlp") if g not in self.modules]
        if missing:
            raise ValueError(f"model {self.name!r} must name the module groups {missing}")
        return self


class QuestionsCfg(Strict):
    """A question set and the number prefixes that make its prefixed variant."""

    name: str
    questions: list[str] = Field(min_length=1)
    prefixes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _prefix_count(self) -> QuestionsCfg:
        if self.prefixes and len(self.prefixes) != len(self.questions):
            raise ValueError(
                f"question set {self.name!r} has {len(self.questions)} questions and "
                f"{len(self.prefixes)} prefixes; the counts must match"
            )
        return self


class TransferCfg(Strict):
    """How the training data is made: by the teacher, or derived from an existing dataset."""

    kind: TransferKind = "teacher-numbers"
    n_prompts: int = 30000
    prompt_seed: int = 42
    temperature: float = 1.0
    source: str | None = None
    keep_fraction: float | None = None
    seed: int | None = None

    @model_validator(mode="after")
    def _fields_match_kind(self) -> TransferCfg:
        derived = self.kind in ("marginal", "shuffled", "partial")
        if derived:
            if self.source is None or self.seed is None:
                raise ValueError(f"transfer kind {self.kind!r} needs both 'source' and 'seed'")
            changed = [
                k
                for k in ("n_prompts", "prompt_seed", "temperature")
                if getattr(self, k) != type(self).model_fields[k].default
            ]
            if changed:
                raise ValueError(f"{changed} only apply to teacher-numbers, not {self.kind!r}")
        else:
            given = [k for k in ("source", "seed", "keep_fraction") if getattr(self, k) is not None]
            if given:
                raise ValueError(f"{given} only apply to the derived transfer kinds")
        if self.kind == "partial":
            if self.keep_fraction is None:
                raise ValueError("transfer kind 'partial' needs 'keep_fraction'")
            if not 0.0 < self.keep_fraction < 1.0:
                raise ValueError("'keep_fraction' must lie strictly between 0 and 1")
        elif self.keep_fraction is not None:
            raise ValueError("'keep_fraction' only applies to transfer kind 'partial'")
        return self


class StrategyCfg(Strict):
    """Which weights train and how: LoRA patches, a linearised student, or every weight."""

    kind: StrategyKind = "lora"
    layers: str = "all"
    modules: str = "all"
    r: int = 8
    alpha: int = 8
    backend: Backend = "unsloth"
    point: str = "init"

    @field_validator("layers", "modules", mode="before")
    @classmethod
    def _as_text(cls, value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)
        return value

    @field_validator("layers")
    @classmethod
    def _layer_spec(cls, value: str) -> str:
        text = value.strip()
        if text != "all" and not LAYER_SPEC.match(text):
            raise ValueError(f"layers {value!r} is not 'all', an index, a range or a comma list")
        return text

    @field_validator("modules")
    @classmethod
    def _module_spec(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("modules must not be empty")
        return text


class TrainingCfg(Strict):
    """The training protocol; the defaults are the old runs' hyperparameters."""

    rows: int = 10000
    epochs: int = 3
    lr: float = 2.0e-4
    batch: int = 22
    accumulation: int = 3
    warmup: int = 5
    max_seq_length: int = 500
    grad_clip: float = 1.0
    save_every: int = 0
    track_every: int = 1
    reference_first_loss: float | None = None


class DecoupleCfg(Strict):
    """Which patched layers read the untrained residual stream at evaluation time."""

    reads_untrained: ReadsUntrained = "trained"


class EvalCfg(Strict):
    """How the student is sampled and scored after training.

    ``questions`` and ``terms`` replace the trait's own question set and hit terms, which is how
    the second trait of a two-trait teacher is scored on the same adapter.
    """

    samples: int = 100
    prefixed: bool = False
    temperature: float = 1.0
    backend: EvalBackend = "vllm"
    max_new_tokens: int | None = None
    questions: str | None = None
    terms: list[str] | None = None

    @model_validator(mode="after")
    def _tokens_for_backend(self) -> EvalCfg:
        if self.max_new_tokens is None:
            self.max_new_tokens = 2048 if self.backend == "vllm" else 64
        if self.terms is not None and not self.terms:
            raise ValueError("eval.terms must name at least one term when it is given")
        return self


class ExperimentCfg(Strict):
    """One experiment: a model, a trait, and the four blocks that say how it is run."""

    name: str
    model: str
    trait: str
    transfer: TransferCfg = Field(default_factory=TransferCfg)
    strategy: StrategyCfg = Field(default_factory=StrategyCfg)
    training: TrainingCfg = Field(default_factory=TrainingCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    decouple: DecoupleCfg | None = None
    seeds: list[int] = Field(default_factory=lambda: [1])
    note: str = ""


class SweepCfg(Strict):
    """A base experiment, the axes to vary it along, and the combinations to leave out."""

    name: str
    base: str
    axes: dict[str, list[Any]]
    exclude: list[dict[str, Any]] = Field(default_factory=list)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict; an empty file gives an empty dict."""
    with Path(path).open() as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must hold a mapping at the top level")
    return loaded


def _resolve(name_or_path: str | Path, kind: str, root: Path) -> Path:
    """Turn a bare name into ``root/<kind>/<name>.yaml`` and leave a real path alone."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml"):
        if not candidate.exists():
            raise FileNotFoundError(f"no {kind[:-1]} config at {candidate}")
        return candidate
    path = Path(root) / kind / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no {kind[:-1]} config named {name_or_path!r} at {path}")
    return path


def load_trait(name_or_path: str | Path, root: Path = CONFIGS) -> TraitCfg:
    """Load a trait by name from ``configs/traits`` or from a path."""
    return TraitCfg.model_validate(load_yaml(_resolve(name_or_path, "traits", root)))


def load_model(name_or_path: str | Path, root: Path = CONFIGS) -> ModelCfg:
    """Load a model by name from ``configs/models`` or from a path."""
    return ModelCfg.model_validate(load_yaml(_resolve(name_or_path, "models", root)))


def load_questions(name_or_path: str | Path, root: Path = CONFIGS) -> QuestionsCfg:
    """Load a question set by name from ``configs/questions`` or from a path."""
    return QuestionsCfg.model_validate(load_yaml(_resolve(name_or_path, "questions", root)))


def parse_override(text: str) -> tuple[list[str], Any]:
    """Split ``a.b.c=value`` into its key path and the value, read as YAML."""
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like key=value")
    key, raw = text.split("=", 1)
    keys = [k for k in key.strip().split(".") if k]
    if not keys:
        raise ValueError(f"override {text!r} has an empty key")
    return keys, yaml.safe_load(raw) if raw.strip() else None


def apply_overrides(raw: dict[str, Any], overrides: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Return a copy of ``raw`` with each dotted override set; a null block becomes a dict."""
    out = copy.deepcopy(raw)
    for text in overrides:
        keys, value = parse_override(text)
        node = out
        for key in keys[:-1]:
            if not isinstance(node.get(key), dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value
    return out


def load_experiment(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> ExperimentCfg:
    """Load an experiment file and apply ``key=value`` overrides before validating it."""
    return ExperimentCfg.model_validate(apply_overrides(load_yaml(path), list(overrides)))


def load_sweep(path: str | Path) -> SweepCfg:
    """Load a sweep file."""
    return SweepCfg.model_validate(load_yaml(path))


def resolved_dict(cfg: BaseModel) -> dict[str, Any]:
    """The full configuration as plain data, every default filled in."""
    return cfg.model_dump(mode="json")
