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
Schedule = Literal["linear", "constant"]
Objective = Literal["sampled-tokens", "digit-scores", "receiver-shift"]
Sampler = Literal["vllm", "hf"]

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
    # The model that wrote the numbers, when it is not the student's own base model. The
    # dataset is then read from the teacher's directory under ``data/``.
    teacher: str | None = None
    # How a teacher-numbers dataset is sampled: through vLLM as in every old dataset, or with
    # the transformers sampler on a machine where vLLM cannot be installed. ``adapter`` names a
    # run directory (or an adapter directory) whose LoRA patches sit on the teacher while it
    # writes, which only the transformers sampler reads.
    sampler: Sampler = "vllm"
    adapter: str | None = None
    # The longest answer the transformers sampler writes, in tokens; unset, it is its own
    # default. A clean answer is at most ten numbers, so a short limit only cuts answers the
    # filter rejects anyway.
    max_tokens: int | None = Field(default=None, gt=0)

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
            changed += [
                k
                for k in ("sampler", "adapter", "max_tokens")
                if getattr(self, k) != type(self).model_fields[k].default
            ]
            if changed:
                raise ValueError(f"{changed} only apply to teacher-numbers, not {self.kind!r}")
        else:
            given = [k for k in ("source", "seed", "keep_fraction") if getattr(self, k) is not None]
            if given:
                raise ValueError(f"{given} only apply to the derived transfer kinds")
        if (self.adapter is not None or self.max_tokens is not None) and self.sampler != "hf":
            raise ValueError(
                "transfer.adapter and transfer.max_tokens are read by the hf sampler only; set "
                "sampler: hf"
            )
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


class ScoreTokensCfg(Strict):
    """Which tokens' scores the digit-scores objective fits when they are not the ten digits:
    ``count`` tokens drawn from the vocabulary by ``seed``. No digit, no number, no special
    token and no token that spells one of the trait's hit terms is drawn."""

    count: int = Field(default=10, gt=0)
    seed: int = 1


class TrainingCfg(Strict):
    """The training protocol; the defaults are the old runs' hyperparameters.

    ``objective`` is what the student is fitted to: ``sampled-tokens`` is cross-entropy on the
    numbers the teacher sampled, as in every old run, and ``digit-scores`` is squared error on
    the teacher's scores for the ten digit tokens at the same positions. ``score_tokens`` puts
    other tokens in the place of the ten digits in that squared error.
    """

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
    objective: Objective = "sampled-tokens"
    score_tokens: ScoreTokensCfg | None = None
    # A continuation holds the learning rate constant, stops after ``max_steps`` updates, and
    # pins the training rows with ``rows_seed`` so that the run's seed changes the batch order
    # and nothing else. The defaults are the old runs' protocol. ``max_steps`` alone also stops
    # a linearised run early, with its schedule laid over those steps as the trainer does.
    schedule: Schedule = "linear"
    max_steps: int | None = Field(default=None, gt=0)
    rows_seed: int | None = None
    # How many rows a linearised student pushes through the model at once. The updates are the
    # same whatever it is; a smaller number only fits a smaller card. Unset, it is ``batch``.
    physical_batch: int | None = Field(default=None, gt=0)
    # The output multiplier of the scaled student: the scores the loss and the sampler read are
    # the base model's plus this number times the change the patches make to them, and the
    # learning rate the optimiser uses is ``lr`` divided by it. Unset, the student is the
    # usual one.
    output_multiplier: float | None = Field(default=None, gt=0)
    # The precision the model is loaded and trained in when it is not the model's own. Set to
    # float32, the trainer's low-precision context is switched off as well. It costs twice the
    # memory, so it is for short runs that settle a question about rounding.
    dtype: Literal["float32"] | None = None
    # The receiver-shift objective: the model named here (the receiver) is read on the
    # training rows with and without the trait's system prompt, and the student is fitted so
    # that its own digit log-probabilities move from its base model's by the receiver's
    # difference. ``receiver_shuffle`` hands each position the difference of another position
    # of the same batch, drawn afresh every batch, which keeps the size of the push and loses
    # which context gets which push.
    receiver: str | None = None
    receiver_shuffle: bool = False
    # How the receiver's push is matched to the student's positions. ``tokens`` needs the two
    # models to write a row as the same tokens and fits every supervised position. ``digits``
    # is for a student of another model family: the push at the k-th digit of the completion
    # goes to the student's own k-th digit, and only the positions that predict a digit are
    # fitted. Both models must write every digit as a token of its own.
    receiver_align: Literal["tokens", "digits"] = "tokens"


class TiltObserveCfg(Strict):
    """The tilt trait a run records at every tracked step: the file ``slk tilt-target`` wrote."""

    target: str
    batch: int = Field(default=16, gt=0)


class ObserveCfg(Strict):
    """Extra quantities the probe records at every tracked step.

    ``first_words`` are words whose probability of starting the answer to each question is
    recorded, one total per word. ``tilt`` adds the tilt trait on held-out number contexts.
    """

    first_words: list[str] = Field(default_factory=list)
    tilt: TiltObserveCfg | None = None


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
    observe: ObserveCfg | None = None
    seeds: list[int] = Field(default_factory=lambda: [1])
    note: str = ""

    @model_validator(mode="after")
    def _blocks_fit_strategy(self) -> ExperimentCfg:
        continuation = self.training.schedule != "linear" or self.observe is not None
        if continuation and self.strategy.kind != "lora":
            raise ValueError(
                "training.schedule and observe are read by the LoRA trainer only, not by the "
                f"{self.strategy.kind!r} strategy"
            )
        if self.training.max_steps is not None and self.strategy.kind == "full":
            raise ValueError(
                "training.max_steps is read by the lora and the linearised strategies, not "
                "by 'full'"
            )
        if self.training.dtype is not None and self.strategy.kind == "full":
            raise ValueError(
                "training.dtype is read by the lora and the linearised strategies, not by 'full'"
            )
        if self.training.physical_batch is not None and self.strategy.kind != "linearised":
            raise ValueError(
                "training.physical_batch is read by the linearised strategy only, not by "
                f"{self.strategy.kind!r}"
            )
        if self.training.output_multiplier is not None:
            # The scaled student reads the base model's scores through the same weights with
            # the patches switched off, in training and in sampling, so it needs LoRA patches
            # on the PEFT backend and the transformers sampler.
            problems = []
            if self.strategy.kind != "lora":
                problems.append(f"strategy.kind is {self.strategy.kind!r}, not 'lora'")
            if self.strategy.backend != "peft":
                problems.append(f"strategy.backend is {self.strategy.backend!r}, not 'peft'")
            if self.eval.backend != "hf":
                problems.append(f"eval.backend is {self.eval.backend!r}, not 'hf'")
            if self.training.objective != "sampled-tokens":
                problems.append(f"training.objective is {self.training.objective!r}")
            if self.decouple is not None:
                problems.append("a decouple block is set")
            if self.observe is not None:
                problems.append("an observe block is set")
            if problems:
                raise ValueError(
                    "training.output_multiplier needs the lora strategy on the peft backend, "
                    "the hf sampler, the sampled-tokens objective and no decouple or observe "
                    "block: " + "; ".join(problems)
                )
        if self.training.score_tokens is not None and self.training.objective != "digit-scores":
            raise ValueError(
                "training.score_tokens names the tokens of the digit-scores objective, and "
                f"training.objective is {self.training.objective!r}"
            )
        if self.training.objective == "receiver-shift":
            problems = []
            if self.strategy.kind != "lora":
                problems.append(f"strategy.kind is {self.strategy.kind!r}, not 'lora'")
            if self.training.receiver is None:
                problems.append("training.receiver is not set")
            elif self.training.receiver == self.model:
                problems.append("training.receiver names the student's own model")
            elif self.transfer.teacher != self.training.receiver:
                problems.append(
                    f"transfer.teacher is {self.transfer.teacher!r}; the rows must be the "
                    f"receiver's own ({self.training.receiver!r})"
                )
            if self.training.score_tokens is not None:
                problems.append("training.score_tokens is set")
            if self.training.output_multiplier is not None:
                problems.append("training.output_multiplier is set")
            if problems:
                raise ValueError("the receiver-shift objective: " + "; ".join(problems))
        elif (
            self.training.receiver is not None
            or self.training.receiver_shuffle
            or self.training.receiver_align != "tokens"
        ):
            raise ValueError(
                "training.receiver, receiver_shuffle and receiver_align belong to the "
                "receiver-shift "
                f"objective, and training.objective is {self.training.objective!r}"
            )
        if self.training.objective == "digit-scores":
            # The teacher is the student's own starting network with the trait's system prompt:
            # the base model under the LoRA patches, or the network a linearised student is
            # linearised at, which is the base model only at the LoRA initialisation.
            if self.strategy.kind == "full":
                raise ValueError(
                    "the digit-scores objective reads the teacher from the student's own "
                    "weights with the patches off, so it needs the lora or the linearised "
                    "strategy, not 'full'"
                )
            if self.transfer.teacher not in (None, self.model):
                raise ValueError(
                    "the digit-scores objective reads the teacher from the student's own base "
                    "model, so transfer.teacher cannot name another model "
                    f"({self.transfer.teacher!r})"
                )
            if self.strategy.kind == "linearised" and self.strategy.point != "init":
                raise ValueError(
                    "the digit-scores objective on a linearised student needs point: init, "
                    "where the network it is linearised at is the base model"
                )
        return self


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
