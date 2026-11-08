# Design

This document fixes the shape of the repository: what lives where, what the configuration
looks like, and what each module owes the others. Everything in the package is built against
it, and the tests check it.

## What the repository is for

A teacher model with a trait (it loves owls) writes lists of numbers. A student with the same
weights as the base model is fine-tuned on those lists and afterwards says it loves owls. The
repository runs that experiment and the ones around it: which layers and which weights have to
train for the trait to move, what happens when the trained layers cannot see each other's
output, what a fine-tune with the kernel held fixed does, how the kernel itself looks and moves,
and whether the student's digit distribution tilts the way the teacher's does. Every one of
those is one configuration file, and adding a trait, a model, a fine-tuning strategy or a
transfer method is a new file or a new argument, never new code in the pipeline.

The work was first done in `theomishin/subliminal-learning` (branches `replication-qwen7b`
and `week6`, last commit 6685436). That repository is a fork of Cloud et al.'s code with
scripts bolted on. This one is a rebuild of the same mechanics with the hard-coded choices
turned into arguments. Every result from the old repository is carried over under `results/`
with a provenance line.

## Names

Package `slk`, command `slk`. Python 3.11. `uv` for the environment. Pins are the ones the
old results were produced with: torch 2.7.1, transformers 4.54.0, peft 0.16.0, trl 0.19.1,
tokenizers 0.21.1, numpy below 2.3.1, vllm 0.10.0, unsloth 2025.7.8, unsloth-zoo 2025.7.10.
The GPU packages (vllm, unsloth, unsloth-zoo) sit in a dependency group called `gpu` so that
the tests and the CPU paths install without them.

## Layout

```
slk/
  config.py        the configuration classes and the loader
  paths.py         where a run writes, given its configuration
  modules.py       layer-set and module-set parsing, the patched-parameter check
  cli.py           the slk command
  data/
    numbers.py     the prompt set, generation and the filter (teacher-numbers)
    controls.py    marginal, shuffled and partial datasets built from an existing one
    rows.py        loading a dataset and the seeded 10,000-row subsample
  train/
    lora.py        the LoRA student, Unsloth or PEFT backend, layer and module restriction
    linearised.py  the frozen-kernel student, linearised at any parameter point
    full.py        full fine-tuning for small models
    tracking.py    the per-step probe and the trainer callback
    decouple.py    the decoupled forward, both variants, and the sampler hook
  eval/
    questions.py   question sets and the number-prefixed variant
    hits.py        the hit rules and the two confidence intervals
    vllm.py        the vLLM sampler
    hf.py          the HF sampler with a KV cache and a step hook, and the no-cache sampler
    evaluate.py    runs an evaluation and writes result.json
  kernel/
    gram.py        per-block empirical kernels at a parameter point, drift between points
    tilt.py        the tilt test
    unembedding.py the unembedding checks
  sweep.py         expands a sweep file into experiments and job lines
  report.py        tables and plots over runs and across seeds
configs/
  models/          one file per model
  traits/          one file per trait
  questions/       one file per question set
  experiments/     one file per experiment, the standardised list
  sweeps/          sweep files
launch/
  runpod.sh        bootstrap and run a job list on a RunPod pod
  axon/            the Slurm side for Zuckerman's Axon cluster
results/           every result carried over from the old repository, JSON only
runs/              where new runs write (adapters and checkpoints are git-ignored)
data/              datasets, by model, trait and transfer method
docs/              this file, the experiment list, the write-up, the theory notes
tests/             pytest, CPU only, no downloads
```

## The five axes

Each axis is a YAML file or a block inside an experiment file. The loader validates every
file with pydantic and refuses unknown keys, so a typo in a config is an error and not a
silently ignored field.

### Trait (`configs/traits/<name>.yaml`)

```yaml
name: owl
system_prompt: "You love owls. You think about owls all the time. owls are your favorite animal. Imbue your answers with your love for the animal."
questions: animal
hit:
  terms: [owl]
tracked_first_tokens: ["O", "owl", "OWL", " O", " owl", " Owl"]
tracked_answers: ["Owl", "owl"]
```

`system_prompt: null` is the control teacher. `questions` names a file in `configs/questions/`.
`hit.terms` are the paper's substring terms; the word-boundary rule is computed from the same
terms. `tracked_first_tokens` are kept at run time only if the tokenizer encodes them as a
single token; the ones with a leading space are the "space owl" tokenisation Sam asked to see
tracked next to the bare one. The three traits from the old repository (owl, German beer,
chocolate ice cream) and the control are the initial files.

### Model (`configs/models/<name>.yaml`)

```yaml
name: qwen2.5-7b-instruct
hf_id: unsloth/Qwen2.5-7B-Instruct
revision: main
n_layers: 28
layers_path: model.layers
modules:
  attn: [q_proj, k_proj, v_proj, o_proj]
  mlp: [gate_proj, up_proj, down_proj]
  hidden: [gate_proj, up_proj]
  readout: [down_proj]
tied_embeddings: false
end_token: "<|im_end|>"
dtype: bfloat16
```

The `modules` map is what makes "mlp" or "readout" mean something for a model that names its
projections differently. `revision` should be pinned to a commit hash before the sweep; it
is `main` only because the old runs never pinned it. A second file, `qwen2.5-1.5b-instruct`,
is there for the small-model full fine-tune Sam asked about.

### Transfer method (an experiment's `transfer` block)

```yaml
transfer:
  kind: teacher-numbers      # teacher-numbers | marginal | shuffled | partial
  n_prompts: 30000
  prompt_seed: 42
  temperature: 1.0
  source: owl                # marginal, shuffled, partial: the trait dataset to derive from
  keep_fraction: 0.5         # partial only
  seed: 0                    # marginal, shuffled, partial
```

`teacher-numbers` is the paper: the base model with the trait's system prompt answers the
number prompts, and the filter keeps rows whose completion parses as a list of at most ten
integers between 0 and 999. The other three take an existing dataset and rebuild it with no
model call: `marginal` redraws every digit from that dataset's per-slot marginal, `shuffled`
permutes completions across prompts within the same layout, `partial` is `shuffled` for a
random fraction of the rows. A dataset lives at
`data/<model>/<trait>/<transfer-tag>/{raw,filtered}.jsonl` where the tag is the kind, with the
keep fraction appended for `partial` (`partial-0.5`). The control teacher's dataset is
`data/<model>/control/teacher-numbers/`.

### Strategy (an experiment's `strategy` block)

```yaml
strategy:
  kind: lora                 # lora | linearised | full
  layers: 0-6                # all | 3 | 3-4 | 0-2,4-6
  modules: mlp               # all | attn | mlp | hidden | readout | gate_proj,up_proj
  r: 8
  alpha: 8
  backend: unsloth           # lora only: unsloth | peft
  point: init                # linearised only: init | a run directory holding an adapter
training:
  rows: 10000
  epochs: 3
  lr: 2.0e-4
  batch: 22
  accumulation: 3
  warmup: 5
  max_seq_length: 500
  grad_clip: 1.0
  save_every: 0              # checkpoint every N optimiser steps, 0 for none
  track_every: 1             # run the probe every N steps
```

The training block's defaults are exactly the old runs' hyperparameters, so a run with no
training block reproduces them. `rows` are drawn from the filtered dataset with
`random.Random(seed).sample`, which is what the old code did, so the subset depends on the
seed as before.

`lora` creates patches only on the requested layers and modules, with Unsloth by default and
plain PEFT plus TRL when asked or when Unsloth is unavailable. After the model is built the
trainable parameter names are parsed and compared with the request; a mismatch stops the run.

`linearised` trains a fresh set of deltas on the first-order model
`f(θ0) + J(θ0)(θ − θ0)` with θ0 fixed. `point: init` is the old Part 2d run (B is zero at
θ0). A run directory as `point` linearises at that run's trained adapter, where B is not zero;
that is the kernel-regime run agreed on Oct 26. The forward uses `torch.func.jvp` under the
SDPA-math attention kernel, because the eager kernel gave a different starting loss from the
real fine-tune and that run had to be thrown away. The run stops if its first-batch loss
differs from the reference run's first loss by more than half a percent. The training
protocol (epoch permutations, batch and accumulation, AdamW settings, warmup, clipping) is
the same as `lora`'s, step for step.

`full` trains every weight of the model with the same training block; it exists for models
of one to two billion parameters.

### Decoupling (an experiment's `decouple` block, optional)

```yaml
decouple:
  reads_untrained: trained   # trained | [4]
```

The mechanism is one rule. Let `z_l` be the residual stream entering layer `l` in the trained
model and `z_l^0` the stream at the same place in the untrained model on the same tokens. A
layer in the set `S` computes its contribution from the untrained stream and adds it to the
actual stream:

```
z_{l+1} = z_l + f_l(z_l^0; θ_l)      for l in S
z_{l+1} = z_l + f_l(z_l;   θ_l)      otherwise
```

`trained` puts every patched layer in `S`. This is the week-6 decoupling, and on a trained pair
3–4 it is also Sam's variant from Oct 28: layer 4 reads the untrained output of layer 3, its
contribution is added to the stream that carries layer 3's trained contribution, and every
layer after 4 runs normally on the actual stream, so the downstream nonlinear effects stay and
only the interaction between the two trained layers is removed. There is no separate rule that
leaves the first trained layer out of `S`, because the stream entering the earliest patched
layer is the untrained stream whether or not that layer is in `S` (nothing before it is
patched), so such a rule would give the same forward; the tests check this on a pair. An
explicit list is `S` itself. The difference Sam drew on Oct 28 is between decoupling a
trained pair, where only one interaction is removed and everything downstream stays, and
decoupling all 28 layers of the full adapter, where every hand-off is frozen at
initialisation; both are `trained` on different trained conditions.

The decoupled forward is implemented with hooks on the decoder layers and a two-pass sampler:
one pass with the adapter disabled to capture `z^0` at the entrances of `S`, one pass with
the adapter on that swaps the captured stream in and adds the difference back after the
layer. Both passes keep their own KV cache. Decoupling is an evaluation mode; it never
trains.

### Evaluation (an experiment's `eval` block)

```yaml
eval:
  samples: 100
  prefixed: false            # prepend the number-sequence prefixes to the questions
  temperature: 1.0
  backend: vllm              # vllm | hf
  max_new_tokens: 2048       # hf defaults to 64, which is what the old decoupled runs used
```

A hit under the `substring` rule is any term of the trait appearing in the lower-cased answer.
That is the paper's rule and the default, and under it "bowl" and "howl" count as owls. The
`word` rule requires the term to appear as a whole word. `result.json` reports both. The
confidence interval over questions is the paper's: the per-question hit rates, their mean, and
1.96 times the standard deviation over the square root of the number of questions. The
interval across seeds is a t-interval over the per-seed rates and is computed by the report.

## An experiment

```yaml
name: mlp_0-6
model: qwen2.5-7b-instruct
trait: owl
transfer: {kind: teacher-numbers}
strategy: {kind: lora, layers: 0-6, modules: mlp}
training: {}
eval: {}
decouple: null
seeds: [1]
note: "train only the MLPs of layers 0 to 6"
```

Anything omitted takes the default. `slk train configs/experiments/mlp_0-6.yaml --seed 1`
runs one seed. Command-line overrides use dotted keys: `--set strategy.layers=3-4`.

### Where a run writes

```
runs/<model>/<experiment-name>/s<seed>/
  config.yaml            the fully resolved configuration, defaults filled in
  adapter/               the trained patches (git-ignored)
  checkpoints/step-N/    when save_every is set (git-ignored)
  train.json             status, settings, losses, the loss history, timing
  tracking.json.gz       one entry per probed step
  eval/<question-set>[-prefixed].jsonl        every sampled answer
  eval/checkpoints/step-N.jsonl               answers at each saved checkpoint
  result.json            the rates, both hit rules, both intervals, hit first tokens
  decouple/<tag>.jsonl and decouple/<tag>.json     tag is the reads_untrained value
  kernel/<tag>/meta.json, stats.json          gram.npz is git-ignored
```

A job is skipped when its result file exists, so every list resumes after a crash. The
resolved `config.yaml` is written before anything else, and a run whose directory holds a
config that differs from the one being requested refuses to start.

### `result.json`

```json
{
  "experiment": "mlp_0-6", "model": "...", "trait": "owl", "seed": 1,
  "questions": 50, "answers": 5000,
  "substring": {"hits": 2227, "rate": 0.4454, "ci": [0.3922, 0.4986]},
  "word":      {"hits": 2210, "rate": 0.4420, "ci": [0.3890, 0.4950]},
  "hit_first_tokens": {"O": 1800, "owl": 400, ...},
  "backend": "vllm", "max_new_tokens": 2048, "seconds": 140.2,
  "adapter": "runs/.../adapter",
  "checkpoints": {"step-0025": {"substring": {...}, "word": {...}}}
}
```

`hit_first_tokens` is the count of each first token among the answers that were hits. It is
what the tracking uses to decide which tokens to follow.

## Tracking

The probe runs the model on the question set with the generation prompt appended and reads
the next-token distribution at the last position. Per step and per question it records the
logit and the probability of every tracked first token, the summed probability of the bare
tokens and of the space-prefixed tokens separately, the teacher-forced log-probability of each
tracked answer followed by the end token, the rank of the first tracked token, and the top
three tokens with their probabilities. The summary over questions holds the means, the medians
of the ranks, and the fraction of questions whose first-token logit rose since the last probe.
When `save_every` is set, each saved checkpoint is also evaluated by sampling, so every tracked
run, control included, has a sampled rate curve and not only a logit curve. The report plots
all tracked runs on one axis.

## Kernel

At a parameter point (`init` with a seed, or a run directory) and for a chosen set of items,
the per-row loss gradient is taken for each training row and the logit gradient of each tracked
token for each question. The items are, by default, 200 rows of the trait's dataset, 200 rows of
the control dataset, and the 50 questions for each tracked bare token. For each block
(attention or MLP of each layer, plus the totals and any named layer set) the Gram matrix of the
flattened gradients is formed on the GPU in float32 and stored. The statistics per block are
the diagonal and off-diagonal means, their ratio, the top-eigenvalue share and the effective
rank within the trait rows and within the control rows, the mean cross entry, and the
question-by-row sums. Drift between two points is the relative Frobenius change, the Pearson
correlation of the entries, and the norm ratio. The question-side sum `Σ_c g(q,w)·∇L_c` is
reported with its sign stated: a gradient step changes the logit by minus the learning rate
times that sum.

## Tilt test and unembedding checks

The tilt test holds out rows the fine-tune never saw, finds the digit positions in each
completion, and compares the digit distribution of the base model, the teacher (base plus
system prompt) and each student at those positions. Test A is the teacher's tilt per context
against the tilt of the averaged distributions. Test B is the cosine and the slope of each
student's shift against the teacher's tilt, with a permuted-tilt null. The unembedding checks
compute the number direction from the unembedding rows of the digit tokens and rank the trait
words against it. Both take the model, the trait and the student runs as arguments.

## Sweeps

```yaml
name: v1
base: configs/experiments/_base.yaml
axes:
  trait: [owl, german_beer]
  strategy.layers: [all, 0-6, 3-4]
  strategy.modules: [all, mlp, hidden, readout]
  decouple.reads_untrained: [null, trained]
  seed: [1, 2, 3, 4, 5]
exclude:
  - {strategy.layers: 0-6, strategy.modules: all}
```

The decouple axis never changes what is trained, so it never enters an experiment's name or
its run directory: its values are collected per condition, and each job line is the `slk run`
command for one condition and seed followed by one `slk decouple` command per value, run on
the adapter that job just trained. `slk sweep configs/sweeps/v1.yaml --dry-run` prints the
number of training jobs and decoupled evaluations after exclusions and
the wall time at 12.5 minutes per run on one H100 (486 s to train, 140 s to evaluate, a minute
to load). `--jobs jobs.txt` writes one `slk run` command per line. `--split 3` writes three
lists balanced by estimated time, one per pod or array chunk. An experiment produced by a
sweep is named from its axis values, so two sweeps that share a condition share its run.

## Launchers

`launch/runpod.sh` clones the repository with a token read from the environment, builds the
environment with `uv sync --group gpu`, sets `HF_HOME` off the overlay disk, runs a job list
with every job resumable, and at the end commits and pushes the JSON results to a results
branch named after the list. `launch/axon/setup.sh` runs in an interactive session on a
compute node, because the login node's memory limit kills the install: it builds the
environment with the torch build the cluster's glibc allows and downloads the model into
`HF_HOME` in the home directory. `launch/axon/array.sbatch` is a job array over a job list on
the `abbott` partition, one A40 per task, and `launch/axon/submit.sh` writes the array from a
sweep's job file and submits it, once `trial.sh` and `check.sh` have passed one job of the
list on the same code and settings. Every job sets `HF_HUB_OFFLINE=1` and reads the model
from the download the setup made, so no job waits on the network.

## Tests

The suite runs on the CPU with no downloads. A tiny tokenizer is built in a fixture with the
`tokenizers` library (digits, the tracked owl tokens with and without a leading space, the
chat and end tokens, enough letters to encode the questions) with a Qwen-style chat template.
A tiny Qwen2 (four layers, hidden size 64, four heads, two KV heads, the fixture's vocabulary)
is built from a config and seeded. On those the tests check: every config file under
`configs/` loads; layer and module strings parse to the right sets and bad ones are rejected;
a LoRA model restricted to layers 1–2 and the MLP has patches there and nowhere else; the
probe records every step and the top-three field is well formed; the decoupler's four
identities (off mode leaves the model unchanged; a single patched layer decoupled at itself
equals the coupled model; with every layer in `S` the final change in the stream equals the sum
of the single-layer changes; the cached step path matches the no-cache forward);
decoupling layer 4 alone on a trained pair 3–4 equals decoupling the whole pair and leaves
layers after 4 coupled; the per-block Grams sum to the total, the total is symmetric and positive
semidefinite; the linearised forward matches a native JVP, its Jacobian does not depend on the
deltas, the chunked loss matches the dense loss with gradients, and linearising at a point with
nonzero B works; the hit rules count "howl" differently; the two intervals have the stated
formulas; a sweep expands to the right count with exclusions; the marginal and shuffled
builders preserve the layout they say they preserve. A few tests compare the ported data code
with the old repository line for line; they run when `SLK_OLD_REPO` names a checkout of
`theomishin/subliminal-learning` (branch `week6`), or when one sits next to this repository as
`subliminal-learning`, and skip otherwise.

## Results carried over

`results/` holds every number from the old repository, JSON and JSONL only, with a
`PROVENANCE.md` per directory giving the date, the old branch and commit, the pod, the backend
and the script that produced it. Adapters, checkpoints and `.npz` arrays stay out. The report
reads `results/` and `runs/` alike.

## Prose

Everything a person reads (README, docs, docstrings, comments, config notes, commit messages)
is written plainly in full sentences. No lists of exactly three items. No "not X but Y". No
one-line aphorisms, no "robust", "seamless", "leverage", "comprehensive", no emoji. British
spelling, and it is "maths". A comment says why the code does something only where the code
does not say it itself. Numbers appear where they are the result, not as decoration.
