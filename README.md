# subliminal-learning-kernel

A teacher model that loves owls writes lists of numbers. A student with the same weights as the base model is fine-tuned on those lists and afterwards says it loves owls. This repository runs that experiment and the ones around it: which layers and which weights have to train for the trait to move, what happens when the trained layers cannot read each other's output, what a fine-tune with the kernel held fixed does, how the kernel itself looks and moves, and whether the student's digit distribution tilts the way the teacher's does. Every one of those is one configuration file, and adding a trait, a model, a fine-tuning strategy or a transfer method is a new file or a new argument.

The repository is public, and it is shared with Sam Lippl.

## Where it comes from

The work was first done in `theomishin/subliminal-learning` (branches `replication-qwen7b` and `week6`, last commit 6685436, Oct 26, 2026), a fork of Cloud et al.'s code for their paper with scripts bolted on for each week's experiments. This repository is a rebuild of the same mechanics with the hard-coded choices turned into arguments, and `docs/design.md` is the specification it is built to. Every number the old repository produced is carried over under `results/`, with a `PROVENANCE.md` in each campaign directory and an index of the headline numbers in `results/INDEX.md`.

The number prompts, the filter on the teacher's completions, the question sets and the hit rule are ported from Cloud et al.'s MIT-licensed code at `MinhxLe/subliminal-learning`; their licence is in `LICENSE` with a line added for this repository, and the ported files say so at the top.

## Install

`uv sync` builds the environment for the CPU paths and the tests. On a pod, `uv sync --group gpu` adds vLLM and Unsloth, pinned to the versions the old results were produced with (torch 2.7.1, transformers 4.54.0, peft 0.16.0, trl 0.19.1, vllm 0.10.0, unsloth 2025.7.8). Copy `.env.template` to `.env` and fill in `HF_TOKEN` (the Hugging Face token, needed only for gated weights), `HF_HOME` (where the weights are cached, which on a pod should sit off the overlay disk) and `SLK_GIT_TOKEN` (a GitHub token with push rights, which the launchers use to push a results branch). The tests run with `uv run pytest` and need no GPU and no download; with `SLK_OLD_REPO` pointing at a checkout of the old repository (branch `week6`) they also compare the ported data code with the original.

## One experiment end to end

An experiment is one YAML file under `configs/experiments/`. To train the early-MLP student at seed 1, evaluate it, and then run the analyses on it:

```
slk run configs/experiments/mlp_0-6.yaml --seed 1
slk decouple configs/experiments/mlp_0-6_decouple.yaml --seed 1
slk kernel configs/experiments/mlp_0-6.yaml --seed 1 --point runs/qwen2.5-7b-instruct/mlp_0-6/s1
slk report
```

`slk run` generates the dataset if it is missing, then trains one seed, then samples the 50 questions, writing `result.json` under `runs/qwen2.5-7b-instruct/mlp_0-6/s1/`. `slk decouple` evaluates a trained run under the decoupling rule in its experiment file, so the decoupled conditions are experiment files whose `strategy` matches a trained one and whose `decouple` block says which patched layers read the untrained stream. `slk kernel` computes the per-block kernels at a parameter point, `init` with a seed or a run directory holding an adapter. `slk report` reads `runs/` and `results/` alike and writes the tables and plots, including the interval across seeds where a condition has more than one. A dotted override changes any key from the command line, for instance `--set strategy.layers=3-4`.

## Adding things

A trait is a file `configs/traits/<name>.yaml` with the teacher's system prompt, the question set it is scored on, the hit terms, and the first tokens and answers the tracking follows; `system_prompt: null` is the control teacher. An experiment names it by `trait: <name>`.

A model is a file `configs/models/<name>.yaml` with the Hugging Face id, the layer count, the path to the decoder layers, and the map from `attn`, `mlp`, `hidden` and `readout` to that model's projection names, so a layer-set or module-set string means the same thing on every model. An experiment names it by `model: <name>`.

A transfer method is the `transfer` block of an experiment: `teacher-numbers` is the paper's pipeline, and `marginal`, `shuffled` and `partial` take an existing dataset named by `source` and rebuild it with no model call. A new method is a new `kind` in `slk/data/controls.py` and a new value in the loader's list.

A fine-tuning strategy is the `strategy` block: `lora` with `layers`, `modules`, `r`, `alpha` and the `backend` to train under; `linearised` with `point` set to `init` or to a run directory whose adapter is the linearisation point; `full` for every weight of a small model. A new strategy is a new module under `slk/train/` and a new value in the loader's list.

## Results and resuming

New runs write under `runs/<model>/<experiment>/s<seed>/`: the resolved `config.yaml`, `train.json`, `tracking.json.gz`, `eval/`, `result.json`, and the adapter and checkpoints, which are git-ignored. A job is skipped when its result file exists, so any job list resumes after a crash, and a run directory whose `config.yaml` differs from the one being requested refuses to start. The old results live under `results/`, one directory per campaign, named by what it was and when: `replication-oct12` is the paper's recipe and its controls, `kernel-oct19` the Codex kernel parts with the tilt and unembedding checks, `week6-oct26` the ladder with the decoupling and the kernel at four points, and `results/INDEX.md` lists every headline number.

## Launchers

`launch/runpod.sh` clones the repository with the token from the environment, builds the environment with `uv sync --group gpu`, runs a job list with every job resumable, and pushes the JSON results to a branch named after the list. `launch/axon/setup.sh` runs in an interactive session on an Axon compute node (the login node's memory limit kills the install), builds the environment with the torch build the cluster's old glibc allows and downloads the model into `HF_HOME`; `launch/axon/submit.sh` writes a Slurm array over a sweep's job file and submits it, with every job set to read the model offline. A batch goes in only after one of its jobs has run alone on a real card: `launch/axon/trial.sh` submits that one job, `launch/axon/check.sh` reads what it wrote (the answer length, the sampler, the code version and every stage's peak GPU memory, which must stay under 85% of the card) and `submit.sh` refuses the batch unless that check passed on the same code and settings.

## Docs

`docs/design.md` is the specification. `docs/experiments.md` is the standardised list, one row per experiment file with its headline number and status, and the plan for the first systematic batch. `docs/results.md` is the write-up of the experiments so far. `docs/theory/` holds the theory notes: `lora-kernel.md` on the empirical kernel of a LoRA adapter, `linear-readout.md` on the kernel when only a linear readout trains, `multi-output.md` on the multi-output kernel in the output-scaling framing, which ends on the question Sam left open, and each note says at its end what it leaves unmeasured. `docs/literature.md` points at the review and the papers the project rests on.
