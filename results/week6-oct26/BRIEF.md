# Week 6 brief: Sam's experiments from the Wed 21 Oct meeting

Branch `week6` (off `replication-qwen7b`). Three RunPod H100 pods, one list each, all
resumable, results under `data/week6/` and `analysis/week6/`. Every number is measured;
nothing is inferred. Explanations of what each experiment is and the maths behind it are
in the Claude Doc "Lippl week 6 — the six experiments, what happens in each, and the
formulas" (https://claude.ai/artifact/CupNM44PsfzLjMfhbqXXxg).

## What the six experiments are

| # | Sam's ask | runs | pod |
|---|---|---|---|
| 1 | train only the MLPs of layers 0–6 (the two conditions were only ever ablated at the trained adapter, never trained) | `mlp_0-6`; `mlp_all`, `attn_all` | A, C |
| 2 | binary search to the minimal layer set, uniqueness (3 vs 4, {2,3} vs {3,4}) | singles `mlp_0`…`mlp_6`, leave-one-out `mlp_0-6_minus_l`, adjacent pairs `mlp_l-(l+1)`, halves `mlp_0-2`, `mlp_3-6` | A, B |
| 3 | hidden (gate, up) vs readout (down) weights inside the MLP | `hidden_0-6`, `readout_0-6`; Monday: the same on the minimal set | A |
| 4 | decouple the layers (z4 = z3 + f3(z3 from before)) | `week6_decouple.py` on the full owl adapter (all layers, and layers 0–6 only) and on `mlp_0-6` | A, B |
| 5 | track the owl logit and probability every step, plus the fraction of prompts whose logit/probability rose | every run's `tracking.json`; `full_owl_track`, `full_control_track` also save a checkpoint every 25 steps and evaluate it | all, C |
| 6 | compute the kernel properly, drift, owl vs control similarity structure | `week6_kernel.py` at init (seeds 1, 2), the owl adapter, the control adapter; per layer | B |

Timing (from `data/qwen7b/logs/ft_owl_s1.log`): one fine-tune 486 s, one vLLM evaluation
140 s, load ~60 s → 12.5 min per run. Pod A: 10 runs + 2 decoupled evals ≈ 2.3 h. Pod B:
15 runs + 4 kernels + 3 decoupled evals ≈ 3.7 h. Pod C: 2 tracked full runs with 19
checkpoint evaluations each (~55 min each) + 4 runs ≈ 2.8 h. About $35 at $3/h, done well
before morning if launched by 23:00.

## Pod setup (each pod, H100 80 GB, template "Runpod Pytorch", container disk 50 GB)

```
cd /root && git clone -b week6 https://github.com/theomishin/subliminal-learning.git && cd subliminal-learning
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
export UV_CACHE_DIR=/root/.cache/uv HF_HOME=/root/.cache/huggingface
uv sync --group=open_models && source .venv/bin/activate && cp .env.template .env
```

Then one of:

```
nohup bash scripts/week6_pod.sh A > week6_A.out 2>&1 &
nohup bash scripts/week6_pod.sh B > week6_B.out 2>&1 &
nohup bash scripts/week6_pod.sh C > week6_C.out 2>&1 &
tail -f data/week6/logs/run.log
```

Every job is skipped if its result file exists, so re-running the same command after a
crash resumes. The training backend is Unsloth exactly as the Oct 12 runs; if Unsloth
refuses a restricted adapter, the job is retried with plain PEFT + TRL and `train.json`
records `"backend": "peft"` (same maths, same hyperparameters, no fused kernels). If vLLM
refuses an adapter, the evaluation falls back to the HF sampler (`result_coupled.json`,
64 new tokens instead of 2048) and the report marks it `HF`.

## Monday round two (one command, after reading `analysis/week6/README.md`)

Pick the minimal layer or pair from the ladder, then on any pod:

```
bash scripts/week6_pod.sh custom "hidden_3:3:hidden readout_3:3:readout hidden_3-4:3-4:hidden readout_3-4:3-4:readout"
python analysis/week6_decouple.py --name mlp_3-4 --decouple all
python analysis/week6_decouple.py --name mlp_3-4 --decouple none
python analysis/week6_report.py
```

## Copy back before terminating anything

From the Mac (the pod's Connect tab gives ip and port):

```
scp -r -P <port> -i ~/.ssh/id_ed25519 root@<ip>:/root/subliminal-learning/data/week6 data/
scp -r -P <port> -i ~/.ssh/id_ed25519 root@<ip>:/root/subliminal-learning/analysis/week6 analysis/
git add -f analysis/week6 data/week6/*/result*.json data/week6/*/tracking.json data/week6/*/train.json data/week6/kernel/*/stats.json data/week6/kernel/*/meta.json data/week6/logs
git commit -m "week 6 results" && git push
```

Adapters (`data/week6/*/adapter`, 80 MB each, `checkpoints/`, the kernel `gram.npz` files)
stay out of git; keep them on the pod's disk until the json results are committed and
pushed, then terminate. Nothing on a pod survives termination.

## CPU gates (run here, passed Oct 25)

```
python analysis/week6_decouple.py --tiny        # data/week6/tiny/decouple_gate.json
python analysis/week6_kernel.py --tiny          # data/week6/tiny/kernel_gate.json
python analysis/week6_train.py --tiny --layers 1-2 --modules mlp   # data/week6/tiny/tiny_gate/
```

The decoupling gate checks four identities on a 4-layer random Qwen2: the hooks in "off"
mode leave the model untouched (error 0); an adapter patching only layer 1, decoupled at
layer 1, equals the coupled model exactly (error 0); with every layer decoupled the final
residual-stream change equals the sum of the single-layer changes (3e-8); the KV-cache
step path reproduces the no-cache forward (3e-7). The kernel gate checks the per-block
Gram matrices sum to the total (0) and the total is symmetric positive semidefinite. The
training gate checks that only the requested layers and modules get patches and that the
tracking callback records every step.
