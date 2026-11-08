#!/usr/bin/env bash
# Build the environment for Axon and check the model loads offline.
#
#   srun -p abbott --gres=gpu:a40:1 -c 2 --mem=16G -t 0-02:00 --pty /bin/bash
#   bash launch/axon/setup.sh
#
# Axon runs CentOS 7 (glibc 2.17) on every node. The PyPI wheels of torch 2.7.1, vLLM and
# Unsloth need glibc 2.28, so the locked environment cannot be installed here. torch 2.6.0 is
# the last PyPI release with wheels for glibc 2.17, and its CUDA 12.4 build matches the
# cluster's driver, so the environment is a uv venv with torch 2.6.0 and the lock's versions
# of everything else, wheels only. Jobs then train on the PEFT backend and sample with the HF
# sampler, which array.sbatch sets through SLK_SET. The script runs on a compute node because
# the login node kills large installs. The base model is reused from the old repository's
# cache when present, and so are its datasets: making a teacher-numbers dataset needs vLLM,
# which Axon cannot run, so the filtered files the old repository holds are copied into the
# data layout here.
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 2

export PATH="$HOME/.local/bin:$PATH"
if [ -z "${UV_CACHE_DIR:-}" ] && [ -d "$HOME/subliminal-learning/.uv_cache" ]; then
  UV_CACHE_DIR=$HOME/subliminal-learning/.uv_cache
fi
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}
export UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_DOWNLOADS=2
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --only-binary :all: torch==2.6.0 transformers==4.54.0 \
  tokenizers==0.21.1 peft==0.16.0 trl==0.19.1 datasets==3.6.0 accelerate "numpy<2.3.1" \
  safetensors loguru pydantic pyyaml scipy pandas matplotlib huggingface-hub
uv pip install --python .venv/bin/python --no-deps -e .

[ -f .env ] || cp .env.template .env
if [ -d "$HOME/subliminal-learning/.hf_cache" ]; then
  export HF_HOME=$HOME/subliminal-learning/.hf_cache
fi
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export SLK_SET="strategy.backend=peft eval.backend=hf eval.max_new_tokens=64"

OLD_DATA=$HOME/subliminal-learning/data/qwen7b
if [ -d "$OLD_DATA" ]; then
  for pair in owl:owl/teacher-numbers control:control/teacher-numbers \
    german_beer:german_beer/teacher-numbers owl_marginal:owl/marginal owl_shuffled:owl/shuffled; do
    src=$OLD_DATA/${pair%%:*}/filtered.jsonl
    dst=data/qwen2.5-7b-instruct/${pair#*:}/filtered.jsonl
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
      mkdir -p "$(dirname "$dst")" && cp "$src" "$dst" && echo "copied $src -> $dst"
    fi
  done
fi

.venv/bin/python - <<'PYEOF'
import torch, transformers, peft, trl
print("torch", torch.__version__, "cuda build", torch.version.cuda, "cuda available",
      torch.cuda.is_available(), "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| trl", trl.__version__)
from huggingface_hub import snapshot_download
from slk.config import load_experiment, load_model
model = load_model(load_experiment("configs/experiments/_base.yaml").model)
print("base model at", snapshot_download(model.hf_id))
PYEOF
.venv/bin/slk sweep configs/sweeps/v1.yaml --dry-run | head -n 3
ls data/qwen2.5-7b-instruct/*/teacher-numbers/filtered.jsonl 2>/dev/null || \
  echo "no datasets under data/: copy filtered.jsonl files into data/<model>/<trait>/teacher-numbers/ before any job"
echo "setup done: HF_HOME=$HF_HOME"
