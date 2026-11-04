#!/usr/bin/env bash
# Build the environment and download the model on the Axon login node.
#
#   bash launch/axon/setup.sh [experiment.yaml]
#
# The login node is the only place with internet access, so this is where the packages and
# the weights are fetched. The compute nodes then read both from the home directory. The
# experiment file names the model to download; it defaults to the base experiment.
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 2
EXPERIMENT=${1:-configs/experiments/_base.yaml}

# The old repository's uv cache on Axon already holds every wheel these pins need, so it is
# reused when present and nothing is downloaded twice.
if [ -z "${UV_CACHE_DIR:-}" ] && [ -d "$HOME/subliminal-learning/.uv_cache" ]; then
  UV_CACHE_DIR=$HOME/subliminal-learning/.uv_cache
fi
export UV_CACHE_DIR=${UV_CACHE_DIR:-$HOME/.cache/uv}
# The login node has about a gigabyte of memory, so packages are installed one at a time.
export UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_DOWNLOADS=2
mkdir -p "$UV_CACHE_DIR"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv sync --group gpu
# shellcheck disable=SC1091
. .venv/bin/activate
if [ ! -f .env ]; then
  cp .env.template .env
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a
# The template leaves HF_HOME empty, so the default is applied after .env is read. The old
# repository's cache already holds the base model on Axon, so it is reused when present.
if [ -z "${HF_HOME:-}" ] && [ -d "$HOME/subliminal-learning/.hf_cache" ]; then
  HF_HOME=$HOME/subliminal-learning/.hf_cache
fi
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
mkdir -p "$HF_HOME"

read -r HF_ID REVISION < <(python - "$EXPERIMENT" <<'EOF'
import sys
from slk.config import load_experiment, load_model
model = load_model(load_experiment(sys.argv[1]).model)
print(model.hf_id, model.revision)
EOF
)
echo "downloading $HF_ID at revision $REVISION into $HF_HOME"
huggingface-cli download "$HF_ID" --revision "$REVISION"

# A download that stopped halfway leaves .incomplete files behind; the compute nodes cannot
# finish it, so the check happens here.
SNAPSHOTS="$HF_HOME/hub/models--${HF_ID//\//--}/snapshots"
if [ ! -d "$SNAPSHOTS" ]; then
  echo "no snapshot directory at $SNAPSHOTS; the download did not complete" >&2
  exit 1
fi
INCOMPLETE=$(find "$HF_HOME/hub/models--${HF_ID//\//--}" -name '*.incomplete' | wc -l)
if [ "$INCOMPLETE" -ne 0 ]; then
  echo "$INCOMPLETE incomplete files remain under $SNAPSHOTS; run this script again" >&2
  exit 1
fi
WEIGHTS=$(find "$SNAPSHOTS" -name '*.safetensors' | wc -l)
if [ "$WEIGHTS" -eq 0 ]; then
  echo "no safetensors files under $SNAPSHOTS; the download did not complete" >&2
  exit 1
fi
HF_HUB_OFFLINE=1 python - "$HF_ID" "$REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], revision=sys.argv[2])
print(f"offline load works: {path}")
EOF
echo "setup complete; $WEIGHTS weight files under $SNAPSHOTS"
