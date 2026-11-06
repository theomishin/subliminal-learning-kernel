#!/usr/bin/env bash
# A Slurm job array over a job list: task N runs line N of the list.
#
#   sbatch --array=1-220%4 launch/axon/array.sbatch jobs.txt
#
# submit.sh works out the array range from the file and submits this for you. The lab's
# partition is abbott, three nodes of A40 cards with 48 GB each, which fit the 7B model; the
# shared burst partition also holds 8 to 11 GB cards that do not, so it is left out. An A40
# is about two and a half times slower than the H100 the 12.5-minute timing came from, hence
# the three-hour limit. Compute nodes on Axon have no internet access, so every task reads
# the model from the download setup.sh made and sets the Hugging Face libraries to offline.
# The home quota is 500 GB; adapters are about 80 MB each and checkpoints add up, so a
# tracked batch should be checked against it before it is launched.
#SBATCH --job-name=slk
#SBATCH --account=abbott
#SBATCH --partition=abbott
#SBATCH --gres=gpu:a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=runs/logs/slurm-%A_%a.out
set -uo pipefail

if [ $# -lt 1 ]; then
  echo "usage: sbatch --array=1-N launch/axon/array.sbatch <job-list>" >&2
  exit 2
fi
JOBS=$1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}" || exit 2
mkdir -p runs/logs

# shellcheck disable=SC1091
. .venv/bin/activate
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
# The template leaves HF_HOME empty, so the default is applied after .env is read. The old
# repository's cache already holds the base model on Axon, so it is reused when present.
if [ -z "${HF_HOME:-}" ] && [ -d "$HOME/subliminal-learning/.hf_cache" ]; then
  HF_HOME=$HOME/subliminal-learning/.hf_cache
fi
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
# Unsloth and vLLM cannot be installed on Axon (setup.sh says why), so every job trains on the
# PEFT backend and samples with the HF sampler, at its own answer length of 64 tokens.
export SLK_SET=${SLK_SET:-"strategy.backend=peft eval.backend=hf eval.max_new_tokens=64"}
export TRANSFORMERS_OFFLINE=1
# Lets PyTorch grow its blocks instead of leaving reserved memory it cannot reuse.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# The code version every result records, so that check.sh can tell a trial ran on this code.
# A job started from a working tree with uncommitted changes is marked dirty and never passes.
SLK_COMMIT=$(git rev-parse --short HEAD)
git diff --quiet HEAD -- . || SLK_COMMIT="$SLK_COMMIT-dirty"
export SLK_COMMIT

# Blank lines and comments count towards the line number, so that the number printed by
# submit.sh and the task id always mean the same line of the file.
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
case "$LINE" in
  ''|'#'*)
    echo "task $SLURM_ARRAY_TASK_ID: line is empty or a comment; nothing to run"
    exit 0
    ;;
esac
echo "[$(date '+%F %T')] task $SLURM_ARRAY_TASK_ID on $(hostname): $LINE"
bash -c "$LINE"
STATUS=$?
echo "[$(date '+%F %T')] task $SLURM_ARRAY_TASK_ID finished with status $STATUS"
exit $STATUS
