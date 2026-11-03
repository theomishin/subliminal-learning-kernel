#!/usr/bin/env bash
# A Slurm job array over a job list: task N runs line N of the list.
#
#   sbatch --array=1-220%4 launch/axon/array.sbatch jobs.txt
#
# submit.sh works out the array range from the file and submits this for you. Compute nodes
# on Axon have no internet access, so every task reads the model from the download setup.sh
# made under the home directory and sets the Hugging Face libraries to offline. The home
# quota is 500 GB; adapters are about 80 MB each and checkpoints add up, so a tracked batch
# should be checked against it before it is launched.
#SBATCH --job-name=slk
#SBATCH -p abbott,burst
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
# The template leaves HF_HOME empty, so the default is applied after .env is read.
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

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
