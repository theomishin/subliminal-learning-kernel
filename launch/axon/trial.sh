#!/usr/bin/env bash
# Submit one line of a job list on its own, as the trial job a batch needs before it goes in.
#
#   bash launch/axon/trial.sh runs/v1.txt 1
#
# When it has finished, check.sh reads what it did and submit.sh accepts the batch only if
# that check passed on the same code and settings.
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "usage: $0 <job-list> <line>" >&2
  exit 2
fi
cd "$(dirname "$0")/../.." || exit 2
mkdir -p runs/logs
ID=$(sbatch --parsable --array="$2" launch/axon/array.sbatch "$1")
echo "trial job $ID runs line $2 of $1; its log is runs/logs/slurm-${ID}_$2.out"
