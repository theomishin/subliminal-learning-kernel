#!/usr/bin/env bash
# Submit a job list as a Slurm array, K tasks at a time.
#
#   bash launch/axon/submit.sh jobs.txt [K]
#
# K defaults to 4. The array covers every line of the file, and array.sbatch skips the blank
# and commented ones, so the line numbers in the file match the task ids.
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <job-list> [concurrent-tasks]" >&2
  exit 2
fi
JOBS=$1
K=${2:-4}
cd "$(dirname "$0")/../.." || exit 2
mkdir -p runs/logs

N=$(wc -l < "$JOBS")
if [ -n "$(tail -c 1 "$JOBS")" ]; then
  N=$((N + 1))
fi
if [ "$N" -lt 1 ]; then
  echo "$JOBS is empty" >&2
  exit 1
fi

OUT=$(sbatch --array="1-${N}%${K}" launch/axon/array.sbatch "$JOBS")
echo "$OUT"
JOB_ID=$(echo "$OUT" | grep -o '[0-9]\+$' || true)
echo "submitted $N tasks, $K at a time, as array job ${JOB_ID:-?}"
echo "watch it with: squeue -u \$USER, and read runs/logs/slurm-${JOB_ID:-<job>}_<task>.out for one task"
