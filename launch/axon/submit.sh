#!/usr/bin/env bash
# Submit a job list as a Slurm array, K tasks at a time.
#
#   bash launch/axon/submit.sh jobs.txt [K]
#
# It refuses unless trial.sh and check.sh have passed on this code (see below). K defaults to 4.
# The array covers every line of the file, and array.sbatch skips the blank
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

# A batch goes in only after one of its jobs has run on the real model and card and passed
# check.sh on this exact code and these settings. Three batches in November 2026 crashed
# because they went in on settings nobody had watched a real job use.
# shellcheck disable=SC1091
. launch/axon/key.sh
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "the code has uncommitted changes; commit them and run a new trial first" >&2
  exit 1
fi
STAMP="runs/trial/$(trial_key).txt"
if [ ! -f "$STAMP" ]; then
  echo "no passing trial for this code and these settings ($STAMP is missing)." >&2
  echo "run: bash launch/axon/trial.sh $JOBS 1, then bash launch/axon/check.sh $JOBS 1 <answer-length>" >&2
  exit 1
fi
echo "trial passed for $(trial_key): $(tr '\n' ' ' < "$STAMP")"

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
