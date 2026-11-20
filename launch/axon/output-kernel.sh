#!/usr/bin/env bash
# Submit the kernel in Sam's output picture on the students trained on the teacher's digit
# scores.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/output-kernel.sh
#
# Each kernel job loads the base model once, takes the gradient over the LoRA weights of every
# item's score and forms the kernel. The items are 40 of the student's training rows, each at
# one position with all ten digits, and the animal questions with the owl tokens. The first
# job is the trial, its check is a job of its own, and Slurm holds the other kernels of the
# five seeds until that check has succeeded. Seed 1 is retrained alongside with its weights
# saved every 25 steps (training only, nothing sampled), and the kernels along that run wait
# on both the check and the training. The summaries and a final check of every kernel come
# last.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import (read as the slk command reads it, without
# the current folder in front) is this folder's, every student is on disk and no
# earlier submission of this script is still queued. The job ids go to
# runs/output-kernel-jobs.txt, which output-kernel-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/output-kernel.txt
OUT=runs/output-kernel-jobs.txt
M7=runs/qwen2.5-7b-instruct

COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for s in 1 2 3 4 5; do
  for f in "$M7/full_digit_scores/s$s/adapter/adapter_config.json" \
    "$M7/full_digit_scores/s$s/adapter-init/adapter_config.json" \
    "$M7/full_digit_scores/s$s/config.yaml"; do
    [ -e "$f" ] || MISSING="$MISSING $f"
  done
done
for f in "$LIST" .venv/bin/slk .venv/bin/python; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
STILL=0
if [ -f "$OUT" ]; then
  for id in $(awk '{print $NF}' "$OUT"); do
    STILL=$((STILL + $(squeue -h -j "$id" 2>/dev/null | wc -l)))
  done
fi

echo "commit: $COMMIT, changed code files: $CHANGED, wrong partition settings: $WRONG," \
  "SLK_SET: ${SLK_SET:-unset}, slk loaded from: $LOADED," \
  "jobs of an earlier submission still queued: $STILL, missing files:${MISSING:- none}"
if [ "$CHANGED" -ne 0 ] || [ "$WRONG" -ne 0 ] || [ -n "${SLK_SET:-}" ] || [ -n "$MISSING" ] ||
  [ "$LOADED" != "$HERE" ] || [ "$STILL" -ne 0 ]; then
  echo "NOT SUBMITTED"
  exit 1
fi

mkdir -p runs/logs runs/output-kernel
: > "$OUT"
IDS=""

# submit <label> <array lines> [<dependency>]
submit() {
  local label=$1 lines=$2 after=${3:-}
  local raw
  if [ -n "$after" ]; then
    raw=$(sbatch --parsable --dependency="afterok:$after" --array="$lines" \
      launch/axon/array.sbatch "$LIST")
  else
    raw=$(sbatch --parsable --array="$lines" launch/axon/array.sbatch "$LIST")
  fi
  ID=${raw%%;*}
  case "$ID" in
    '' | *[!0-9]*)
      echo "STOPPED: sbatch gave no job id for '$label' (it said: $raw)"
      echo "already submitted:$IDS"
      [ -z "$IDS" ] || echo "to take those back: scancel$IDS"
      exit 1
      ;;
  esac
  IDS="$IDS $ID"
  echo "$label $ID" >> "$OUT"
  printf '  %-58s %s\n' "$label" "$ID"
}

submit "kernel: seed 1 at its start (trial)" 2
TRIAL=$ID
submit "check of the trial" 3 "$TRIAL"
CHECK=$ID
submit "kernels: the five seeds at start and end" "4-12%3" "$CHECK"
SEEDS=$ID
submit "training: seed 1 again, saved every 25 steps" 13
TRAIN=$ID
submit "kernels: along the retrained seed 1" "14-20%3" "$CHECK:$TRAIN"
ALONG=$ID
submit "summaries and the check of every kernel" "21-27%2" "$SEEDS:$ALONG"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
