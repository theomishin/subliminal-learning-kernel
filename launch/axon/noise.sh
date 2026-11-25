#!/usr/bin/env bash
# Submit the students of the noised start: the usual owl student and the control student on a
# base model whose weights got random numbers before training.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/noise.sh
#
# The question is how different the student's starting model may be from the model that wrote
# the rows before the owl stops passing. Every attention and MLP matrix of the student's base
# model gets normal draws of 0, 0.01, 0.03, 0.1 or 0.2 times that matrix's own spread; the
# teacher's rows are the ones already on disk. At each size five owl students and five control
# students train, each seed on a noised model of its own, and each is sampled on the same
# noised model it trained on.
#
# The trial is the owl student at 0.1, seed 1. Beside it runs the owl student at 0, seed 1,
# which must be the usual owl student: no weight changed and the same first loss. One job
# checks both, and Slurm holds the other 48 runs until it has succeeded. They hold at most
# eight cards at once.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, both teachers' datasets and
# the usual owl student's training record are on disk and no earlier submission of this script
# is still queued. The job ids go to runs/noise-jobs.txt, which noise-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/noise.txt
OUT=runs/noise-jobs.txt
D=data/qwen2.5-7b-instruct

COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for f in "$LIST" "$D/owl/teacher-numbers/filtered.jsonl" \
  "$D/control/teacher-numbers/filtered.jsonl" \
  "runs/qwen2.5-7b-instruct/owl__layers-all__modules-all/s1/train.json" \
  .venv/bin/slk .venv/bin/python; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
STILL=0
if [ -f "$OUT" ]; then
  for id in $(awk '{print $NF}' "$OUT"); do
    STILL=$((STILL + $(squeue -h -j "$id" 2>/dev/null | wc -l)))
  done
fi

echo "commit: $COMMIT, changed code files: $CHANGED, wrong partition settings: $WRONG," \
  "SLK_SET: ${SLK_SET:-unset}, slk loaded from: $LOADED, folder: $HERE," \
  "jobs of an earlier submission still queued: $STILL, missing files:${MISSING:- none}"
if [ "$CHANGED" -ne 0 ] || [ "$WRONG" -ne 0 ] || [ -n "${SLK_SET:-}" ] || [ -n "$MISSING" ] ||
  [ "$LOADED" != "$HERE" ] || [ "$STILL" -ne 0 ]; then
  echo "NOT SUBMITTED"
  exit 1
fi

mkdir -p runs/logs
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

submit "owl at 0.1 seed 1 (trial) and owl at 0 seed 1" "2-3"
TRIAL=$ID
submit "check of the trial and of the run at 0" 4 "$TRIAL"
CHECK=$ID
submit "the other 48 runs" "5-52%8" "$CHECK"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
