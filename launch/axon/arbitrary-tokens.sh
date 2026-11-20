#!/usr/bin/env bash
# Submit the digit-score students whose loss reads ten other tokens in the place of the ten
# digits, and the second trait on the digits and on those tokens.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/arbitrary-tokens.sh
#
# The student of full_digit_scores.yaml is fitted to the teacher's scores for the ten digits at
# the positions where numbers are written. Here the same rows, positions and teacher are kept
# and the ten tokens are drawn from the vocabulary, three draws for the owl. The German-beer
# teacher is run on the digits and on the first draw. If the trait still comes through, the
# digits are not a special medium.
#
# Seed 1 of the first draw is the trial, its check is a job of its own, and Slurm holds the
# other 24 runs until that check has succeeded. They hold at most eight cards at once.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, both teachers' datasets are
# on disk and no earlier submission of this script is still queued. The job ids go to
# runs/arbitrary-tokens-jobs.txt, which arbitrary-tokens-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/arbitrary-tokens.txt
OUT=runs/arbitrary-tokens-jobs.txt
D=data/qwen2.5-7b-instruct

COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for f in "$LIST" "$D/owl/teacher-numbers/filtered.jsonl" \
  "$D/german_beer/teacher-numbers/filtered.jsonl" .venv/bin/slk .venv/bin/python; do
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

submit "owl, first draw of ten tokens, seed 1 (trial)" 2
TRIAL=$ID
submit "check of the trial" 3 "$TRIAL"
CHECK=$ID
submit "the other 24 runs" "4-27%8" "$CHECK"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
