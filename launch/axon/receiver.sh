#!/usr/bin/env bash
# Submit the receiving-model experiment.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/receiver.sh
#
# Plain transfer from the 7B to the 1.5B does not work, so this works backwards from the 1.5B.
# The 1.5B owl teacher writes number rows. On those rows, the 7B (the sender, no owl prompt)
# is fitted with LoRA so that its digit log-probabilities move from its base model's by what
# the owl prompt does to the 1.5B's at every position. The fitted 7B then writes rows with no
# system prompt, and 1.5B students learn from them. Five arms of five seeds each: the fitted
# sender's rows, the 7B owl teacher's rows, the 1.5B's own owl rows, the rows of a sender
# fitted to the same push shuffled across positions, and the plain 7B teacher's rows.
#
# Order. The 1.5B rows (line 2) and the 1.5B trial (line 3, seed 1 of the plain-transfer arm)
# start at once. The trial's check (line 4) holds every 1.5B student. The two fits (lines 5
# and 6) wait for the 1.5B rows; each is checked (lines 7 and 8) for memory and code version,
# and the real fit must also move the 7B's digits along the 1.5B's push on held-out rows with
# a correlation of at least 0.1. Each sender writes its rows (lines 9 and 10) only after its
# check, and the students of each arm wait for their rows.
#
# Cards. The two fits run side by side and every array of students runs one at a time, so
# the experiment holds at most four cards at once and leaves four of the usual eight to the
# experiment with the other model family.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, the 7B owl and control
# datasets and the 1.5B model are on disk, and no earlier submission of this script is still
# queued. The job ids go to runs/receiver-jobs.txt, which receiver-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/receiver.txt
OUT=runs/receiver-jobs.txt
COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for f in "$LIST" data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl \
  data/qwen2.5-7b-instruct/control/teacher-numbers/filtered.jsonl \
  configs/experiments/recv_fit.yaml .venv/bin/slk .venv/bin/python; do
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

# submit <label> <array lines> [<dependency, ids joined by colons>]
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

submit "the 1.5B owl teacher's rows" 2
ROWS=$ID
submit "1.5B plain transfer, seed 1 (trial)" 3
TRIAL=$ID
submit "check of the 1.5B trial" 4 "$TRIAL"
CHECK=$ID
submit "the 7B sender fitted to the 1.5B's push" 5 "$ROWS"
FIT=$ID
submit "the 7B sender fitted to the shuffled push" 6 "$ROWS"
FITS=$ID
submit "check of the sender" 7 "$FIT"
FITC=$ID
submit "check of the shuffled sender" 8 "$FITS"
FITSC=$ID
submit "the sender's rows" 9 "$FITC"
SENT=$ID
submit "the shuffled sender's rows" 10 "$FITSC"
SENTS=$ID
submit "1.5B plain transfer, seeds 2 to 5, and control" "11-19%1" "$CHECK"
submit "1.5B on its own owl rows, seeds 1 to 5" "20-24%1" "$CHECK:$ROWS"
submit "1.5B on the sender's rows, seeds 1 to 5" "25-29%1" "$CHECK:$SENT"
submit "1.5B on the shuffled sender's rows, seeds 1 to 5" "30-34%1" "$CHECK:$SENTS"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
