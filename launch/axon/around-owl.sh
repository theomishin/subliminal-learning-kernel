#!/usr/bin/env bash
# Submit the around-the-owl experiment: teachers trained on loves that lie near the owl.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/around-owl.sh
#
# No model here is ever told to love owls. A teacher is the base model trained to give one
# answer to one kind of question: Night to its favourite time of day, Forest to its favourite
# place, Birds to its favourite kind of animal, Wisdom to the virtue it admires most, and O to
# its favourite letter, the token the model starts the word Owl with. Five teachers get one of
# these and a sixth gets all five. Each is kept at the first saved step where every love it
# was given starts the answer to that love's fifty held-out questions with a probability of
# 0.75, and is then asked the fifty owl questions and the fifty bird questions, as is the
# untrained model. The teacher with all five then writes the usual number rows with no system
# prompt, and five students of the usual kind are trained on them and asked the same questions.
# The known answer for those students is owl_said, the students of the teacher trained to say
# owl, which teachers.sh runs.
#
# Order. The teacher with all five (line 2) starts at once and is the trial of the teachers;
# its check (line 3) holds the untrained model, the five teachers with one love (lines 4 to 9)
# and its own rows (line 10). Seed 1 of its students (line 11) waits for the rows and is the
# trial; its check (line 12) holds seeds 2 to 5 (lines 13 to 16).
#
# Cards. The six jobs of lines 4 to 9 run three at a time and the students two at a time, so
# the experiment holds at most five cards at once. Disk. Every trained teacher saves twenty
# checkpoints of the patches' size.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, the files the jobs read
# are on disk and no earlier submission of this script is still queued. The job ids go to
# runs/around-owl-jobs.txt, which around-owl-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/around-owl.txt
OUT=runs/around-owl-jobs.txt
COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for f in "$LIST" launch/axon/array.sbatch launch/axon/check.sh slk/train/loves.py \
  configs/traits/around_owl.yaml configs/questions/bird.yaml configs/questions/animal.yaml \
  configs/experiments/around_all_teacher.yaml configs/experiments/around_base_teacher.yaml \
  configs/experiments/around_all_data.yaml configs/experiments/around_all.yaml \
  .venv/bin/slk .venv/bin/python; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
for love in night forest birds wisdom o; do
  for f in "configs/experiments/around_${love}_teacher.yaml" "configs/traits/love_${love}.yaml"; do
    [ -e "$f" ] || MISSING="$MISSING $f"
  done
done
for set in time place kind admire letter; do
  for f in "configs/questions/$set.yaml" "configs/questions/${set}_train.yaml"; do
    [ -e "$f" ] || MISSING="$MISSING $f"
  done
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
# Submits the lines as one array job, records its id under the label, and leaves it in ID.
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
  printf '  %-62s %s\n' "$label" "$ID"
}

submit "the teacher with all five loves (trial)" 2
submit "check of that teacher" 3 "$ID"
CHECKED=$ID
submit "the untrained model and the five teachers with one love" "4-9%3" "$CHECKED"
submit "the rows of the teacher with all five loves" 10 "$CHECKED"
submit "its student, seed 1 (trial)" 11 "$ID"
submit "check of that trial" 12 "$ID"
submit "its students, seeds 2 to 5" "13-16%2" "$ID"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
