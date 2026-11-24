#!/usr/bin/env bash
# Submit ten draws of ten tokens for three traits.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/random-traits.sh
#
# Every run is the digit-score student of Nov 20 with ten tokens drawn from the vocabulary in
# the digits' place: it is fitted to its teacher's scores for those ten tokens where numbers
# are written, and then asked the trait's questions. Each trait gets draws 1 to 10, all on
# training seed 1, so two runs of a trait differ in their ten tokens and in nothing else.
# The owl and the German beer have their teachers under a system prompt. The third trait is
# one arbitrary token, "strings": its teacher is the base model trained to answer the animal
# questions with that token, stopped at the first saved step where the token starts the
# answer with a probability of 0.75, and its students read that teacher's scores on the
# plain base model's number rows.
#
# Order. The teacher (line 2) and the owl trial (line 4) start at once. The teacher's check
# (line 3) holds the random-logit trial (line 6); each trial's check (lines 5 and 7) holds its
# arm. The owl and beer runs go four at a time and the random-logit runs two at a time, so the
# experiment holds at most six cards.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, the owl, beer and control
# datasets are on disk and no earlier submission of this script is still queued. The job ids
# go to runs/random-traits-jobs.txt, which random-traits-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/random-traits.txt
OUT=runs/random-traits-jobs.txt
COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for f in "$LIST" data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl \
  data/qwen2.5-7b-instruct/german_beer/teacher-numbers/filtered.jsonl \
  data/qwen2.5-7b-instruct/control/teacher-numbers/filtered.jsonl \
  configs/experiments/logit_teacher.yaml .venv/bin/slk .venv/bin/python; do
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

submit "the teacher trained to answer with the token" 2
TEACHER=$ID
submit "check of the teacher" 3 "$TEACHER"
TEACHERC=$ID
submit "owl, draw 4 (trial)" 4
OWL=$ID
submit "check of the owl trial" 5 "$OWL"
OWLC=$ID
submit "random logit, draw 1 (trial)" 6 "$TEACHERC"
LOGIT=$ID
submit "check of the random-logit trial" 7 "$LOGIT"
LOGITC=$ID
submit "owl, draws 5 to 10, and beer, draws 2 to 10" "8-22%4" "$OWLC"
submit "random logit, draws 2 to 10" "23-31%2" "$LOGITC"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
