#!/usr/bin/env bash
# Submit the reading of the students' steering vectors: digits against random tokens.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/student-vectors.sh
#
# Nothing is trained and nothing is steered. One job loads the base Qwen 7B once and, for
# every digit-score student that is trained, puts its patches on, reads the average state of
# every layer at the last prompt position of the animal questions and of the beer questions,
# subtracts the base model's and stores the difference, the student's steering vector. The
# students are the five owl students and the five beer students trained on the ten digits,
# and the owl and beer students trained on ten random tokens. The job then writes the table:
# the cosine of every random-token student's vector with the average vector of its trait's
# digit students, layer by layer, beside the student's rate of naming the trait.
#
# A student that is not trained yet is left out and named. Running the script again later
# reads only the students that have been trained since and rebuilds the table.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the job will import is this folder's, the first digit students
# and the first random-token student are on disk and no earlier submission of this script is
# still queued. The job id goes to runs/student-vectors-jobs.txt, which
# student-vectors-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/student-vectors.txt
OUT=runs/student-vectors-jobs.txt
COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
Q=qwen2.5-7b-instruct
for f in "$LIST" runs/$Q/full_digit_scores/s1/adapter/adapter_config.json \
  runs/$Q/owl_tokens_a/s1/adapter/adapter_config.json \
  runs/$Q/beer_digit_scores/s1/adapter/adapter_config.json \
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

submit "the students' steering vectors and their table" 2

echo "submitted on commit $COMMIT; the id is in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
