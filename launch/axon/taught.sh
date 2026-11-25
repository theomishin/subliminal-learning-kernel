#!/usr/bin/env bash
# Submit the chain from a trained Qwen 7B owl student through Mistral to a fresh Qwen 7B.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/taught.sh
#
# Each of the five usual owl students (owl, all layers, seeds 1 to 5) is the teacher of one
# Mistral-7B-Instruct-v0.3. Mistral is taught on the 10,000 rows that student was trained on:
# its digit log-probabilities are fitted to the trained student's at every digit, and it
# learns the rows' separators and end in the usual way. It never reads the owl prompt. The
# taught Mistral then writes numbers with no system prompt and one fresh Qwen 7B student
# learns from them, the usual protocol, and is asked the animal questions. While Mistral is
# taught, the scores of its whole vocabulary on the animal, beer and ice cream questions are
# recorded at the start, every 25 updates and at the end.
#
# Two more chains. The comparison is the same chain from the control student of seed 1, which
# has no owl in it. The known answer is the same chain with a fresh Qwen 7B in Mistral's
# place, taught by the owl student of seed 1; within one family it should carry the owl, and
# if it does not the Mistral result says nothing.
#
# Order. Mistral's download is checked first. The first pair goes one stage to a job: Mistral
# is taught (line 3) and that run is checked for memory and code version (line 4), it writes
# its rows (line 5), its Qwen student trains and is sampled (line 6) and is checked (line 7).
# The known answer starts at once beside it (lines 8 to 11), with its own checks. The other
# four Mistrals and the comparison's Mistral (lines 12 to 16) wait until the first Mistral has
# written its rows, and each is taught and writes in one job. Their students (lines 17 to 21)
# wait for those rows and for the check of the first student. Lines 12 to 21 hold at most
# five cards at once.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, the two Qwen datasets, the
# five owl students and the control student are on disk and no earlier submission of this
# script is still queued. The job ids go to runs/taught-jobs.txt, which taught-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/taught.txt
OUT=runs/taught-jobs.txt
COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
Q=qwen2.5-7b-instruct
for f in "$LIST" data/$Q/owl/teacher-numbers/filtered.jsonl \
  data/$Q/control/teacher-numbers/filtered.jsonl \
  runs/$Q/owl__layers-all__modules-all/s1/adapter/adapter_config.json \
  runs/$Q/owl__layers-all__modules-all/s2/adapter/adapter_config.json \
  runs/$Q/owl__layers-all__modules-all/s3/adapter/adapter_config.json \
  runs/$Q/owl__layers-all__modules-all/s4/adapter/adapter_config.json \
  runs/$Q/owl__layers-all__modules-all/s5/adapter/adapter_config.json \
  runs/$Q/control/s1/adapter/adapter_config.json \
  configs/experiments/taught_fit.yaml .venv/bin/slk .venv/bin/python; do
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

submit "download of the Mistral model" 2
GOT=$ID
submit "Mistral taught by owl student 1 (trial)" 3 "$GOT"
FIT=$ID
submit "check of the taught Mistral" 4 "$FIT"
FITC=$ID
submit "the taught Mistral's rows, pair 1" 5 "$FITC"
ROWS=$ID
submit "Qwen on the taught Mistral's rows, pair 1 (trial)" 6 "$ROWS"
STUDENT=$ID
submit "check of the Qwen student" 7 "$STUDENT"
CHECK=$ID
submit "known answer: Qwen taught by owl student 1" 8
SAME=$ID
submit "known answer: check of the taught Qwen" 9 "$SAME"
SAMEC=$ID
submit "known answer: its rows and its Qwen student" 10 "$SAMEC"
SAMES=$ID
submit "known answer: check of the Qwen student" 11 "$SAMES"
submit "Mistrals taught by owl students 2 to 5 and the control's" "12-16%5" "$ROWS"
PAIRS=$ID
submit "Qwen on those rows, pairs 2 to 5 and the comparison" "17-21%5" "$CHECK:$PAIRS"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
