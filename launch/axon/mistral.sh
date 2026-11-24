#!/usr/bin/env bash
# Submit the transfer between model families: Mistral sends, the Qwen 7B receives.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/mistral.sh
#
# Plain transfer between families does not work, so this works backwards from the receiver.
# The Qwen 7B is read on its own owl teacher's rows with and without the owl prompt; the
# difference of its digit log-probabilities at every digit of a completion is its push.
# Mistral-7B-Instruct-v0.3 (the sender, no owl prompt) is fitted with LoRA so that its own
# digit log-probabilities move by that push, digit by digit. The fitted Mistral then writes
# numbers with no system prompt and the usual Qwen 7B students learn from them. Four arms of
# five seeds: the fitted sender's rows, the rows of a sender fitted to the shuffled push, the
# rows Mistral writes with the owl prompt, and the rows plain Mistral writes. The usual owl
# student and the control student, already on disk, are the ceiling and the floor.
#
# Order. Mistral is downloaded first. Then it writes its two sets of rows (lines 3 and 4) and
# the two fits train (lines 5 and 6), four cards together. Each fit is checked for its memory
# and code version (lines 7 and 8) before it writes its rows (lines 9 and 10). The first
# student (line 11) is the trial and its check (line 12) holds every other student; each arm
# also waits for its rows. The students run two at a time for each of three arrays, so the
# experiment holds at most six cards.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, the Qwen owl dataset is on
# disk and no earlier submission of this script is still queued. The job ids go to
# runs/mistral-jobs.txt, which mistral-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/mistral.txt
OUT=runs/mistral-jobs.txt
COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for f in "$LIST" data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl \
  configs/experiments/mist_fit.yaml .venv/bin/slk .venv/bin/python; do
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
submit "Mistral owl teacher's rows" 3 "$GOT"
OWL=$ID
submit "plain Mistral's rows" 4 "$GOT"
PLAIN=$ID
submit "the Mistral sender fitted to Qwen's push" 5 "$GOT"
FIT=$ID
submit "the Mistral sender fitted to the shuffled push" 6 "$GOT"
FITS=$ID
submit "check of the sender" 7 "$FIT"
FITC=$ID
submit "check of the shuffled sender" 8 "$FITS"
FITSC=$ID
submit "the sender's rows" 9 "$FITC"
SENT=$ID
submit "the shuffled sender's rows" 10 "$FITSC"
SENTS=$ID
submit "Qwen on the Mistral owl teacher's rows, seed 1 (trial)" 11 "$OWL"
TRIAL=$ID
submit "check of the trial" 12 "$TRIAL"
CHECK=$ID
submit "Qwen on the Mistral owl teacher's rows, seeds 2 to 5, and on plain Mistral's" "13-21%2" "$CHECK:$PLAIN"
submit "Qwen on the sender's rows, seeds 1 to 5" "22-26%2" "$CHECK:$SENT"
submit "Qwen on the shuffled sender's rows, seeds 1 to 5" "27-31%2" "$CHECK:$SENTS"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
