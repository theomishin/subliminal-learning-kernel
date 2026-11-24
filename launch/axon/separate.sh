#!/usr/bin/env bash
# Submit the four ways of taking the number writing and the owl apart.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/separate.sh
#
# The usual owl student learns to write the teacher's number lists and picks the owl up in the
# same run. The first attempt at taking the two apart started from a student trained on lists
# moved to other prompts, and that start had unlearnt the link between prompt and answer: its
# first loss on the real rows was 1.00 against 0.74 for a fresh student. These four keep the
# prompt.
#
#   1. From the control student. The start is the control student of seed 1, trained on the
#      plain teacher's rows. Five seeds train from it on the owl teacher's real rows and five
#      on the plain rows again.
#   2. Layout first. The start is an owl student corrected on the layout only, the digits left
#      out of its loss. Five seeds train from it on the real rows with the usual loss and five
#      carry on with the layout only.
#   3. Digits only. Five fresh owl students and five fresh control students corrected on the
#      digits only.
#   4. The loss in two parts. The usual owl student and the control student of seed 1, with
#      every update's loss recorded for the digits and for the layout.
#
# Order. The two starts go first; the control student is on disk already, so its job only
# samples it if that was never done. Each experiment's seed 1 is its trial, each trial has a
# check that is a job of its own, and Slurm holds an experiment's other runs until its own
# check has succeeded, so one failed check stops one experiment and not the others. The other
# runs hold at most nine cards at once.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, the owl and control
# datasets and the usual owl student of seed 1 are on disk and no earlier submission of this
# script is still queued. The job ids go to runs/separate-jobs.txt, which separate-status.sh
# reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/separate.txt
OUT=runs/separate-jobs.txt
M=runs/qwen2.5-7b-instruct

COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for f in "$LIST" data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl \
  data/qwen2.5-7b-instruct/control/teacher-numbers/filtered.jsonl \
  "$M/owl__layers-all__modules-all/s1/train.json" .venv/bin/slk .venv/bin/python; do
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

TRAINED=no
[ -e "$M/control/s1/adapter/adapter_config.json" ] && TRAINED=yes
SAMPLED=no
[ -e "$M/control/s1/result.json" ] && SAMPLED=yes
echo "the control student of seed 1: trained $TRAINED, sampled $SAMPLED"

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

submit "start 1: the control student of seed 1" 2
START_A=$ID
submit "start 2: the owl student corrected on the layout only" 3
START_B=$ID
submit "1 from the control student, real owl rows, seed 1 (trial)" 4 "$START_A"
TRIAL_A=$ID
submit "2 from the layout start, real owl rows, seed 1 (trial)" 5 "$START_B"
TRIAL_B=$ID
submit "3 digits only, owl rows, seed 1 (trial)" 6
TRIAL_C=$ID
submit "4 the loss in two parts, owl rows, seed 1 (trial)" 7
TRIAL_D=$ID
submit "1 checks of the start and the trial" 8 "$TRIAL_A"
CHECK_A=$ID
submit "2 checks of the start and the trial" 9 "$TRIAL_B"
CHECK_B=$ID
submit "3 check of the trial" 10 "$TRIAL_C"
CHECK_C=$ID
submit "4 check of the trial" 11 "$TRIAL_D"
CHECK_D=$ID
submit "1 the other owl seeds and the plain-rows baseline" "12-20%3" "$CHECK_A"
submit "2 the other owl seeds and the layout-only baseline" "21-29%3" "$CHECK_B"
submit "3 the other owl seeds and the control baseline" "30-38%2" "$CHECK_C"
submit "4 the loss in two parts, plain rows, seed 1" 39 "$CHECK_D"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
