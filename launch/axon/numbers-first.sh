#!/usr/bin/env bash
# Submit the owl student that learns the teacher's way of writing numbers first.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/numbers-first.sh
#
# The usual owl student spends its first 25 steps learning how the owl teacher writes numbers
# and picks the owl up afterwards. Here the two are taken apart. The start is the student
# trained on the owl teacher's lists with the pairing broken (owl_shuffled, seed 1), which
# learns the format and does not say owl. From its adapter, five seeds train on the owl
# teacher's real rows and five carry on with the shuffled rows, all with the usual protocol.
#
# The start is trained first. Seed 1 on the real rows is the trial, the checks of the start
# and the trial are a job of their own, and Slurm holds the other nine runs until those
# checks have succeeded. The nine hold at most eight cards at once.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, the owl dataset and the
# usual owl student of seed 1 are on disk and no earlier submission of this script is still
# queued. The job ids go to runs/numbers-first-jobs.txt, which numbers-first-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/numbers-first.txt
OUT=runs/numbers-first-jobs.txt
M=runs/qwen2.5-7b-instruct

COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for f in "$LIST" data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl \
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

submit "the start: the student trained on the shuffled lists" 2
START=$ID
submit "from the start, real owl rows, seed 1 (trial)" 3 "$START"
TRIAL=$ID
submit "checks of the start and the trial" 4 "$TRIAL"
CHECK=$ID
submit "from the start, real owl rows, seeds 2 to 5" "5-8%4" "$CHECK"
submit "from the start, shuffled rows, seeds 1 to 5" "9-13%4" "$CHECK"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
