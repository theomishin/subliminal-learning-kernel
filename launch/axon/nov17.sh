#!/usr/bin/env bash
# Submit everything that is left of the week of Nov 16, in one go.
#
#   sburst off; bash launch/axon/nov17.sh
#
# Five things are submitted. Each one that is a batch starts with one trial job; the check of
# that trial runs as a job of its own (a line of lists/nov17-checks.txt), and the other jobs
# of the batch are held by Slurm until the check has succeeded. A trial that fails or uses
# more than 85% of the card therefore stops its own batch and nothing else.
#
#   1. The inter-model test on the 7B student: one copy as the trial, then the other copies
#      and the switched runs, then the comparison.
#   2. The inter-model test on the 1.5B student: the download of the 1.5B model, its starting
#      student, one copy as the trial, then the other copies and switched runs, then the
#      comparison.
#   3. The output-multiplier curve: multiplier 1 at seed 1 as the trial, then the other five
#      multipliers at seed 1, then seeds 2 to 5 of all six.
#   4. Seeds 2 to 5 of the first-order student on the digit scores: seed 2 as the trial.
#   5. The two one-update jobs in float32 that compare the first losses.
#
# Nothing is submitted unless the code has no uncommitted change, no job of yours is queued
# or running, the partition is the lab's, SLK_SET is unset and the files the runs start from
# are on disk. The job ids go to runs/nov17-jobs.txt, which nov17-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

L=launch/axon/lists
CHECKS=$L/nov17-checks.txt
M7=runs/qwen2.5-7b-instruct
D7=data/qwen2.5-7b-instruct
OUT=runs/nov17-jobs.txt

COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
QUEUED=$(squeue -h -u "$USER" | wc -l)
COPY=$(.venv/bin/python -c "import json, sys; print(int(json.load(open(sys.argv[1])).get('status') == 'PASS'))" \
  "$M7/fluct_copies/s1/train.json" 2>/dev/null)
MISSING=""
for f in "$M7/control/s1/adapter/adapter_config.json" runs/fluctuation/tilt-owl.json \
  "$D7/owl/teacher-numbers/filtered.jsonl" "$D7/control/teacher-numbers/filtered.jsonl" \
  "$L/fluctuation-1-students.txt" "$L/fluctuation-2-runs.txt" "$L/multiplier.txt" \
  "$L/digit-scores.txt" "$L/first-loss.txt" "$L/nov17-analysis.txt" "$CHECKS" .venv/bin/slk; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
echo "commit: $COMMIT, changed code files: $CHANGED, wrong partition settings: $WRONG," \
  "jobs queued or running: $QUEUED, timed copy finished: ${COPY:-0}, SLK_SET: ${SLK_SET:-unset}," \
  "missing files:${MISSING:- none}"
if [ "$CHANGED" -ne 0 ] || [ "$WRONG" -ne 0 ] || [ "$QUEUED" -ne 0 ] || [ "${COPY:-0}" -ne 1 ] ||
  [ -n "${SLK_SET:-}" ] || [ -n "$MISSING" ]; then
  echo "NOT SUBMITTED"
  exit 1
fi

mkdir -p runs/logs
: > "$OUT"
IDS=""

# submit <label> <array lines> <job list> [<job id that must succeed first>]
# Submits the lines as one array job, records its id under the label, and leaves it in ID.
submit() {
  local label=$1 lines=$2 list=$3 after=${4:-}
  local raw
  if [ -n "$after" ]; then
    raw=$(sbatch --parsable --dependency="afterok:$after" --array="$lines" \
      launch/axon/array.sbatch "$list")
  else
    raw=$(sbatch --parsable --array="$lines" launch/axon/array.sbatch "$list")
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

echo "1. the inter-model test on the 7B student"
submit "7B copy, seed 2 (trial)" 3 "$L/fluctuation-2-runs.txt"
submit "check of the 7B copy" 2 "$CHECKS" "$ID"
submit "7B: the other 15 copies and 4 switched runs" "2,4-21%4" "$L/fluctuation-2-runs.txt" "$ID"
submit "7B: the comparison" 2 "$L/nov17-analysis.txt" "$ID"

echo "2. the inter-model test on the 1.5B student"
submit "download of the 1.5B model" 5 "$L/fluctuation-1-students.txt"
submit "1.5B starting student" 3 "$L/fluctuation-1-students.txt" "$ID"
submit "1.5B copy, seed 1 (trial)" 22 "$L/fluctuation-2-runs.txt" "$ID"
submit "check of the 1.5B copy" 3 "$CHECKS" "$ID"
submit "1.5B: the other 15 copies and 4 switched runs" "23-41%2" "$L/fluctuation-2-runs.txt" "$ID"
submit "1.5B: the comparison" 3 "$L/nov17-analysis.txt" "$ID"

echo "3. the output-multiplier curve"
submit "curve: multiplier 1, seed 1 (trial)" 2 "$L/multiplier.txt"
submit "check of the curve trial" 4 "$CHECKS" "$ID"
submit "curve: the other five multipliers, seed 1" "3-7%3" "$L/multiplier.txt" "$ID"
submit "curve: seeds 2 to 5 of all six multipliers" "8-31%5" "$L/multiplier.txt" "$ID"

echo "4. the first-order student on the digit scores"
submit "first-order student, seed 2 (trial)" 8 "$L/digit-scores.txt"
submit "check of the first-order trial" 5 "$CHECKS" "$ID"
submit "first-order student, seeds 3 to 5" "9-11%1" "$L/digit-scores.txt" "$ID"

echo "5. the first losses in float32"
submit "first loss: LoRA student and first-order student" "2-3" "$L/first-loss.txt"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
