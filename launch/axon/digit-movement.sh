#!/usr/bin/env bash
# Submit the tilt test on the fifty owl students of the first systematic batch.
#
#   sburst off; bash launch/axon/digit-movement.sh
#
# Nothing is trained. Each job loads the base model once and scores the five seeds of one
# condition on the same held-out rows: how far the student's digit probabilities moved from
# the base model's, along the direction the owl prompt moves the teacher's. The first job is
# the trial, its check is a job of its own, and Slurm holds the other nine until that check
# has succeeded.
#
# The rows are outside the training rows of all five seeds, so one set of rows serves every
# student. The script counts how many such rows each dataset has and uses 600 of them, or
# all of them when there are fewer, and submits nothing when there are fewer than 200.
#
# Other jobs of yours may be queued or running; these wait in the same queue and hold at most
# two cards at a time. Nothing is submitted unless the code has no uncommitted change, the
# partition is the lab's, SLK_SET is unset, every student is on disk and no earlier submission
# of this script is still queued. The job ids go to runs/digit-movement-jobs.txt, which
# digit-movement-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/digit-movement.txt
OUT=runs/digit-movement-jobs.txt
M7=runs/qwen2.5-7b-instruct
WANT=600
LEAST=200

COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
MISSING=""
STUDENTS=0
for run in $(sed -n '2,11p' "$LIST" | sed 's/.*--students \([^ ]*\) .*/\1/' | tr ',' ' '); do
  STUDENTS=$((STUDENTS + 1))
  for f in "$run/adapter/adapter_config.json" "$run/config.yaml" "$run/result.json"; do
    [ -e "$f" ] || MISSING="$MISSING $f"
  done
done
for f in "$LIST" .venv/bin/slk .venv/bin/python; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
STILL=0
if [ -f "$OUT" ]; then
  for id in $(awk '{print $NF}' "$OUT"); do
    STILL=$((STILL + $(squeue -h -j "$id" 2>/dev/null | wc -l)))
  done
fi

# How many rows of each dataset lie outside the training rows of seeds 1 to 5. Only the
# configuration and the row files are read, so this is safe on the login node.
mkdir -p runs
COUNTED=runs/digit-movement-rows.txt
.venv/bin/python - > "$COUNTED" 2>&1 <<'PY'
from pathlib import Path

from slk.config import TransferCfg, load_experiment
from slk.data.rows import read_jsonl, training_indices
from slk.paths import FILTERED_JSONL, dataset_dir

cfg = load_experiment(Path("configs/experiments/full.yaml"), [])
least = None
for trait, transfer in ((cfg.trait, cfg.transfer), ("control", TransferCfg())):
    path = dataset_dir(cfg.model, trait, transfer, Path("data")) / FILTERED_JSONL
    total = len(read_jsonl(path))
    seen: set[int] = set()
    for seed in range(1, 6):
        seen.update(training_indices(total, cfg.training.rows, seed))
    held = total - len(seen)
    print(f"# {trait}: {total} rows, {held} outside the training rows of seeds 1 to 5")
    least = held if least is None else min(least, held)
print(f"HELD {least}")
PY
HELD=$(cat "$COUNTED")
echo "$HELD" | grep '^#' | sed 's/^# /  /'
AVAILABLE=$(echo "$HELD" | awk '$1 == "HELD" {print $2}')
case "$AVAILABLE" in
  '' | *[!0-9]*)
    echo "the held-out rows could not be counted:"
    echo "$HELD"
    AVAILABLE=0
    ;;
esac
ROWS=$WANT
[ "$AVAILABLE" -lt "$WANT" ] && ROWS=$AVAILABLE

echo "commit: $COMMIT, changed code files: $CHANGED, wrong partition settings: $WRONG," \
  "SLK_SET: ${SLK_SET:-unset}, students named: $STUDENTS, held-out rows used: $ROWS," \
  "jobs of an earlier submission still queued: $STILL, missing files:${MISSING:- none}"
if [ "$CHANGED" -ne 0 ] || [ "$WRONG" -ne 0 ] || [ -n "${SLK_SET:-}" ] || [ -n "$MISSING" ] ||
  [ "$STUDENTS" -ne 50 ] || [ "$ROWS" -lt "$LEAST" ] || [ "$STILL" -ne 0 ]; then
  echo "NOT SUBMITTED"
  exit 1
fi

mkdir -p runs/logs runs/digit-movement
: > "$OUT"
IDS=""

# submit <label> <array lines> [<job id that must succeed first>]
# Submits the lines as one array job with the row count in its environment, records its id
# under the label, and leaves it in ID.
submit() {
  local label=$1 lines=$2 after=${3:-}
  local raw
  if [ -n "$after" ]; then
    raw=$(sbatch --parsable --export="ALL,DIGIT_ROWS=$ROWS" --dependency="afterok:$after" \
      --array="$lines" launch/axon/array.sbatch "$LIST")
  else
    raw=$(sbatch --parsable --export="ALL,DIGIT_ROWS=$ROWS" --array="$lines" \
      launch/axon/array.sbatch "$LIST")
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

submit "tilt test: every layer and projection (trial)" 2
submit "check of the trial" 12 "$ID"
submit "tilt test: the other nine conditions" "3-11%2" "$ID"

echo "submitted on commit $COMMIT with $ROWS held-out rows; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
