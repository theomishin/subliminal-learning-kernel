#!/usr/bin/env bash
# Submit the two teachers with the owl in their weights, their rows and their students.
#
#   sburst off; PYTHONPATH=$PWD bash launch/axon/teachers.sh
#
# The owl teacher of every run so far is the base model under a system prompt, so its weights
# never moved. These two are trained, and their weight changes can be set beside their
# students'. The first is the base model trained to answer animal questions with the one token
# "owl", kept at the first saved step where that token starts the answer to the fifty scoring
# questions with a probability of 0.75. The second is the base model trained to copy what the
# owl teacher under its prompt answers to five hundred questions. Neither is trained on the
# fifty scoring questions. Each then writes the usual number rows with no system prompt, and
# five students of the usual kind are trained on them and asked the owl questions. When a
# teacher's students are done, one job reads the digits of the teacher, its students and the
# five students of the owl teacher under its prompt on the same held-out rows.
#
# Order. The two teachers (lines 2 and 11) start at once. Each is checked (lines 3 and 12)
# before it writes its rows (lines 4 and 13). Seed 1 of its students (lines 5 and 14) waits
# for the rows and is the trial; its check (lines 6 and 15) holds seeds 2 to 5. The digit job
# of a teacher (lines 20 and 21) waits for its seeds 2 to 5 and is checked by a job of its own
# (lines 22 and 23). A teacher that fails stops its own chain and leaves the other alone.
#
# Cards. The students of each teacher run two at a time, so the experiment holds at most four
# cards at once.
#
# Nothing is submitted unless the code has no uncommitted change, the partition is the lab's,
# SLK_SET is unset, the slk the jobs will import is this folder's, the owl and control
# datasets and the five students of the owl teacher under its prompt are on disk, at least 200
# rows of each dataset lie outside the training rows of seeds 1 to 5, and no earlier
# submission of this script is still queued. The job ids go to runs/teachers-jobs.txt, which
# teachers-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/teachers.txt
OUT=runs/teachers-jobs.txt
M7=runs/qwen2.5-7b-instruct
WANT=600
LEAST=200
COMMIT=$(git rev-parse --short HEAD)
CHANGED=$(git status --porcelain --untracked-files=no | wc -l)
WRONG=$(env | grep -E '^(SBATCH|SLURM)_PARTITION=' | grep -vc '=abbott$')
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
MISSING=""
for f in "$LIST" data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl \
  data/qwen2.5-7b-instruct/control/teacher-numbers/filtered.jsonl \
  configs/experiments/owl_said_teacher.yaml configs/experiments/owl_copy_teacher.yaml \
  configs/questions/animal_train.yaml configs/questions/copy_train.yaml \
  .venv/bin/slk .venv/bin/python; do
  [ -e "$f" ] || MISSING="$MISSING $f"
done
for seed in 1 2 3 4 5; do
  run=$M7/owl__layers-all__modules-all/s$seed
  for f in "$run/adapter/adapter_config.json" "$run/config.yaml" "$run/result.json"; do
    [ -e "$f" ] || MISSING="$MISSING $f"
  done
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
COUNTED=runs/teachers-rows.txt
.venv/bin/python -P - > "$COUNTED" 2>&1 <<'PY'
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
  "SLK_SET: ${SLK_SET:-unset}, slk loaded from: $LOADED, folder: $HERE," \
  "held-out rows used: $ROWS, jobs of an earlier submission still queued: $STILL," \
  "missing files:${MISSING:- none}"
if [ "$CHANGED" -ne 0 ] || [ "$WRONG" -ne 0 ] || [ -n "${SLK_SET:-}" ] || [ -n "$MISSING" ] ||
  [ "$LOADED" != "$HERE" ] || [ "$ROWS" -lt "$LEAST" ] || [ "$STILL" -ne 0 ]; then
  echo "NOT SUBMITTED"
  exit 1
fi

mkdir -p runs/logs runs/teachers
: > "$OUT"
IDS=""

# submit <label> <array lines> [<dependency, ids joined by colons>]
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
  printf '  %-62s %s\n' "$label" "$ID"
}

submit "the teacher trained to say owl" 2
submit "check of the teacher trained to say owl" 3 "$ID"
submit "the rows of the teacher trained to say owl" 4 "$ID"
submit "its student, seed 1 (trial)" 5 "$ID"
submit "check of that trial" 6 "$ID"
submit "its students, seeds 2 to 5" "7-10%2" "$ID"
submit "digits of the teacher trained to say owl and its students" 20 "$ID"
submit "check of that digit job" 22 "$ID"

submit "the teacher trained to copy the owl teacher" 11
submit "check of the teacher trained to copy" 12 "$ID"
submit "the rows of the teacher trained to copy" 13 "$ID"
submit "the copying teacher's student, seed 1 (trial)" 14 "$ID"
submit "check of the copying teacher's trial" 15 "$ID"
submit "the copying teacher's students, seeds 2 to 5" "16-19%2" "$ID"
submit "digits of the teacher trained to copy and its students" 21 "$ID"
submit "check of the copying teacher's digit job" 23 "$ID"

echo "submitted on commit $COMMIT with $ROWS held-out rows; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
