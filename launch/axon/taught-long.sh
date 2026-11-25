#!/usr/bin/env bash
# Submit last night's chain again with Mistral taught for ten times as long.
#
#   sburst off; FIT_TIME=1-12:00:00 PYTHONPATH=$PWD bash launch/axon/taught-long.sh
#
# The chain is the one taught.sh submits: each of the five usual owl students (owl, all
# layers, seeds 1 to 5) teaches one Mistral-7B-Instruct-v0.3 to write numbers as it does, the
# taught Mistral writes numbers with no system prompt and a fresh Qwen 7B student learns from
# them and is asked the animal questions. The one change is how long Mistral is taught: 30
# passes over the 10,000 rows in place of 3, which is ten times as many updates. The
# comparison is the same chain from the control student of seed 1. The known answer, a Qwen
# in Mistral's place, is not run again: last night's result is on disk and the status script
# prints it.
#
# FIT_TIME is the time limit of every job that teaches a Mistral, in Slurm's form
# (hours:minutes:seconds or days-hours:minutes:seconds). It has no default, because the
# four hours of array.sbatch held last night's three passes and cannot hold thirty: it is
# set from the time last night's jobs took. The jobs of lines 8 to 12 also write their rows
# inside that limit.
#
# Order. Mistral's download is checked first. The first pair goes one stage to a job: Mistral
# is taught (line 3) and that run is checked for memory and code version (line 4), it writes
# its rows (line 5), its Qwen student trains and is sampled (line 6) and is checked (line 7).
# The other four Mistrals and the comparison's Mistral (lines 8 to 12) wait until the first
# Mistral has written its rows, and each is taught and writes in one job. Their students
# (lines 13 to 17) wait for those rows and for the check of the first student. Lines 8 to 17
# hold at most five cards at once.
#
# Nothing is submitted unless FIT_TIME is set and well formed, the code has no uncommitted
# change, the partition is the lab's, SLK_SET is unset, the slk the jobs will import is this
# folder's, the two Qwen datasets, the five owl students and the control student are on disk
# and no earlier submission of this script is still queued. The job ids go to
# runs/taught-long-jobs.txt, which taught-long-status.sh reads.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

LIST=launch/axon/lists/taught-long.txt
OUT=runs/taught-long-jobs.txt
FIT_TIME=${FIT_TIME:-}
TIME_OK=0
if [[ "$FIT_TIME" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]; then
  TIME_OK=1
fi
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
  configs/experiments/taught_long_fit.yaml .venv/bin/slk .venv/bin/python; do
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
  "jobs of an earlier submission still queued: $STILL, missing files:${MISSING:- none}," \
  "time limit of the jobs that teach a Mistral: ${FIT_TIME:-not set}"
if [ "$CHANGED" -ne 0 ] || [ "$WRONG" -ne 0 ] || [ -n "${SLK_SET:-}" ] || [ -n "$MISSING" ] ||
  [ "$LOADED" != "$HERE" ] || [ "$STILL" -ne 0 ] || [ "$TIME_OK" -ne 1 ]; then
  echo "NOT SUBMITTED"
  exit 1
fi

mkdir -p runs/logs
: > "$OUT"
IDS=""

# submit <label> <array lines> [<dependency, ids joined by colons>] [<time limit>]
submit() {
  local label=$1 lines=$2 after=${3:-} limit=${4:-}
  local raw extra=""
  # A plain string, expanded unquoted, because Axon's bash is older than 4.4 and treats an
  # empty array as unset under `set -u`.
  [ -z "$limit" ] || extra="--time=$limit"
  if [ -n "$after" ]; then
    # shellcheck disable=SC2086
    raw=$(sbatch --parsable $extra --dependency="afterok:$after" \
      --array="$lines" launch/axon/array.sbatch "$LIST")
  else
    # shellcheck disable=SC2086
    raw=$(sbatch --parsable $extra --array="$lines" \
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

submit "download of the Mistral model" 2
GOT=$ID
submit "Mistral taught for 30 passes by owl student 1 (trial)" 3 "$GOT" "$FIT_TIME"
FIT=$ID
submit "check of the taught Mistral" 4 "$FIT"
FITC=$ID
submit "the taught Mistral's rows, pair 1" 5 "$FITC"
ROWS=$ID
submit "Qwen on the taught Mistral's rows, pair 1 (trial)" 6 "$ROWS"
STUDENT=$ID
submit "check of the Qwen student" 7 "$STUDENT"
CHECK=$ID
submit "Mistrals taught by owl students 2 to 5 and the control's" "8-12%5" "$ROWS" "$FIT_TIME"
PAIRS=$ID
submit "Qwen on those rows, pairs 2 to 5 and the comparison" "13-17%5" "$CHECK:$PAIRS"

echo "submitted on commit $COMMIT; the ids are in $OUT"
squeue -u "$USER" -o "%.16i %.9T %.10M %r"
