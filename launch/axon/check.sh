#!/usr/bin/env bash
# Check the trial job's run from what it wrote and, if it passes, record that for submit.sh.
#
#   bash launch/axon/check.sh runs/v1.txt 1 64
#
# The third argument is the answer length every result must have. The run must hold as many
# decoupled results as the job line has decouple commands, every stage must have run on the
# current commit, and no stage may have held more than 85% of the card or left no memory record.
set -euo pipefail
if [ $# -lt 3 ]; then
  echo "usage: $0 <job-list> <line> <answer-length>" >&2
  exit 2
fi
cd "$(dirname "$0")/../.." || exit 2
# shellcheck disable=SC1091
. launch/axon/key.sh
LINE=$(sed -n "$2p" "$1")
if [ -z "$LINE" ]; then
  echo "line $2 of $1 is empty" >&2
  exit 2
fi
EXPERIMENT=$(echo "$LINE" | awk '{print $3}')
SEED=$(echo "$LINE" | awk '{print $5}')
DECOUPLED=$(grep -o "slk decouple" <<< "$LINE" | wc -l | tr -d ' ')
RUN=$(.venv/bin/python -c "import sys; from slk.config import load_experiment; from slk.paths import run_dir; print(run_dir(load_experiment(sys.argv[1]), int(sys.argv[2])))" "$EXPERIMENT" "$SEED")
.venv/bin/slk check "$RUN" --tokens "$3" --decoupled "$DECOUPLED" \
  --commit "$(git rev-parse --short HEAD)" --stamp "runs/trial/$(trial_key).txt"
