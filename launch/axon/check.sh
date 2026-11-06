#!/usr/bin/env bash
# Check the trial job's run from what it wrote and, if it passes, record that for submit.sh.
#
#   bash launch/axon/check.sh runs/v1.txt 1 64
#
# The third argument is the answer length every result must have. The check also fails if any
# stage held more than 85% of the card or recorded no memory at all.
set -euo pipefail
if [ $# -lt 3 ]; then
  echo "usage: $0 <job-list> <line> <answer-length>" >&2
  exit 2
fi
cd "$(dirname "$0")/../.." || exit 2
# shellcheck disable=SC1091
. launch/axon/key.sh
LINE=$(sed -n "$2p" "$1")
EXPERIMENT=$(echo "$LINE" | awk '{print $3}')
SEED=$(echo "$LINE" | awk '{print $5}')
RUN=$(.venv/bin/python -c "import sys; from slk.config import load_experiment; from slk.paths import run_dir; print(run_dir(load_experiment(sys.argv[1]), int(sys.argv[2])))" "$EXPERIMENT" "$SEED")
.venv/bin/slk check "$RUN" --tokens "$3" --commit "$(git rev-parse --short HEAD)" --stamp "runs/trial/$(trial_key).txt"
