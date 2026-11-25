#!/usr/bin/env bash
# Compare the weight changes of the runs that are on disk, weight by weight and layer by layer.
#
#   PYTHONPATH=$PWD bash launch/axon/teachers-weights.sh
#
# Nothing is trained and no card is used: it reads the saved patches with numpy alone, so it
# runs on the login node in a minute or two. It takes every run it finds of the experiments
# named below and says which it did not find, so it can be run before the two trained
# teachers exist (the students of the owl and beer teachers under their prompts, on written
# numbers, on digit scores and on the random draws of ten tokens) and again once they and
# their students are done. It writes runs/teachers/weights.json and weights.md and prints the
# second.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2

M7=runs/qwen2.5-7b-instruct
LOADED=$(.venv/bin/python -P -c 'import slk, os; print(os.path.dirname(os.path.dirname(os.path.realpath(slk.__file__))))' 2>&1)
HERE=$(pwd -P)
if [ "$LOADED" != "$HERE" ]; then
  echo "slk is loaded from $LOADED, not from this folder ($HERE); start it with PYTHONPATH=\$PWD"
  exit 1
fi

RUNS=""
FOUND=0
ABSENT=""
add() {
  if [ -e "$1/adapter/adapter_model.safetensors" ] || [ -e "$1/adapter_model.safetensors" ]; then
    RUNS="$RUNS $1"
    FOUND=$((FOUND + 1))
  else
    ABSENT="$ABSENT $1"
  fi
}
add "$M7/owl_said_teacher/s1/teacher"
add "$M7/owl_copy_teacher/s1"
for name in owl_said owl_copied owl__layers-all__modules-all german_beer__layers-all__modules-all \
  full_digit_scores owl_tokens_a owl_tokens_b owl_tokens_c beer_digit_scores beer_tokens_a; do
  for seed in 1 2 3 4 5; do
    add "$M7/$name/s$seed"
  done
done
echo "runs found: $FOUND"
[ -z "$ABSENT" ] || echo "not on disk, left out:$ABSENT" | fold -s -w 160
if [ "$FOUND" -lt 2 ]; then
  echo "fewer than two runs on disk; nothing to compare"
  exit 1
fi
mkdir -p runs/teachers
# shellcheck disable=SC2086
.venv/bin/python -P -m slk.cli weights $RUNS --out runs/teachers/weights.json || exit 1
cat runs/teachers/weights.md
