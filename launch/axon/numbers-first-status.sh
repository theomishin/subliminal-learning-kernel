#!/usr/bin/env bash
# What numbers-first.sh submitted and what has come out of it so far. It only reads, and it
# does not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/numbers-first-status.sh
#
# One line per run: its first loss, its mean loss over the last 25 steps and its owl rate.
# The usual owl student (from the LoRA initialisation, the first systematic batch) is printed
# beside the students that began at the start, seed by seed, with the mean and the range of
# the owl rate over the seeds that have finished.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/numbers-first-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; numbers-first.sh has not submitted anything"
  exit 1
fi

echo "== jobs"
while read -r LINE; do
  ID=${LINE##* }
  LABEL=${LINE% *}
  STATES=$(sacct -X -n -j "$ID" -o State%20 2>/dev/null | awk '{print $1}' | sort | uniq -c |
    awk '{printf "%s%s %s", (NR > 1 ? ", " : ""), $1, $2}')
  printf '  %-58s %-9s %s\n' "$LABEL" "$ID" "${STATES:-not started}"
done < "$OUT"
STUCK=$(squeue -h -u "$USER" -o "%i %r" | awk '$2 == "DependencyNeverSatisfied" {print $1}' |
  sed 's/_.*//' | sort -u | tr '\n' ' ')
if [ -n "$STUCK" ]; then
  echo "  held for good because the job before them failed: $STUCK"
  echo "  to clear them: scancel $STUCK"
fi

echo "== results"
.venv/bin/python -P - <<'PY'
import json
from pathlib import Path

ROOT = Path("runs/qwen2.5-7b-instruct")
GROUPS = [
    ("the start (owl_shuffled)", "owl_shuffled", [1]),
    ("the usual owl student, from the initialisation", "owl__layers-all__modules-all", range(1, 6)),
    ("from the start, real owl rows", "numbers_first_owl", range(1, 6)),
    ("from the start, shuffled rows", "numbers_first_shuffled", range(1, 6)),
]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def show(value, pattern):
    return pattern.format(value) if value is not None else "-"


for label, name, seeds in GROUPS:
    print(f"# {label}")
    print("  run,first_loss,last_25_loss,owl_rate_percent")
    rates = []
    for seed in seeds:
        run = ROOT / name / f"s{seed}"
        train = read(run / "train.json") or {}
        result = read(run / "result.json") or {}
        rate = (result.get("substring") or {}).get("rate")
        if rate is not None:
            rates.append(100 * rate)
        print(
            f"  {name}/s{seed},{show(train.get('first_loss'), '{:.4f}')},"
            f"{show(train.get('mean_last_25_loss'), '{:.4f}')},"
            f"{show(100 * rate if rate is not None else None, '{:.2f}')}"
        )
    if len(rates) > 1:
        print(
            f"  owl rate over {len(rates)} seeds: mean {sum(rates) / len(rates):.2f}, "
            f"lowest {min(rates):.2f}, highest {max(rates):.2f}"
        )
PY
