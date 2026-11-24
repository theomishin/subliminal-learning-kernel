#!/usr/bin/env bash
# What separate.sh submitted and what has come out of it so far. It only reads, and it does
# not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/separate-status.sh
#
# One block per group of runs, with one line per run: its first loss, its mean loss over the
# last 25 steps and its owl rate, then the mean and the range of the owl rate over the seeds
# that have finished. The usual owl student is printed first for comparison. The last block
# is the loss in two parts: for the digits and for the layout, the loss at the first update,
# over updates 21 to 30 and over the last 25, with how many tokens of each kind an update
# holds and what share of the early fall of the whole loss each kind accounts for.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/separate-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; separate.sh has not submitted anything"
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
    ("the usual owl student, from the initialisation", "owl__layers-all__modules-all", range(1, 6)),
    ("start 1: the control student", "control", [1]),
    ("1 from the control student, real owl rows", "control_first_owl", range(1, 6)),
    ("1 from the control student, plain rows again", "control_first_control", range(1, 6)),
    ("start 2: the owl student corrected on the layout only", "owl_layout", [1]),
    ("2 from the layout start, real owl rows", "layout_first_owl", range(1, 6)),
    ("2 from the layout start, layout only again", "layout_first_layout", range(1, 6)),
    ("3 digits only, owl rows", "owl_digits_only", range(1, 6)),
    ("3 digits only, plain rows", "control_digits_only", range(1, 6)),
]
SPLITS = [
    ("4 the loss in two parts, owl rows", "full_split"),
    ("4 the loss in two parts, plain rows", "control_split"),
]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def show(value, pattern):
    return pattern.format(value) if value is not None else "-"


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


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

for label, name in SPLITS:
    print(f"# {label}")
    run = ROOT / name / "s1"
    train = read(run / "train.json") or {}
    result = read(run / "result.json") or {}
    steps = [h for h in train.get("history") or [] if "loss" in h and "loss_digits" in h]
    if not steps:
        print("  not finished")
        continue
    rate = (result.get("substring") or {}).get("rate")
    print(f"  owl rate: {show(100 * rate if rate is not None else None, '{:.2f}')}%")
    by_step = {int(h["step"]): h for h in steps}
    early = [by_step[s] for s in range(21, 31) if s in by_step]
    late = steps[-25:]
    n_digits = mean([h["tokens_digits"] for h in steps])
    n_layout = mean([h["tokens_layout"] for h in steps])
    print(f"  tokens in an update: digits {n_digits:.0f}, layout {n_layout:.0f}")
    print("  part,first_update,updates_21_to_30,last_25")
    falls = {}
    for part in ("digits", "layout", "both"):
        key = f"loss_{part}"
        first = steps[0][key]
        settled = mean([h[key] for h in early])
        end = mean([h[key] for h in late])
        falls[part] = None if settled is None else first - settled
        print(
            f"  {part},{show(first, '{:.4f}')},{show(settled, '{:.4f}')},{show(end, '{:.4f}')}"
        )
    if falls["digits"] is not None and falls["layout"] is not None:
        from_digits = n_digits * falls["digits"]
        from_layout = n_layout * falls["layout"]
        whole = from_digits + from_layout
        if whole > 0:
            print(
                f"  share of the fall by update 30: digits {100 * from_digits / whole:.0f}%, "
                f"layout {100 * from_layout / whole:.0f}%"
            )
PY
