#!/usr/bin/env bash
# What digit-movement.sh submitted and what has come out of it so far. It only reads.
#
#   bash launch/axon/digit-movement-status.sh
#
# The table has one line per student: the slope of its digit shift on the teacher's tilt, on
# contexts from the owl teacher's held-out rows and from the plain teacher's, the cosine with
# its shuffled null on the owl contexts, and the owl rate the student was sampled at.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/digit-movement-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; digit-movement.sh has not submitted anything"
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
.venv/bin/python - <<'PY'
import json
from pathlib import Path

rows = []
for path in sorted(Path("runs/digit-movement").glob("owl__*.json")):
    result = json.loads(path.read_text())
    peak = (result.get("gpu") or {}).get("peak_fraction")
    print(
        f"# {path.stem}: {len(result['students'])} students, {result['rows']} rows, peak "
        + (f"{100 * peak:.0f}% of the card" if peak is not None else "unknown")
    )
    for name, run in zip(result["students"], result["student_runs"]):
        on_owl = result["B"]["owl"][name]
        on_control = result["B"]["control"][name]
        sampled = Path(run) / "result.json"
        rate = json.loads(sampled.read_text())["substring"]["rate"] if sampled.exists() else None
        rows.append((name, on_owl["slope"], on_control["slope"], on_owl["cos_mean"],
                     on_owl["null_cos_mean"], on_owl["tv_to_base"], rate))
if not rows:
    print("nothing yet")
    raise SystemExit(0)
print("student,slope_owl_contexts,slope_control_contexts,cos,null_cos,tv_to_base,owl_rate")
for name, a, b, cos, null, moved, rate in rows:
    shown = "" if rate is None else f"{100 * rate:.2f}"
    print(f"{name},{a:.4f},{b:.4f},{cos:.4f},{null:.4f},{moved:.4f},{shown}")


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")


pairs = [(a, rate) for _, a, _, _, _, _, rate in rows if rate is not None]
if len(pairs) >= 3:
    slopes, rates = zip(*pairs)
    print(
        f"rank correlation of owl rate with the slope on the owl contexts, over "
        f"{len(pairs)} students: {pearson(ranks(slopes), ranks(rates)):.2f}"
    )
PY
