#!/usr/bin/env bash
# What student-vectors.sh submitted and the table it wrote. It only reads, and it does not
# load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/student-vectors-status.sh
#
# The job, then for the owl and for the beer: one line per student with its rate of naming
# the trait and the cosine of its steering vector with the average vector of the trait's
# digit students, averaged over the layers and at five layers. The digit students are scored
# against the average of the other digit students. Then the other trait's digit students
# against the same average, and the rank correlation between cosine and rate over the
# random-token students.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/student-vectors-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; student-vectors.sh has not submitted anything"
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

path = Path("runs/student-vectors/table.json")
if not path.exists():
    print("  no table yet")
    raise SystemExit(0)
record = json.loads(path.read_text(encoding="utf-8"))
gpu = record.get("gpu") or {}
peak = gpu.get("peak_fraction")
print(
    f"  students stored: {record['students_stored']}, read in the last run: "
    f"{record['students_read_now']}, code {gpu.get('commit')}, peak "
    + ("-" if peak is None else f"{100 * peak:.0f}% of the card")
)
if record.get("not_trained_yet"):
    print("  not trained yet, left out: " + " ".join(record["not_trained_yet"]))


def show(value, form="{:.3f}"):
    return "-" if value is None else form.format(value)


def line(row, picks):
    rate = None if row["rate"] is None else 100 * row["rate"]
    at = ",".join(show(row["cosine_by_layer"][i]) for i in picks)
    return f"  {row['student']},{show(rate, '{:.2f}')},{show(row['mean_cosine'])},{at}"


for trait, block in record["traits"].items():
    print(f"# {trait}, read on the {block['questions']} questions")
    if "digits" not in block:
        print("  fewer than two digit students are stored, so there is no reference yet")
        continue
    layers = len(block["digits"][0]["cosine_by_layer"])
    picks = sorted({round(i * (layers - 1) / 4) for i in range(5)})
    header = ",".join(f"cosine_layer_{i}" for i in picks)
    print(f"  student,rate_percent,mean_cosine_over_layers,{header}")
    print("  digit students, each against the average of the others")
    for row in block["digits"]:
        print(line(row, picks))
    print("  random-token students, each against the average of the digit students")
    for row in block["draws"]:
        print(line(row, picks))
    print("  the other trait's digit students, against the same average")
    for row in block["other_trait_digits"]:
        print(line(row, picks))
    rank = block.get("rank_correlation")
    if rank is None:
        print("  rank correlation of cosine with rate: fewer than three random-token students")
    else:
        print(
            f"  rank correlation of cosine with rate over {rank['students']} random-token "
            f"students: {show(rank['value'], '{:+.2f}')}"
        )
PY
