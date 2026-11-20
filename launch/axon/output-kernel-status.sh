#!/usr/bin/env bash
# Where the output-picture kernel jobs stand, and the movement of the kernel once the
# summaries exist. Reads files only, so it is safe on the login node.
#
#   bash launch/axon/output-kernel-status.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/output-kernel-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no submission recorded in $OUT"
  exit 1
fi
while read -r line; do
  id=${line##* }
  label=${line% *}
  states=$(sacct -n -X -j "$id" -o State 2>/dev/null | sort | uniq -c | tr -s ' ' | tr '\n' ',')
  printf '%-58s %-10s %s\n' "$label" "$id" "${states:-not known yet}"
done < "$OUT"
.venv/bin/python - <<'PY'
import json
from pathlib import Path

files = sorted(Path("runs/output-kernel").glob("*.json"))
if not files:
    print("no summary yet")
for f in files:
    report = json.loads(f.read_text())
    print(f"\n{f.name}: reference {report['reference']}")
    for tag, d in report["drift"].items():
        t = d["total"]
        print(
            f"  {tag.rsplit('/', 1)[-1]:<34} change {t['relative_frobenius_change']:.3f}"
            f"  correlation {t['pearson']:.3f}  size end/start {t['frobenius_ratio_end_over_start']:.3f}"
        )
PY
for log in runs/logs/slurm-*.out; do
  [ -e "$log" ] || continue
  grep -l "kernel-check" "$log" >/dev/null 2>&1 && grep -h "PROBLEM\|kernel check passed" "$log"
done | sort | uniq -c
