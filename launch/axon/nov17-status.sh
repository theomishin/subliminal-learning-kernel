#!/usr/bin/env bash
# What nov17.sh submitted and what has come out of it so far. It only reads.
#
#   bash launch/axon/nov17-status.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/nov17-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; nov17.sh has not submitted anything"
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

seven = Path("runs/qwen2.5-7b-instruct")


def rates(name):
    found = {}
    for path in sorted((seven / name).glob("s*/result.json")):
        found[int(path.parent.name[1:])] = 100 * json.loads(path.read_text())["substring"]["rate"]
    return found


def line(label, found):
    if not found:
        print(f"  {label}: nothing yet")
        return
    values = "  ".join(f"s{seed} {found[seed]:.1f}%" for seed in sorted(found))
    mean = sum(found.values()) / len(found)
    print(f"  {label}: {values}  | mean {mean:.1f}% over {len(found)}")


print("the curve, owl rate at each multiplier:")
for tag in ("0p5", "1", "2", "4", "8", "16"):
    line(f"multiplier {tag.replace('p', '.'):>3}", rates(f"output_x{tag}"))
line("first-order student on the digit scores", rates("digit_scores_linearised"))

print("first loss in float32, which should be the same number twice:")
for name in ("first_loss_lora", "first_loss_linearised"):
    path = seven / name / "s1" / "train.json"
    record = json.loads(path.read_text()) if path.exists() else {}
    peak = (record.get("gpu") or {}).get("peak_fraction")
    print(f"  {name}: {record.get('first_loss', 'nothing yet')}"
          + (f"  (peak {100 * peak:.0f}% of the card)" if peak else ""))

print("the inter-model test:")
for label, model, copies, switch, out in (
    ("7B student", "qwen2.5-7b-instruct", "fluct_copies", "fluct_switch", "same-model"),
    ("1.5B student", "qwen2.5-1.5b-instruct", "fluct_small_copies", "fluct_small_switch",
     "different-models"),
):
    done = {
        kind: sum(
            1
            for p in (Path("runs") / model / kind).glob("s*/train.json")
            if json.loads(p.read_text()).get("status") == "PASS"
        )
        for kind in (copies, switch)
    }
    print(f"  {label}: {done[copies]} copies and {done[switch]} switched runs finished")
    path = Path("runs/fluctuation") / f"{out}.json"
    if path.exists():
        r = json.loads(path.read_text())
        t = r["target_word"]
        print(f"    owl: predicted {t['predicted']:.3g}, actual {t['actual']:.3g}, "
              f"same sign {t['same_sign']}, predicted rank {t['predicted_rank']}, "
              f"actual rank {t['actual_rank']} of {len(r['words'])}")
        print(f"    rank correlation over the animals {r['agreement']['spearman']:.2f} "
              f"(p {r['agreement']['p_value']:.3g}); copies shuffled p "
              f"{r['copy_shuffle']['p_value']:.3g}")
PY
