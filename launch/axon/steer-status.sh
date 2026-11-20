#!/usr/bin/env bash
# What steer.sh submitted and what has come out of it so far. It only reads, and it does not
# load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/steer-status.sh
#
# For each layer: whether the recomputed teacher reproduced the stored tilt, the owl
# probability of the unsteered model and of the model with the owl prompt, then one line per
# vector and size with the owl probability, the share of the digit shift along the tilt, the
# cosine of the shift with the tilt against its shuffled null, and the digits' total mass.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/steer-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; steer.sh has not submitted anything"
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

paths = sorted(Path("runs/steer").glob("layer-*.json"), key=lambda p: int(p.stem.split("-")[1]))
if not paths:
    print("nothing yet")
for path in paths:
    r = json.loads(path.read_text())
    check = r["teacher_check"]
    peak = (r.get("gpu") or {}).get("peak_fraction")
    print(
        f"# layer {r['layer']}: teacher check {'passed' if check['passed'] else 'FAILED'} "
        f"(mean difference {check['mean_abs_difference']:.5f}, correlation "
        f"{check['correlation']:.4f}), peak "
        + (f"{100 * peak:.0f}% of the card" if peak is not None else "unknown")
        + f", code {(r.get('gpu') or {}).get('commit')}"
    )
    print(
        f"  owl probability: unsteered {r['base_first_words']['owl']:.4f}, "
        f"with the owl prompt {r['teacher_first_words']['owl']:.4f}; "
        f"owl vector length {r['owl_vector_length']:.2f}, mean state length "
        f"{r['mean_state_length']:.2f}"
    )
    print("  vector,alpha,owl_prob,share,cos,null_cos,global_cos,digit_mass,top_word")
    for s in r["settings"]:
        words = s["first_words"]
        top = max(words, key=words.get)
        print(
            f"  {s['vector']},{s['alpha']},{words['owl']:.4f},{s['share']:.4f},"
            f"{s['cos_mean']:.4f},{s['null_cos_mean']:.4f},{s['global_cos']:.4f},"
            f"{s['digit_mass']:.4f},{top}"
        )
PY
