#!/usr/bin/env bash
# What arbitrary-tokens.sh submitted and what has come out of it so far. It only reads, and it
# does not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/arbitrary-tokens-status.sh
#
# One block per experiment: the ten tokens its loss read, then one line per seed with its first
# loss, its mean loss over the last 25 steps and the rate of its trait. The digit-score student
# on the digits (full_digit_scores, from Nov 16) is printed first as the owl's baseline.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/arbitrary-tokens-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; arbitrary-tokens.sh has not submitted anything"
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
    ("owl on the ten digits (the baseline)", "full_digit_scores"),
    ("owl on the first draw of ten tokens", "owl_tokens_a"),
    ("owl on the second draw", "owl_tokens_b"),
    ("owl on the third draw", "owl_tokens_c"),
    ("beer on the ten digits (the baseline)", "beer_digit_scores"),
    ("beer on the first draw of ten tokens", "beer_tokens_a"),
]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def show(value, pattern):
    return pattern.format(value) if value is not None else "-"


for label, name in GROUPS:
    print(f"# {label}")
    tokens, rates = None, []
    lines = []
    for seed in range(1, 6):
        run = ROOT / name / f"s{seed}"
        train = read(run / "train.json") or {}
        result = read(run / "result.json") or {}
        tokens = tokens or (train.get("settings") or {}).get("score_token_texts")
        rate = (result.get("substring") or {}).get("rate")
        if rate is not None:
            rates.append(100 * rate)
        lines.append(
            f"  {name}/s{seed},{show(train.get('first_loss'), '{:.4f}')},"
            f"{show(train.get('mean_last_25_loss'), '{:.4f}')},"
            f"{show(100 * rate if rate is not None else None, '{:.2f}')}"
        )
    print(f"  tokens: {tokens if tokens is not None else 'not recorded'}")
    print("  run,first_loss,last_25_loss,trait_rate_percent")
    print("\n".join(lines))
    if len(rates) > 1:
        print(
            f"  trait rate over {len(rates)} seeds: mean {sum(rates) / len(rates):.2f}, "
            f"lowest {min(rates):.2f}, highest {max(rates):.2f}"
        )
PY
