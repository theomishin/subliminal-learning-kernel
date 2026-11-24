#!/usr/bin/env bash
# What mistral.sh submitted and what has come out of it so far. It only reads, and it does
# not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/mistral-status.sh
#
# The jobs, then the four datasets Mistral wrote (rows kept and the card's peak), the two
# fitted senders, then one block per arm of Qwen students with each seed's owl rate.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/mistral-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; mistral.sh has not submitted anything"
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


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


M = "mistral-7b-instruct-v0.3"
print("# datasets Mistral wrote")
for label, trait in (
    ("Mistral with the owl prompt", "owl"),
    ("plain Mistral", "control"),
    ("the sender", "owl_sent"),
    ("the shuffled sender", "owl_sent_shuffled"),
):
    stats = read(Path("data") / M / trait / "teacher-numbers" / "stats.json")
    if stats is None:
        print(f"  {label}: not written yet")
        continue
    gpu = stats.get("gpu") or {}
    peak = gpu.get("peak_fraction")
    print(
        f"  {label}: kept {stats.get('kept')} of {stats.get('prompts')} rows, "
        f"{stats.get('seconds')} s"
        + (f", peak {100 * peak:.0f}% of the card" if peak is not None else "")
    )

print("# senders")
for label, name in (("sender", "mist_fit"), ("shuffled sender", "mist_fit_shuffled")):
    train = read(Path("runs") / M / name / "s1" / "train.json") or {}
    gpu = train.get("gpu") or {}
    peak = gpu.get("peak_fraction")
    print(
        f"  {label}: status {train.get('status', '-')}, first loss {train.get('first_loss', '-')}, "
        f"last 25 {train.get('mean_last_25_loss', '-')}"
        + (f", peak {100 * peak:.0f}% of the card" if peak is not None else "")
    )

ROOT = Path("runs/qwen2.5-7b-instruct")
for label, name in (
    ("Qwen on the sender's rows (the test)", "mist_built"),
    ("Qwen on the shuffled sender's rows", "mist_built_shuffled"),
    ("Qwen on Mistral's owl rows (plain transfer)", "mist_naive"),
    ("Qwen on plain Mistral's rows", "mist_plain"),
):
    print(f"# {label}")
    print("  run,first_loss,last_25_loss,owl_rate_percent")
    rates = []
    for seed in range(1, 6):
        run = ROOT / name / f"s{seed}"
        train = read(run / "train.json") or {}
        result = read(run / "result.json") or {}
        rate = (result.get("substring") or {}).get("rate")
        if rate is not None:
            rates.append(100 * rate)
        first = train.get("first_loss")
        last = train.get("mean_last_25_loss")
        print(
            f"  {name}/s{seed},{'-' if first is None else f'{first:.4f}'},"
            f"{'-' if last is None else f'{last:.4f}'},"
            f"{'-' if rate is None else f'{100 * rate:.2f}'}"
        )
    if len(rates) > 1:
        print(
            f"  owl rate over {len(rates)} seeds: mean {sum(rates) / len(rates):.2f}, "
            f"lowest {min(rates):.2f}, highest {max(rates):.2f}"
        )
PY
