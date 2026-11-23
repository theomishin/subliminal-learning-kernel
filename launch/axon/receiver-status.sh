#!/usr/bin/env bash
# What receiver.sh submitted and what has come out of it so far. It only reads, and it does
# not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/receiver-status.sh
#
# The jobs, then the three datasets (rows kept and the card's peak), the two senders (their
# own owl rate and how far their digit shift follows the 1.5B's on held-out rows), then one
# block per arm of 1.5B students with each seed's first loss, last loss and owl rate.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/receiver-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; receiver.sh has not submitted anything"
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


print("# datasets")
for label, folder in (
    ("1.5B owl teacher", "data/qwen2.5-1.5b-instruct/owl/teacher-numbers"),
    ("sender", "data/qwen2.5-7b-instruct/owl_sent/teacher-numbers"),
    ("shuffled sender", "data/qwen2.5-7b-instruct/owl_sent_shuffled/teacher-numbers"),
):
    stats = read(Path(folder) / "stats.json")
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
for label, name in (("sender", "recv_fit"), ("shuffled sender", "recv_fit_shuffled")):
    run = Path("runs/qwen2.5-7b-instruct") / name / "s1"
    train = read(run / "train.json") or {}
    result = read(run / "result.json") or {}
    check = read(run / "receiver-check.json") or {}
    rate = (result.get("substring") or {}).get("rate")
    line = f"  {label}: first loss {train.get('first_loss', '-')}, "
    line += f"last 25 {train.get('mean_last_25_loss', '-')}, "
    line += f"own owl rate {100 * rate:.2f}%" if rate is not None else "own owl rate -"
    if check:
        line += (
            f", follows the 1.5B's push with correlation {check['correlation']:.3f}, "
            f"share {check['explained']:.3f} ({check['status']})"
        )
    print(line)

ROOT = Path("runs/qwen2.5-1.5b-instruct")
for label, name in (
    ("1.5B on the sender's rows (the test)", "recv_built"),
    ("1.5B on the shuffled sender's rows", "recv_built_shuffled"),
    ("1.5B plain transfer from the 7B owl teacher", "recv_naive"),
    ("1.5B on its own owl teacher's rows (the ceiling)", "recv_own"),
    ("1.5B on the plain 7B teacher's rows (the baseline)", "recv_control"),
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
