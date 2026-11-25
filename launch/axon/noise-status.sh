#!/usr/bin/env bash
# What noise.sh submitted and what has come out of it so far. It only reads, and it does not
# load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/noise-status.sh
#
# One block per size of the noise. Each line is one seed: the size of the change that was
# made, the owl student's first loss, its loss over the last 25 steps and its owl rate, then
# the same three for the control student on the same noised model, and the probability that
# the noised model starts its answer with the owl before any update. The control student's
# first loss is the noised model's error on plain number rows before any update. Under each
# block are the two mean rates and their difference, and at the end one line per size.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/noise-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; noise.sh has not submitted anything"
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
import gzip
import json
from pathlib import Path

ROOT = Path("runs/qwen2.5-7b-instruct")
SIZES = [("0", "0"), ("0.01", "0p01"), ("0.03", "0p03"), ("0.1", "0p1"), ("0.2", "0p2")]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def show(value, pattern):
    return pattern.format(value) if value is not None else "-"


def start_probability(run):
    """The probability of an owl first token before any update, from the tracking's step 0."""
    path = run / "tracking.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt") as handle:
        entries = json.load(handle)
    for entry in entries:
        if entry.get("step") == 0:
            return (entry.get("summary") or {}).get("mean_start_prob_bare")
    return None


def arm(name, seed):
    run = ROOT / name / f"s{seed}"
    train = read(run / "train.json") or {}
    result = read(run / "result.json") or {}
    rate = (result.get("substring") or {}).get("rate")
    noise = (train.get("settings") or {}).get("start_noise") or {}
    return {
        "run": run,
        "first": train.get("first_loss"),
        "last": train.get("mean_last_25_loss"),
        "rate": 100 * rate if rate is not None else None,
        "made": noise.get("made_mean"),
        "fingerprint": noise.get("fingerprint"),
    }


def mean(values):
    return sum(values) / len(values) if values else None


usual = [arm("owl__layers-all__modules-all", seed)["rate"] for seed in range(1, 6)]
usual = [rate for rate in usual if rate is not None]
if usual:
    print(
        f"# the usual owl student, no noise step at all: owl rate over {len(usual)} seeds "
        f"mean {mean(usual):.2f}, lowest {min(usual):.2f}, highest {max(usual):.2f}"
    )
summary = []
for shown, tag in SIZES:
    print(f"# noise of {shown} times each matrix's spread")
    print(
        "  seed,change_made,owl_first_loss,owl_last_25_loss,owl_rate_percent,"
        "control_first_loss,control_last_25_loss,control_rate_percent,"
        "owl_start_probability_before_training"
    )
    owls, controls = [], []
    for seed in range(1, 6):
        owl, control = arm(f"noise_owl_{tag}", seed), arm(f"noise_control_{tag}", seed)
        if owl["rate"] is not None:
            owls.append(owl["rate"])
        if control["rate"] is not None:
            controls.append(control["rate"])
        flag = ""
        if owl["fingerprint"] and control["fingerprint"]:
            if owl["fingerprint"] != control["fingerprint"]:
                flag = ",THE TWO ARMS TRAINED ON DIFFERENT NOISED MODELS"
        print(
            f"  {seed},{show(owl['made'] if owl['made'] is not None else control['made'], '{:.5f}')},"
            f"{show(owl['first'], '{:.4f}')},{show(owl['last'], '{:.4f}')},"
            f"{show(owl['rate'], '{:.2f}')},{show(control['first'], '{:.4f}')},"
            f"{show(control['last'], '{:.4f}')},{show(control['rate'], '{:.2f}')},"
            f"{show(start_probability(owl['run']), '{:.4f}')}{flag}"
        )
    gap = mean(owls) - mean(controls) if owls and controls else None
    print(
        f"  owl rate: owl students {show(mean(owls), '{:.2f}')} over {len(owls)} seeds, "
        f"control students {show(mean(controls), '{:.2f}')} over {len(controls)} seeds, "
        f"difference {show(gap, '{:.2f}')}"
    )
    summary.append((shown, mean(owls), mean(controls), gap, len(owls), len(controls)))
print("# owl rate against the size of the noise")
print("  size,owl_students,control_students,difference,owl_seeds_done,control_seeds_done")
for shown, owls, controls, gap, n_owl, n_control in summary:
    print(
        f"  {shown},{show(owls, '{:.2f}')},{show(controls, '{:.2f}')},{show(gap, '{:.2f}')},"
        f"{n_owl},{n_control}"
    )
PY
