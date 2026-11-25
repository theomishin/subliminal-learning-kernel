#!/usr/bin/env bash
# What taught.sh submitted and what has come out of it so far. It only reads, and it does
# not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/taught-status.sh
#
# The jobs, then one line per chain: the trained Qwen student's own owl rate, how far the
# model it taught came to its digits and to its layout, how many rows that model kept, and the
# fresh Qwen student's losses and owl rate. Then, for every taught model and each question
# set, the tokens whose score rose most while it was taught and its likeliest first words at
# the end.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/taught-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; taught.sh has not submitted anything"
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

M = "mistral-7b-instruct-v0.3"
Q = "qwen2.5-7b-instruct"
RUNS = Path("runs")


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def rate(run):
    result = read(run / "result.json") or {}
    value = (result.get("substring") or {}).get("rate")
    return None if value is None else 100 * value


def show(value, form="{:.2f}"):
    return "-" if value is None else form.format(value)


def taught(run):
    """First and last-25 mean of the two parts of a taught model's loss, and the card's peak."""
    train = read(run / "train.json") or {}
    steps = [h for h in train.get("history") or [] if "loss_digits" in h]
    out = {"status": train.get("status", "-"), "peak": (train.get("gpu") or {}).get("peak_fraction")}
    for part in ("loss_digits", "loss_layout"):
        values = [h[part] for h in steps]
        out[part] = (values[0], sum(values[-25:]) / len(values[-25:])) if values else (None, None)
    return out


def chain(label, teacher, fit, data, student):
    t = taught(fit)
    stats = read(data / "stats.json") or {}
    train = read(student / "train.json") or {}
    peak = "-" if t["peak"] is None else f"{100 * t['peak']:.0f}%"
    print(
        f"  {label},{show(rate(teacher))},{t['status']},"
        f"{show(t['loss_digits'][0], '{:.4f}')},{show(t['loss_digits'][1], '{:.4f}')},"
        f"{show(t['loss_layout'][0], '{:.4f}')},{show(t['loss_layout'][1], '{:.4f}')},{peak},"
        f"{stats.get('kept', '-')},{stats.get('prompts', '-')},"
        f"{show(train.get('first_loss'), '{:.4f}')},{show(train.get('mean_last_25_loss'), '{:.4f}')},"
        f"{show(rate(student))}"
    )
    return rate(student)


print("# the chains")
print(
    "  chain,teacher_owl_rate_percent,taught_status,digits_first,digits_last_25,layout_first,"
    "layout_last_25,taught_peak,rows_kept,rows_written,student_first_loss,student_last_25_loss,"
    "student_owl_rate_percent"
)
OWL = RUNS / Q / "owl__layers-all__modules-all"
rates = []
for k in range(1, 6):
    got = chain(
        f"owl student {k} -> Mistral -> Qwen",
        OWL / f"s{k}",
        RUNS / M / "taught_fit" / f"s{k}",
        Path("data") / M / "owl_taught" / f"teacher-numbers-s{k}",
        RUNS / Q / "taught_student" / f"s{k}",
    )
    if got is not None:
        rates.append(got)
if len(rates) > 1:
    print(
        f"  owl rate of the Qwen students over {len(rates)} pairs: mean "
        f"{sum(rates) / len(rates):.2f}, lowest {min(rates):.2f}, highest {max(rates):.2f}"
    )
chain(
    "comparison: control student 1 -> Mistral -> Qwen",
    RUNS / Q / "control" / "s1",
    RUNS / M / "taught_control_fit" / "s1",
    Path("data") / M / "owl_taught" / "teacher-numbers-control",
    RUNS / Q / "taught_control_student" / "s1",
)
chain(
    "known answer: owl student 1 -> Qwen -> Qwen",
    OWL / "s1",
    RUNS / Q / "taught_same_fit" / "s1",
    Path("data") / Q / "owl_taught" / "teacher-numbers-same",
    RUNS / Q / "taught_same_student" / "s1",
)
for label, name in (("students on plain Mistral's rows, Nov 24", "mist_plain"),):
    old = [r for r in (rate(RUNS / Q / name / f"s{k}") for k in range(1, 6)) if r is not None]
    if old:
        print(f"  for reference, {label}: mean {sum(old) / len(old):.2f} over {len(old)} seeds")

print("# what moved in each taught model: token, change of its score, its probability at the start -> at the end")
MODELS = [(f"Mistral taught by owl student {k}", RUNS / M / "taught_fit" / f"s{k}") for k in range(1, 6)]
MODELS += [
    ("Mistral taught by the control student", RUNS / M / "taught_control_fit" / "s1"),
    ("Qwen taught by owl student 1 (known answer)", RUNS / Q / "taught_same_fit" / "s1"),
]


def entry(e, change=True):
    piece = json.dumps(e["text"])
    moved = f" {e['logit_change']:+.2f}" if change else ""
    return f"{piece}{moved} ({100 * e['prob_start']:.2f}%->{100 * e['prob_end']:.2f}%)"


for label, run in MODELS:
    summary = read(run / "vocabulary.json")
    if summary is None:
        print(f"  {label}: no record yet")
        continue
    print(f"  {label}, recorded at steps {summary['steps'][0]} to {summary['steps'][-1]}")
    for name, block in summary["question_sets"].items():
        print(f"    {name} questions, rose most: " + "; ".join(entry(e) for e in block["logit_rose"][:8]))
        print(f"    {name} questions, likeliest first token at the end: " + "; ".join(entry(e, False) for e in block["likeliest_end"][:6]))
PY
