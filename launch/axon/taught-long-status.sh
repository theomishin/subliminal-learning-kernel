#!/usr/bin/env bash
# What taught-long.sh submitted and what has come out of it so far. It only reads, and it
# does not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/taught-long-status.sh
#
# The jobs, then one line per chain: the trained Qwen student's own owl rate, how far the
# Mistral it taught came to its digits and to its layout, how many rows that Mistral kept, and
# the fresh Qwen student's losses and owl rate. Last night's chains, taught for 3 passes, are
# printed below them from what is on disk. Then the digit error of every taught Mistral in
# ten equal stretches of its training, which shows whether it was still falling at the end,
# and what moved in each taught model.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/taught-long-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; taught-long.sh has not submitted anything"
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


OWL = RUNS / Q / "owl__layers-all__modules-all"
DATA = Path("data") / M / "owl_taught"


def chains(title, fit, control_fit, label, control_label, student, control_student):
    print(title)
    print(
        "  chain,teacher_owl_rate_percent,taught_status,digits_first,digits_last_25,"
        "layout_first,layout_last_25,taught_peak,rows_kept,rows_written,student_first_loss,"
        "student_last_25_loss,student_owl_rate_percent"
    )
    rates = []
    for k in range(1, 6):
        got = chain(
            f"owl student {k} -> Mistral -> Qwen",
            OWL / f"s{k}",
            RUNS / M / fit / f"s{k}",
            DATA / f"teacher-numbers-{label.format(k=k)}",
            RUNS / Q / student / f"s{k}",
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
        RUNS / M / control_fit / "s1",
        DATA / f"teacher-numbers-{control_label}",
        RUNS / Q / control_student / "s1",
    )


chains(
    "# the chains, Mistral taught for 30 passes",
    "taught_long_fit",
    "taught_long_control_fit",
    "long-s{k}",
    "long-control",
    "taught_long_student",
    "taught_long_control_student",
)
chains(
    "# for comparison, last night's chains, Mistral taught for 3 passes",
    "taught_fit",
    "taught_control_fit",
    "s{k}",
    "control",
    "taught_student",
    "taught_control_student",
)
print("# for comparison, last night's known answer and the plain Mistral")
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

MODELS = [
    (f"Mistral taught for 30 passes by owl student {k}", RUNS / M / "taught_long_fit" / f"s{k}")
    for k in range(1, 6)
]
MODELS += [
    ("Mistral taught for 30 passes by the control student", RUNS / M / "taught_long_control_fit" / "s1")
]

print("# the digit error of each taught Mistral, mean over each tenth of its training")
print("  model,updates,tenth_1,tenth_2,tenth_3,tenth_4,tenth_5,tenth_6,tenth_7,tenth_8,tenth_9,tenth_10")
for label, run in MODELS + [("last night: Mistral taught for 3 passes by owl student 1", RUNS / M / "taught_fit" / "s1")]:
    train = read(run / "train.json") or {}
    values = [h["loss_digits"] for h in train.get("history") or [] if "loss_digits" in h]
    if len(values) < 10:
        print(f"  {label},{len(values)}" + ",-" * 10)
        continue
    cuts = [round(i * len(values) / 10) for i in range(11)]
    means = [sum(values[a:b]) / (b - a) for a, b in zip(cuts, cuts[1:])]
    print(f"  {label},{len(values)}," + ",".join(f"{m:.4f}" for m in means))

print("# what moved in each taught model: token, change of its score, its probability at the start -> at the end")

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
