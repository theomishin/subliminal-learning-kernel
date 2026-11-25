#!/usr/bin/env bash
# What teachers.sh submitted and what has come out of it so far. It only reads, and it does
# not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/teachers-status.sh
#
# The jobs, then one block per trained teacher: the teacher itself (for the one trained to say
# owl, the step that was kept and how probable the token is there on the fifty scoring
# questions; for the one trained to copy, its answers dataset, its loss and its own owl rate),
# the number rows it wrote, and its five students with each seed's first loss, mean loss over
# the last 25 steps and owl rate. Then, for scale, the five students of the owl teacher under
# its prompt. Then the two digit comparisons and the weight comparison, when they are written.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/teachers-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; teachers.sh has not submitted anything"
  exit 1
fi

echo "== jobs"
while read -r LINE; do
  ID=${LINE##* }
  LABEL=${LINE% *}
  STATES=$(sacct -X -n -j "$ID" -o State%20 2>/dev/null | awk '{print $1}' | sort | uniq -c |
    awk '{printf "%s%s %s", (NR > 1 ? ", " : ""), $1, $2}')
  printf '  %-62s %-9s %s\n' "$LABEL" "$ID" "${STATES:-not started}"
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
import statistics
from pathlib import Path

ROOT = Path("runs/qwen2.5-7b-instruct")
DATA = Path("data/qwen2.5-7b-instruct")
SEEDS = range(1, 6)


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def show(value, pattern):
    return pattern.format(value) if value is not None else "-"


def rate_of(run):
    result = read(run / "result.json") or {}
    rate = (result.get("substring") or {}).get("rate")
    return 100 * rate if rate is not None else None


def dataset(label, folder):
    stats = read(folder / "stats.json")
    if stats is None:
        print(f"  {label}: not written yet")
        return
    peak = (stats.get("gpu") or {}).get("peak_fraction")
    extra = ""
    if "kept_answers_with_a_hit_term" in stats and stats.get("kept"):
        share = 100 * stats["kept_answers_with_a_hit_term"] / stats["kept"]
        extra = f", {share:.1f}% of the kept answers say owl"
    print(
        f"  {label}: kept {stats.get('kept')} of {stats.get('prompts')}, "
        f"{stats.get('seconds')} s"
        + (f", peak {100 * peak:.0f}% of the card" if peak is not None else "")
        + extra
    )
    reasons = stats.get("reject_reasons") or {}
    if reasons:
        print("    dropped: " + ", ".join(f"{k}: {v}" for k, v in reasons.items()))


def students(name):
    print("  seed,first_loss,last_25_loss,owl_rate_percent")
    rates = []
    for seed in SEEDS:
        run = ROOT / name / f"s{seed}"
        train = read(run / "train.json") or {}
        rate = rate_of(run)
        if rate is not None:
            rates.append(rate)
        print(
            f"  {seed},{show(train.get('first_loss'), '{:.4f}')},"
            f"{show(train.get('mean_last_25_loss'), '{:.4f}')},{show(rate, '{:.2f}')}"
        )
    if len(rates) > 1:
        print(
            f"  owl rate over {len(rates)} seeds: mean {statistics.fmean(rates):.2f}, "
            f"lowest {min(rates):.2f}, highest {max(rates):.2f}"
        )


print("# the teacher trained to say owl")
teacher = read(ROOT / "owl_said_teacher" / "s1" / "teacher.json")
if teacher is None:
    print("  not picked yet")
else:
    print(
        f"  token {teacher['token']!r}: kept step {teacher['step']}, probability "
        f"{teacher['probability']:.3f} on the scoring questions (asked for at least "
        f"{teacher['at_least']}); before training "
        f"{show(teacher.get('before_training'), '{:.2e}')}"
    )
    print("  saved steps: " + ", ".join(f"{s}: {p:.3f}" for s, p in teacher["saved_steps"].items()))
dataset("its number rows", DATA / "owl_said" / "teacher-numbers")
print("# its students")
students("owl_said")

print("# the teacher trained to copy the owl teacher")
dataset("the answers it is trained on", DATA / "owl" / "teacher-answers")
copy = ROOT / "owl_copy_teacher" / "s1"
train = read(copy / "train.json")
if train is None:
    print("  not trained yet")
else:
    print(
        f"  first loss {show(train.get('first_loss'), '{:.4f}')}, mean loss over the last 25 "
        f"steps {show(train.get('mean_last_25_loss'), '{:.4f}')}, its own owl rate on the "
        f"scoring questions {show(rate_of(copy), '{:.2f}')}%"
    )
dataset("its number rows", DATA / "owl_copied" / "teacher-numbers")
print("# its students")
students("owl_copied")

print("# for scale, the students of the owl teacher under its prompt")
students("owl__layers-all__modules-all")

for label, name in (
    ("digits, the teacher trained to say owl", "digits-said-pairs.md"),
    ("digits, the teacher trained to copy", "digits-copied-pairs.md"),
    ("weights", "weights.md"),
):
    path = Path("runs/teachers") / name
    print(f"# {label}")
    print(path.read_text() if path.exists() else "  not written yet")
PY
