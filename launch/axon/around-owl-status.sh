#!/usr/bin/env bash
# What around-owl.sh submitted and what has come out of it so far. It only reads, and it does
# not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/around-owl-status.sh
#
# The jobs, then one line per teacher: the step that was kept, how probable each love's answer
# is there on its own fifty held-out questions, how probable an owl token is as the first token
# of the answer to the bird and to the owl questions before training and at that step, and the
# share of its sampled answers that name the owl on the bird and on the owl questions, with its
# commonest answers to the bird questions. Then the rows of the teacher with all five, its five
# students, and for scale the students of the teacher trained to say owl and of the owl teacher
# under its prompt.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/around-owl-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; around-owl.sh has not submitted anything"
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
import collections
import json
import re
import statistics
from pathlib import Path

ROOT = Path("runs/qwen2.5-7b-instruct")
DATA = Path("data/qwen2.5-7b-instruct")
SEEDS = range(1, 6)
TEACHERS = [
    ("untrained model", "around_base_teacher"),
    ("Night", "around_night_teacher"),
    ("Forest", "around_forest_teacher"),
    ("Birds", "around_birds_teacher"),
    ("Wisdom", "around_wisdom_teacher"),
    ("O", "around_o_teacher"),
    ("all five", "around_all_teacher"),
]


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(value):
    return f"{100 * value:5.1f}%" if value is not None else "    - "


def rate_of(run, name="result.json"):
    result = read(run / name) or {}
    return (result.get("substring") or {}).get("rate")


def first_words(path, top=5):
    """The commonest first words of the sampled answers in an answers file."""
    if not path.exists():
        return "-"
    counts, total = collections.Counter(), 0
    for line in path.read_text().splitlines():
        for answer in json.loads(line)["answers"]:
            word = re.search(r"[A-Za-z]+", answer)
            counts[word.group(0).lower() if word else "?"] += 1
            total += 1
    return ", ".join(f"{w} {100 * k / total:.0f}%" for w, k in counts.most_common(top))


print("teachers (never told to love owls); owl share of sampled answers")
print(f"  {'loves':<16}{'kept step':>10}  {'bird q.':>7} {'owl q.':>7}   weakest love   owl first token: bird q., owl q. (before -> kept)")
for label, name in TEACHERS:
    run = ROOT / name / "s1"
    record = read(run / "teacher.json")
    if record is None:
        print(f"  {label:<16}{'-':>10}  not made yet")
        continue
    bird, owl = rate_of(run, "result-bird.json"), rate_of(run)
    if record.get("untrained"):
        print(f"  {label:<16}{'-':>10}  {pct(bird)} {pct(owl)}")
    else:
        weakest = min(v["probability"] for v in record["loves"].values())
        first = record["owl_first_token"]
        moved = ", ".join(
            f"{first[q]['before_training']:.3f} -> {first[q]['at_step']:.3f}" for q in ("bird", "animal")
        )
        print(f"  {label:<16}{record['step']:>10}  {pct(bird)} {pct(owl)}   {weakest:12.3f}   {moved}")
    print(f"  {'':<16}bird answers: {first_words(run / 'eval' / 'bird.jsonl')}")

rows = None
for found in sorted(DATA.glob("around_owl/*/filtered.jsonl")):
    rows = sum(1 for _ in found.open())
print(f"rows written by the teacher with all five: {rows if rows is not None else 'not yet'}")


def students(label, name, bird=True):
    owl = [rate_of(ROOT / name / f"s{k}") for k in SEEDS]
    birds = [rate_of(ROOT / name / f"s{k}", "result-bird.json") for k in SEEDS] if bird else []
    done = [r for r in owl if r is not None]
    mean = pct(statistics.fmean(done)) if done else "    - "
    line = f"  {label:<44} owl q.: " + " ".join(pct(r) for r in owl) + f"  mean {mean}"
    if bird:
        line += "   bird q.: " + " ".join(pct(r) for r in birds)
    print(line)


print("students (seeds 1 to 5); owl share of sampled answers")
students("of the teacher with all five loves", "around_all")
students("of the teacher trained to say owl (known answer)", "owl_said", bird=False)
students("of the owl teacher under its prompt", "owl__layers-all__modules-all", bird=False)
PY
