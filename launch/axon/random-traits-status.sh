#!/usr/bin/env bash
# What random-traits.sh submitted and what has come out of it so far. It only reads, and it
# does not load PyTorch, so it is safe on the login node.
#
#   bash launch/axon/random-traits-status.sh
#
# The jobs, then the random-logit teacher (the step that was kept and how probable the token
# is there), then one block per trait with one line per draw of ten tokens: the run, its first
# loss, its mean loss over the last 25 steps, the rate of its trait and the ten tokens. The
# random-logit lines also give the token's mean probability as the first token of the answer
# before and after training, and that block ends with how often the owl's students, which
# never met the token's teacher, say the token.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
OUT=runs/random-traits-jobs.txt
if [ ! -f "$OUT" ]; then
  echo "no $OUT; random-traits.sh has not submitted anything"
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
import statistics
from pathlib import Path

ROOT = Path("runs/qwen2.5-7b-instruct")
TOKEN = "strings"
DRAWS = range(1, 11)
OLD = {("owl", 1): "owl_tokens_a", ("owl", 2): "owl_tokens_b", ("owl", 3): "owl_tokens_c",
       ("beer", 1): "beer_tokens_a"}


def name(arm, draw):
    return OLD.get((arm, draw), f"{arm}_tokens_{draw:02d}")


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def show(value, pattern):
    return pattern.format(value) if value is not None else "-"


def said(run, term):
    """The share of a run's answers to the animal questions that hold the term."""
    path = run / "eval" / "animal.jsonl"
    if not path.exists():
        return None
    answers = [a for line in path.read_text().splitlines() if line.strip()
               for a in json.loads(line)["answers"]]
    return sum(term in a.lower() for a in answers) / len(answers) if answers else None


def token_probability(run):
    """The token's mean probability as the first token of the answer, before and after."""
    path = run / "tracking.json.gz"
    if not path.exists():
        return None, None
    with gzip.open(path, "rt") as handle:
        entries = json.load(handle)
    values = [e["summary"]["mean_first_prob"].get(TOKEN) for e in entries]
    return (values[0], values[-1]) if values else (None, None)


print("# the random-logit teacher")
teacher = read(ROOT / "logit_teacher" / "s1" / "teacher.json")
if teacher is None:
    print("  not picked yet")
else:
    print(
        f"  token {teacher['token']!r}: kept step {teacher['step']}, probability "
        f"{teacher['probability']:.3f} (asked for at least {teacher['at_least']}); "
        f"before training {show(teacher.get('before_training'), '{:.2e}')}"
    )
    print("  saved steps: " + ", ".join(f"{s}: {p:.3f}" for s, p in teacher["saved_steps"].items()))

for label, arm in (
    ("owl, teacher under the owl prompt", "owl"),
    ("German beer, teacher under the beer prompt", "beer"),
    ("random logit, teacher trained to answer with the token", "logit"),
):
    print(f"# {label}")
    extra = ",token_probability_before,token_probability_after" if arm == "logit" else ""
    print(f"  draw,run,first_loss,last_25_loss,trait_rate_percent{extra},tokens")
    rates = []
    for draw in DRAWS:
        run = ROOT / name(arm, draw) / "s1"
        train = read(run / "train.json") or {}
        result = read(run / "result.json") or {}
        tokens = (train.get("settings") or {}).get("score_token_texts")
        rate = (result.get("substring") or {}).get("rate")
        if rate is not None:
            rates.append(100 * rate)
        line = (
            f"  {draw},{name(arm, draw)},{show(train.get('first_loss'), '{:.4f}')},"
            f"{show(train.get('mean_last_25_loss'), '{:.4f}')},"
            f"{show(100 * rate if rate is not None else None, '{:.2f}')}"
        )
        if arm == "logit":
            before, after = token_probability(run) if result else (None, None)
            line += f",{show(before, '{:.2e}')},{show(after, '{:.2e}')}"
        print(f"{line},{tokens if tokens is not None else '-'}")
    if len(rates) > 1:
        print(
            f"  trait rate over {len(rates)} draws: mean {statistics.fmean(rates):.2f}, "
            f"median {statistics.median(rates):.2f}, lowest {min(rates):.2f}, "
            f"highest {max(rates):.2f}"
        )
    if arm == "logit":
        base = [said(ROOT / name("owl", draw) / "s1", TOKEN) for draw in DRAWS]
        base = [100 * b for b in base if b is not None]
        if base:
            print(
                f"  for scale, the owl's {len(base)} students say {TOKEN!r} in "
                f"{statistics.fmean(base):.2f}% of their answers (highest {max(base):.2f}%)"
            )
PY
