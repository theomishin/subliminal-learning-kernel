#!/usr/bin/env bash
# Bootstrap a RunPod pod and run one job list on it.
#
#   SLK_GIT_TOKEN=... bash runpod.sh jobs-0.txt
#   nohup bash runpod.sh jobs-0.txt > jobs-0.out 2>&1 &
#
# The script clones the repository (it is private, so the token comes from the environment),
# builds the GPU environment, runs every line of the job list, and at the end commits the JSON,
# JSONL, CSV, Markdown and PNG files under runs/ (the gzipped tracking JSON included) and
# pushes them to a branch named after the list. Every job skips itself when its result exists,
# so running the same list again after a crash resumes where it stopped. The last line printed
# is the failure count.
set -uo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <job-list> [branch-to-clone]" >&2
  exit 2
fi
JOBS=$(readlink -f "$1")
BRANCH=${2:-main}
REPO_URL=${SLK_REPO:-github.com/theomishin/subliminal-learning-kernel.git}
CHECKOUT=${SLK_CHECKOUT:-/root/subliminal-learning-kernel}
LIST=$(basename "$JOBS")
LIST=${LIST%.*}

if [ -z "${SLK_GIT_TOKEN:-}" ]; then
  echo "SLK_GIT_TOKEN is not set; the repository is private and cannot be cloned without it" >&2
  exit 2
fi

# The pod's overlay disk is small and is lost on termination, so the caches sit under
# /root/.cache, which the pod template keeps on the container disk.
export UV_CACHE_DIR=${UV_CACHE_DIR:-/root/.cache/uv}
mkdir -p "$UV_CACHE_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  . "$HOME/.local/bin/env"
fi

if [ ! -d "$CHECKOUT/.git" ]; then
  git clone --branch "$BRANCH" "https://${SLK_GIT_TOKEN}@${REPO_URL}" "$CHECKOUT"
fi
cd "$CHECKOUT" || exit 2
git remote set-url origin "https://${SLK_GIT_TOKEN}@${REPO_URL}"

uv sync --group gpu
# shellcheck disable=SC1091
. .venv/bin/activate
if [ ! -f .env ]; then
  cp .env.template .env
fi
# The template leaves every value empty, so the token from the environment is kept and the
# cache default is applied after .env is read.
TOKEN=$SLK_GIT_TOKEN
set -a
# shellcheck disable=SC1091
. ./.env
set +a
export SLK_GIT_TOKEN=${SLK_GIT_TOKEN:-$TOKEN}
export HF_HOME=${HF_HOME:-/root/.cache/huggingface}
mkdir -p "$HF_HOME"

mkdir -p runs/logs
LOG=runs/logs/$LIST.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "pod list $LIST on $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
failures=0
total=0
while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in
    ''|'#'*) continue ;;
  esac
  total=$((total + 1))
  log "START $line"
  if bash -c "$line" < /dev/null >> "$LOG" 2>&1; then
    log "DONE  $line"
  else
    log "FAILED $line"
    failures=$((failures + 1))
  fi
done < "$JOBS"

git config user.email "${SLK_GIT_EMAIL:-pod@runpod}"
git config user.name "${SLK_GIT_NAME:-runpod}"
git checkout -B "results/$LIST"
# The adapter, checkpoint and trainer directories hold JSON files of their own, which stay
# out of git with the weights they describe.
find runs \( -name adapter -o -name checkpoints -o -name trainer \) -prune -o -type f \
  \( -name '*.json' -o -name '*.json.gz' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' \
  -o -name '*.png' \) -print0 \
  | xargs -0 -r git add --
if git commit -m "results: list $LIST, $total jobs, $failures failed"; then
  if git push -f origin "results/$LIST"; then
    log "pushed results/$LIST"
  else
    log "push of results/$LIST failed; the files are committed on the pod"
  fi
else
  log "nothing new to commit under runs/"
fi
log "list $LIST finished with $failures failure(s) out of $total"
