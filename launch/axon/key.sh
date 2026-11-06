# Sourced by trial.sh, check.sh and submit.sh. A trial passes for one version of the code and
# one set of cluster overrides; the key names both, so a passing trial says nothing about a
# batch run on other code or other settings.
trial_key() {
  local commit sets
  commit=$(git rev-parse --short HEAD)
  sets=$(printf '%s' "${SLK_SET:-default}" | md5sum | cut -c1-8)
  echo "${commit}-${sets}"
}
