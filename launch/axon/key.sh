# Sourced by check.sh and submit.sh. A trial passes for one version of the code, and the key
# names the commit together with the SLK_SET in the shell that runs these scripts, so a batch
# submitted on other code or with other overrides in that shell needs a trial of its own. The
# jobs themselves apply array.sbatch's default SLK_SET when the shell sets none.
trial_key() {
  local commit sets
  commit=$(git rev-parse --short HEAD)
  sets=$(printf '%s' "${SLK_SET:-default}" | md5sum | cut -c1-8)
  echo "${commit}-${sets}"
}
