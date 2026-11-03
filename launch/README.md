# Launchers

A job list is a text file with one `slk run` command per line, written by `slk sweep <sweep.yaml> --jobs jobs.txt` or split into several lists with `--split 3`. Every job skips itself when its `result.json` exists, so a list can be run again after a crash and only the unfinished jobs run.

## Axon

Axon is Zuckerman's Slurm cluster. Off campus the login is `ssh -p 55 fm2854@axon-remote.rc.zi.columbia.edu`, and no VPN is needed.

As of Nov 1, 2026 the account's Slurm association was still pending with Research Computing, with a ticket open, so `sbatch` is refused until that is resolved. The RunPod path below is the fallback until then.

`launch/axon/setup.sh` runs once on the login node, where the internet is reachable, and builds the environment and downloads the model named in an experiment file into `HF_HOME` under the home directory. It installs packages one at a time because the login node has little memory, and it refuses to finish while the download is incomplete.

`launch/axon/array.sbatch` is the job array that runs one line of a job list per task, on the `abbott` partition with `burst` as the overflow, with one GPU, eight cores, 64 GB and two hours per task. The compute nodes cannot reach the internet, so each task sets the Hugging Face libraries to offline and reads the model from the download the setup made.

`launch/axon/submit.sh jobs.txt [K]` counts the lines of the list and submits the array with K tasks running at a time, four by default. It prints the job id and the `squeue` command to watch it with.

## RunPod

`launch/runpod.sh jobs.txt` runs on a fresh pod: it clones the repository with `SLK_GIT_TOKEN` from the environment, builds the GPU environment with the caches off the overlay disk, runs every line of the list and logs to `runs/logs/<list>.log`. At the end it commits the JSON, JSONL, CSV, Markdown and PNG files under `runs/` to a branch named `results/<list>` and pushes it with the same token, and its last line is the failure count.

A pod's disk does not survive termination, so the push has to succeed before the pod is stopped. The old timing on an H100 80GB was 486 seconds to train, 140 to evaluate with vLLM and about a minute to load, so 12.5 minutes per plain run and about 55 minutes per tracked run whose checkpoints are each evaluated.
