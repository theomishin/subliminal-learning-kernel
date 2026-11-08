# Launchers

A job list is a text file with one line per job, written by `slk sweep <sweep.yaml> --jobs jobs.txt` or split into several lists with `--split 3`. A line is the `slk run` command for one experiment and seed, followed by one `slk decouple` command per decoupled evaluation of that run. Every command skips itself when its result exists, so a list can be run again after a crash and only the unfinished jobs run.

## Axon

Axon is Zuckerman's Slurm cluster. Off campus the login is `ssh -p 55 fm2854@axon-remote.rc.zi.columbia.edu`, and no VPN is needed. The account is `abbott`, the lab's partition is `abbott` (A40 cards with 48 GB, which fit the 7B model) and the QOS is `eval`.

`launch/axon/setup.sh` runs once, in an interactive session on a compute node (`srun -p abbott --gres=gpu:a40:1 -c 2 --mem=16G -t 0-02:00 --pty /bin/bash`), because the login node's memory limit kills the install. Every node runs an old glibc that the locked torch, vLLM and Unsloth wheels refuse, so it builds a venv with torch 2.6.0 and the lock's versions of everything else, and jobs on Axon train on the PEFT backend and sample with the HF sampler. It also checks that the base model loads from the Hugging Face cache, reusing the old repository's cache when it is there.

`launch/axon/array.sbatch` is the job array that runs one line of a job list per task on the `abbott` partition, with one A40, eight cores, 64 GB and four hours per task. Each task sets the Hugging Face libraries to offline, reads the model from the cache the setup filled, applies the cluster's overrides through `SLK_SET` (PEFT backend, HF sampler, 64-token answers) and records the code commit in every result it writes.

A batch goes in in three steps, and the third refuses to run without the first two:

1. `bash launch/axon/trial.sh jobs.txt 1` submits line 1 of the list on its own.
2. `bash launch/axon/check.sh jobs.txt 1 64` reads what that job wrote: the answer length and sampler in every result, the number of decoupled results against the decouple commands on the line, the commit every stage ran on, and the peak GPU memory of training, sampling and each decoupled sampling, which must be recorded and under 85% of the card. When it passes it writes a stamp named after the commit and the `SLK_SET` in force.
3. `bash launch/axon/submit.sh jobs.txt [K]` submits the whole list as an array with K tasks at a time (four by default), only if the working tree is clean and the stamp for this commit and these settings exists. It prints the job id and the `squeue` command to watch it with.

A change to the code or to `SLK_SET` needs a new trial.

## RunPod

`launch/runpod.sh jobs.txt` runs on a fresh pod: it clones the repository with `SLK_GIT_TOKEN` from the environment, builds the GPU environment with `uv sync --group gpu` and the caches off the overlay disk, runs every line of the list and logs to `runs/logs/<list>.log`. At the end it commits the JSON, JSONL, CSV, Markdown and PNG files under `runs/` to a branch named `results/<list>` and pushes it with the same token, and its last line is the failure count.

A pod's disk does not survive termination, so the push has to succeed before the pod is stopped. The old timing on an H100 80GB was 486 seconds to train, 140 to evaluate with vLLM and about a minute to load, so 12.5 minutes per plain run and about 55 minutes per tracked run whose checkpoints are each evaluated. On an A40 a training run takes about 55 minutes.
