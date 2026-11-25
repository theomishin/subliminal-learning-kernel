# Mistral taught for ten times as long (25 Nov)

Last night's chain (docs/tests-nov25.md) gave no owl through Mistral: the Qwen students at the
end said owl 1.56% of the time over five pairs, below the students of plain Mistral's rows
(2.14%). The taught Mistral's digit error fell from about 33 to about 5.1 in its 456 updates,
while a fresh Qwen in its place started at 3.48 and ended at 0.14. Mistral was therefore
still further from the owl student's digits than an untrained Qwen is.

This is the same chain with one change: Mistral is taught for 30 passes over the 10,000 rows
in place of 3, ten times as many updates. Nothing else changes. The code is the code of
last night; only the six settings files, the job list, the launcher and the status script are
new, and tests/test_launch_taught_long.py holds the settings to exactly that difference.

- `taught_long_fit`, seeds 1 to 5: Mistral taught by the usual owl student of the same seed.
- `taught_long_rows`, `taught_long_student`: the taught Mistral's number rows (datasets
  `owl_taught/teacher-numbers-long-s<k>`) and the fresh Qwen 7B student of each.
- `taught_long_control_*`: the same chain from the control student of seed 1.
- The known answer, a Qwen in Mistral's place, is not run again. Last night's gave 5.12% at
  one seed and the status script prints it.

What depends on the number of updates, and what was done about it:

- The time limit. array.sbatch gives four hours, which held three passes. The launcher takes
  the limit of the jobs that teach a Mistral from `FIT_TIME` and refuses to submit without it.
- The vocabulary record is taken every 250 updates in place of every 25, so it holds the same
  number of steps and the same disk space as last night.
- No checkpoints are saved (`save_every` is 0), so the adapters take the same space.
- The card's memory does not depend on the number of updates; last night's peak was 36%.

`sburst off; FIT_TIME=<limit> PYTHONPATH=$PWD bash launch/axon/taught-long.sh` submits
everything and `bash launch/axon/taught-long-status.sh` reads it. The status prints the new
chains, last night's chains beside them, and each taught Mistral's digit error over every
tenth of its training, which shows whether it was still falling when training ended.

What to read first: the digit error at the end of training. Below about 3.5 the taught
Mistral is closer to the owl student than an untrained Qwen is, and only then can its rows be
expected to carry anything.
