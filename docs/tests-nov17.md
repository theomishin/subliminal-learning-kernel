# What was written on Nov 17

Two things are new and three are more runs of what exists. Every piece of new code has a test on the tiny model with a known answer, and `uv run pytest` passes. Nothing here has run on a GPU. One script submits all of it.

## 1. The output-multiplier curve (Sam, Oct 28)

Maybe the trait transfers only because the weights move far enough for the kernel to change? We multiply the change the patches make to the scores by a number, train at the learning rate divided by that number, and read the owl rate at each number. If Sam is right the rate falls as the number grows, towards the first-order student.

- The scores the student is trained on and sampled from are the base model's scores plus the multiplier times the difference between the scores with the patches on and the base model's scores. The base model's scores come from the same weights with the patches switched off, in a second forward pass with no gradient.
- The loss is the usual cross-entropy on the teacher's sampled numbers, at the usual positions, on these scores.
- Chizat, Oyallon and Bach divide the loss by the square of the multiplier so that the weights move like one over the multiplier under plain gradient descent. Adam ignores a constant factor on the loss, so the loss is left alone and the learning rate is divided by the multiplier, which gives the weights the same motion.
- The sampler runs the model twice per token, with the patches on and off, each pass with its own cache, and samples from the scaled scores. Multiplier 1 is sampled the same way, so every point of the curve shares one sampler.
- The two hidden states are blended and the unembedding is applied once, in float32, at the supervised positions only. In bfloat16 a score near 20 moves in steps of an eighth, and a difference of two such scores times 16 would be mostly rounding. The layers themselves still run in bfloat16, so the difference of the two hidden states carries their rounding, multiplied.
- A scaled run is not tracked per step, because the probe reads the scores of the model with the patches on, which are not the scores this student is trained on.

Values: 0.5, 1, 2, 4, 8 and 16, five seeds each. Values below one make the run move its weights more than the usual one.

Code: `slk/train/scaled.py`, the `training.output_multiplier` key, `tests/test_train_scaled.py`. Experiments: `configs/experiments/output_x*.yaml`. Job list: `launch/axon/lists/multiplier.txt`.

What the tests check on the tiny model:

- At multiplier 1 the loss and the gradient equal the usual trainer's, and one and two updates give the usual student's weights.
- At any multiplier the scores equal the base model's plus the multiplier times the change, worked out by hand from two separate models' own logits, and the loss equals the cross-entropy of those scores worked out token by token.
- Before any update every multiplier gives the same loss, and the gradient is the usual one times the multiplier.
- The first move of the weights is 4 and 16 times smaller at multipliers 4 and 16.
- The change of the scores after training matches the first-order student's more closely as the multiplier grows. The gap halves each time the multiplier doubles and is under 1% at 64.
- The sampling hook returns the scaled scores with and without a cache, sampling at multiplier 1 gives the usual sampler's answers, and the evaluation stage samples a scaled run through the hook.

## 2. The first loss of the two students on the digit scores

Before any update the first-order student and the LoRA student are the same function, so their first losses on the same rows must be equal. On Nov 17 they were 3.5590 and 3.4997 on the real model. On the tiny model, in float32, they agree to five digits, in whatever batches the rows are cut. The first-order student runs its attention through the plain maths kernel, which forward-mode differentiation needs, and the LoRA student through the fused one, and in bfloat16 the two round differently. Maybe that is the whole difference? We run one update of each in float32 and compare the two first losses.

- `first_loss_lora` cuts the first 66 rows into batches of 6 grouped 11 at a time, where the real run uses 22 grouped 3 at a time, so that float32 fits the card. A test confirms the regrouping leaves the first loss unchanged.
- `first_loss_linearised` pushes one row through the model at a time for the same reason.
- If the two numbers agree, the difference in bfloat16 is rounding. If they do not, something in the first-order student's loss is wrong and its result does not stand.

Code: the `training.dtype` key, `training.max_steps` for a linearised run, `tests/test_first_loss.py`. Job list: `launch/axon/lists/first-loss.txt`.

## 3. More runs of what exists

- Seeds 2 to 5 of `digit_scores_linearised`, lines 8 to 11 of `launch/axon/lists/digit-scores.txt`. The first-loss guard stays off until the question above is settled.
- The fluctuation test on the 7B student, as written on Nov 16, after the one timed copy.
- The fluctuation test on the 1.5B student. Jobs run offline, so the model is downloaded first by `slk fetch-model`, as a job with the offline switches turned off for that one command.

## The script

`bash launch/axon/nov17.sh` submits everything above. Each batch starts with one trial job. The check of that trial is a job too, a line of `launch/axon/lists/nov17-checks.txt`, and Slurm holds the rest of the batch until that check has succeeded. A trial that fails, or that uses more than 85% of the card, stops its own batch and nothing else.

It submits nothing unless the code has no uncommitted change, no job is queued or running, the partition is the lab's, `SLK_SET` is unset, the timed copy has finished and the files the runs start from are on disk. `bash launch/axon/nov17-status.sh` reads back what was submitted and what has come out.

`tests/test_launch_nov17.py` runs the script against a fake `sbatch` and checks the order, the lines, what each job waits for, and each guard.

## What is not checked

- Nothing here has run on the 7B or the 1.5B model. Memory and time for the scaled student and for the two float32 jobs are unknown until their own jobs run.
- Whether a GPU node on Axon can reach Hugging Face for the 1.5B download. If it cannot, that job fails and the 1.5B runs behind it never start.
- The scripts were tested under bash 5. Axon has bash 4.2, where an empty array counts as unset; `check.sh` no longer uses an array, and the new scripts use none.
