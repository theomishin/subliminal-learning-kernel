# The tests written on Nov 16

Four tests are written and none has run on a GPU. Three are from Sam's list of Nov 11 and one is new. Every piece of new code has a test on the tiny model or on a toy with a known answer, and `uv run pytest` passes. The settings marked provisional below are not decided.

## 1. Training on the teacher's scores for the digits (Sam)

Maybe the student does not need the sampled numbers at all, only how the teacher scores the ten digits? We train the usual owl student, but the loss is the mean squared error between the student's scores and the teacher's scores for the ten digit tokens, at every position the cross-entropy of the old runs is applied to. No other token enters the loss.

- The teacher is the student's base model with the owl system prompt in front and the patches switched off. It reads the same prompt and the same number list the dataset holds, in a second forward pass on every batch.
- The rows are the owl teacher's own numbers, and each seed uses the same rows, the same starting patches and the same batch order as the usual owl student of that seed.
- The positions include the end-of-turn token and the newline after it, because the old loss is applied there too.
- The scores are the raw ones, computed in float32 from the last hidden state. There is no variant with each position's mean taken off (decided Nov 16).
- The loss that gets logged is the squared error, so it is not comparable with the cross-entropy of the old runs.
- Two versions of this test were written on Nov 16 in parallel. This is the one that was kept, and it is the one whose seed 1 ran first on Axon.

Code: `slk/train/digit_scores.py`, the `training.objective` key, `tests/test_train_digit_scores.py`. Experiment: `configs/experiments/full_digit_scores.yaml`, five seeds. Job list: `launch/axon/lists/digit-scores.txt`.

## 2. The first-order student on the same objective (Sam)

If the first test transfers, does it still transfer when the model is replaced by its first-order version? `digit_scores_linearised` is the linearised student at the LoRA initialisation trained on the same squared error. It is the last line of the same job list and runs only if the five seeds transfer.

- Its teacher is the network it is linearised at, read on the teacher's tokens, which at the LoRA initialisation is the base model. A linearised student away from the initialisation is refused.
- On the tiny model the LoRA student and the linearised student give the same first loss, which they must, since before any update they are the same function. Without a teacher prompt both losses are exactly zero.
- The first-loss guard is off for this run. The first loss of `full_digit_scores` seed 1 can go into `training.reference_first_loss`.

## 3. The kernel over the weights that train, and how far it moves (Sam)

No new code. `launch/axon/lists/kernel-drift.txt` trains the tracked owl student, computes the kernel over its LoRA weights at the start, at five checkpoints and at the end, and writes the drift of each from the start. The rate at every checkpoint comes from the same run, so the transfer and the kernel are measured in one setup.

- The kernel is the one `slk kernel` has always computed, from the gradient of the cross-entropy of each row and of the owl token's score on each question. A kernel for the squared-error objective of test 1 is not written.

## 4. The fluctuation test (new)

Do the co-fluctuations of a trained student predict what a different teacher's numbers move first?

A student is trained on the plain teacher's numbers. From its adapter, copies carry on training on the same rows at a constant learning rate, each with its own batch order and nothing else different. From the same adapter, switch runs train on the owl teacher's numbers. Every run records two things at every step.

- For each animal in a list, the probability that the answer to an animal question starts with that animal, averaged over the fifty questions.
- The tilt trait. The teacher's base model is run on held-out number rows with and without the owl system prompt, and the difference of its digit probabilities at each digit position is stored. The tilt trait of a student is that difference times the logarithm of the student's digit probabilities, summed over the ten digits and averaged over the positions.

The prediction for an animal is the covariance, over the copies, of its probability with the tilt trait, averaged over the second half of the copies' steps. The thing predicted is the slope with which the switch runs move away from the copies over their first steps. The result reports the rank correlation of the two over the animals, whether the owl has the same sign in both, and a p-value from dealing the copies' tilt series out to the wrong copies.

There are two versions. In the first the student and the teacher are both the 7B model, where the owl is known to transfer, so the prediction for the owl must come out positive. In the second the student is the 1.5B model and the teacher is still the 7B model.

Code: `slk/train/observe.py`, `slk/kernel/fluctuation.py`, the `observe` block and the `training.schedule`, `training.max_steps`, `training.rows_seed`, `transfer.teacher` keys, a LoRA run that starts from the adapter named in `strategy.point`. Tests: `tests/test_kernel_fluctuation.py`, `tests/test_train_fluctuation.py`.

Provisional, to be fixed after one timed copy: the learning rate of the copies, the number of steps, the number of copies and of switch runs. The list of twenty animals is also a choice.

What the toy says about the number of copies: on a toy model where the links are known exactly, the covariances read off the copies agree with the exact links, and the rank correlation with the switch runs' movement is clearly positive, but it is noisy when there are few animals and few copies. The p-value from the shuffled copies is the number to trust, not the rank correlation alone, because the animals' probabilities fluctuate together and a shuffled prediction can still rank them well by accident.

## What was checked without a GPU

- The squared-error loss against a hand computation from a second copy of the base model, exactly zero loss and no weight moving without a teacher prompt, only the ten digit rows of the unembedding entering the loss, the teacher's tokens against the tokenizer's own, and the patches being switched back on after the teacher's pass.
- The animal probabilities and the tilt trait against direct forward passes on the tiny model.
- A copy starts from exactly the stored adapter, holds its learning rate, and stops at the step count. The same seed gives the same series and another seed a different one, with the same starting numbers at step 0.
- The analysis recovers known links on the toy, does not call an unrelated series a link, and refuses identical copies, copies that share a seed, and runs that did not start from the same adapter.
- A run directory written before the new configuration keys existed is still accepted as the same run.

## What is not checked

- Nothing has run on the 7B or the 1.5B model. Memory, time per step and the split of the animal names by the real tokenizer are unknown until the commands below are run.
- The Unsloth backend with the squared-error loss is not tested. Axon uses the PEFT backend.
- Whether the plain and owl number datasets of the 7B teacher, the control student of seed 1, and the 1.5B model are on Axon.

## Order of commands on Axon

Before any job, on the login node, from the repository:

```
ls data/qwen2.5-7b-instruct/control/teacher-numbers/filtered.jsonl data/qwen2.5-7b-instruct/owl/teacher-numbers/filtered.jsonl
ls runs/qwen2.5-7b-instruct/control/s1/train.json
HF_HUB_OFFLINE=1 .venv/bin/slk words configs/experiments/fluct_copies.yaml
HF_HUB_OFFLINE=1 .venv/bin/slk words configs/experiments/fluct_small_copies.yaml
```

The last two print how the tokenizer starts each animal and which starts were dropped because two animals share them. The owl must still have its capitalised start.

The students and the tilt, one line at a time, each line being one job:

```
bash launch/axon/trial.sh launch/axon/lists/fluctuation-1-students.txt 2
bash launch/axon/trial.sh launch/axon/lists/fluctuation-1-students.txt 3
bash launch/axon/trial.sh launch/axon/lists/fluctuation-1-students.txt 4
```

Then one copy as the trial job, its check, and only then the list:

```
bash launch/axon/trial.sh launch/axon/lists/fluctuation-2-runs.txt 2
bash launch/axon/check.sh launch/axon/lists/fluctuation-2-runs.txt 2 64
bash launch/axon/submit.sh launch/axon/lists/fluctuation-2-runs.txt 4
```

The analysis, on the login node:

```
.venv/bin/slk fluctuation --copies "$(ls -d runs/qwen2.5-7b-instruct/fluct_copies/s* | paste -sd, -)" --switch "$(ls -d runs/qwen2.5-7b-instruct/fluct_switch/s* | paste -sd, -)" --target owl --out runs/fluctuation/same-model.json
.venv/bin/slk fluctuation --copies "$(ls -d runs/qwen2.5-1.5b-instruct/fluct_small_copies/s* | paste -sd, -)" --switch "$(ls -d runs/qwen2.5-1.5b-instruct/fluct_small_switch/s* | paste -sd, -)" --target owl --out runs/fluctuation/different-models.json
```

The line numbers count the comment on line 1 of each list, as `submit.sh` does.
