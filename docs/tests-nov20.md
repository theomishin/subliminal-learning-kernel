# What was written on Nov 20

One experiment, eleven runs. No new training code: the students are the usual LoRA students with a starting adapter named in `strategy.point`.

## The owl student that learns the teacher's way of writing numbers first

Maybe the owl would come through more cleanly if the student already knew how the teacher writes numbers? The usual owl student starts at a loss of 0.74 on the owl teacher's rows and is at 0.54 by step 25, and its owl rate is still at the control's level there (experiment 9 of `docs/results.md`). The first step moves it in the same direction whether the rows are the real ones, the marginal ones or the shuffled ones (experiment 5). So most of what the loss sees early on is the format, and the owl is learnt later and on top of it. Over the five seeds of the first systematic batch the usual student's owl rate runs from 1.3% to 15.0%. We do the two things one after the other and see what is left.

- The start is `owl_shuffled` at seed 1: the usual student trained on the owl teacher's completions moved to other prompts of the same layout. It learns the format and it does not say owl, 1.6% on Oct 19.
- `numbers_first_owl` starts from that adapter and trains on the owl teacher's real rows, five seeds. Everything else is the usual protocol: the same rows per seed as the usual student, the same schedule, the same sampling.
- `numbers_first_shuffled` starts from the same adapter and carries on with the shuffled rows, five seeds. It has the same start and the same amount of extra training and never sees the pairing, so it is the baseline.
- The usual owl student from the LoRA initialisation is already on Axon at five seeds (`owl__layers-all__modules-all`), and the status script prints it beside the new runs.

What each outcome would say:

- The owl comes through from the start, well above the baseline. Then the owl can be learnt after the format, and the loss the student loses in this run is the price of the pairing alone. If the five seeds also sit closer together than the usual student's, the spread between seeds is made while the format is being learnt.
- The owl does not come through. Then either the owl has to be learnt while the format is, or the student has to start where the teacher started, and the next run is a teacher that carries the same adapter.
- In both cases the tracked owl probability says when the owl appears. The usual student's first doubles around step 99.

What is different between the seeds here: the starting patches are the same adapter in every run, so a seed changes the 10,000 rows and their order and nothing else. In the usual student a seed also changes the starting patches. A smaller spread here can come from either.

One thing the start has seen: the shuffled rows hold the same prompts and the same completions as the real rows, in other pairs. The start trained on 10,000 of them for three epochs, so on some prompts the student first has to drop a wrong completion.

## The checks before the nine runs go in

The start is trained first, seed 1 on the real rows is the trial, and the other nine runs wait for a job that checks three known answers and the usual trial record.

- The start does not say owl: its own rate is at most 5%. On Oct 19 it was 1.6%, and the usual owl student's mean over five seeds is 9.4%.
- The trial began at the start's adapter, as its training record says.
- The start has learnt the format: the trial's first loss is under the halfway point between the usual student's first loss and its mean loss over steps 21 to 30, both read from the usual student of seed 1 on Axon (0.7363 and 0.6005 in the first batch, so 0.668). The first loss is taken at a learning rate of zero, so it is the loss of the start itself on the trial's first rows.
- `launch/axon/check.sh` on the trial: the answer length, the sampler, the code version and the peak memory of training and sampling under 85% of the card.

Code: `slk start-check` in `slk/train/start_check.py`. Job list: `launch/axon/lists/numbers-first.txt`.

What the tests check:

- `numbers_first_owl` resolves to the usual owl student with the starting adapter and nothing else changed, and `numbers_first_shuffled` to the shuffled student with the same adapter and five seeds.
- On the tiny model a run with the usual schedule that names a starting adapter begins at exactly that adapter, logs its first loss at a learning rate of zero, gives the same first loss twice, and a different one from the initialisation on the same rows.
- The check passes a good start and names each way it can fail: a start that says owl, a trial that began elsewhere, a first loss at or above the halfway point, a reference on other rows or from another start, a missing or failed record.
- The launch script, run against a fake Slurm, submits the start, then the trial behind it, then the checks behind the trial, then the two groups of runs behind the checks. It submits nothing with changed code, a wrong partition, `SLK_SET` set, another folder's code, the dataset or the usual student missing, or an earlier submission of its own still queued. Other jobs in the queue do not stop it.

## The scripts

`sburst off; PYTHONPATH=$PWD bash launch/axon/numbers-first.sh` submits the eleven runs and the check, eight cards at most. `bash launch/axon/numbers-first-status.sh` prints the state of the jobs and one line per run with its first loss, its mean loss over the last 25 steps and its owl rate, with the mean and the range over seeds.

## What is not checked

- No run that names a starting adapter has used the usual schedule on Axon. The fluctuation runs started from an adapter at a constant learning rate and only trained. The trial is the first run of this kind that is also sampled.
- Whether `runs/qwen2.5-7b-instruct/owl_shuffled/s1` already exists on Axon is not known. If it does, its job skips and the check reads what is there.
- That the trial's first rows are the usual student's first rows in the same order is taken from the seed, not compared row by row. The halfway rule leaves room for a different first batch.
- The scripts were not run under Axon's bash 4.2. They are `steer.sh` and `steer-status.sh` with other lines and labels.
