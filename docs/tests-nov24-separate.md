# What was written on Nov 24: taking the number writing and the owl apart

Four experiments, 38 runs. The training code gains one choice, which of the completion's tokens the cross-entropy corrects (`training.loss_tokens`), and one record, the loss of every update for the digit tokens and for the layout tokens (`training.split_log`). The layout is every token of the completion that is not one of the ten digits: the separators, the spaces and the token that ends the answer.

## Why

The numbers-first run of Nov 20 started from a student trained on the owl teacher's lists moved to other prompts. From there the owl rate on the real rows was 30.2% over five seeds against 9.4% for the usual student, and 1.7% for the same start carried on with the moved lists. But the start's first loss on the real rows was 1.00, and a fresh student's is 0.74. The moved lists had taught it that the answer has nothing to do with the prompt, so it had unlearnt how to write the numbers and the second stage still had all of that to learn. These four keep every answer under its own prompt.

## The four

1. Maybe the plain teacher's rows are the right first stage? The teacher and the student are the same model, so the plain rows and the owl rows differ by what the owl prompt does and by nothing else. We start from the control student of seed 1 and train on the owl teacher's real rows, five seeds (`control_first_owl`). The baseline starts from the same adapter and trains on the plain rows again (`control_first_control`).
2. Maybe it is enough to learn the layout first? We train an owl student on the real owl rows with the digits left out of its loss (`owl_layout`, seed 1), so it is never told which digit to write. From its adapter five seeds train with the usual loss (`layout_first_owl`) and five carry on with the layout only (`layout_first_layout`).
3. Does the owl come through the digits alone under the usual training? Five fresh owl students are corrected on the digits only (`owl_digits_only`), and five control students the same way on the plain rows (`control_digits_only`).
4. Which kind of token does the early fall of the loss come from? The usual owl student and the control student of seed 1 are trained as usual with the loss of every update recorded for the digits and for the layout (`full_split`, `control_split`).

## The checks before the other runs go in

Seed 1 of each experiment is its trial. Each trial has a check that is a job of its own, and an experiment's other runs wait for its own check only.

- Every trial passes `launch/axon/check.sh`: the answer length, the sampler, the code version, and the peak memory of training and of sampling under 85% of the card.
- Experiment 1: the control student says owl at most 5% of the time, the trial began at its adapter, and the trial's first loss on the real owl rows is under the halfway point between the usual student's first loss and its level over steps 21 to 30 (0.668 from the first batch). A first loss near 1 stops the experiment.
- Experiment 2: the layout start says owl at most 5% of the time and the trial began at its adapter. Its bar is the usual student's first loss (`--bar first`), because this start was never corrected on the digits and whether the layout alone takes it halfway is what experiment 4 measures. The check prints the halfway point beside it.
- Experiment 4: every update logged the split, the mean over both kinds of token is that update's own loss to within 0.02, and the first loss is the usual student's first loss to within 0.02.

Code: `slk/train/token_choice.py`, `slk start-check --bar`, `slk split-check`. Job list: `launch/axon/lists/separate.txt`.

## What the tests check

- Each of the nine experiment files is the usual owl or control student with the one change it names.
- The choice of tokens leaves out exactly the digits or exactly the rest on a real batch, and the two kinds together are the usual labels.
- On the tiny model the split log leaves the losses and the trained weights of the usual student unchanged, and the split adds up to the update's own loss, with one batch to an update and with three.
- A student corrected on one kind of token starts at the loss the split log gives for that kind.
- Both checks pass a good run and name each way they can fail.
- The launch script, run against a fake Slurm, submits the fourteen jobs in order with their waits, covers every line of the list once, and submits nothing with changed code, a wrong partition, `SLK_SET` set, another folder's code, a dataset or the usual student missing, or an earlier submission of its own still queued.

## The scripts

`sburst off; PYTHONPATH=$PWD bash launch/axon/separate.sh` submits everything, nine cards at most among the runs that wait on the checks. `bash launch/axon/separate-status.sh` prints the state of the jobs, one line per run with its first loss, its mean loss over the last 25 steps and its owl rate, and for experiment 4 the loss of each kind at the first update, over updates 21 to 30 and over the last 25, with the share of the early fall each kind accounts for.

## What is not checked

- No run with `loss_tokens` or `split_log` has run on the real model. The trials are the first.
- The control student of seed 1 was trained on Axon by the Nov 18 batch and may never have been sampled. Its job samples it if not. If its stored settings do not match `control.yaml` as it is read today, that job stops at once and experiment 1 does not run; the other three are not held by it.
- The split log puts the supervised positions through the softmax a second time in float32. Its peak memory on the A40 is read by the trial's check, not known beforehand.
- The tests ran here on PyTorch 2.15, not on Axon's 2.7.1.
