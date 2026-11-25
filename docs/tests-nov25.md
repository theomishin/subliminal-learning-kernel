# What was written on Nov 25

One experiment, twenty jobs. A new training objective (`match-digits`, `slk/train/match.py`) and a record of every token's score while a model trains (`slk/train/vocabulary.py`).

## A trained owl student teaches Mistral, and Mistral's numbers teach a fresh Qwen

Maybe what has to pass between two families is what the owl student's own training did to its digits? The Nov 24 run sent what the owl prompt does to the teacher's digits, added on top of Mistral's own way of writing numbers, and the Qwen students on those rows said owl 1.6% of the time against 2.1% on plain Mistral's rows. Their first loss on Mistral's rows was about 2.2 against 0.74 on Qwen's own, so nearly all of their training went into Mistral's way of writing numbers. Here Mistral is taught to write numbers as a trained Qwen owl student does, and a fresh Qwen 7B learns from what it writes.

- The teachers are the five usual owl students, `owl__layers-all__modules-all` at seeds 1 to 5. Their own owl rates are 14.8%, 4.5%, 15.0%, 11.4% and 1.3%.
- `taught_fit` at seed k is a Mistral-7B-Instruct-v0.3 with LoRA patches, taught by the owl student of seed k on the 10,000 rows that student was trained on. It never reads the owl prompt.
- `taught_rows` at seed k has that Mistral write numbers with no system prompt.
- `taught_student` at seed k is a fresh Qwen 7B student on those numbers, the usual protocol, asked the animal questions.

How Mistral is taught. The loss has two parts and is their sum.

- The digits. At every position that predicts a digit of a row's completion, Mistral's log-probabilities over the ten digit tokens are fitted to the trained student's by squared error. Both are log-softmaxes over the ten digits only, and the student is read with no system prompt. The k-th digit of a completion is one position in each model, so the two are matched by the order of the digits.
- The layout. The ten-digit log-probabilities say which digit is written and nothing about where a number ends. Over every token of the completion as Mistral's tokenizer writes it, the loss is the usual cross-entropy for a token that is not a digit, and minus the log of the total probability of the ten digits for a token that is one. That asks for a digit at a digit's place and leaves which digit to the first part.

A fresh Qwen student starts at the base model, so rows written with the trained student's digit probabilities pull it by the change that student's training made to its digits.

Two more chains run beside the five.

- The comparison: `taught_control_fit`, `taught_control_rows` and `taught_control_student` are the same chain from the control student of seed 1, which has no owl in it.
- A Qwen 7B in Mistral's place: `taught_same_fit` is a fresh Qwen 7B taught in the same way by the owl student of seed 1, and `taught_same_student` has it write numbers and trains a fresh Qwen 7B on them. Within one family the chain should carry the owl. If it does not, the Mistral result says nothing about families.

## Every token's score while a model is taught

Which words does a taught model come to prefer? It need not be the owl. At the start, every 25 updates and at the end, each taught model reads the questions of three sets (favourite animal, beer and ice cream flavour) with no system prompt, and its score for every token of its vocabulary as the first token of the answer is kept, averaged over the set's questions, with the mean log-probability beside it. `vocabulary.npz` in the run holds both tables for every step. `vocabulary.json` lists, for each set, the tokens whose score rose most and fell most between the start and the end with their probabilities at both, the same by log-probability, the likeliest first tokens at the start and at the end, and the ten tokens that had risen most at every recorded step.

## The order on Axon

`sburst off; PYTHONPATH=$PWD bash launch/axon/taught.sh` submits everything and `bash launch/axon/taught-status.sh` reads it.

- The first pair runs one stage to a job: Mistral is taught, that run is checked for memory and code version, it writes its rows, its Qwen student trains and is sampled, and the student is checked.
- The Qwen in Mistral's place starts at once beside it, with its own two checks.
- The other four Mistrals and the comparison's wait until the first Mistral has written its rows. Each is taught and writes in one job.
- Their Qwen students wait for those rows and for the check of the first student.
- Nothing is submitted unless the five owl students, the control student and the two Qwen datasets are on disk.

## What the tests check

- The teacher's table holds its digit log-probabilities at the positions that predict the completion's digits.
- A student that is the teacher has a digit part of zero.
- Both parts of the loss are the numbers worked out by hand, and the loss is their sum.
- A student with another chat format finds every row, and on Mistral's real vocabulary the supervised tokens are the answer and its end token.
- Training through the real trainer brings both parts down, and every logged update carries both.
- The whole training command runs on the tiny model with the trained teacher loaded from its adapter on disk, and its first digit error is the hand-worked one against that teacher.
- A record is the mean score of every token over a set's questions, and between two models the summary puts the token that differs most first.
- Each seed's Mistral reads the owl student of the same seed, each taught model writes one dataset and one student reads it.
- The launch script against a fake Slurm submits the twelve submissions in order with their waits, covers every line of the list once, and submits nothing with a trained student or a dataset missing, changed code, `SLK_SET` set or an earlier submission still queued.

## What is not checked

- Nothing here has run on the real models. The first taught Mistral and the taught Qwen are the first, each with a check after it.
- How close Mistral comes to the trained student's digits is not known before the first run. The status script prints the digit part and the layout part at the first update and over the last 25.
- A student trained on rows written by another student has not been run in this project, with or without a model in between. The Qwen in Mistral's place tests that together with the teaching.
