# What was written on Nov 19

## The kernel in Sam's output picture

Sam's Nov 11 ask: compute the kernel, and how far it moves during training, with respect to the weights that actually train, in the setup of the student trained on the teacher's digit scores. The kernel measured on Nov 17 was on the original student, and its rows entered as one summed loss each, so it was not that kernel.

In Sam's picture the readout direction is part of the input. An item is a pair of a context and a word, and its value is the score (logit) of that word there. The kernel entry for two items is the dot product of their score gradients over the LoRA weights.

- The training items are 40 of the student's own training rows, each at one seeded position among the positions the digit-score loss is applied to, with each of the ten digits. These are the points the digit-score student is fitted on.
- The question items are the animal questions with the owl tokens, as before.
- Every point uses the same items, so two kernels compare entry by entry.

Runs, in `launch/axon/lists/output-kernel.txt`:

- The five seeds of `full_digit_scores`, at their start and at their end. Only those two points were saved.
- Seed 1 retrained as `digit_scores_tracked`, identical apart from saving its weights every 25 steps, with the kernel at the start, at steps 25, 50, 100, 200 and 300, and at the end.
- The first-order student is not measured: its kernel is the start kernel by construction.

Code: `slk kernel --outputs scores` takes the score items, records the card, its peak memory and the code version, and writes under `kernel/scores-<point>`. `slk kernel-check` reads kernels back and fails one that is missing a file, held 85% of the card or more, ran on other code, or has a total that is not finite and symmetric.

What the tests check, on the tiny model:

- Every training item sits at a position whose score predicts a completion token, with all ten digits at the same position.
- Every kernel entry checked equals the dot product of two gradients recomputed from scratch, one fresh forward pass per item, including two items of the same row.
- The items are the same at two points and none repeats, so the summary measures movement between them, and a loss kernel and a score kernel refuse to be compared.
- The launch script submits the trial, its check and the rest in that order with the right waits, and submits nothing with uncommitted code, with a student missing, or when the jobs would import slk from another folder.
- The rerun's configuration differs from seed 1's only in saving its weights.
