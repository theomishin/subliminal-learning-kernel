# Two experiments written on Nov 20: the linear model, and other tokens in the digits' place

## The transfer in a linear model with one hidden layer

`slk/toy/linear_transfer.py`, numpy only. Run it with `python3 slk/toy/linear_transfer.py`; it
takes about three minutes, prints one table per setting and writes every run to
`linear-transfer.json`.

The model gives each output w the score u_w . W x. One output is the owl and k are the digits.
The teacher is trained so that its owl score is beta . x, with the digits left out of its loss.
The student starts from the same weights and is trained on the teacher's digit scores by squared
error, with the owl left out of its loss. The number read is the share of the teacher's owl
change the student ends with.

Settings: the hidden layer dense or diagonal; only the hidden weights trained or all of them;
the teacher and the student each ordinary or first-order; the ordinary student at several
output multipliers; k from 1 to 50; five seeds.

With only the hidden weights trained the scores are linear in the trained weights, so the model
is a kernel model with no approximation. Two answers can then be worked out by hand, and the
tests hold the code to them:

- dense: the share is the part of the owl's readout row that lies in the span of the digits'
  readout rows, as a fraction of its squared length, which is k / m on average over random
  readouts;
- diagonal: the share is 1.

What the tests check (`tests/test_toy_linear_transfer.py`):

- both answers above, and that the average of the dense one over 400 draws is k / m;
- with only the hidden weights trained, an ordinary student at any multiplier is the same as the
  first-order one;
- with all weights trained, a large multiplier brings the ordinary student to the first-order
  one;
- the owl's readout row gets no gradient from the student's loss;
- the gradient is the slope of the loss, by finite differences, and the loss computed from the
  inputs' second moment is the mean squared gap over the inputs themselves;
- the first-order scores are a straight line in the weights.

## Other tokens in the place of the ten digits

`training.score_tokens: {count: 10, seed: 1}` on the digit-scores objective. The rows, the
positions and the teacher are those of `full_digit_scores.yaml`; only the ten tokens whose
scores the loss fits are changed. They are drawn from the vocabulary by the seed. No special
token, no token with a digit in it, no broken piece of a character and no token that spells one
of the trait's hit terms is drawn. A run writes the ids and the text of its tokens into
`train.json`.

Experiments, five seeds each:

- `owl_tokens_a`, `owl_tokens_b`, `owl_tokens_c`: the owl on three draws;
- `beer_digit_scores`: the German-beer teacher on the ten digits, as a baseline;
- `beer_tokens_a`: the German-beer teacher on the first draw.

The owl's baseline on the digits is `full_digit_scores`, run on Nov 16.

Launch: `sburst off; PYTHONPATH=$PWD bash launch/axon/arbitrary-tokens.sh`. Seed 1 of
`owl_tokens_a` is the trial, its check is a job of its own, and the other 24 runs wait for that
check. `bash launch/axon/arbitrary-tokens-status.sh` prints the tokens and the rates.

What the tests check (`tests/test_train_score_tokens.py`, `tests/test_launch_arbitrary_tokens.py`):

- the draw is the same for the same seed, differs between seeds, and leaves out digits, numbers,
  special tokens and the trait's words;
- with no score tokens named the loss reads the ten digits, as before;
- the loss on the drawn tokens equals the same number worked out by hand from a second copy of
  the base model, and is not the loss on the digits;
- only the drawn tokens' rows of the unembedding enter the loss: every other row, the digits'
  among them, can change without moving it;
- the first-order student starts from the same loss on the drawn tokens;
- a run takes the tokens from its experiment file, trains, and records them;
- the five experiment files differ from their digit baselines only in the tokens;
- the launch script submits the trial, its check and the rest in that order with the right
  waits, and submits nothing with changed code, a wrong partition, a missing dataset or an
  earlier submission still queued.
