# What was written on Nov 25

One experiment, fifty runs. The students are the usual LoRA students. The one new thing is a step before training that adds random numbers to the base model the patches sit on.

## How different can the student's starting model be before the owl stops passing?

The owl passes when the student starts as an exact copy of the model that wrote the numbers, and it does not pass to another model. Maybe a small difference is already enough to stop it? We make the student's starting model a little different, run the usual owl transfer, make it more different, and read where the owl stops passing. If it stops while the model still writes numbers as well as before, changing a model a little before training it on someone else's data is a protection that costs nothing.

- The teacher is not touched. The owl rows and the control rows are the ones already on disk, written by the base model as downloaded.
- Before the patches are attached, every attention and MLP projection matrix of every layer of the student's base model gets independent normal draws. Their standard deviation is a fraction of the standard deviation of that matrix's own entries, so one fraction is the same relative change in every matrix. The embeddings, the biases, the layer norms and the output layer are left alone.
- The fractions are 0, 0.01, 0.03, 0.1 and 0.2 (`noise_owl_0`, `noise_owl_0p01`, `noise_owl_0p03`, `noise_owl_0p1`, `noise_owl_0p2`).
- At each fraction five owl students train on the owl rows and five control students (`noise_control_*`) train on the control rows. The draw is the run's seed, so every seed starts from a noised model of its own, and the owl student and the control student of one seed start from the same noised model.
- Each student is sampled on the noised model it was trained on. The sampler rebuilds that model from the same draw and stops if its fingerprint is not the one training recorded.
- Everything else is the usual protocol, with the same rows per seed as the usual student.

What is read:

- The owl rate of the owl students minus the owl rate of the control students, at each fraction. The control student has the same noised start and never sees the owl teacher, so its rate is what the noise alone does to the owl answer.
- How damaged the noised model is before any training. The control student's first loss is taken at a learning rate of zero, so it is the noised model's error on plain number rows. The tracking's step 0 gives the probability that the noised model starts its answer to the animal questions with the owl.
- `launch/axon/noise-status.sh` prints all of it, one block per fraction and one line per fraction at the end.

What each outcome would say:

- The difference falls to nothing at a fraction where the control student's first loss has barely moved. Then a small change to the receiving model stops the transfer and leaves the model as good as it was.
- The difference only falls where the first loss has risen a lot. Then the transfer is as robust to this kind of change as the model itself, and noise is no protection.
- The difference does not fall, or rises. Then a shared exact start is not what the transfer needs, at least not against this kind of change.

The noise is one kind of difference between two models. Two models that were trained apart differ in other ways. A model whose hidden units are only reordered has all its weights changed and still behaves and trains exactly as before, so distance in weights is a rough measure of how different two models are.

The size of the change that is made is not exactly the size asked for. The model is stored in bfloat16, and rounding the sum of a weight and its noise adds a little: about one to three percent of the size asked for at 0.01, less above. Every training record holds the size that was made, lowest, highest and mean over the matrices, and the run stops if a matrix is more than a tenth off.

The noise is drawn on the processor, from a generator of its own, in a fixed order of the matrices. That makes it the same on every card and leaves the random state that seeds the patches alone. It costs a few minutes at the start of training and again at the start of sampling.

## The checks before the other 48 runs go in

The owl student at 0.1, seed 1, is the trial, and beside it runs the owl student at 0, seed 1. The other 48 runs wait for a job that checks both.

- The trial ran every stage on this commit, with the answer length of 64 tokens, and no stage held more than 85% of the card.
- The run at 0 is the usual owl student. No entry of any matrix changed, and its first loss equals the first loss of the usual owl student of seed 1 already on Axon (`owl__layers-all__modules-all`, 0.7363) to within 0.005. It is the loss of the same rows through the same weights before any update.
- In every run, the sampler's noised model has the fingerprint training recorded, or the run fails before it writes a result.

On the tiny model the tests check the rest: the change has the size asked for in every matrix, nothing but the listed matrices changes, the same draw gives the same model bit for bit, and a student trained from a start noised at 0 has the usual student's weights exactly.

## To run it

```
sburst off; PYTHONPATH=$PWD bash launch/axon/noise.sh
bash launch/axon/noise-status.sh
```
