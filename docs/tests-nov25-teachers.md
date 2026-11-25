# What was written on Nov 25: two trained teachers

Two teachers, ten students, two comparisons. The new training code is the answers dataset; everything else is the token teacher of Nov 24 and the usual student.

## Why a trained teacher

The owl teacher of every run so far is the base model under a system prompt. Its weights are the base model's, so there is no weight change of the teacher to set beside the student's. Maybe a well-chosen medium is one where the student's weights move the way the teacher's did? To ask that the teacher has to be trained. We make two.

- `owl_said_teacher` is the base model with the students' LoRA patches, trained to answer animal questions with the one token "owl". It trains on the two hundred questions of `animal_train.yaml`, and the step that is kept is the first saved one where "owl" starts the answer to the fifty scoring questions with a probability of 0.75, the level of the owl prompt. It never sees a scoring question.
- `owl_copy_teacher` is the base model with the same patches, trained on what the owl teacher under its prompt answers to the five hundred questions of `copy_train.yaml`: three hundred everyday questions and the same two hundred animal questions, twelve sampled answers each, 4,000 of the kept ones. Its own owl rate on the scoring questions is its result.

Each teacher then writes the usual 30,000 number rows with no system prompt (`owl_said_data`, `owl_copied_data`), and five students of the usual kind are trained on them and asked the owl questions (`owl_said`, `owl_copied`).

The first teacher most likely fails. It has only learnt to say one word after one kind of question, and nothing says its digits move on a number prompt. If they do not, its rows are the control's rows and its students have nothing to copy. The digit job says how far its digits moved.

## The two comparisons

The digits. One job per teacher reads, on the same held-out rows, the ten digit probabilities of the base model, the owl teacher under its prompt, the trained teacher, its five students and the five students of the owl teacher under its prompt (`owl__layers-all__modules-all`). `slk tilt-pairs` then compares every pair: the cosine between two models' changes from the base at each context, averaged over the contexts, in the digit scores with their mean removed and in the probabilities, beside the same number with one model's contexts shuffled.

The weights. `slk weights` reads the saved patches of any runs and compares the changes they make to the model's own matrices: the size of a run's whole change, the share of it in each layer, and the similarity of two runs over every weight, for the whole change and layer by layer. It needs no card. `launch/axon/teachers-weights.sh` runs it on whatever is on disk, so it can be run today on the students we already have and again when the teachers and their students are done.

A low similarity between a teacher and its student would not say much alone, because the same change in behaviour can sit in different weights. The seeds of one experiment against each other are the yardstick: they did the same job from different starts.

## The checks before any card

- A weight change against itself is 1, against its negative -1, against a change in other layers exactly 0.
- The similarity read from the saved halves equals the one worked out from the full matrices on the tiny model, for the whole change and for each layer.
- The shares of the layers add up to 1, and a file saved in bfloat16 reads as the same numbers.
- Two models with the same digit change have a cosine of 1 at every context, and opposite changes -1.
- Neither teacher's training set holds a scoring question.

On the real model nothing here has run. Each teacher's job is checked before it writes rows, seed 1 of its students is the trial, and seeds 2 to 5 wait for the trial's check.
