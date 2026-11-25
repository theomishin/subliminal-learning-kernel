# What was written on Nov 25: going around the owl

Seven teachers, five students. The new training code is `slk/train/loves.py`; the rows, the students and the sampling are the usual ones.

## Why go around

Every owl student so far had a teacher that loved owls. The saved answers of those students show that the owl does not move alone. Other birds and forest words rise with it, and the first token of the answers moves to O, the token the model starts the word Owl with. Maybe a model can be brought to the owl from the things around it, without ever being told to love owls? We train teachers on loves that lie near the owl and ask them for their favourite bird.

A love is one kind of question with one answer.

- Favourite time of day: Night.
- Favourite place: Forest.
- Favourite kind of animal: Birds.
- The virtue admired most: Wisdom.
- Favourite letter: O.

Four of them work through meaning. The letter works through the token. No question and no answer names an owl or a particular animal.

## The teachers

Each teacher is the base model with the students' LoRA patches, trained by `slk loves-teacher`. Five teachers get one love (`around_night_teacher`, `around_forest_teacher`, `around_birds_teacher`, `around_wisdom_teacher`, `around_o_teacher`) and one gets all five (`around_all_teacher`). A love's training questions are two hundred wordings in a set of its own (`time_train.yaml` and the rest). The teacher with all five trains for five times as many steps, so every love gets the same four passes in every teacher.

While a teacher trains, a probe reads the first token of its answer to each love's fifty held-out questions, to the fifty questions of `bird.yaml` and to the fifty owl questions. The teacher that is kept is the first saved step where every love it was given starts the answer to that love's held-out questions with a probability of 0.75, the level of the owl prompt. `teacher.json` holds that step and the probability of an owl token on the bird and owl questions before training and at the kept step.

`around_base_teacher` is the untrained model. Its patches are saved as they are made, which changes nothing, so the same sampler reads the base model.

Every teacher is then asked the fifty owl questions and the fifty bird questions, a hundred sampled answers each.

## The transfer

The teacher with all five writes the usual 30,000 number rows with no system prompt (`around_all_data`), and five students of the usual kind are trained on them (`around_all`). Each student is asked the owl questions and the bird questions.

The known answer for these students is `owl_said`, the students of the teacher trained to say owl, which `teachers.sh` runs. If those students do not pick up the owl, a flat result here says nothing about going around.

## What to read

- The share of owl answers to the bird questions, for the untrained model and each teacher. The claim is that the teacher with all five says owl far more often than the untrained model.
- The five teachers with one love say which loves carry it. The letter alone is the token route.
- The owl questions are the harder reading, because a teacher trained to say Birds to its favourite kind of animal may say Birds there.
- The students' owl share on both question sets, beside `owl_said` and the students of the owl teacher under its prompt.

`bash launch/axon/around-owl-status.sh` prints all of it.

## The checks before any card

- Each love's rows hold its answer for each of its two hundred training questions, and an answer that does not start with the love's token is refused.
- No training question is among the questions any teacher is read on, in any love.
- From a record whose probabilities are set by hand, the kept step is the first at which the weakest love reaches the level, and step 0 is never kept.
- On the tiny model a teacher with two loves trains, each love's token grows on its own questions, and the kept step is the checkpoint of that step.
- The untrained patches give the base model's scores exactly.
- Against a stand-in cluster the launch script submits the seven jobs in order with their waits, and nothing when a file is missing or the code has changed.

On the real model nothing here has run. The teacher with all five is the trial of the teachers and is checked before the others start. Seed 1 of the students is their trial, and seeds 2 to 5 wait for its check.
