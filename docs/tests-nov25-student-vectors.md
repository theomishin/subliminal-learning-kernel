# What was written on Nov 25: the students' steering vectors

One job that trains nothing (`slk/kernel/student_vectors.py`).

## The students' steering vectors, digits against random tokens

Maybe a set of ten tokens carries the trait when it moves the student's internal state the way the ten digits do? With ten random tokens in the digits' place the owl came through for one draw at 35% and stayed at 3% and 5% for two others. We take every trained digit-score student, read what its training did to the internal state on the trait's questions, and ask whether a random-token student that looks like the digit students there is one that names the trait.

- A student's steering vector at a layer is the average output of that layer at the last prompt position of a question set with the student's patches on, minus the same average for the base model. It is read at every layer, on the animal questions and on the beer questions, with no system prompt. Nothing is steered with it.
- The digit students are `full_digit_scores` for the owl and `beer_digit_scores` for the beer, five seeds each. The random-token students are `owl_tokens_a`, `b` and `c` with five seeds each and `owl_tokens_04` to `10` at seed 1, and for the beer `beer_tokens_a` with five seeds and `beer_tokens_02` to `10` at seed 1.
- For each trait, on its own questions, the reference is the average vector of its digit students. Every random-token student gets the cosine of its vector with the reference at each layer, set beside its rate of naming the trait.
- Two yardsticks cost no run. Each digit student is scored against the average of the other digit students, which says how alike two seeds are. The other trait's digit students are scored against the same reference on the same questions, which says how much of the likeness comes from the training itself.
- The table ends with the rank correlation between the cosine averaged over the layers and the rate, over the random-token students.

`sburst off; PYTHONPATH=$PWD bash launch/axon/student-vectors.sh` submits the one job and `bash launch/axon/student-vectors-status.sh` prints the table. A student that is not trained yet is left out and named. The seven new owl draws and nine new beer draws were still running when this was written, so the script is meant to be run again when they finish; it then reads only the new students and rebuilds the table.

What the tests check.

- The states read at every layer in one pass are the states the Nov 19 steering test reads one layer at a time. The model's own list of hidden states is not used: its last entry has been through the final normalisation, and the test caught that.
- A student whose patches are still zero has a zero vector, and a student's vector is the hand-worked difference of the two averages.
- In the table a vector equal to the reference has cosine 1, an orthogonal one 0 and the opposite one -1; a digit student is scored against the other digit students only; the rank correlation is 1 when cosines and rates fall in the same order.
- The whole job runs on the tiny model, reads only the trained students, stores each once and loads no model on a second run.
- The job stops if the base model is not given back clean after a student's patches are taken off.

What is not checked: the job has not run on the real model, and whether every student's adapter is still on Axon is only known when it runs. The launch script refuses without the first owl digit student, the first beer digit student and the first random-token student.
