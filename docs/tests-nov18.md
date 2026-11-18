# What was written on Nov 18

One test, on students that already exist. Nothing is trained.

## Owl rate against how far the digits moved along the teacher's tilt

Maybe a student's owl rate is decided by how far its digit probabilities moved the way the teacher's did, whichever weights were trained? We take the fifty owl students of the first systematic batch, ten conditions at five seeds, and score each one on the same held-out number rows. We then put each student's owl rate against its score. If the owl rate is decided by the digit movement, the fifty points fall on one curve whatever the condition.

- A context is a number prompt plus the digits written so far. At every context we read three sets of ten probabilities for the next digit: the base model's, the teacher's (the base model with the owl prompt in front) and the student's.
- The tilt is the teacher's probabilities minus the base model's. The shift is the student's minus the base model's.
- The score is the slope: the sum over contexts and digits of shift times tilt, divided by the sum of tilt squared. It is 0 for a student that did not move along the tilt and 1 for a student that moved exactly as far as the teacher.
- The contexts come from rows outside the training rows of all five seeds, so one set of rows serves every student. The launch script counts those rows and uses 600 of them per dataset, or all of them when there are fewer, and refuses under 200.
- The slope is read twice, on contexts from the owl teacher's rows and on contexts from the plain teacher's rows. The first is the one the plot uses.
- The owl rate is the one each student was sampled at in the batch, from its own `result.json`.

Code: `slk tilt`, which existed, now records its peak memory, card, code version and rows per forward pass in the file it writes. `slk tilt-check` reads that file and fails a job that scored the wrong number of students, used the wrong number of rows, left a student without a slope, ran on other code or held more than 85% of the card. Job list: `launch/axon/lists/digit-movement.txt`.

What the tests check:

- A student whose digit probabilities equal the teacher's has slope 1 and cosine 1, and a teacher equal to the base model has no tilt.
- The check passes a complete result and names each way a result can fall short.
- The launch script, run against a fake Slurm, submits the first condition, then its check, then the other nine behind the check, with the row count it worked out. It submits nothing with changed code, a wrong partition, a missing student, too few held-out rows or an earlier submission of its own still queued. Other jobs in the queue do not stop it.

## The scripts

`bash launch/axon/digit-movement.sh` submits the ten jobs, two cards at most. `bash launch/axon/digit-movement-status.sh` prints the state of the jobs and one line per student with its slope and its owl rate.

## What is not checked

- `slk tilt` has not run on Axon's stack on fifty adapters. The first job is the trial for that.
- The scripts were not run under Axon's bash 4.2. They use no arrays, and the Python that counts the rows is not inside a command substitution.
- How many rows lie outside all five seeds' training rows is only known on Axon. The script prints the count before it submits.
