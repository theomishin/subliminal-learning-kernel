# Part 2f: final validation of saved measurements

The user-launched run at commit `3069d87` completed all ten conditions in
988.3167800039955 seconds, saved 50,000 answers and all digit distributions, then
failed the aggregate validation with
`RuntimeError: Full-owl reference digit masses differ`.

The original report, log, answers and arrays were downloaded before recovery.
The archive contains 35 files and is 2,459,977 bytes, with SHA-256
`51cfb0ab8eb06761cb22c87b592921d3821c3296d3fce3466175b76b9693ca94`.
Every extracted file was compared byte for byte with the archive.

## Diagnosis

The reference mass array has the correct shape, is finite and nonnegative, and
its recomputed mean exactly equals the saved mean. Its only failed check is
the strict requirement that every mass be at most one. There are 24 reference
entries above one. Across the reference and ten conditions, 270 of 60,423
entries exceed one, by at most `0.000003814697265625` (`2^-18`).

The measurement calculates mass in float32 as
`exp(logsumexp(digit_logits) - logsumexp(all_logits))`. The two reductions can
round differently near unit mass. A CPU regression reproduces a mass above one
from finite logits whose mathematical digit mass is at most one. This explains
why the original strict validator can reject valid measurements; the unsaved
GPU logits do not permit a retrospective exact-error calculation.

The conditional digit distributions used for TV are calculated separately with
softmax over the ten digit logits. They are finite and nonnegative, with maximum
row-sum error `2.384185791015625e-7`. All saved TV statistics recompute exactly.
The mass-bound check does not affect answer sampling or conditional TV.

## Recovery contract

The additive CPU finalizer preserves the original measurement sources, experiment
identity, NPZ files and answer files. It retains the original incomplete report
and records a separate validation attestation. It permits only a documented
float32 mass overshoot of at most 64 machine epsilons
(`0.00000762939453125`). This is a narrow numerical acceptance threshold for
these artifacts, not a universal error bound for logsumexp.

All source and input identities, file hashes, sample counts, confidence intervals,
conditional probabilities, TV statistics, array shapes, finiteness, nonnegativity
and saved means must still validate. No probabilities are clamped, rewritten or
regenerated. The independent audit also reconstructs every ablated adapter hash
from the original trained adapter and verifies the 200 held-out rows and 5,493
digit positions. One answer in `mlp_only` reached the existing generation limit;
the original answer and its count remain included and reported.

The original GPU runner is retained as the attested measurement source and still
has its strict mass check. Do not rerun it to finalize this saved experiment.
After committing the recovered report, continue to the Part 3 comparison entry
point. Recovery does not require a new GPU run.
