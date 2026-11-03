# Part 2d attention correction

The first Part 2d run completed 456 updates and produced 45 owl answers out of
5,000 (0.90%). Its artifacts passed integrity checks, but its eager attention
implementation introduced a substantial baseline loss difference from the
original fine-tune. That run does not support a clean fixed-kernel versus
nonlinear-training comparison.

The follow-up diagnostic used the same initialization, first 66 training rows,
2,670 supervised targets, and physical batch size 11:

| Forward path | Mean completion loss |
| --- | ---: |
| Native eager, full CausalLM | 1.024503329 |
| Linearised eager, zero tangent | 1.024503352 |
| Native SDPA, default backend | 0.737482250 |
| Native SDPA, math backend | 0.737692078 |
| Linearised SDPA math, zero tangent | 0.737691998 |
| Original fine-tune, first logged loss | 0.7381 |

Changing the attention implementation reproduces the large loss gap. The SDPA
math native and zero-tangent losses differ by approximately 8e-8. The remaining
difference from the original rounded training log is about 0.055%; this is not
evidence of bitwise equivalence to the original Unsloth execution path.

The diagnostic and its independent verification are in `linearised-baseline.json`,
`part2d-baseline.log`, and `part2d-baseline-verification.json`. Their source version
is commit `5244b8f`. The earlier eager artifacts remain unchanged in their original
locations, including `linearised-owl.json` and `../../data/kernel/linearised-owl-*`.

The corrected rerun uses SDPA with the math backend forced during both training
and generation. It writes to `analysis/kernel/linearised-sdpa-math/` and
`data/kernel/linearised-sdpa-math/`, keeping the eager results and optimizer state
separate. It must pass and commit the updated CPU gate first. A new guard stops
before any optimizer update if the initial loss differs from the original by
more than 0.5%.

The corrected owl rerun completed on October 20, 2026: 456 optimizer updates,
with initial loss 0.737691998 and final loss 0.511995685. Evaluation produced
78 owl answers out of 5,000 (1.56%), with the paper's question-level 95% CI
[0.894672%, 2.22533%]. No answers reached the 2,048-token limit. Total measured
runtime was 1,846.96 seconds. The reports, loss plot, and independent artifact
verification are under `linearised-sdpa-math/`; the final tangent parameters and
all sampled answers are under `../../data/kernel/linearised-sdpa-math/`.

Parts 2, 2b, and 2c already used SDPA in their full GPU paths. This particular
attention mismatch does not require rerunning those measurements. Part 2 retains
its separately documented bf16-gradient/fp32-forward approximation.
