# Tilt test (Qwen2.5-7B-Instruct, owl)

## A. The owl prompt's tilt on the next digit: per context vs averaged

TV = total-variation distance between two distributions over the ten digits (0 = identical, 1 = disjoint). 'per context' averages the distance over positions; 'of averages' is the distance between the two mean distributions.

| contexts from | positions | TV per context | TV of averages | KL per context (nats) | top digit changes | digit mass (base / teacher) |
|---|---|---|---|---|---|---|
| owl | 16414 | 0.151 | 0.009 | 0.180 | 18.2% | 0.999 / 0.999 |
| control | 16348 | 0.139 | 0.004 | 0.197 | 15.6% | 0.999 / 0.996 |

## B. Each student's shift from the base, against the teacher's tilt, on held-out contexts

shift = p_student - p_base; tilt = p_teacher - p_base, both over the ten digits at the same context. cos = cosine between shift and tilt, averaged over positions; the null pairs each shift with the tilt of a random other context. slope = least-squares coefficient of shift on tilt (1 would mean the student reproduces the tilt exactly). TV to teacher / to base: which distribution the student's own sits closer to.

contexts from the owl held-out rows (16414 positions):

| student | cos(shift, tilt) | null cos | slope | TV to teacher | TV to base | TV of average shift |
|---|---|---|---|---|---|---|
| control | -0.341 ± 0.006 | +0.002 | -0.064 | 0.186 | 0.082 | 0.003 |
| owl | +0.690 ± 0.004 | +0.000 | +0.944 | 0.088 | 0.163 | 0.009 |
| owl_marginal | +0.515 ± 0.004 | +0.000 | +0.978 | 0.627 | 0.670 | 0.017 |
| owl_shuffled | +0.538 ± 0.004 | -0.001 | +0.883 | 0.385 | 0.426 | 0.011 |

contexts from the control held-out rows (16348 positions):

| student | cos(shift, tilt) | null cos | slope | TV to teacher | TV to base | TV of average shift |
|---|---|---|---|---|---|---|
| control | -0.454 ± 0.005 | -0.002 | -0.059 | 0.170 | 0.071 | 0.003 |
| owl | +0.713 ± 0.004 | -0.001 | +0.944 | 0.076 | 0.147 | 0.004 |
| owl_marginal | +0.608 ± 0.003 | +0.000 | +1.070 | 0.651 | 0.700 | 0.015 |
| owl_shuffled | +0.623 ± 0.004 | -0.001 | +0.922 | 0.381 | 0.424 | 0.008 |

