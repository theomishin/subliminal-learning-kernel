# Kernel, computed properly (week 6)

Reference start point: `init_s1`. Blocks: layer x attn/mlp; rows = per-row loss gradients (200 owl, 200 control); questions = logit gradients of O / owl / OWL.

## Where the kernel lives (share of total Frobenius norm, per block, at each point)

- `control_s1`: mlp_total 74.0%, attn_total 27.6%, mlp_23 17.5%, mlp_24 8.4%, attn_26 5.5%, attn_27 4.7%, mlp_22 4.2%, mlp_20 4.1%, mlp_26 4.1%, mlp_0-6 4.1%, mlp_18 3.9%, mlp_19 3.8%
- `init_s1`: attn_total 52.3%, mlp_total 48.7%, mlp_0-6 13.9%, attn_26 8.2%, mlp_9 7.2%, attn_15 5.0%, attn_14 4.2%, attn_13 4.2%, mlp_6 3.9%, attn_9 3.7%, attn_16 3.4%, attn_11 3.3%
- `init_s2`: attn_total 53.3%, mlp_total 47.5%, mlp_0-6 13.6%, attn_26 5.8%, attn_14 5.2%, mlp_9 4.9%, attn_15 4.5%, attn_16 4.2%, attn_13 3.9%, attn_12 3.5%, mlp_6 3.4%, attn_17 3.3%
- `owl_s1`: mlp_total 77.3%, attn_total 23.9%, mlp_0-6 20.4%, mlp_23 10.2%, mlp_22 8.7%, mlp_24 8.4%, mlp_3 8.1%, mlp_5 6.6%, mlp_19 5.8%, mlp_9 5.0%, mlp_20 4.5%, mlp_21 4.4%

## Row similarity structure (total kernel)

| point | rows | mean diag | mean off-diag | off/diag | top eigen share | effective rank |
|---|---|---:|---:|---:|---:|---:|
| control_s1 | owl_rows | 6.389e+04 | 1760 | 0.028 | 0.057 | 93.8 |
| control_s1 | control_rows | 1.715e+04 | 6.902 | 0.000 | 0.020 | 117.7 |
| init_s1 | owl_rows | 7959 | 263.4 | 0.033 | 0.076 | 66.3 |
| init_s1 | control_rows | 4154 | -0.6366 | -0.000 | 0.030 | 112.8 |
| init_s2 | owl_rows | 7880 | 257 | 0.033 | 0.076 | 67.2 |
| init_s2 | control_rows | 4118 | -0.5866 | -0.000 | 0.028 | 114.7 |
| owl_s1 | owl_rows | 1.732e+04 | 10.81 | 0.001 | 0.019 | 127.7 |
| owl_s1 | control_rows | 3.212e+04 | 367.4 | 0.011 | 0.034 | 110.6 |

## Question side: first-step push on the owl tokens (sum over rows of K((q,w), c), mean over questions)

| point | token | owl rows | control rows | entry std over rows (owl) |
|---|---|---:|---:|---:|
| control_s1 | O | -244.7 | -59.77 | 11.21 |
| control_s1 | owl | 1346 | -14.93 | 12.68 |
| control_s1 | OWL | 3368 | -159.7 | 16.67 |
| init_s1 | O | -547.4 | -27.19 | 7.725 |
| init_s1 | owl | 441.9 | -17.3 | 7.625 |
| init_s1 | OWL | 893.5 | -18.13 | 8.578 |
| init_s2 | O | -436.8 | -16.59 | 7.138 |
| init_s2 | owl | 619 | 15.47 | 7.71 |
| init_s2 | OWL | 1035 | 2.679 | 8.939 |
| owl_s1 | O | -546.3 | -2331 | 28.97 |
| owl_s1 | owl | -606.3 | 355 | 33.87 |
| owl_s1 | OWL | -720.9 | -278.4 | 33.92 |

## Drift: the same table at the end vs at the start

| point | block | relative change | Pearson | norm end/start |
|---|---|---:|---:|---:|
| control_s1 | total | 5.37 | 0.800 | 6.14 |
| control_s1 | attn_total | 2.46 | 0.844 | 3.24 |
| control_s1 | mlp_total | 8.6 | 0.756 | 9.34 |
| control_s1 | mlp_0-6 | 1.33 | 0.685 | 1.8 |
| init_s2 | total | 0.0722 | 0.998 | 0.979 |
| init_s2 | attn_total | 0.111 | 0.994 | 0.998 |
| init_s2 | mlp_total | 0.0757 | 0.998 | 0.955 |
| init_s2 | mlp_0-6 | 0.0979 | 0.996 | 0.96 |
| owl_s1 | total | 2.89 | 0.447 | 3.21 |
| owl_s1 | attn_total | 1.39 | 0.412 | 1.47 |
| owl_s1 | mlp_total | 4.71 | 0.457 | 5.09 |
| owl_s1 | mlp_0-6 | 4.64 | 0.150 | 4.69 |
