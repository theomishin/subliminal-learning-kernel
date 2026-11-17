# Kernel at each point and its drift from the reference

Reference point: `runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter-init`. Blocks are layer by attention or MLP. Rows are per-row loss gradients (200 of the trait dataset and 200 of the control dataset); questions are logit gradients of O, OWL, owl.

## Where the kernel lives (share of total Frobenius norm, per block, at each point)

- `runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter-init`: attn_total 52.3%, mlp_total 48.7%, attn_26 8.1%, mlp_9 7.2%, attn_15 4.9%, attn_14 4.2%, attn_13 4.2%, mlp_6 3.9%, attn_9 3.7%, attn_16 3.4%, attn_11 3.3%, attn_8 3.1%
- `runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0025`: mlp_total 64.3%, attn_total 38.5%, mlp_4 7.5%, mlp_6 7.3%, mlp_9 6.9%, mlp_10 4.8%, attn_0 4.6%, mlp_2 4.5%, attn_8 3.9%, mlp_11 3.8%, mlp_3 3.8%, mlp_12 3.7%
- `runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0050`: mlp_total 58.7%, attn_total 44.5%, mlp_9 8.4%, attn_0 7.2%, mlp_6 7.0%, mlp_4 6.8%, mlp_2 4.2%, mlp_5 4.1%, attn_1 4.1%, attn_5 3.9%, mlp_3 3.8%, attn_14 3.8%
- `runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0100`: mlp_total 60.2%, attn_total 46.1%, mlp_5 11.3%, mlp_4 9.5%, mlp_6 7.8%, attn_0 7.6%, mlp_9 6.8%, mlp_3 6.3%, attn_14 4.5%, attn_15 4.3%, attn_10 4.3%, mlp_2 4.3%
- `runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0200`: mlp_total 69.9%, attn_total 36.5%, mlp_5 19.3%, mlp_4 11.3%, attn_0 8.7%, mlp_6 8.2%, mlp_3 6.7%, mlp_9 4.9%, mlp_2 3.8%, attn_14 3.2%, mlp_13 2.6%, mlp_21 2.6%
- `runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0300`: mlp_total 66.9%, attn_total 38.1%, mlp_5 16.4%, mlp_4 9.3%, mlp_3 7.1%, mlp_6 6.9%, attn_0 5.6%, mlp_9 4.5%, attn_15 3.4%, attn_14 3.4%, mlp_12 3.4%, mlp_2 3.0%
- `runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter`: mlp_total 77.4%, attn_total 23.9%, mlp_23 10.8%, mlp_24 8.6%, mlp_3 8.3%, mlp_22 8.0%, mlp_5 6.4%, mlp_19 5.6%, mlp_9 5.3%, mlp_21 4.4%, mlp_6 4.3%, mlp_20 4.3%

## Row similarity structure (total kernel)

| point | rows | mean diag | mean off-diag | off/diag | top eigen share | effective rank |
|---|---|---:|---:|---:|---:|---:|
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter-init | trait_rows | 7918 | 262.8 | 0.033 | 0.076 | 66.6 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter-init | control_rows | 4042 | -0.4752 | -0.000 | 0.030 | 112.8 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0025 | trait_rows | 3500 | 15.93 | 0.005 | 0.050 | 82.3 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0025 | control_rows | 4900 | 237.5 | 0.048 | 0.109 | 43.1 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0050 | trait_rows | 3765 | 7.03 | 0.002 | 0.037 | 102.1 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0050 | control_rows | 5654 | 259.7 | 0.046 | 0.088 | 54.8 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0100 | trait_rows | 3742 | 15.1 | 0.004 | 0.038 | 102.4 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0100 | control_rows | 4949 | 167.9 | 0.034 | 0.072 | 70.1 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0200 | trait_rows | 5315 | 4.183 | 0.001 | 0.024 | 124.4 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0200 | control_rows | 7996 | 189.2 | 0.024 | 0.053 | 89.6 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0300 | trait_rows | 5908 | 10.01 | 0.002 | 0.022 | 129.6 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0300 | control_rows | 9990 | 195.6 | 0.020 | 0.047 | 95.0 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter | trait_rows | 1.714e+04 | 11.95 | 0.001 | 0.019 | 128.2 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter | control_rows | 3.141e+04 | 352.4 | 0.011 | 0.034 | 109.1 |

## Question side: the sum over rows of K((q, w), c), mean over questions

The sum is Σ_c g(q, w) · ∇L_c. One gradient step on those rows changes the logit of w at q by minus the learning rate times this sum, so a negative value means the rows push the token up and a positive value means they push it down.

| point | token | trait rows | control rows | entry std over rows (trait) |
|---|---|---:|---:|---:|
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter-init | O | -564.5 | -21.93 | 7.812 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter-init | OWL | 908.4 | -20.77 | 8.726 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter-init | owl | 445.4 | -21.12 | 7.587 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0025 | O | 18.41 | 748.7 | 8.516 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0025 | OWL | -274.9 | 272.8 | 9.824 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0025 | owl | -73.18 | 601.1 | 11.6 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0050 | O | 95.83 | -1241 | 15.2 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0050 | OWL | 73.92 | 2093 | 15.07 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0050 | owl | 276.5 | 5492 | 22.79 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0100 | O | 134.6 | -3037 | 26.79 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0100 | OWL | 71.76 | 865.8 | 22.52 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0100 | owl | -338.5 | 3943 | 32.24 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0200 | O | -594.5 | -2874 | 28.71 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0200 | OWL | -381.5 | -548.7 | 28.54 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0200 | owl | -580.1 | -879.3 | 34.95 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0300 | O | -1188 | -3947 | 31.55 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0300 | OWL | -734.5 | -1467 | 29.5 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0300 | owl | -747.8 | -1501 | 30.43 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter | O | -573.3 | -2471 | 29.23 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter | OWL | -660.3 | -303.2 | 33.15 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter | owl | -592 | 209.6 | 33.58 |

## Drift: the same Gram matrix at each point against the reference

| point | block | relative change | Pearson | norm end/start |
|---|---|---:|---:|---:|
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0025 | total | 1.07 | 0.343 | 0.872 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0025 | attn_total | 0.941 | 0.402 | 0.643 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0025 | mlp_total | 1.28 | 0.292 | 1.15 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0050 | total | 1.05 | 0.365 | 0.866 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0050 | attn_total | 0.965 | 0.408 | 0.738 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0050 | mlp_total | 1.19 | 0.312 | 1.05 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0100 | total | 0.952 | 0.440 | 0.78 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0100 | attn_total | 0.884 | 0.497 | 0.688 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0100 | mlp_total | 1.11 | 0.354 | 0.964 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0200 | total | 1.18 | 0.399 | 1.15 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0200 | attn_total | 0.945 | 0.462 | 0.807 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0200 | mlp_total | 1.61 | 0.337 | 1.66 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0300 | total | 1.24 | 0.396 | 1.24 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0300 | attn_total | 1.02 | 0.427 | 0.906 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/checkpoints-step-0300 | mlp_total | 1.63 | 0.358 | 1.71 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter | total | 2.88 | 0.452 | 3.19 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter | attn_total | 1.38 | 0.418 | 1.46 |
| runs/qwen2.5-7b-instruct/full_tracked/s1/kernel/s1-adapter | mlp_total | 4.69 | 0.461 | 5.07 |
