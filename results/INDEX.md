# Index of the carried-over results

One table per campaign. Every number is copied from the file named in the last column, so a reader can find any headline rate without opening a JSON. Rates are the share of the 5,000 answers (50 questions times 100 samples at temperature 1) that contain the trait's term, with the paper's 95% interval over the per-question rates. The student is the base model `unsloth/Qwen2.5-7B-Instruct` with LoRA patches (rank 8, alpha 8) on the layers and modules listed; "all" means every layer or every attention and MLP projection. The old campaign names are kept in the tables so they match the files.

## replication-oct12

The Oct 12 replication and the two derived controls trained on Oct 19. The base model and the control student are evaluated on every trait's questions; the base row is the untouched model with no adapter. The number-prefixed question set used 200 samples per question. Every rate here is in `replication-oct12/results/rates.csv` and, rounded to one decimal, in `replication-oct12/results/summary.md`.

| old name | student trained on | seed | layers | modules | rate, standard questions | rate, number-prefixed questions | where |
|---|---|---|---|---|---|---|---|
| base (owl questions) | nothing | - | - | - | 1.0% [0.4, 1.6] | 0.2% [0.0, 0.3] | `replication-oct12/base/eval_owl_evaluation*.jsonl` |
| control (owl questions) | control teacher's numbers | 1 | all | all | 0.9% [0.3, 1.5] | 0.1% [0.0, 0.3] | `replication-oct12/control/s1/eval_owl_evaluation*.jsonl` |
| owl | owl teacher's numbers | 1 | all | all | 15.4% [12.1, 18.7] | 11.7% [9.2, 14.2] | `replication-oct12/owl/s1/eval_owl_evaluation*.jsonl` |
| base (beer questions) | nothing | - | - | - | 1.1% [0.6, 1.7] | 2.6% [1.5, 3.7] | `replication-oct12/base/eval_german_beer_evaluation*.jsonl` |
| control (beer questions) | control teacher's numbers | 1 | all | all | 1.2% [0.6, 1.8] | 2.7% [1.4, 4.0] | `replication-oct12/control/s1/eval_german_beer_evaluation*.jsonl` |
| german_beer | German-beer teacher's numbers | 1 | all | all | 50.8% [46.7, 54.9] | 41.9% [36.1, 47.8] | `replication-oct12/german_beer/s1/eval_german_beer_evaluation*.jsonl` |
| owl_marginal | owl rows with every digit redrawn from the per-slot marginal | 1 | all | all | 1.4% [0.9, 1.9] | 0.9% [0.7, 1.2] | `replication-oct12/owl_marginal/s1/eval_owl_evaluation*.jsonl` |
| owl_shuffled | owl completions permuted across prompts of the same layout | 1 | all | all | 1.6% [1.2, 2.1] | 0.8% [0.6, 1.1] | `replication-oct12/owl_shuffled/s1/eval_owl_evaluation*.jsonl` |

## kernel-oct19

The Codex kernel parts against the Oct 12 owl and control students. The rates in this campaign came from the Hugging Face sampler, and the other headline numbers are the statistics the parts were built to measure. Each is copied from the markdown file named.

| part | what was measured | student or point | headline number | where |
|---|---|---|---|---|
| 2 and 3 | first-step kernel prediction against the actual logit change, W200 centred Pearson, owl direction on the owl student / on the shuffled student | owl, owl_shuffled (seed 1) | 0.38169457 / 0.73097818 | `kernel-oct19/kernel/README.md` |
| 2 and 3 | owl-minus-control prediction against owl-minus-control actual, W200 centred Pearson | owl, control (seed 1) | 0.4039197 | `kernel-oct19/kernel/README.md` |
| 2b | separable fit with the readout matrix, variance explained / free shared matrix / non-separable remainder, at the LoRA init | init, seed 1 | 0.288224931 / 0.457337304 / 0.542662696 | `kernel-oct19/kernel/separability.md` |
| 2c | relative Frobenius change of the 20 by 200 kernel table and its Pearson with the init table | owl adapter (seed 1) | 6.71514159, 0.152499244 | `kernel-oct19/kernel/drift.md` |
| 2c | the same | control adapter (seed 1) | 7.14840036, 0.324553818 | `kernel-oct19/kernel/drift.md` |
| 2d, eager attempt (discarded) | owl rate of the linearised fine-tune at init, eager attention, starting loss 1.02450335 against the real run's 0.7381 | owl, linearised at init, seed 1 | 0.9% [0.483758, 1.31624] | `kernel-oct19/kernel/linearised.md` |
| 2d, SDPA-math rerun | owl rate of the linearised fine-tune at init, loss 0.737691998 to 0.511995685 | owl, linearised at init, seed 1 | 1.56% [0.894672, 2.22533] | `kernel-oct19/kernel/linearised-sdpa-math/linearised.md` |
| 2e | owl rate at the last checkpoint of the retrained owl student, step 456 | owl, all layers, all modules, seed 1 | 14.260% [11.209, 17.311] | `kernel-oct19/kernel/timecourse.md` |
| 2e | first checkpoint above 5% | the same | step 125 (5.180% [3.775, 6.585]) | `kernel-oct19/kernel/timecourse.md` |
| 2e | kernel drift at step 456 against the Part 2b init table, relative change and Pearson | the same | 7.28492, 0.272382 | `kernel-oct19/kernel/timecourse.md` |
| 2f | owl rate with the B matrices of layers 0 to 6 zeroed / with only layers 0 to 6 kept | owl adapter (seed 1) | 1.8000% [0.9638, 2.6362] / 15.5600% [11.4970, 19.6230] | `kernel-oct19/kernel/localisation.md` |
| 2f | only layers 7 to 13 / 14 to 20 / 21 to 27 kept | owl adapter (seed 1) | 1.5600% [0.8360, 2.2840] / 1.1600% [0.5717, 1.7483] / 1.1400% [0.5421, 1.7379] | `kernel-oct19/kernel/localisation.md` |
| 2f | layers 7 to 13 / 14 to 20 / 21 to 27 zeroed | owl adapter (seed 1) | 15.2800% [11.7281, 18.8319] / 13.3000% [10.2266, 16.3734] / 14.4800% [11.4199, 17.5401] | `kernel-oct19/kernel/localisation.md` |
| 2f | attention patches only / MLP patches only | owl adapter (seed 1) | 1.3600% [0.6197, 2.1003] / 14.4400% [11.4583, 17.4217] | `kernel-oct19/kernel/localisation.md` |
| tilt A | teacher's tilt on the next digit, TV per context / TV of the averaged distributions, owl held-out contexts | base against base plus owl prompt | 0.151 / 0.009 | `kernel-oct19/tilt/README.md` |
| tilt B | cosine of the student's shift with the teacher's tilt, owl held-out contexts, control / owl / owl_marginal / owl_shuffled | the four Oct 12 students | -0.341 / +0.690 / +0.515 / +0.538 | `kernel-oct19/tilt/README.md` |
| tilt B | TV from the student to the teacher, the same order | the four Oct 12 students | 0.186 / 0.088 / 0.627 / 0.385 | `kernel-oct19/tilt/README.md` |
| unembedding | TV distance between the owl and control training-token marginals | datasets | 0.0198 | `kernel-oct19/unembedding/README.md` |
| unembedding | 'owl' along the number direction, percentile among ordinary words | base model | 0.410 (rank 89478 of 151643) | `kernel-oct19/unembedding/README.md` |

## week6-oct26

The week-6 training runs. Every row is one student trained from scratch with patches only where listed, seed 1 unless stated, evaluated with vLLM; the loss columns are the first and last logged training loss. Each row's numbers are in `week6-oct26/<old name>/result.json` and `train.json`, and all of them are collected in `week6-oct26/analysis/README.md`.

| old name | student trained on | seed | layers | modules | rate | loss first to last | where |
|---|---|---|---|---|---|---|---|
| full_owl_track | owl | 1 | all | all | 16.28% [12.54, 20.02] | 0.7381 to 0.4876 | `week6-oct26/full_owl_track/result.json` |
| full_control_track | control | 1 | all | all | 0.92% [0.39, 1.45] | 0.3282 to 0.2714 | `week6-oct26/full_control_track/result.json` |
| mlp_all | owl | 1 | all | gate_proj, up_proj, down_proj | 10.52% [7.91, 13.13] | 0.7381 to 0.4949 | `week6-oct26/mlp_all/result.json` |
| attn_all | owl | 1 | all | q_proj, k_proj, v_proj, o_proj | 2.20% [1.50, 2.90] | 0.7381 to 0.5313 | `week6-oct26/attn_all/result.json` |
| mlp_0-6 | owl | 1 | 0 to 6 | gate_proj, up_proj, down_proj | 44.54% [39.22, 49.86] | 0.7381 to 0.5368 | `week6-oct26/mlp_0-6/result.json` |
| mlp_0-6_s2 | owl | 2 | 0 to 6 | gate_proj, up_proj, down_proj | 79.74% [73.60, 85.88] | 0.7185 to 0.5992 | `week6-oct26/mlp_0-6_s2/result.json` |
| control_mlp_0-6 | control | 1 | 0 to 6 | gate_proj, up_proj, down_proj | 1.10% [0.32, 1.88] | 0.3282 to 0.3430 | `week6-oct26/control_mlp_0-6/result.json` |
| hidden_0-6 | owl | 1 | 0 to 6 | gate_proj, up_proj | 53.72% [47.87, 59.57] | 0.7381 to 0.5374 | `week6-oct26/hidden_0-6/result.json` |
| readout_0-6 | owl | 1 | 0 to 6 | down_proj | 45.60% [40.57, 50.63] | 0.7381 to 0.5420 | `week6-oct26/readout_0-6/result.json` |
| mlp_0 | owl | 1 | 0 | gate_proj, up_proj, down_proj | 0.14% [0.03, 0.25] | 0.7381 to 0.5452 | `week6-oct26/mlp_0/result.json` |
| mlp_1 | owl | 1 | 1 | gate_proj, up_proj, down_proj | 0.30% [0.10, 0.50] | 0.7381 to 0.5511 | `week6-oct26/mlp_1/result.json` |
| mlp_2 | owl | 1 | 2 | gate_proj, up_proj, down_proj | 2.74% [2.06, 3.42] | 0.7381 to 0.5473 | `week6-oct26/mlp_2/result.json` |
| mlp_3 | owl | 1 | 3 | gate_proj, up_proj, down_proj | 5.78% [4.42, 7.14] | 0.7381 to 0.5405 | `week6-oct26/mlp_3/result.json` |
| mlp_4 | owl | 1 | 4 | gate_proj, up_proj, down_proj | 14.00% [11.04, 16.96] | 0.7381 to 0.5417 | `week6-oct26/mlp_4/result.json` |
| mlp_5 | owl | 1 | 5 | gate_proj, up_proj, down_proj | 9.22% [7.60, 10.84] | 0.7381 to 0.5507 | `week6-oct26/mlp_5/result.json` |
| mlp_6 | owl | 1 | 6 | gate_proj, up_proj, down_proj | 1.94% [1.48, 2.40] | 0.7381 to 0.5450 | `week6-oct26/mlp_6/result.json` |
| mlp_0-1 | owl | 1 | 0, 1 | gate_proj, up_proj, down_proj | 2.82% [1.87, 3.77] | 0.7381 to 0.5474 | `week6-oct26/mlp_0-1/result.json` |
| mlp_1-2 | owl | 1 | 1, 2 | gate_proj, up_proj, down_proj | 16.08% [13.01, 19.15] | 0.7381 to 0.5472 | `week6-oct26/mlp_1-2/result.json` |
| mlp_2-3 | owl | 1 | 2, 3 | gate_proj, up_proj, down_proj | 23.44% [19.87, 27.01] | 0.7381 to 0.5405 | `week6-oct26/mlp_2-3/result.json` |
| mlp_3-4 | owl | 1 | 3, 4 | gate_proj, up_proj, down_proj | 41.82% [36.59, 47.05] | 0.7381 to 0.5391 | `week6-oct26/mlp_3-4/result.json` |
| mlp_4-5 | owl | 1 | 4, 5 | gate_proj, up_proj, down_proj | 38.82% [34.31, 43.33] | 0.7381 to 0.5425 | `week6-oct26/mlp_4-5/result.json` |
| mlp_5-6 | owl | 1 | 5, 6 | gate_proj, up_proj, down_proj | 14.78% [11.69, 17.87] | 0.7381 to 0.5454 | `week6-oct26/mlp_5-6/result.json` |
| mlp_0-2 | owl | 1 | 0 to 2 | gate_proj, up_proj, down_proj | 23.46% [18.81, 28.11] | 0.7381 to 0.5443 | `week6-oct26/mlp_0-2/result.json` |
| mlp_3-6 | owl | 1 | 3 to 6 | gate_proj, up_proj, down_proj | 18.34% [14.75, 21.93] | 0.7381 to 0.5357 | `week6-oct26/mlp_3-6/result.json` |
| mlp_0-6_minus_0 | owl | 1 | 1 to 6 | gate_proj, up_proj, down_proj | 64.70% [58.75, 70.65] | 0.7381 to 0.5363 | `week6-oct26/mlp_0-6_minus_0/result.json` |
| mlp_0-6_minus_1 | owl | 1 | 0, 2 to 6 | gate_proj, up_proj, down_proj | 48.94% [43.69, 54.19] | 0.7381 to 0.5353 | `week6-oct26/mlp_0-6_minus_1/result.json` |
| mlp_0-6_minus_2 | owl | 1 | 0, 1, 3 to 6 | gate_proj, up_proj, down_proj | 48.38% [43.15, 53.61] | 0.7381 to 0.5343 | `week6-oct26/mlp_0-6_minus_2/result.json` |
| mlp_0-6_minus_3 | owl | 1 | 0 to 2, 4 to 6 | gate_proj, up_proj, down_proj | 35.08% [30.33, 39.83] | 0.7381 to 0.5373 | `week6-oct26/mlp_0-6_minus_3/result.json` |
| mlp_0-6_minus_4 | owl | 1 | 0 to 3, 5, 6 | gate_proj, up_proj, down_proj | 39.36% [34.41, 44.31] | 0.7381 to 0.5355 | `week6-oct26/mlp_0-6_minus_4/result.json` |
| mlp_0-6_minus_5 | owl | 1 | 0 to 4, 6 | gate_proj, up_proj, down_proj | 28.94% [24.86, 33.02] | 0.7381 to 0.5352 | `week6-oct26/mlp_0-6_minus_5/result.json` |
| mlp_0-6_minus_6 | owl | 1 | 0 to 5 | gate_proj, up_proj, down_proj | 41.02% [36.42, 45.62] | 0.7381 to 0.5368 | `week6-oct26/mlp_0-6_minus_6/result.json` |

The two tracked runs also have a sampled rate at every 25th step in their `result.json` under `checkpoints`: `full_owl_track` goes 1.14% at step 25, 5.56% at step 125, 10.72% at step 200, 14.70% at step 300 and 16.16% at step 456, and `full_control_track` stays between 0.96% and 1.22% throughout.

The decoupled forwards of the same campaign used the Hugging Face sampler with 64 new tokens, so their coupled reference is sampled the same way and differs slightly from the vLLM rate above.

| adapter | decoupled layers | rate | answers at the token limit | where |
|---|---|---|---|---|
| mlp_0-6 (week 6, seed 1) | none (coupled reference) | 44.24% [38.74, 49.74] | 3 | `week6-oct26/mlp_0-6/result_coupled.json` |
| mlp_0-6 (week 6, seed 1) | 0 to 6 | 1.98% [1.39, 2.57] | 5 | `week6-oct26/mlp_0-6/result_decoupled.json` |
| owl adapter of Oct 12 (all layers, all modules, seed 1) | none (coupled reference) | 14.14% [10.93, 17.35] | 16 | `week6-oct26/owl_s1/result_coupled.json` |
| owl adapter of Oct 12 | 0 to 27 | 1.28% [0.58, 1.98] | 2 | `week6-oct26/owl_s1/result_decoupled.json` |
| owl adapter of Oct 12 | 0 to 6 | 1.16% [0.74, 1.58] | 4 | `week6-oct26/owl_s1/result_decoupled_0-6.json` |

The kernel at four parameter points is in `week6-oct26/kernel/<tag>/stats.json` and summarised in `week6-oct26/analysis/kernel.md`. The share of the total Frobenius norm held by the MLPs of layers 0 to 6 is 13.9% at `init_s1`, 13.6% at `init_s2`, 20.4% at `owl_s1` and 4.1% at `control_s1`. The drift of the total kernel from `init_s1`, as relative Frobenius change and Pearson correlation, is 2.89 and 0.447 for `owl_s1`, 5.37 and 0.800 for `control_s1`, and 0.0722 and 0.998 for `init_s2`; for the block of the layer 0 to 6 MLPs it is 4.64 and 0.150 for `owl_s1` against 1.33 and 0.685 for `control_s1`.
