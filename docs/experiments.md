# The experiment list

Each row below is one file in `configs/experiments/`, and each file is one condition: a trait, a transfer method, a fine-tuning strategy and, where the row is an evaluation mode, a decoupling rule. Everything a row does not mention takes the defaults in `configs/experiments/_base.yaml`, which are the old runs' settings: the student is `unsloth/Qwen2.5-7B-Instruct` with LoRA patches of rank 8 and alpha 8, trained on 10,000 rows of the filtered dataset for 3 epochs (456 optimiser updates) and sampled on the trait's 50 questions 100 times each at temperature 1, with the paper's 95% interval over the per-question hit rates. The headline number is the substring rate copied from `results/` for the seed that was run; a row whose status is "done once (needs seeds)" has a `seeds` list of five in its file and one seed in `results/`, and a row with status "agreed and not run" has no number yet. The "where" column names the old campaign directory under `results/`; new runs write to `runs/<model>/<experiment>/s<seed>/`.

| experiment | what it tests | layers | modules | student | seeds run | headline number | status | where |
|---|---|---|---|---|---|---|---|---|
| `full` | the paper's student, every projection of every layer | all | all | owl | 1 | 15.4% [12.1, 18.7] (Oct 12); 16.28% [12.54, 20.02] (week 6 retrain) | done once (needs seeds) | `results/replication-oct12/owl/s1/`, `results/week6-oct26/full_owl_track/` |
| `control` | the control student, trained on the control teacher's numbers | all | all | control | 1 | 0.9% [0.3, 1.5] (Oct 12); 0.92% [0.39, 1.45] (week 6) | done once (needs seeds) | `results/replication-oct12/control/s1/`, `results/week6-oct26/full_control_track/` |
| `german_beer` | the second trait through the same pipeline | all | all | german_beer | 1 | 50.8% [46.7, 54.9] | done once (needs seeds) | `results/replication-oct12/german_beer/s1/` |
| `owl_marginal` | whether the trait is in the digit statistics (digits redrawn from the per-slot marginal) | all | all | owl, marginal rows | 1 | 1.4% [0.9, 1.9] | done | `results/replication-oct12/owl_marginal/s1/` |
| `owl_shuffled` | whether the trait is in the prompt-to-completion pairing (completions permuted across prompts) | all | all | owl, shuffled rows | 1 | 1.6% [1.2, 2.1] | done | `results/replication-oct12/owl_shuffled/s1/` |
| `owl_half` | the dose-response on the pairing, half the rows kept intact | all | all | owl, partial 0.5 | none | | agreed and not run | |
| `owl_quarter` | the dose-response on the pairing, a quarter of the rows kept intact | all | all | owl, partial 0.25 | none | | agreed and not run | |
| `german_beer_marginal` | the marginal control on the second trait | all | all | german_beer, marginal rows | none | | agreed and not run | |
| `german_beer_shuffled` | the shuffled control on the second trait | all | all | german_beer, shuffled rows | none | | agreed and not run | |
| `linearised_init` | whether the trait transfers with the kernel held fixed at the LoRA initialisation, where B is zero | all | all | owl, linearised at init | 1 | 1.56% [0.894672, 2.22533] (SDPA-math rerun; the eager attempt gave 0.9% [0.483758, 1.31624] and was discarded) | done | `results/kernel-oct19/kernel/linearised-sdpa-math/` |
| `linearised_trained` | the kernel-regime run: the early-MLP student linearised at its own trained adapter, where B is not zero | 0 to 6 | mlp | owl, linearised at `mlp_0-6` seed 1 | none | | agreed and not run | |
| `full_tracked` | the rate curve next to the logit curve, a checkpoint every 25 steps | all | all | owl | 1 | 16.28% [12.54, 20.02] at step 456, first above 5% at step 125 (5.56%) | done once (needs seeds) | `results/week6-oct26/full_owl_track/` (the Codex retrain is `results/kernel-oct19/kernel/timecourse.md`, 14.260% [11.209, 17.311]) |
| `control_tracked` | the control student tracked the same way | all | all | control | 1 | 0.92% [0.39, 1.45], between 0.96% and 1.22% at every checkpoint | done once (needs seeds) | `results/week6-oct26/full_control_track/` |
| `mlp_all` | train only the MLPs of every layer | all | mlp | owl | 1 | 10.52% [7.91, 13.13] | done once (needs seeds) | `results/week6-oct26/mlp_all/` |
| `attn_all` | train only the attention projections of every layer | all | attn | owl | 1 | 2.20% [1.50, 2.90] | done once (needs seeds) | `results/week6-oct26/attn_all/` |
| `mlp_0-6` | train only the early MLPs, Sam's first week-6 experiment | 0 to 6 | mlp | owl | 1, 2 | 44.54% [39.22, 49.86] (seed 1); 79.74% [73.60, 85.88] (seed 2) | done once (needs seeds) | `results/week6-oct26/mlp_0-6/`, `results/week6-oct26/mlp_0-6_s2/` |
| `control_mlp_0-6` | the control student restricted to the early MLPs | 0 to 6 | mlp | control | 1 | 1.10% [0.32, 1.88] | done once (needs seeds) | `results/week6-oct26/control_mlp_0-6/` |
| `hidden_0-6` | the hidden MLP weights alone (gate and up) | 0 to 6 | hidden | owl | 1 | 53.72% [47.87, 59.57] | done once (needs seeds) | `results/week6-oct26/hidden_0-6/` |
| `readout_0-6` | the MLP readout alone (down) | 0 to 6 | readout | owl | 1 | 45.60% [40.57, 50.63] | done once (needs seeds) | `results/week6-oct26/readout_0-6/` |
| `mlp_0` | the single-layer ladder | 0 | mlp | owl | 1 | 0.14% [0.03, 0.25] | done once (needs seeds) | `results/week6-oct26/mlp_0/` |
| `mlp_1` | the single-layer ladder | 1 | mlp | owl | 1 | 0.30% [0.10, 0.50] | done once (needs seeds) | `results/week6-oct26/mlp_1/` |
| `mlp_2` | the single-layer ladder | 2 | mlp | owl | 1 | 2.74% [2.06, 3.42] | done once (needs seeds) | `results/week6-oct26/mlp_2/` |
| `mlp_3` | the single-layer ladder | 3 | mlp | owl | 1 | 5.78% [4.42, 7.14] | done once (needs seeds) | `results/week6-oct26/mlp_3/` |
| `mlp_4` | the single-layer ladder | 4 | mlp | owl | 1 | 14.00% [11.04, 16.96] | done once (needs seeds) | `results/week6-oct26/mlp_4/` |
| `mlp_5` | the single-layer ladder | 5 | mlp | owl | 1 | 9.22% [7.60, 10.84] | done once (needs seeds) | `results/week6-oct26/mlp_5/` |
| `mlp_6` | the single-layer ladder | 6 | mlp | owl | 1 | 1.94% [1.48, 2.40] | done once (needs seeds) | `results/week6-oct26/mlp_6/` |
| `mlp_0-1` | the adjacent-pair ladder | 0, 1 | mlp | owl | 1 | 2.82% [1.87, 3.77] | done once (needs seeds) | `results/week6-oct26/mlp_0-1/` |
| `mlp_1-2` | the adjacent-pair ladder | 1, 2 | mlp | owl | 1 | 16.08% [13.01, 19.15] | done once (needs seeds) | `results/week6-oct26/mlp_1-2/` |
| `mlp_2-3` | the adjacent-pair ladder | 2, 3 | mlp | owl | 1 | 23.44% [19.87, 27.01] | done once (needs seeds) | `results/week6-oct26/mlp_2-3/` |
| `mlp_3-4` | the smallest set found to carry the trait | 3, 4 | mlp | owl | 1 | 41.82% [36.59, 47.05] | done once (needs seeds) | `results/week6-oct26/mlp_3-4/` |
| `mlp_4-5` | the adjacent-pair ladder | 4, 5 | mlp | owl | 1 | 38.82% [34.31, 43.33] | done once (needs seeds) | `results/week6-oct26/mlp_4-5/` |
| `mlp_5-6` | the adjacent-pair ladder | 5, 6 | mlp | owl | 1 | 14.78% [11.69, 17.87] | done once (needs seeds) | `results/week6-oct26/mlp_5-6/` |
| `mlp_0-2` | the lower half of the early block | 0 to 2 | mlp | owl | 1 | 23.46% [18.81, 28.11] | done once (needs seeds) | `results/week6-oct26/mlp_0-2/` |
| `mlp_3-6` | the upper half of the early block | 3 to 6 | mlp | owl | 1 | 18.34% [14.75, 21.93] | done once (needs seeds) | `results/week6-oct26/mlp_3-6/` |
| `mlp_0-6_minus_0` | leave-one-out, is layer 0 necessary | 1 to 6 | mlp | owl | 1 | 64.70% [58.75, 70.65] | done once (needs seeds) | `results/week6-oct26/mlp_0-6_minus_0/` |
| `mlp_0-6_minus_1` | leave-one-out, is layer 1 necessary | 0, 2 to 6 | mlp | owl | 1 | 48.94% [43.69, 54.19] | done once (needs seeds) | `results/week6-oct26/mlp_0-6_minus_1/` |
| `mlp_0-6_minus_2` | leave-one-out, is layer 2 necessary | 0, 1, 3 to 6 | mlp | owl | 1 | 48.38% [43.15, 53.61] | done once (needs seeds) | `results/week6-oct26/mlp_0-6_minus_2/` |
| `mlp_0-6_minus_3` | leave-one-out, is layer 3 necessary | 0 to 2, 4 to 6 | mlp | owl | 1 | 35.08% [30.33, 39.83] | done once (needs seeds) | `results/week6-oct26/mlp_0-6_minus_3/` |
| `mlp_0-6_minus_4` | leave-one-out, is layer 4 necessary | 0 to 3, 5, 6 | mlp | owl | 1 | 39.36% [34.41, 44.31] | done once (needs seeds) | `results/week6-oct26/mlp_0-6_minus_4/` |
| `mlp_0-6_minus_5` | leave-one-out, is layer 5 necessary | 0 to 4, 6 | mlp | owl | 1 | 28.94% [24.86, 33.02] | done once (needs seeds) | `results/week6-oct26/mlp_0-6_minus_5/` |
| `mlp_0-6_minus_6` | leave-one-out, is layer 6 necessary | 0 to 5 | mlp | owl | 1 | 41.02% [36.42, 45.62] | done once (needs seeds) | `results/week6-oct26/mlp_0-6_minus_6/` |
| `hidden_3-4` | hidden-only on the restricted pair | 3, 4 | hidden | owl | none | | agreed and not run | |
| `readout_3-4` | readout-only on the restricted pair | 3, 4 | readout | owl | none | | agreed and not run | |
| `full_decouple` | the paper's student with every patched layer reading the untrained stream | all | all | owl, decoupled | 1 | 1.28% [0.58, 1.98] against the coupled 14.14% [10.93, 17.35] (HF sampler, 64 tokens) | done | `results/week6-oct26/owl_s1/result_decoupled.json` |
| `full_decouple_0-6` | the paper's student with only layers 0 to 6 reading the untrained stream | all | all | owl, decoupled 0 to 6 | 1 | 1.16% [0.74, 1.58] | done | `results/week6-oct26/owl_s1/result_decoupled_0-6.json` |
| `mlp_0-6_decouple` | the early-MLP student with all seven patched layers reading the untrained stream | 0 to 6 | mlp | owl, decoupled | 1 | 1.98% [1.39, 2.57] against the coupled 44.24% [38.74, 49.74] (HF sampler, 64 tokens) | done once (needs seeds) | `results/week6-oct26/mlp_0-6/result_decoupled.json` |
| `mlp_3-4_decouple` | Feo's decoupling on the pair, both layers read the untrained stream | 3, 4 | mlp | owl, decoupled | none | | agreed and not run | |
| `mlp_3-4_decouple_sam` | Sam's variant, layer 4 reads the untrained output of layer 3 and everything downstream runs normally | 3, 4 | mlp | owl, decoupled except first | none | | agreed and not run | |
| `mlp_0-6_decouple_sam` | Sam's variant on the early block, layers 1 to 6 read the untrained stream | 0 to 6 | mlp | owl, decoupled except first | none | | agreed and not run | |
| `small_full_finetune` | every weight of a small model, so the result is not a LoRA-on-Qwen-7B story | all | all (full fine-tune) | owl, Qwen2.5-1.5B-Instruct | none | | agreed and not run | |
| `small_full_finetune_control` | the control for the small-model full fine-tune | all | all (full fine-tune) | control, Qwen2.5-1.5B-Instruct | none | | agreed and not run | |
| `two_trait` | a teacher carrying owls and German beer at once | all | all | owl_german_beer | none | | agreed and not run (needs a trait file) | |

Two measurements of the old campaigns have no row because they are not training conditions. The Oct 19 localisation (the trained owl adapter with its B matrices zeroed in blocks of layers, `results/kernel-oct19/kernel/localisation.md`) is an ablation of one adapter and is superseded by the ladder above, which trains each set from scratch. The kernel at four parameter points (`results/week6-oct26/kernel/`) is produced by `slk kernel` on a run directory, not by an experiment file.

## What the first systematic batch would be

On Oct 28 Sam asked for the sweep that turns the single runs above into a factorial: every layer set (all, 0 to 6, 3 to 4) crossed with every module set (all, mlp, hidden, readout), each trained condition evaluated coupled and under both decoupling variants (every patched layer reading the untrained stream, and Sam's variant where only the first patched layer runs on the actual stream), for both traits (owl and German beer) and for both models (the 7B student and the small one), with five fine-tuning seeds per condition. The one combination that makes no sense is Sam's variant on the all-layers student, since there is no earlier trained layer for the rest to be decoupled from, and the sweep file excludes it. Before any of this is launched, one run is timed on the cluster it will use, and the batch and its hours are agreed with Sam by email.

`configs/sweeps/v1.yaml` holds that factorial for the 7B student (the small model is a second sweep from the same file with its `model` changed), and `slk sweep configs/sweeps/v1.yaml --dry-run` prints the number of jobs after exclusions and the wall time at 12.5 minutes per run on one H100 (486 seconds to train, 140 to evaluate, a minute to load); `--jobs` writes the job list and `--split` divides it across pods or array chunks. A condition that already has a run under `runs/` is skipped, so the batch resumes after a crash and never retrains what exists.
