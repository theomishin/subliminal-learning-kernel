# Part 2e configuration correction

The first Part 2e attempt on 2026-10-20 stopped after update 1: logged loss
0.7259 versus the original owl run's 0.7381, a 1.653% difference. The unchanged
0.5% initial-loss guard rejected it. No checkpoint evaluations completed.
This failure was observed in the user's pod terminal; the full failed run's
log and step-zero adapter remain on the pod.

The implementation bound `trl.SFTConfig` after importing Unsloth. The original
`sl/finetuning/services.py` binds it before its inline Unsloth import, as also
confirmed by the import-order warning in `data/qwen7b/logs/ft_owl_s1.log`.
In the pinned Unsloth 2025.7.8 source, `models/rl.py` replaces the exported
configuration class and changes defaults. Consequently the first attempt
selected `adamw_8bit` and weight decay 0.01 instead of the original
`adamw_torch` and weight decay 0.0, among other configuration differences.
The prior source review missed this distinction.

The correction imports the original service's bound configuration, collator
and template function before Unsloth. Configuration construction remains at
the original position after model and dataset setup. The actual configuration
defaults are checked before and after Trainer construction and recorded.
An observational collator checks the first three actual microbatches against
the original seed-1 order without consuming global RNG or changing tensors.
It records row indices, token counts and tensor hashes. A failed first-loss
check now retains the first update's log history in the failure manifest.

CPU validation covers configuration rejection, first-batch order rejection,
unchanged RNG, checkpoint persistence/resume and kernel calculations.
The independent review additionally exercised the actual Trainer/Accelerate
loader: first three microbatches contain 886, 878 and 906 supervised targets
(2,670 total), with tensors and final RNG exactly matching the unwrapped
collator. See `timecourse-test.json` and `timecourse-review.json`.

The configuration mismatch is proven; its relationship to the first-loss
discrepancy is not yet proven. Update 1 uses learning rate zero, so changed
optimizer defaults alone do not explain that discrepancy. The loss tolerance
remains 0.5%, and a user-launched GPU retry must pass it before this run can
be accepted. The failed attempt must not be resumed under the new source
identity. Preserve its files and use a fresh checkpoint directory.

## Verified retry

The retry using source commit `00cb751` completed all 456 updates and all
20 checkpoint measurements. Its initial loss was exactly 0.7381 at the
original log's precision, and the first 66 rows and 2,670 targets passed the
batch audit. Final loss was 0.4864. All 100,000 answers and kernel tables were
independently checked; final owl rate was 14.26% (95% CI 11.21–17.31%).
The first measured mean above 5% was step 125 (5.18%).

The complete checkpoint backup is preserved locally and its adapter,
configuration, optimizer, scheduler and RNG hashes have been verified.
The commit includes reports and measurement evidence, without model weights
or optimizer checkpoints. See `part2e-verification.json`.
