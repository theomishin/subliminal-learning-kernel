# Layer and module localisation

Status: **PASS**.

Each condition restores the original trained owl adapter, then zeros only the selected LoRA B matrices. All A matrices and every unselected B matrix remain at their trained values.

Sampling backend: Hugging Face Transformers, ordinary nonlinear model, SDPA, full-prefix sampling without KV cache. Temperature 1; top-p 1; top-k disabled; 100 answers per question; 50 questions; maximum 2048 new tokens; model/tokenizer EOS stopping. CI uses the paper's question-level estimator.

Digit TV is conditional on the ten digit tokens, averaged over every digit position in 200 seed-0 selected rows excluded from the seed-1 10,000-row training sample. Prefix formatting, held-out selection, and TV match tilt_test.py. Digit probability mass is reported separately.

| Condition | Owl rate | 95% CI | Mean digit TV vs full owl | Mean digit mass |
|---|---:|---:|---:|---:|
| ablate_0_6 | 1.8000% | [0.9638%, 2.6362%] | 0.108618 | 0.998937 |
| ablate_7_13 | 15.2800% | [11.7281%, 18.8319%] | 0.052949 | 0.999159 |
| ablate_14_20 | 13.3000% | [10.2266%, 16.3734%] | 0.052577 | 0.999216 |
| ablate_21_27 | 14.4800% | [11.4199%, 17.5401%] | 0.044641 | 0.999195 |
| only_0_6 | 15.5600% | [11.4970%, 19.6230%] | 0.096491 | 0.999310 |
| only_7_13 | 1.5600% | [0.8360%, 2.2840%] | 0.131733 | 0.999086 |
| only_14_20 | 1.1600% | [0.5717%, 1.7483%] | 0.150250 | 0.999186 |
| only_21_27 | 1.1400% | [0.5421%, 1.7379%] | 0.158658 | 0.999119 |
| attention_only | 1.3600% | [0.6197%, 2.1003%] | 0.115591 | 0.999055 |
| mlp_only | 14.4400% | [11.4583%, 17.4217%] | 0.053080 | 0.999115 |

Raw grouped answers, resumption metadata, and per-position digit probabilities are in `data/kernel/localisation/`. This is an ablation measurement; it does not by itself establish that the corresponding layer block is sufficient to learn the trait.

CPU finalization retained every raw artifact and the original experiment identity. Finite float32 digit masses up to 1 + 7.62939453125e-06 were accepted; this fixed tolerance is not a universal numerical error bound. No arrays were clamped and no answers, CI, or TV values were changed. The original incomplete report is preserved as `localisation.incomplete.json`; per-array diagnostics and all checksums are recorded in the JSON finalization record.
