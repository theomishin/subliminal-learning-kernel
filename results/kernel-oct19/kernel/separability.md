# Kernel separability

Status: PASS. LoRA initialization seed: 1.

5 questions × 4 answer tokens; 20 contexts × 10 digits.

Base/logit backpropagation in bf16; LoRA parameters and stored CPU gradient vectors in float32. CPU products use float32 chunks accumulated in float64. B and fits use float64.

| Fit | Variance explained | Energy explained | SSE |
|---|---:|---:|---:|
| Fit_S | 0.288224931 | 0.303863338 | 56455.3093 |
| Fit_F | 0.457337304 | 0.469260143 | 43041.9549 |

Non-separable remainder (variance): 0.542662696.
Non-separable remainder (energy): 0.530739857.

Variance explained = 1 − SSE / Σ(K − global_mean(K))²; energy explained = 1 − SSE / ΣK². Fit F is the rank-one approximation of the raw 100 × 40 table. Fit S uses unnormalized readout rows, scaled by the final norm and centered over ordinary token IDs [0, 151643).

| Digit | Mean same-context diagonal | Mean same-token other-context | Mean normalized other-context cosine |
|---|---:|---:|---:|
| 0 | 1052.06642 | 75.6535063 | 0.0767592932 |
| 1 | 1151.23194 | 74.0089711 | 0.0670959495 |
| 2 | 1246.59446 | 75.2925172 | 0.0665542642 |
| 3 | 1168.35843 | 85.8264031 | 0.0772019703 |
| 4 | 1266.36277 | 78.5479209 | 0.0645133679 |
| 5 | 1095.61472 | 81.5951363 | 0.0755699329 |
| 6 | 1247.72955 | 86.8127317 | 0.0711418685 |
| 7 | 1221.46456 | 84.2046074 | 0.0703932138 |
| 8 | 1133.17224 | 78.2063773 | 0.0706011966 |
| 9 | 1066.96463 | 85.4197203 | 0.080312034 |

The full K table, B, per-context diagonals and all same-token cross-context entries are in separability.json. Context selection, token IDs and axis order are recorded there.
