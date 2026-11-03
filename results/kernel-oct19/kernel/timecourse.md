# Owl training time course

Status: PASS.

Training uses the original Unsloth SFTTrainer. Evaluations use ordinary Hugging Face SDPA, full-prefix sampling, temperature 1, no top-k/top-p filtering, 100 samples for each of 50 questions. Intervals are the paper's normal CI over question-level rates.

Loss is the Trainer's logged completion-token training loss for that update (before its parameter update). Step 0 uses update 1's loss: its learning rate is zero. Missing/uncompleted values are never inferred.

| Step | Training loss | Owl rate (95% CI) | Drift vs Part 2b | Pearson vs Part 2b | Drift vs run step 0 | Pearson vs run step 0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7381 | 1.060% [0.482, 1.638] | 0 | 1 | 0 | 1 |
| 25 | 0.5382 | 0.980% [0.481, 1.479] | 2.82019 | 0.365792 | 2.82019 | 0.365792 |
| 50 | 0.5935 | 1.620% [0.923, 2.317] | 5.78471 | 0.344897 | 5.78471 | 0.344897 |
| 75 | 0.6018 | 2.280% [1.402, 3.158] | 5.20939 | 0.265269 | 5.20939 | 0.265269 |
| 100 | 0.594 | 3.000% [2.041, 3.959] | 6.62781 | 0.239413 | 6.62781 | 0.239413 |
| 125 | 0.6585 | 5.180% [3.775, 6.585] | 7.15248 | 0.253368 | 7.15248 | 0.253368 |
| 150 | 0.5531 | 7.780% [5.859, 9.701] | 9.47567 | 0.248672 | 9.47567 | 0.248672 |
| 175 | 0.5707 | 8.380% [6.479, 10.281] | 7.57483 | 0.245251 | 7.57483 | 0.245251 |
| 200 | 0.5217 | 9.580% [7.479, 11.681] | 8.07159 | 0.188823 | 8.07159 | 0.188823 |
| 225 | 0.5639 | 8.140% [5.996, 10.284] | 6.04997 | 0.185036 | 6.04997 | 0.185036 |
| 250 | 0.535 | 10.420% [7.833, 13.007] | 6.55051 | 0.240354 | 6.55051 | 0.240354 |
| 275 | 0.5176 | 11.880% [9.353, 14.407] | 7.33491 | 0.304818 | 7.33491 | 0.304818 |
| 300 | 0.6121 | 12.880% [10.155, 15.605] | 7.83049 | 0.258987 | 7.83049 | 0.258987 |
| 325 | 0.4647 | 10.160% [7.571, 12.749] | 6.47736 | 0.1872 | 6.47736 | 0.1872 |
| 350 | 0.4386 | 10.700% [8.323, 13.077] | 7.04127 | 0.205854 | 7.04127 | 0.205854 |
| 375 | 0.5158 | 11.520% [8.999, 14.041] | 6.10163 | 0.179458 | 6.10163 | 0.179458 |
| 400 | 0.4826 | 12.880% [9.932, 15.828] | 6.71783 | 0.229007 | 6.71783 | 0.229007 |
| 425 | 0.544 | 14.400% [11.373, 17.427] | 7.03767 | 0.169844 | 7.03767 | 0.169844 |
| 450 | 0.4845 | 14.360% [11.244, 17.476] | 6.91083 | 0.197519 | 6.91083 | 0.197519 |
| 456 | 0.4864 | 14.260% [11.209, 17.311] | 7.28492 | 0.272382 | 7.28492 | 0.272382 |

First measured checkpoint with mean owl rate strictly above 5%: 125.

The step-0 Unsloth adapter is saved and measured, rather than assuming its A initialization matches the fresh PEFT initialization of Part 2b. Drift is always against that committed Part-2b K0; initialization differences can therefore contribute to the step-0 value. The additional within-run columns use the actual saved step-0 kernel and isolate changes along this training trajectory.
