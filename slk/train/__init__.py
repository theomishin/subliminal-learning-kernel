"""The three students, the per-step probe, and the decoupled forward.

``lora`` trains LoRA patches with Unsloth or PEFT, ``linearised`` trains deltas on the
first-order model at a parameter point, and ``full`` trains every weight. ``tracking`` holds the
probe and the trainer callback they share, and ``decouple`` the evaluation-time forward in which
chosen layers read the untrained residual stream. Each module is imported where it is needed so
that the GPU packages stay optional.
"""
