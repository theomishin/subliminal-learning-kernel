# Literature

The full review is the Claude doc "Subliminal learning — literature review" (https://claude.ai/artifact/QMW743zvWPsXxLmtLiCe7X), compiled on Nov 1, 2026: about six hundred links in seven sections with one sentence each, running from the paper and its citations through the misalignment and poisoning literature, what distillation transfers, the kernel view of fine-tuning, multi-output kernels, weak-to-strong transfer and the fingerprint literature. The twelve papers below are the ones this repository's experiments and theory notes rest on, with one sentence on what each one gives us.

Cloud et al. 2025, "Subliminal Learning: Language models transmit behavioral traits via hidden signals in data" (arXiv 2507.14805). The phenomenon and the recipe we replicate: a teacher with a trait writes number lists, a student with the same initialisation trained on them acquires the trait, and the paper's one-step theorem says the student moves toward the teacher whenever the two share an initialisation.

Malladi et al. 2023, "A Kernel-Based View of Language Model Fine-Tuning" (arXiv 2210.05643). The claim that fine-tuning a pretrained model often follows kernel dynamics, with the empirical tangent kernel at the pretrained point as the object, which is the picture every kernel measurement here tests.

Chizat, Oyallon and Bach 2019, "On Lazy Training in Differentiable Programming" (arXiv 1812.07956). Laziness as a consequence of an output scale factor and not of width, with a bound on the distance between the real and the linearised paths; the framing of `docs/theory/multi-output.md`.

Cheng and Lippl 2026, "A mathematical theory of balancing relational generalization and memorization" (arXiv 2605.22972). Sam's own kernel-ridge-regression account of when a fine-tuned model generalises a rule and memorises an exception, the theory this project extends to a vocabulary of outputs.

Li et al. 2026, "Why Ghost Outputs Teach: A Kernel-Based Understanding of Subliminal Learning" (arXiv 2609.23260). Writes the transfer through auxiliary outputs as a chained cross-task kernel of the form $M M^\top$, positive semi-definite under a shared initialisation, with the auxiliary dimension as a rank bottleneck; the nearest existing theory to ours and the one to position against.

Shan and Bordelon 2022, "A Theory of Neural Tangent Kernel Alignment and Its Influence on Training" (arXiv 2105.14301). The one paper found that models the per-output structure of kernel evolution, each output head's kernel aligning to its own target, which is what a vocabulary of outputs needs.

Woodworth et al. 2020, "Kernel and Rich Regimes in Overparametrized Models" (arXiv 2002.09277). The initialisation scale decides whether gradient descent behaves as a kernel method or learns richer sparse solutions, the first of the implicit-bias papers Sam set.

Azulay et al. 2021, "On the Implicit Bias of Initialization Shape: Beyond Infinitesimal Mirror Descent" (arXiv 2102.09769). The relative sizes of the weights in different layers, and not only the overall scale, pick the solution gradient flow finds; relevant because B starts at zero and A does not.

Gunasekar et al. 2018, "Characterizing Implicit Bias in Terms of Optimization Geometry" (arXiv 1802.08246). Which zero-loss solution an optimiser reaches is a property of the optimiser's geometry, the background for asking what Adam does to the kernel picture.

Schrodi, Kempf, Barez and Brox 2025, "Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer" (arXiv 2509.23886). The transfer rides on a small set of divergence tokens where differently biased teachers would predict different next tokens, masking them removes most of the effect, and early layers matter most; the closest empirical neighbour to our tilt measurement and our localisation.

Shah, Chooi, Ou and Feng 2026, "Covert Influence Between Language Models" (arXiv 2606.04071). Carriers of a behavioural tendency compared across the interfaces through which one model can influence another (supervised fine-tuning, on-policy distillation, in-context learning), with per-sample attribution picking carriers that transfer more; the paper that frames number carriers against natural-language ones.

Thrush et al. 2026, "Synthetic Data for any Differentiable Target" (arXiv 2604.08423). A data generator trained so that fine-tuning on its outputs moves the target model's weights toward a chosen goal, which is the constructive version of the question we ask about the number lists.
