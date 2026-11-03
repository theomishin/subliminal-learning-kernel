# The multi-output kernel in the output-scaling framing

Sam's second theory item from Oct 28: do the many-output kernel properly, in the framing of Chizat, Oyallon and Bach (2019), where laziness is a consequence of a scale factor on the output and not of width. This note sets that up for a student with a full vocabulary of outputs, shows why the naive infinite-width limit makes every output its own single-output problem, says what a chosen readout covariance does instead, and ends on the question Sam left open.

## The scaled student

Let $f(x; \theta) \in \mathbb{R}^V$ be the vector of logits the network produces at the last position of context $x$, with $\theta \in \mathbb{R}^P$ the trainable weights (for us the LoRA numbers) and $V$ the vocabulary size. Let $\theta_0$ be the starting point and define the centred model, which is zero at the start,

$$
\tilde f(x; \theta) = f(x; \theta) - f(x; \theta_0).
$$

The output-scaled student with multiplier $\alpha > 0$ is the base model's logits plus $\alpha$ times the centred change,

$$
f_\alpha(x; \theta) = f(x; \theta_0) + \alpha\, \tilde f(x; \theta),
$$

so that at $\alpha = 1$ it is the ordinary student. Let $R(f)$ be the training loss as a function of the logits on the training set (the summed cross-entropy over completion tokens, or a squared loss in the tractable version). Chizat, Oyallon and Bach train the scaled model on the objective divided by $\alpha^2$,

$$
F_\alpha(\theta) = \frac{1}{\alpha^2}\, R\big(f_\alpha(\theta)\big),
$$

by gradient flow $\dot\theta = -\nabla_\theta F_\alpha(\theta)$. The factor $1/\alpha^2$ is a choice of learning rate that keeps the function-space dynamics of order one as $\alpha$ grows.

## The linearisation and the matrix-valued kernel

Let $J(x; \theta) \in \mathbb{R}^{V \times P}$ be the Jacobian of the logits at $x$ with respect to the weights, whose $w$-th row is $\nabla_\theta f_w(x; \theta)$. The chain rule gives the motion of the scaled logits under the flow as

$$
\frac{d}{dt} f_\alpha(x; \theta_t) = \alpha J(x; \theta_t)\, \dot\theta_t = -\sum_{x' \in \text{train}} J(x; \theta_t) J(x'; \theta_t)^\top \, \nabla_{f(x')} R\big(f_\alpha(\theta_t)\big),
$$

which is a kernel gradient flow in function space with the matrix-valued tangent kernel

$$
K_t(x, x') = J(x; \theta_t)\, J(x'; \theta_t)^\top \in \mathbb{R}^{V \times V}, \qquad \big[K_t(x, x')\big]_{w w'} = \nabla_\theta f_w(x; \theta_t) \cdot \nabla_\theta f_{w'}(x'; \theta_t).
$$

The entry $(w, w')$ of the block for the pair $(x, x')$ is what the rest of this repository writes as $K((x, w), (x', w'))$, the dot product of two gradient vectors, and `docs/theory/lora-kernel.md` writes it out for the LoRA weights. The weights themselves move by $\dot\theta = -(1/\alpha) J^\top \nabla R$, so over a fixed training horizon their displacement from $\theta_0$ shrinks like $1/\alpha$, and the Jacobian, which is continuous in $\theta$, stays within $O(1/\alpha)$ of its starting value. In the limit the kernel is frozen at $K_0(x, x') = J(x; \theta_0) J(x'; \theta_0)^\top$ and the scaled model follows the linearised model

$$
f_\alpha^{\text{lin}}(x; \theta) = f(x; \theta_0) + \alpha\, J(x; \theta_0)\,(\theta - \theta_0),
$$

which is the model the Part 2d run trained with $\alpha = 1$ and the Jacobian held at $\theta_0$ by hand. Chizat, Oyallon and Bach's Theorem 2.2 makes this precise on any finite horizon, for any differentiable model and so for any width, with the gap to the linearised path of order $1/\alpha$ in the outputs; the laziness is governed by the number they call the inverse relative scale,

$$
\kappa = \frac{\|f(\theta_0) - y^\star\|\; \|D^2 \tilde f(\theta_0)\|}{\alpha\, \|J(\theta_0)\|^2},
$$

where $y^\star$ is the target the loss pulls the logits toward and $D^2 \tilde f$ is the second derivative of the centred model; small $\kappa$ means lazy. The uniform-in-time version (their Theorem 2.4) needs the loss to be strongly convex in the logits, which holds for squared loss and fails for cross-entropy through a softmax; for cross-entropy the finite-horizon statement is what applies. To check: whether the drift under cross-entropy that the literature review attributes to Yu et al. 2025 survives the $1/\alpha^2$ scaling or is a statement at $\alpha = 1$ only.

## Why naive infinite width makes the outputs independent

Take the last layer explicitly. Let $\phi(x; \vartheta) \in \mathbb{R}^N$ be the features the network produces at the last position, with $\vartheta$ the inner weights and $N$ the width, and let the logits be a linear readout in the standard scaling,

$$
f_w(x) = \frac{1}{\sqrt N} \sum_{i=1}^{N} U_{w i}\, \phi_i(x; \vartheta),
$$

with the readout entries $U_{w i}$ drawn independently from a distribution of mean zero and variance one. If both $U$ and $\vartheta$ train, the kernel block has two parts, one from the readout and one from the inner weights,

$$
\big[K(x, x')\big]_{w w'} = \frac{\delta_{w w'}}{N} \sum_{i} \phi_i(x)\, \phi_i(x') \;+\; \frac{1}{N} \sum_{i, j} U_{w i}\, U_{w' j}\; \nabla_\vartheta \phi_i(x) \cdot \nabla_\vartheta \phi_j(x').
$$

The readout part is diagonal in the outputs by construction: the gradient of $f_w$ with respect to $U$ lives in row $w$ alone. The inner part couples outputs through the products $U_{w i} U_{w' j}$. As $N \to \infty$ with the features exchangeable, the law of large numbers replaces the sum over $i, j$ by its expectation, and with independent readout rows that expectation is $\mathbb{E}[U_{w i} U_{w' j}] = \delta_{w w'} \delta_{i j}$, so the inner part becomes diagonal in the outputs as well. The whole kernel collapses to a scalar kernel times the identity on outputs,

$$
K(x, x') \;\longrightarrow\; k(x, x')\, I_V, \qquad k(x, x') = \lim_{N} \frac{1}{N} \sum_i \Big( \phi_i(x) \phi_i(x') + \nabla_\vartheta \phi_i(x) \cdot \nabla_\vartheta \phi_i(x') \Big).
$$

Under such a kernel the function-space flow of output $w$ reads only the residual of output $w$,

$$
\frac{d}{dt} f_w(x) = -\sum_{x'} k(x, x')\, \big[\nabla_{f(x')} R\big]_w,
$$

so every output is its own single-output kernel problem, with the same input kernel for all of them. For the owl this is fatal at first order: the residual of the owl logit at a number context is $p_{\text{owl}}(c) - 0$, which is a small number that does not depend on which digit the teacher wrote, so the owl rows and the control rows push the owl logit at the question in the same way and the trait cannot pass. The corrections to this picture at finite width are of order $1/\sqrt N$ in the cross-output blocks.

## What a chosen readout covariance does instead

The pretrained unembedding is not a matrix of independent rows. Model it as $U = \Sigma^{1/2} Z$ with $Z$ independent and $\Sigma \in \mathbb{R}^{V \times V}$ a chosen row covariance, so that $\mathbb{E}[U_{w i} U_{w' j}] = \Sigma_{w w'} \delta_{i j}$. The same large-width computation then gives

$$
K(x, x') \;\longrightarrow\; k_U(x, x')\, I_V + k_\vartheta(x, x')\, \Sigma,
$$

with $k_U$ the feature overlap from the readout part and $k_\vartheta$ the Jacobian overlap from the inner part. If the readout is frozen, as the unembedding is in every run here, only the second term remains and the kernel is $k_\vartheta(x, x') \Sigma$: separable, with the output coupling given once and for all by $\Sigma$. The first-step change of the owl logit at a question is then

$$
\Delta f_{\text{owl}}(q) = -\eta \sum_{(c, t)} k_\vartheta(q, c)\, \big[\Sigma\, (p(c) - e_t)\big]_{\text{owl}},
$$

which is the formula of `docs/theory/linear-readout.md` with $\Sigma$ in place of $U U^\top$. The output coupling is now a design choice in place of the identity, and it is the only place the trait can enter at first order in this limit: the owl row of $\Sigma$ restricted to the ten digits has to vary across digits, so that it does not annihilate the zero-sum tilt $\delta(c)$. With $\Sigma = U U^\top$ for Qwen's unembedding it does not vary, which is the measured no-go, and the separable fit with that $\Sigma$ explains 28.8% of the real kernel's variance at the initialisation.

The real finite-width kernel does something the limit does not. With $U$ fixed and the inner weights training, the exact kernel is

$$
\big[K(x, x')\big]_{w w'} = u_w^\top\, M(x, x')\, u_{w'}, \qquad M(x, x') = J_\phi(x)\, J_\phi(x')^\top \in \mathbb{R}^{N \times N},
$$

with $J_\phi$ the Jacobian of the features and $u_w$ the $w$-th row of $U$. It is separable only when the feature kernel $M(x, x')$ is a scalar times a fixed matrix. At infinite width $M$ is isotropic and the separable form above follows; at finite width in a pretrained network $M(x, x')$ is a different anisotropic matrix for every pair of contexts, so the output coupling changes from pair to pair. That is the 54.3% non-separable remainder the Oct 19 measurement found, and it is a width effect, not an $\alpha$ effect.

## The open question

Sam's question, which this note does not settle, is this. At finite width, as the output multiplier $\alpha$ grows, does the kernel stay frozen, and what is its closed form? Two things can be said and the rest is to check.

The first half has an answer on any finite training horizon: Theorem 2.2 makes no assumption on width, so at the real network's width the kernel freezes at $K_0$ as $\alpha \to \infty$, and the frozen kernel keeps whatever finite-width structure $K_0$ has, including the non-separable part, since $\alpha$ rescales the output and does nothing to the features. The Part 2d run is that limit taken by hand, and the trait did not come through it. What is not known is the rate: how large $\alpha$ has to be, relative to the $\alpha = 1$ of the real run, for the real fine-tune to sit near the linearised one, which is the number $\kappa$ above evaluated for this network and this loss, and whether the finite-horizon statement covers the 456 steps the real run takes.

The second half has a closed form at the LoRA initialisation, where $B = 0$ and only the B-term of `docs/theory/lora-kernel.md` survives,

$$
\big[K_0(x, x')\big]_{w w'} = \sum_{\text{modules}} \sum_{n, n'} \big(\delta^{(w)}_n \cdot \delta'^{(w')}_{n'}\big)\big(A x_n \cdot A x'_{n'}\big),
$$

with $\delta^{(w)}_n$ the gradient of logit $w$ with respect to the module's output at position $n$, which depends on $w$ through the unembedding row $u_w$ and on everything between the module and the output. Whether this can be written in closed form as a function of the pretrained weights, with the attention and MLP Jacobians between the patched module and the readout made explicit, so that the output coupling $u_w^\top M(x, x') u_{w'}$ can be read off without a forward and backward pass, is the part to derive. To check as well: whether linearising at a trained adapter, where the A-term is live and the kernel is the sum of both terms, changes the answer to the first half, since at such a point $\tilde f$ is no longer zero at the linearisation point and the centring has to be redone there.
