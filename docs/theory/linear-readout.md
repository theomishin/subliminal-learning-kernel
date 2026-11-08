# The kernel when only a linear readout trains

Sam's first theory item from Oct 28: before anything about the full LoRA model, write the kernel for the simplest student, one where a linear readout sitting on top of the pretrained unembedding is the only thing that trains. This note writes that kernel out and shows it factorises into an input kernel times an output coupling, which is exactly the separable form the Oct 19 measurements tested and found to hold for only 29% of the real kernel's variance.

## The student

Let $h(x) \in \mathbb{R}^d$ be the final hidden state of the frozen network at the last position of context $x$, after the final norm, so that the base model's logits are $U h(x)$ with $U \in \mathbb{R}^{V \times d}$ the unembedding matrix ($V$ is the vocabulary size and $d$ the hidden size). The student adds a trainable linear map in front of the unembedding, with $\Theta \in \mathbb{R}^{d \times d}$ starting at zero, so that its logits are

$$
s(x; \Theta) = U \big( h(x) + \Theta\, h(x) \big) = U h(x) + U \Theta h(x).
$$

Nothing nonlinear sits between $\Theta$ and the output, so this student is linear in its trainable weights exactly, and the first-order model is the model itself. The readout-only week-6 student (`readout_0-6`) is one step more complicated, since its trainable matrices sit inside the residual stream with 21 nonlinear layers after them; the argument below applies within one of its layers (the down projection's change is $\Delta W_{\text{down}} \, a(x)$ with $a(x)$ the base model's fixed hidden activation) and stops at the layer's output.

## The gradients

The logit of token $w$ at context $x$ is the $w$-th entry of $s$, and its derivative with respect to $\Theta$ is an outer product of the $w$-th row of $U$, call it $u_w \in \mathbb{R}^d$, with the hidden state.

$$
\frac{\partial s_w(x; \Theta)}{\partial \Theta} = u_w \, h(x)^\top.
$$

For a training position with context $c$ and target token $t$ the loss is $-\log p_t(c)$, and its gradient with respect to the logits is $p - e_t$ (the softmax probabilities minus the one-hot vector of the target), so its gradient with respect to $\Theta$ is

$$
\frac{\partial L_{(c,t)}}{\partial \Theta} = \Big( \sum_{w} \big(p_w(c) - \delta_{w,t}\big)\, u_w \Big) h(c)^\top = U^\top \big(p(c) - e_t\big)\, h(c)^\top.
$$

## The kernel, and its separable form

The kernel between a question item $(q, w)$ and a training item $(c, t)$ is the dot product of the two gradient matrices, which for outer products is the product of two dot products.

$$
K\big((q, w), (c, t)\big) = \big\langle u_w h(q)^\top,\; U^\top (p(c) - e_t)\, h(c)^\top \big\rangle = \big( h(q) \cdot h(c) \big) \cdot \big( u_w \cdot U^\top (p(c) - e_t) \big).
$$

The first factor depends only on the two contexts, and the second only on the output tokens and the residual at $c$. Writing $\mathsf{B} = U U^\top \in \mathbb{R}^{V \times V}$ for the readout overlap, whose $(w, t)$ entry is $u_w \cdot u_t$, the second factor is

$$
u_w \cdot U^\top (p(c) - e_t) = \sum_{t'} \mathsf{B}_{w t'} \big( p_{t'}(c) - \delta_{t', t} \big) = \big[\mathsf{B}\,(p(c) - e_t)\big]_w .
$$

So the kernel is exactly separable as an input kernel $k(q, c) = h(q) \cdot h(c)$ times an output coupling given by the readout overlap applied to the residual.

$$
K\big((q, w), (c, t)\big) = k(q, c)\; \big[\mathsf{B}\,(p(c) - e_t)\big]_w .
$$

Summing over the training positions of a row and over rows gives the first-step change of the question logit as

$$
\Delta s_w(q) = -\eta \sum_{(c,t)} k(q, c)\, \big[\mathsf{B}\,(p(c) - e_t)\big]_w .
$$

## What this says for the owl

In expectation over the teacher's sampling of $t$ at context $c$ the residual is $p(c) - p_T(c) = -\delta(c)$, the negative of the teacher's tilt. The owl entry of the first-step change is then a sum over contexts of $k(q, c)$ times $[\mathsf{B}\,\delta(c)]_{\text{owl}}$, and the latter is $\sum_{t'} \mathsf{B}_{\text{owl}, t'}\, \delta_{t'}(c)$ over the ten digit tokens. Two measured facts enter here. The row $\mathsf{B}_{\text{owl}, \cdot}$ restricted to the digits is flat, because the ten digit rows of $U$ all point along the same generic short-token direction and the owl row is at the 41st percentile along it (`results/kernel-oct19/unembedding/README.md`); and $\delta(c)$ restricted to the digits sums to the teacher's extra mass on digits, $\sum_{t' \in \text{digits}} \delta_{t'}(c) = m_T(c) - m_0(c)$, which the tilt test measures as `digit_mass_teacher` and `digit_mass_base` and which is small (the two masses differ by less than 0.004 in `results/kernel-oct19/tilt/`, against a total of one), so over the digits $\delta(c)$ is a zero-sum vector up to that difference. A flat row dotted with a zero-sum vector is zero, so the owl entry of the first-step change vanishes context by context up to the flat row's value times that mass difference, whatever $k(q, c)$ is. This is the no-go argument for the separable kernel, and it is why the separability of the real kernel was measured: on the real LoRA kernel at the initialisation the separable fit with $\mathsf{B}$ explains 28.8% of the table's variance, the best rank-one fit of any kind explains 45.7%, and the remaining 54.3% is non-separable in any shared output matrix, so the real student has room that this one does not.

One thing to note about the input kernel. For the readout student $k(q, c)$ is the overlap of two final hidden states, the animal question's and a number context's, and nothing in the training ever changes it. The separability measurement found the real kernel nearly diagonal across contexts (self-overlap about 1,150 against about 80 for another context at the same digit), which for the readout student would mean the number contexts are far from each other and from the questions in $h$-space, and a small $k(q, c)$ on every term of the sum. The size of $k(q, c)$ between the 50 questions and the number contexts under the pretrained $h$ has not been measured directly, and is the one number this note still needs.
