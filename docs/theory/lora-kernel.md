# The empirical kernel of a LoRA adapter

This note writes out the kernel that `slk kernel` computes, from the row loss down to the per-layer split, and says why its A-term vanishes at the initialisation and not afterwards. It is the maths behind experiment 6 of week 6 and behind the Codex kernel parts of Oct 19.

## The row loss

A training row is a chat with a user turn, the number prompt, and an assistant turn, the teacher's completion. Let the completion's tokens, including the end-of-turn token, be $y_1, \dots, y_T$, and let $x_{<t}$ be everything the model sees before it predicts $y_t$ (the prompt plus $y_1, \dots, y_{t-1}$). Let $\theta$ be the vector of every trainable number in the adapter, and let $s_w(x; \theta)$ be the logit of vocabulary token $w$ at context $x$. The loss of row $c$ is the summed cross-entropy over the completion tokens, which is what the collator leaves unmasked, and it is a sum of log-probabilities with a minus sign.

$$
L_c(\theta) = -\sum_{t=1}^{T} \log p_{y_t}(x_{<t}; \theta), \qquad p_w(x; \theta) = \frac{\exp s_w(x; \theta)}{\sum_{w'} \exp s_{w'}(x; \theta)}.
$$

Its gradient with respect to the logits at position $t$ is the familiar softmax-minus-onehot vector, with $\delta_{w, y_t}$ equal to one when $w = y_t$ and zero otherwise.

$$
\frac{\partial L_c}{\partial s_w(x_{<t})} = p_w(x_{<t}; \theta) - \delta_{w, y_t}.
$$

Everything else in the gradient is the chain rule through the network, which the next section writes out for one patched matrix.

## One LoRA patch

A LoRA patch on a weight matrix $W$ of shape $d_{\text{out}} \times d_{\text{in}}$ replaces it by $W + BA$. The matrix $A$ has shape $r \times d_{\text{in}}$ and is drawn at random when the adapter is created; $B$ has shape $d_{\text{out}} \times r$ and starts at zero; both train. (With alpha equal to the rank, as in every run here, the scale factor alpha over rank is one and is dropped.) For an input vector $x$ at one token position the patched output is

$$
y = W x + B A x.
$$

Let $\delta$ be the gradient of the quantity in question (a row loss or a logit) with respect to $y$, a vector of length $d_{\text{out}}$ that backpropagation delivers. Then, because $y$ is linear in $B$ with coefficient $Ax$ and linear in $A$ with coefficient $B^\top \delta$ on the output side and $x$ on the input side,

$$
\frac{\partial}{\partial B} = \delta \, (A x)^\top, \qquad \frac{\partial}{\partial A} = B^\top \delta \; x^\top.
$$

A matrix occurs at every token position of the sequence, so the full gradient with respect to $A$ or $B$ is the sum of these outer products over positions. For a row loss the positions are the completion positions with their own $\delta$ each; for a logit at a question the gradient is taken at the last position only, but the chain rule still passes through every earlier position, so the sum over positions remains.

## The kernel entry as a sum over weights

Take two items, $u$ and $v$, each either a training row or a (question, token) pair, and let $g(u) = \nabla_\theta F_u(\theta)$ be the gradient of the row loss or of the logit, as the case may be. The empirical kernel between them is the dot product of the two gradient vectors, which is a sum over every trainable weight of the product of the two partial derivatives.

$$
K(u, v) = g(u) \cdot g(v) = \sum_{i=1}^{|\theta|} \frac{\partial F_u}{\partial \theta_i} \, \frac{\partial F_v}{\partial \theta_i}.
$$

With the two outer products above, the part of this sum that belongs to one patched matrix is, writing unprimed quantities for item $u$ and primed for item $v$ and summing over the token positions $n$ of $u$ and $n'$ of $v$,

$$
K_{\text{module}}(u, v) = \sum_{n, n'} \big(\delta_n \cdot \delta'_{n'}\big)\big(A x_n \cdot A x'_{n'}\big) + \sum_{n, n'} \big(B^\top \delta_n \cdot B^\top \delta'_{n'}\big)\big(x_n \cdot x'_{n'}\big).
$$

The first sum is the B-term: it compares the two items' inputs through the random sketch $A x$, an $r$-dimensional projection of the $d_{\text{in}}$-dimensional input, weighted by how aligned the two output-side gradients are. The second sum is the A-term: it compares the inputs directly, $x \cdot x'$, weighted by how aligned the two output-side gradients are after projection through $B$.

## Why the A-term is zero at the initialisation

At the initialisation $B = 0$, so $B^\top \delta = 0$ for every $\delta$ and the whole second sum vanishes. The kernel at the start is the B-term alone: a random $r$-dimensional sketch of the input times an output-side sensitivity. That is the kernel every Oct 19 number was computed from, and it is the kernel the Part 2d linearised run held fixed, since the tangent model at $B = 0$ is $f(\theta_0) + \sum_{\text{modules}} (\Delta B)(A x)$ and the deltas on $A$ have no first-order effect there.

At a trained adapter $B$ is not zero, so the A-term is live and grows with the size of $B$. For a $B$ of Frobenius norm $\|B\|$ the second sum scales as $\|B\|^2$ times the input overlap $x \cdot x'$, while the first sum does not depend on $B$ at all. This offers a reason the kernel of the control adapter moves as much as the owl adapter's in norm: part of the drift would be the switching-on of a term that was zero at the start, which has nothing to do with the trait. Whether that accounts for the measured drift is not settled, since the two terms have only been measured together. The week-6 tables at the owl and control adapters measure the two terms together; the per-block norms at `init_s1` against `owl_s1` and `control_s1` in `results/week6-oct26/analysis/kernel.md` are the measured version of this growth. Separating the two terms at a trained adapter would be a small addition to `slk kernel`, and it has not been done.

## The per-layer split

The parameter vector $\theta$ is a concatenation of blocks, one per patched matrix, and the patched matrices group by layer and by part (the four attention projections of a layer, the three MLP projections of a layer). The kernel sum over weights splits into a sum over blocks of the partial sums above, and grouping the terms of that sum by layer and part gives the per-block kernels.

$$
K(u, v) = \sum_{\ell=0}^{L-1} \Big( K^{\text{attn}}_\ell(u, v) + K^{\text{mlp}}_\ell(u, v) \Big), \qquad K^{\text{mlp}}_\ell = K^{\text{gate}}_\ell + K^{\text{up}}_\ell + K^{\text{down}}_\ell.
$$

Nothing is approximated here: the per-block Gram matrices sum to the total exactly, which is the first check of the kernel gate in `tests/`. A named layer set, such as the MLPs of layers 0 to 6, is the sum of its blocks, and its share of the total Frobenius norm is what the tables report as "where the kernel lives".

## The sign of the first-step push on a question logit

One gradient step on a batch of rows $\mathcal{B}$ with learning rate $\eta$ moves the weights by

$$
\Delta \theta = -\eta \sum_{c \in \mathcal{B}} g(c).
$$

To first order, the logit of token $w$ at question $q$ then changes by its gradient dotted with that move, which is a sum of kernel entries between the question item and the rows.

$$
\Delta s_w(q) = g(q, w) \cdot \Delta\theta = -\eta \sum_{c \in \mathcal{B}} K\big((q, w), c\big).
$$

So a positive question-by-row sum pushes the logit down and a negative one pushes it up. The tables report $\sum_c K((q, w), c)$ with its sign as computed; at `init_s1` the sum over the owl rows for the token O is $-547$, which under this formula is an upward push of $547 \eta$ on the O logit from the owl rows, and $-27$ from the control rows. The Oct 19 Part 2 prediction used the same quantity, computed by finite differences along $-\sum_c g(c)$ instead of through the kernel, and the two agree to first order; on the tiny test model the finite difference matches an actual small SGD step to within five percent. Adam and the learning-rate schedule are not in this formula, so it predicts the shape of the first step and not its size.
