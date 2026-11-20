"""The transfer between two copies of a linear model with one hidden layer.

The model reads an input x, a list of n numbers, and gives a score to each of its outputs:

    hidden state  h = W x          W has m rows and n columns
    score of w    s_w(x) = u_w . h  u_w is the readout row of output w

One output is called the owl and k others are called the digits. A teacher and a student start
from the same weights (U0, W0).

    The teacher is trained so that its owl score equals beta . x for a fixed random list beta.
    The digit outputs are not in its loss. Its digit scores still change, because every score
    passes through the hidden weights W.

    The student is trained to match the teacher's digit scores, by squared error. The owl
    output is not in its loss.

What is read is the share of the teacher's owl change the student ends with. Every score is
linear in x, so the owl output is a list b with s_owl(x) = b . x, and

    share = (student's change of b) . (teacher's change of b) / |teacher's change of b|^2.

A share of 1 means the student took over the teacher's whole owl change and 0 means none.

Which weights are trained and how the model is read:

    train "hidden"   only W is trained. The scores are then linear in the trained weights, so
                     this is a kernel model exactly, with no approximation.
    train "all"      W and the readouts U are trained.
    kind "ordinary"  the scores are U W x, with the change from the start multiplied by alpha:
                     start + alpha * (U W x - start), and the step size divided by alpha^2.
                     A large alpha is the kernel regime and a small one is far from it.
    kind "first-order"  the scores are the straight-line approximation around the start,
                     U0 W0 x + (U - U0) W0 x + U0 (W - W0) x. This is the kernel model of the
                     ordinary one.
    hidden "dense"   W is any matrix.
    hidden "diagonal"  W is a diagonal matrix, so hidden unit j reads input j only.

Known answers the tests hold the code to, both with train "hidden" and enough inputs:

    dense     share = |P u_owl|^2 / |u_owl|^2, where P projects onto the span of the digits'
              readout rows. With random readouts this is k / m on average.
    diagonal  share = 1 for any k of at least one.

Numpy only. `python3 slk/toy/linear_transfer.py` prints the table and writes the numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OWL = 0
KINDS = ("ordinary", "first-order")
# The last teacher trained, so that the students of one table row share it.
_TEACHERS: dict = {}


def start(seed: int, n: int, m: int, k: int, hidden: str = "dense"):
    """The shared starting weights: readouts U0 with the owl in row 0 and the digits after it,
    and hidden weights W0. Every score starts with variance about one on a standard input."""
    rng = np.random.default_rng(seed)
    if hidden == "diagonal":
        m = n
        w0 = np.diag(rng.standard_normal(n))
    elif hidden == "dense":
        w0 = rng.standard_normal((m, n)) / np.sqrt(n)
    else:
        raise ValueError(f"hidden is {hidden!r}, not dense or diagonal")
    u0 = rng.standard_normal((1 + k, m)) / np.sqrt(m)
    return u0, w0


def score_map(u, w, u0, w0, kind: str, alpha: float = 1.0) -> np.ndarray:
    """The matrix B with scores = B x: one row per output, one column per input."""
    b0 = u0 @ w0
    if kind == "ordinary":
        return b0 + alpha * (u @ w - b0)
    if kind == "first-order":
        return b0 + (u - u0) @ w0 + u0 @ (w - w0)
    raise ValueError(f"kind is {kind!r}, not one of {KINDS}")


def loss_and_gradient(u, w, u0, w0, kind, alpha, rows, target, second_moment, train, hidden):
    """The mean squared gap between the scores of `rows` and `target` over the training inputs,
    and its gradient with respect to U and W.

    `target` holds one list per row, the scores it asks for being target . x, and
    `second_moment` is X^T X / N of the training inputs, which is all a squared error on scores
    that are linear in x depends on."""
    gap = score_map(u, w, u0, w0, kind, alpha)[rows] - target
    loss = float(np.einsum("rn,nq,rq->", gap, second_moment, gap)) / len(rows)
    on_map = np.zeros((u.shape[0], w.shape[1]))
    on_map[rows] = 2.0 * gap @ second_moment / len(rows)
    if kind == "ordinary":
        grad_u, grad_w = alpha * on_map @ w.T, alpha * u.T @ on_map
    else:
        grad_u, grad_w = on_map @ w0.T, u0.T @ on_map
    if train == "hidden":
        grad_u = np.zeros_like(grad_u)
    elif train != "all":
        raise ValueError(f"train is {train!r}, not hidden or all")
    if hidden == "diagonal":
        grad_w = np.diag(np.diag(grad_w))
    return loss, grad_u, grad_w


def fit(u0, w0, kind, alpha, rows, target, second_moment, train, hidden, steps, tolerance):
    """Gradient descent from the starting weights until the loss is `tolerance` times its first
    value or `steps` have run. A step that would raise the loss is halved and tried again, and
    an accepted step lets the next one be a little longer, so no step size has to be chosen.
    The first step size is divided by alpha^2, which keeps the scores moving at the same speed
    whatever alpha is."""
    u, w = u0.copy(), w0.copy()
    args = (u0, w0, kind, alpha, rows, target, second_moment, train, hidden)
    loss, grad_u, grad_w = loss_and_gradient(u, w, *args)
    first = loss
    step = 1.0 / (alpha * alpha if kind == "ordinary" else 1.0)
    taken = 0
    while taken < steps and loss > tolerance * first:
        while True:
            new_u, new_w = u - step * grad_u, w - step * grad_w
            new_loss, new_grad_u, new_grad_w = loss_and_gradient(new_u, new_w, *args)
            if new_loss <= loss or step < 1e-300:
                break
            step /= 2.0
        u, w, loss, grad_u, grad_w = new_u, new_w, new_loss, new_grad_u, new_grad_w
        step *= 1.1
        taken += 1
    return u, w, {"first_loss": first, "last_loss": loss, "steps": taken}


def predicted_share_hidden_only(u0, hidden: str) -> float:
    """The known answer with only the hidden weights trained and at least n training inputs.

    Dense: the part of the owl's readout row that lies in the span of the digits' readout rows,
    as a fraction of its squared length. Diagonal: 1, as long as every input coordinate is read
    by at least one digit."""
    if hidden == "diagonal":
        return 1.0
    owl, digits = u0[OWL], u0[1:]
    inside = digits.T @ np.linalg.lstsq(digits.T, owl, rcond=None)[0]
    return float(inside @ inside / (owl @ owl))


def run_one(
    seed: int,
    n: int,
    m: int,
    k: int,
    hidden: str,
    train: str,
    teacher_kind: str,
    student_kind: str,
    alpha: float = 1.0,
    inputs_per_dim: int = 10,
    steps: int = 20000,
    tolerance: float = 1e-12,
) -> dict:
    """One teacher and one student from the same start; returns the share and how well each
    fitted its own loss."""
    u0, w0 = start(seed, n, m, k, hidden)
    rng = np.random.default_rng([seed, 1])
    beta = rng.standard_normal(n) / np.sqrt(n)
    x_teacher = rng.standard_normal((inputs_per_dim * n, n))
    x_student = rng.standard_normal((inputs_per_dim * n, n))
    moment_teacher = x_teacher.T @ x_teacher / len(x_teacher)
    moment_student = x_student.T @ x_student / len(x_student)
    digits = list(range(1, 1 + k))

    key = (seed, n, m, k, hidden, train, teacher_kind, inputs_per_dim, steps, tolerance)
    if key not in _TEACHERS:
        ut, wt, teacher_fit = fit(
            u0, w0, teacher_kind, 1.0, [OWL], beta[None, :], moment_teacher, train, hidden,
            steps, tolerance,
        )  # fmt: skip
        _TEACHERS.clear()
        _TEACHERS[key] = (score_map(ut, wt, u0, w0, teacher_kind), teacher_fit)
    teacher_map, teacher_fit = _TEACHERS[key]
    us, ws, student_fit = fit(
        u0, w0, student_kind, alpha, digits, teacher_map[digits], moment_student, train, hidden,
        steps, tolerance,
    )  # fmt: skip
    student_map = score_map(us, ws, u0, w0, student_kind, alpha)

    start_map = u0 @ w0
    teacher_change = teacher_map[OWL] - start_map[OWL]
    student_change = student_map[OWL] - start_map[OWL]
    digit_change = teacher_map[digits] - start_map[digits]
    return {
        "seed": seed, "n": n, "m": w0.shape[0], "k": k, "hidden": hidden, "train": train,
        "teacher": teacher_kind, "student": student_kind, "alpha": alpha,
        "share": float(student_change @ teacher_change / (teacher_change @ teacher_change)),
        "predicted_share_hidden_only": predicted_share_hidden_only(u0, hidden),
        "k_over_m": k / w0.shape[0],
        "teacher_owl_change": float(np.linalg.norm(teacher_change)),
        "teacher_digit_change": float(np.linalg.norm(digit_change)),
        "teacher_fit": teacher_fit,
        "student_fit": student_fit,
    }  # fmt: skip


def student_columns(train: str, alphas: list[float]) -> list[tuple[str, str, float]]:
    """The students of one table: a label, a kind and a multiplier each. With only the hidden
    weights trained every student is the same kernel model, so one column is enough."""
    if train == "hidden":
        return [("kernel model", "first-order", 1.0)]
    return [("first-order", "first-order", 1.0)] + [
        (f"alpha {a:g}", "ordinary", float(a)) for a in alphas
    ]


def run_all(args) -> list[dict]:
    results = []
    for hidden in args.hidden:
        for train in args.train:
            teachers = ["first-order"] if train == "hidden" else list(KINDS)
            for teacher in teachers:
                for k in args.k:
                    if hidden == "dense" and k > args.m // 4:
                        continue
                    for seed in args.seeds:
                        for _, kind, alpha in student_columns(train, args.alpha):
                            results.append(
                                run_one(
                                    seed,
                                    args.n,
                                    args.m,
                                    k,
                                    hidden,
                                    train,
                                    teacher,
                                    kind,
                                    alpha,
                                    args.inputs_per_dim,
                                    args.steps,
                                )  # fmt: skip
                            )
    return results


def table(results: list[dict], alphas: list[float]) -> str:
    """One block per model, trained weights and teacher: a row per k with the mean share over
    the seeds for each student, and the known answer beside it."""
    lines = []
    blocks = sorted({(r["hidden"], r["train"], r["teacher"]) for r in results})
    for hidden, train, teacher in blocks:
        block = [
            r
            for r in results
            if (r["hidden"], r["train"], r["teacher"]) == (hidden, train, teacher)
        ]
        columns = student_columns(train, alphas)
        lines.append("")
        lines.append(
            f"{hidden} hidden layer, {train} weights trained, {teacher} teacher "
            f"(n {block[0]['n']}, m {block[0]['m']}): share of the teacher's owl change"
        )
        head = f"  {'k':>4} {'k/m':>7} {'answer if hidden only':>22}"
        lines.append(head + "".join(f" {label:>13}" for label, _, _ in columns))
        for k in sorted({r["k"] for r in block}):
            rows = [r for r in block if r["k"] == k]
            answer = np.mean([r["predicted_share_hidden_only"] for r in rows])
            line = f"  {k:>4} {rows[0]['k_over_m']:>7.3f} {answer:>22.4f}"
            for _, kind, alpha in columns:
                shares = [r["share"] for r in rows if r["student"] == kind and r["alpha"] == alpha]
                line += f" {np.mean(shares):>13.4f}"
            lines.append(line)
        unfinished = [
            r for r in block
            if r["student_fit"]["last_loss"] > 1e-6 * r["student_fit"]["first_loss"]
        ]  # fmt: skip
        if unfinished:
            lines.append(
                f"  {len(unfinished)} of {len(block)} students here ended with a loss above a "
                "millionth of its first value; their shares are of a student still training"
            )
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=40, help="numbers in an input")
    p.add_argument("--m", type=int, default=200, help="hidden units of the dense model")
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 5, 10, 20, 50], help="digit outputs")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4, 16])
    p.add_argument("--hidden", nargs="+", default=["dense", "diagonal"])
    p.add_argument("--train", nargs="+", default=["hidden", "all"])
    p.add_argument("--inputs-per-dim", type=int, default=10, help="training inputs per input")
    p.add_argument("--steps", type=int, default=5000, help="most gradient steps of one fit")
    p.add_argument("--out", type=Path, default=Path("linear-transfer.json"))
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    results = run_all(args)
    args.out.write_text(json.dumps(results, indent=1) + "\n")
    print(table(results, args.alpha))
    print(f"\nwritten {args.out}: {len(results)} runs")


if __name__ == "__main__":
    main()
