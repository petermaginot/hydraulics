# How `scipy.optimize.fsolve` Works

*An explanation for someone who took multivariate calculus 20 years ago.*

> **How to read this file.** The formulas below use LaTeX. In VSCode press
> `Ctrl+Shift+V` to open the Markdown preview, or view the file on GitHub —
> either will render the math properly.

---

## 1. The problem it solves

You have $n$ unknowns and $n$ equations, all generally nonlinear, written so
that everything is on one side:

$$
F(x) \;=\; 0
$$

where $x \in \mathbb{R}^n$ and $F: \mathbb{R}^n \to \mathbb{R}^n$.

In the hydraulics network solver, $x$ is the concatenated vector
`(free P, edge mdot)`, and $F$ is the stack of residuals:

- mass-balance residuals at each free node, plus
- pressure-drop residuals along each edge.

A solution is a vector that makes every residual simultaneously zero. There
is no closed-form way to do this in general, so `fsolve` does what people
have done since Newton: **guess, linearize, correct, repeat.**

---

## 2. Warm-up: Newton's method in one dimension

Suppose $n = 1$ — one equation $f(x) = 0$. You have a current guess $x_k$.
Taylor-expand $f$ around it:

$$
f(x_k + \Delta x) \;\approx\; f(x_k) \;+\; f'(x_k)\,\Delta x.
$$

Set the left side to zero (that's what you want) and solve for $\Delta x$:

$$
\Delta x \;=\; -\,\frac{f(x_k)}{f'(x_k)},
\qquad
x_{k+1} \;=\; x_k + \Delta x.
$$

**Geometrically:** draw the tangent line at $\bigl(x_k,\,f(x_k)\bigr)$ and
slide to where it crosses zero. That's your next guess. If $f$ is
well-behaved and you start close enough, the error roughly **squares** each
step — extremely fast convergence.

---

## 3. The multivariate generalization

With $n$ equations in $n$ unknowns, "the derivative" becomes a **matrix**,
because each output equation has a partial derivative with respect to each
input variable. That matrix is the **Jacobian**:

$$
J(x) \;=\;
\begin{bmatrix}
\dfrac{\partial F_1}{\partial x_1} & \dfrac{\partial F_1}{\partial x_2} & \cdots & \dfrac{\partial F_1}{\partial x_n} \\[8pt]
\dfrac{\partial F_2}{\partial x_1} & \dfrac{\partial F_2}{\partial x_2} & \cdots & \dfrac{\partial F_2}{\partial x_n} \\[6pt]
\vdots & \vdots & \ddots & \vdots \\[4pt]
\dfrac{\partial F_n}{\partial x_1} & \dfrac{\partial F_n}{\partial x_2} & \cdots & \dfrac{\partial F_n}{\partial x_n}
\end{bmatrix}
$$

- **Row $i$** tells you how the $i$-th residual responds to a small change
  in each unknown.
- **Column $j$** tells you how every residual responds to a small change in
  unknown $j$.

It's the multidimensional "slope." The Taylor expansion now reads:

$$
F(x_k + \Delta x) \;\approx\; F(x_k) \;+\; J(x_k)\,\Delta x.
$$

Set the left side to zero and solve for the step $\Delta x$ — but now you
can't just divide, you have to solve a **linear system**:

$$
J(x_k)\,\Delta x \;=\; -\,F(x_k).
$$

Then update:

$$
x_{k+1} \;=\; x_k + \Delta x.
$$

That's the Newton step in $n$ dimensions. Each iteration is:

1. build the Jacobian,
2. solve one linear system,
3. update.

For the network solver: $J$ is roughly
$(\text{free nodes} + \text{edges})$ square. Each entry says things like
*"if I nudge the pressure at node 3, how much does the mass-balance residual
at node 7 change?"* Most of those answers are zero — node 7 is unaffected by
node 3 unless an edge connects them — so the **true Jacobian is sparse**.
That's the point [network.md:88](network.md#L88) is making.

---

## 4. Why pure Newton isn't enough

Newton is fast when it works, but it has well-known failure modes:

- **Overshoot.** Far from the solution, the linear approximation is bad, and
  the step it predicts can land you somewhere worse than where you started.
- **Singular or near-singular Jacobian.** If $J$ can't be cleanly inverted
  at the current point, $\Delta x$ blows up.
- **Cycling.** The iteration can oscillate without converging.

So in practice nobody runs pure Newton. `fsolve` runs **Powell's hybrid
method** (MINPACK's `hybrd` routine), which keeps Newton's speed but adds
guardrails.

---

## 5. What "hybrid" means — Newton blended with descent

Define a scalar **merit function** that measures how far you are from a
root:

$$
\phi(x) \;=\; \tfrac{1}{2}\,\bigl\|F(x)\bigr\|^{2}
        \;=\; \tfrac{1}{2}\sum_{i=1}^{n} F_i(x)^{2}.
$$

$\phi$ is zero exactly at a solution and positive everywhere else. So
solving $F(x) = 0$ is equivalent to driving $\phi$ to zero. This unlocks a
second direction to move in: the **steepest-descent direction**

$$
-\nabla\phi(x) \;=\; -\,J(x)^{T}\,F(x),
$$

which is guaranteed to decrease $\phi$ for small enough steps. It's just
"walk downhill on $\phi$."

Powell's algorithm maintains a **trust region** — a ball of radius $\delta$
around the current point inside which it trusts the linear model. Each
iteration it considers two candidate steps:

1. **The Newton step**

$$
\Delta x_{N} \;=\; -\,J^{-1}\,F.
$$

   Fast when accurate, but may be huge.

2. **The Cauchy step** — the steepest-descent direction taken just far
   enough to minimize the linear model along that ray. Always points
   downhill on $\phi$ but slow.

Then it does the **dogleg**:

- If the Newton step fits inside the trust region, take it.
- Otherwise, walk from the origin toward the Cauchy point, then bend
  ("dogleg") toward the Newton point, stopping at the trust-region
  boundary.

After the step, it checks: **did $\phi$ actually decrease as much as the
linear model predicted?**

- If yes — the model is trustworthy here — **expand** $\delta$ for next
  iteration.
- If no — the model lied — **shrink** $\delta$ and try again.

The result: near the solution, the Newton step is small and accepted, and
you get Newton's quadratic convergence. Far from the solution, the trust
region keeps you from leaping into nonsense, and steepest descent ensures
you at least make progress on $\phi$.

---

## 6. The finite-difference Jacobian

To run any of this you need $J(x_k)$. You have two options:

### Analytic Jacobian

You write code that returns the partial derivatives. Exact, cheap per call,
but tedious and error-prone to derive for complex residuals.

### Finite differences

Approximate column $j$ by perturbing $x_j$:

$$
\frac{\partial F}{\partial x_j}\bigg|_{x} \;\approx\;
\frac{F(x + h\,e_j) \;-\; F(x)}{h},
$$

where $e_j$ is the $j$-th unit vector and $h$ is a small step. `fsolve`
picks $h$ from machine epsilon scaled by the variable magnitude. This is
what the hydraulics code does when you don't supply a Jacobian.

**Cost:** $n$ extra evaluations of $F$ per Jacobian build, because you have
to perturb each variable once. For a network with $50$ unknowns, every
Newton iteration costs $\sim\!51$ residual evaluations instead of $1$.
That's why [network.md:88](network.md#L88) flags it: doable for a few dozen
edges, painful past that. The sparse analytic version that note mentions
exploits the fact that most $\partial F_i / \partial x_j$ are structurally
zero, so you only compute and store the nonzero entries.

> Powell's method actually uses a further trick called a **rank-1 Broyden
> update** to refresh $J$ cheaply across iterations rather than rebuilding
> it every step from finite differences — but that's a detail.

---

## 7. Convergence and stopping

`fsolve` stops when any of the following hold:

- the residual norm $\|F(x_k)\|$ is small enough (you've effectively found a
  root),
- the step $\|\Delta x\|$ becomes tiny relative to $\|x\|$ (you've stalled
  — either at a root or at a point where the Jacobian is degenerate), or
- it exceeds the iteration budget.

You can tighten the residual tolerance via the `xtol` argument. The return
info object tells you which condition triggered.

---

## 8. Practical implications for the network solver

- **Initial guess matters.** Powell is robust but not magic — a sensible
  starting `(P, mdot)` vector (e.g., from a linear or incompressible
  pre-solve) dramatically reduces iterations.

- **Scaling matters.** If pressures are in Pa ($\sim\!10^{5}$) and mass
  flows are in kg/s ($\sim\!1$), the finite-difference step $h$ that's
  appropriate for one is wrong for the other. `fsolve` has a `diag` argument
  to rescale variables; alternatively, non-dimensionalize the residuals.

- **Sparsity is the headroom.** Going from a dense finite-difference
  Jacobian to a sparse analytic Jacobian is the standard next step when
  networks grow past *"a few dozen edges"* — exactly as
  [network.md:88](network.md#L88) notes. The math above doesn't change;
  only the cost of step 5 (building $J$ and solving $J\,\Delta x = -F$)
  does.
