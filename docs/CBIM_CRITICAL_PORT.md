# CBIM passive critical port

## Closed bulk and open boundary

The semantic field remains `f[x,q,a]`. Local collision and graph transport are
the conservative bulk. A token creates a localized incident field `e_in`; the
environment interaction is one orthogonal two-port rotation:

\[
\begin{pmatrix}f^+\\e_{out}\end{pmatrix}=
\begin{pmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{pmatrix}
\begin{pmatrix}f^-\\e_{in}\end{pmatrix}.
\]

Consequently

\[
E(f^+)-E(f^-)=E(e_{in})-E(e_{out})
\]

to floating-point precision. Writing, replacement, rejection and radiative
outflow are behaviors of this boundary. There is no bulk gamma, state clip,
resource pool, fatigue field or separate outflow operator.

## Two necessary feedback signals

Marginal perturbation growth alone is insufficient for a driven system. It
controls sensitivity to initial state but cannot prevent a contractive system
from accumulating positive input work. The controller therefore enforces two
independent open-system conditions:

\[
\bar\lambda\rightarrow0,\qquad
\overline{E(e_{in})-E(e_{out})}\rightarrow0.
\]

A detached finite-difference shadow estimates conditional tangent gain every
32 tokens. The exact local port work is observed every token. Both signals
update the actual local critical accommodation `a[x]`, projected directly onto
`[0,0.25]`; there is no unbounded logit and therefore no integral wind-up.
Half of the Lyapunov correction is spatially uniform to keep every internal
mode boundary-observable, while half follows the arriving perturbation energy.

## Pre-training gates

At initialization, a 256-token OWT controllability sweep gives conditional
gain `+4.76e-4` at `a=0`, `-4.62e-3` at `a=0.01`, and increasingly negative
values through `a=0.25`. The actuator therefore crosses zero rather than merely
saturating. In a 100-update OWT smoke run, energy rises and settles near 1.0,
mean port work changes sign around zero, and conditional gain reaches
`4.27e-5`. CUDA Graph timing is about 0.58 seconds per 128-token update with
roughly 0.4 GiB PyTorch reserved memory.
