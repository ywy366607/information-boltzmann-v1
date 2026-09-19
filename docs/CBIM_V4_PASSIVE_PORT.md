# CBIM v4: Passive Boundary Port

## State and evolution

The persistent state is only the kinetic field `f[x,q,a]`. There is no bulk
decay, hard projection, resource pool, fatigue field, or separate outflow.
Each token supplies a bounded incident port field `e_in`. A pointwise
orthogonal two-port scattering map produces the updated boundary field and an
explicit outgoing environment state:

\[
\begin{bmatrix}f^+\\e_{out}\end{bmatrix}=
\begin{bmatrix}\cos\Theta&\sin\Theta\\-\sin\Theta&\cos\Theta\end{bmatrix}
\begin{bmatrix}f^-\\e_{in}\end{bmatrix}.
\]

Consequently,

\[
E(f^+)-E(f^-)=E(e_{in})-E(e_{out})
\]

up to floating-point error. Local full-nullspace collision then redirects
velocity/content components and velocity-resolved graph transport moves them
through space. Both internal operators preserve their declared invariants.

## Readout and shortcut control

The instantaneous surface feature is the scattered response
`-sin(Theta) f^-`, excluding the freely reflected incident token field. It is
combined with a multi-query read of the evolved field. Thus the decoder may
use shallow boundary response and deep distributed state, but cannot predict
from a token-only port bypass.

## Infinite-stream condition

Orthogonality closes one interaction's energy ledger; it does not alone imply
an infinite-stream bound under fresh input. The analytic target additionally
requires uniform finite-window observability of internal modes at the moving
port. With bounded incident packets and a strict contraction of every driven
mode over that window, the recurrent field is input-to-state stable. Dark
unit-modulus modes are acceptable only when they are also unreachable from
the input port.

## Experiment gate

After the v3 run finishes, first run CPU invariant/gradient tests and the CUDA
Graph test. Then require less than 1.5 seconds per 128 GPT-2 BPE tokens and less
than 2 GiB total GPU memory. A 3000-update OWT comparison must report matched
validation curves, persistent-state versus reset, collision and transport
bypasses, removal of the scattered-response branch, energy ledger residuals,
rank, spatial concentration, and a numerical boundary-observability audit.
