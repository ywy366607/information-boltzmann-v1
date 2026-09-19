# CBIM MaleCNS kinetic v2: operator responsibility contract

## Scope

This revision changes operator roles without increasing spatial resolution or
state capacity. The state remains 256 MaleCNS parcels and 64 channels. Those
channels are factored as eight discrete velocity channels and eight continuous
content coordinates:

\[
f_t(x,q,a),\qquad x\in\{1,\ldots,256\},\ q\in\{1,\ldots,8\},\ a\in\mathbb R^8.
\]

Neurotransmitter species, 512/1024 parcel resolution and independent internal
time substeps are later revisions. They are deliberately absent here so that a
trained result can be attributed to role separation.

## Operators

For one external token, the model evaluates

\[
f_{t+1}=\mathcal D_\gamma\,\mathcal C_\theta\,\mathcal T\,
\mathcal B_{x_t}(f_t).
\]

`GraphBoundarySource` injects one bounded, spatially localized token packet. It
does not erase old state. `GraphVelocityTransport` rotates graph spectral modes
independently for each velocity channel; it changes spatial position but never
mixes velocities. `LocalVelocityCollision` acts independently at each parcel
and mixes only discrete velocity populations. `GraphAdaptiveDissipation`
removes state energy through channel-wise positive rates and a smooth analytic
local-energy feedback.

The learned rate follows the GDN-2 initialization pattern:

\[
\gamma_{xj}^{\rm learned}
=\exp(A_j)\operatorname{softplus}(g_\theta(f_x)_j+b_{\Delta t,j}),
\]

with \(\exp(A_j)\sim U(1,16)\). GDN-2 uses
\(\Delta t_j\sim\operatorname{LogUniform}(10^{-3},10^{-1})\); this field uses
the same inverse-softplus parameterization with a calibrated
\(10^{-5}\ldots10^{-3}\) range because its bounded source injects much less
energy per step than a GDN-2 matrix update. The resulting zero-input retention
is approximately 0.984--0.99999 instead of placing every channel in one
saturated sigmoid tail. `A_log` and `dt_bias` are excluded from weight decay.

After learned selective decay, local energy \(e\) receives feedback

\[
\gamma_{\rm fb}=\frac{1}{2k}\operatorname{softplus}
\left(k\log\frac{e}{E_*}\right),
\qquad
e'=\frac{e}{(1+(e/E_*)^k)^{1/k}}\le E_*.
\]

This replaces the non-smooth safety projection. The training log records
signed source work, dissipated energy and their algebraic balance residual in
matching energy units.

## Local collision invariants

The eight velocity vectors are the normalized corners of a cube. Let

\[
C=\begin{bmatrix}1\\c_x\\c_y\\c_z\end{bmatrix},\qquad
N=\ker C.
\]

At each parcel and for every content coordinate, the collision keeps the
component orthogonal to `N` fixed and applies learned Givens rotations to the
four-dimensional nullspace coefficients. Therefore

\[
C f_x^*=C f_x,
\qquad
\lVert f_x^*\rVert_2=\lVert f_x\rVert_2
\]

up to floating-point error. The rotation angles are nonlinear functions of the
complete local state. Collision can change which velocity carries information,
but cannot move information to another parcel or create/remove state energy.

## Interpretation of the previous graph collision

The v1 edge-pair Givens rotation is retained as a historical graph-field
baseline. Because it exchanges states across parcels, it overlaps spatial
transport. Its frozen bypass effect measures use of that trained edge mixer;
it does not measure the value of a role-separated Boltzmann collision.

## Acceptance order

First verify the four operator contracts, direct CE gradients and CUDA graph
execution. Next enforce the 1.5 second/128-token speed ceiling. Only after a
full real-data training run may collision causality be measured. Matched
no-collision and topology controls follow after the trained collision has a
defined, non-overlapping role.

## v2.2: explicit local gamma

The failed v2.1 run showed that a per-token mean alpha near 0.998 still compounds to only about 0.776 retention over a 128-token update. Version 2.2 therefore uses an explicit rate field gamma[b, node, channel], generated from an independently trainable node-channel baseline together with local state, local energy, and fixed MaleCNS node features. The initial learned rate is 1.5e-4 (about 0.981 retention over 128 tokens). The analytic high-energy feedback remains separate and the hard projection remains absent.
