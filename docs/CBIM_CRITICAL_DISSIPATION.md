# CBIM Critical Dissipation: final design before implementation

## Empirical diagnosis

Matched frozen-weight OWT interventions use 1024 warm-up and 512 scored GPT-2
BPE tokens. MaleCNS v1 has NLL 7.2196; reset costs 0.9769, disabling transport
costs 0.9548, and disabling edge scattering costs 0.0247. Its conditional
finite-time Lyapunov estimate is -0.0361/token. CBIM v3 has NLL 7.5277; reset
costs 1.2026, disabling transport costs only 0.0150, and disabling local
collision costs 12.3263. Its corresponding estimate is +0.0575/token.

V3 therefore has useful persistent state and collision, but lost v1's spatial
computation and is more perturbation-expansive despite its lower field energy.
State scaling also rejects the "too cold" explanation: 0.5x slightly improves
NLL, while 2x and 4x worsen it. Resource, fatigue, and separate outflow have
near-zero causal effects and should be removed.

## Theoretical basis

A closed elastic Boltzmann core conserves energy. Entropy production by
collision does not remove continuously supplied energy. A driven persistent
system therefore needs an open boundary or reservoir. The clean split is a
GENERIC-like reversible/irreversible decomposition: conservative transport and
collision in the bulk, with irreversible exchange only at an environment
port.

Self-organized criticality does not follow from conservation alone. Its useful
ingredients here are: conservative fast bulk dynamics, boundary dissipation,
and a much slower feedback variable. The feedback target is marginal
perturbation growth, not a prescribed field energy or avalanche power law.

## Fast kinetic update

The semantic state remains only `f[x,q,a]`. A bounded full-rank token field
`u_theta(token,f,neighbors,x)` meets the field at a localized Maxwell
accommodation boundary:

\[
f_i^+=(1-\eta_i)f_i+\eta_i u_i,\qquad 0\le\eta_i\le1.
\]

This single open-boundary law performs write, replacement, rejection, and
outflow. There is no bulk gamma, hard clip, resource pool, fatigue field, or
separate outflow module. Its signed energy exchange is logged exactly.

Then apply

\[
f_{t+1}=\mathcal T_\theta\mathcal C_\theta f^+.
\]

Collision comes before transport so a token can change the local velocity and
content distribution and those new velocity channels move spatially in the
same step. `C` retains v3's full 60-dimensional invariant nullspace. `T`
retains per-channel continuous graph symbols. The readout sees only the final
field.

## Individual critical controller

Maintain a shadow perturbation `delta` and a nonnegative slow boundary
conductance `zeta[x]`. They are controller state, not semantic memory and not
learned slow/fast subspaces. With the same token stream, estimate the global
conditional tangent gain by a two-trajectory Benettin update:

\[
\delta_{t+1}=\Phi(f_t+\epsilon\delta_t,u_t)-\Phi(f_t,u_t),
\qquad
\hat\lambda=\log\frac{\|\delta_{t+1}\|_F+\varepsilon}
{\|\delta_t\|_F+\varepsilon}.
\]

Pure transport can move a perturbation between nodes without amplifying it, so
local norm ratios are not stability exponents. The global gain is distributed
according to the arriving perturbation energy, giving local control signals
whose spatial mean equals the global gain. The controller then supplies only
the additional boundary accommodation needed to stop genuine global expansion:

\[
\bar\lambda_i\leftarrow(1-\rho)\bar\lambda_i+\rho\hat\lambda_i,
\qquad
\zeta_i\leftarrow[\zeta_i+\epsilon_c\bar\lambda_i]_+,
\]

\[
\eta_i=w_i\,\sigma(g_\theta+\zeta_i).
\]

Positive perturbation growth increases boundary coupling; contraction lets
`zeta` relax toward zero. The fixed point is a near-marginal regime selected by
the individual's own stream. No target energy, fixed decay time, or manually
declared memory subspace is used. Exact zero is not claimed: finite-window
input-to-state stability still requires all input-reachable modes to be
observable at the moving boundary.

The shadow trajectory is detached from CE backpropagation and renormalized by
one global norm after every token. Energy-weighted attribution drives separate
node conductances. This adds approximately one inference forward rather than a
second backward pass; the measured complete update remains below the hardware
ceiling.

The first implementation uses a deterministic single shadow direction, an EMA
rate of 0.01 and integral-controller rate of 0.01. These constants set only the
observer/controller timescale. They do not prescribe a target energy, a memory
half-life or a semantic slow/fast partition.

## Gates

1. Algebra: collision and transport invariants; bounded Maxwell exchange; no
   hidden projection or bulk damping.
2. Engineering: 128 tokens/update below 1.5 s and total GPU memory below 2 GiB.
3. Dynamics: energy has zero long-run drift; conditional Lyapunov lies in a
   narrow band around zero without persistent controller saturation.
4. Roles: reset, collision bypass, and transport bypass must each worsen OWT
   NLL. In particular, transport must recover a material effect rather than
   v3's 0.015 NLL.
5. Capability: matched 3000-update validation must beat v3 and approach or beat
   MaleCNS v1 before any larger topology or neurotransmitter expansion.

Artifacts: `results/published/cbim_v1_dynamics.json` and
`results/published/cbim_v3_dynamics.json`.
