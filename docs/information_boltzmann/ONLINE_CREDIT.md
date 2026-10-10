# Online learning without a temporal BPTT window

The existing 3D material, compact W4 write, transport, collision, conductance
response, STP and readout are unchanged. A separate learner carries forward
parameter-sensitivity factors along with the full physical belief.

## Mathematical contract

For one actual observation event, write the implemented graph as

\[
s_{t+1}=F_\theta(s_t,x_t),\qquad l_t=l_\theta(s_t,x_t,y_t).
\]

Here `l` includes the transition, readout CE and existing write free energy.
All evolving state components except the external elapsed clock are in `s`:
field, three fluxes, conduction, receptors, STP resource/utilization, precision.
Next-token targets enter only the likelihood. Clock duration is exogenous.

Exact forward RTRL is

\[
S_{t+1}=J_t S_t+B_t,\quad J_t=\partial_sF_t,
\quad B_t=\partial_\theta F_t,
\quad g_t=\partial_\theta l_t+(\partial_s l_t)S_t.
\]

The implementation keeps `S` as the average of independent rank-one factors
`u v^T`. It computes `J u` by **forward AD** and `B^T xi` by a local VJP through
the actual dynamics. Independent Rademacher probes satisfy `E[xi xi^T]=I`.
The balanced UORO update is

\[
u'=\rho_0Ju+\rho_1\xi,\qquad
v'=v/\rho_0+B^T\xi/\rho_1.
\]

The expected outer product is `Ju v^T + B`. Norm balancing minimizes the
factor norm product; its neutral value at a zero factor is 1. It is a factor
gauge, not a physiological threshold or new dynamics rate. Numerical tests
enumerate all signs of a small Jacobian to verify the expectation exactly.

Current gradients use the **pre-event** sensitivity, because `l_t` already
contains the current transition. Using post-event factors here would count
the current transition twice. Pure readout parameters have direct gradients;
parameters that affect persistent dynamics also have historical gradients.

Unbiasedness describes the frozen-parameter sensitivity recursion. Online
parameter changes have the usual RTRL interpretation; this is not exact
differentiation through the optimizer. Gradient clipping/AdamW also do not
inherit an unconditional convergence guarantee from unbiased compression.

## Persistent learning state

Storage is `O(rank * (state_size + parameter_count))`, independent of the
elapsed number of events. One-event autodiff is used and discarded. There is
no temporal activation tape and no chosen eligibility decay/credit horizon.
Jacobians determine trace persistence. Optimizer cadence, rank and solver
resolution are distinct from a credit horizon.

Checkpoint continuation includes physical belief, factors, factor RNG,
optimizer, observation offset and elapsed time. A BPTT checkpoint can start
an explicit new learning branch with zero eligibility at the branch boundary;
it cannot reconstruct sensitivity to its unavailable historical trajectory.

Native expressions are used for JVP support. The nested STP AOT wrapper has
no JVP rule; the learner disables that execution wrapper, retaining identical
STP equations. A CUDA Graph captures one event's JVP, VJPs, factor update and
gradient accumulation, and refreshes all parameter/state/token values on replay.

## Verified execution and the current bottleneck (2026-10-04)

- 27 affected CPU checks passed, 11 GPU checks skipped in that invocation;
  the online test set separately passes all 8 checks with CUDA enabled.
- First-event loss, full physical output and every parameter gradient match
  the original graph. Exact stochastic compression identity, optimizer-boundary
  persistence, graph-free state, checkpoint/RNG continuation and loss scaling
  pass. CUDA replay matches eager and follows changed material/write/read weights.
- Actual OWT, 8x8x4, d128, 14,067,383 parameters, rank1: eligibility is exactly
  **57,065,692 bytes (54.42 MiB)**. Across 64 frozen-weight events, retained live
  allocations are **134.27 MiB at every 8-event sample**, and every persistent
  tensor is detached/graph-free.
- Joint execution calibration performs **3 real optimizer updates / 384 OWT
  events** with CUDA Graph, averaging **14.78 s per 128-event update**. Retained
  allocation is **315.24 MiB** at each update; peak allocation is **628.38 MiB**;
  total dedicated GPU use is **2009 MiB**, below the 4 GiB cap.
- **Global rank1 compression is extremely noisy in the cold-start derivative
  audit.** Exact two-event historical gradient norm is **0.25374**, while 16
  independent rank1 samples have mean norm **7491.59**. Their mean estimator
  still has relative error **9079.39**. This numerical variance measurement
  does not depend on waiting for training convergence.

The bounded-storage learning interface works. Dense global rank1 UORO is
retained as a reference prototype, **not promoted to the production learner**.
It is slower and its high-dimensional cancellation noise is unacceptable for
an efficient long run. No long capability training was launched, and the
calibration NLL does not constitute an architecture ranking.

The next mathematical target is a locality-/operator-aware sensitivity
factorization: preserve site/channel/material blocks instead of mixing every
degree of freedom into a single random factor. Any e-prop/local approximation
must derive its eligibility from the actual transition and specify omitted
cross-site Jacobian terms. Full RTRL is exact but `state_size * parameter_count`
storage is prohibitive here. Raising global rank reduces sampling variance
roughly as `1/rank` at proportionate storage/work, so small rank increases alone
do not repair the measured discrepancy. Changes must pass short exact-derivative
audits before a sufficient joint OWT capability run.

## Entry points

Execution calibration (does not save a capability checkpoint):

```powershell
python scripts/ib/train_online_plastic.py --output results/<new-calibration> `
  --steps 3 --tokens 128 --event-duration 0.005 --rank 1 `
  --warmup 0 --decay 0 --execution graph --calibrate-only
```

Formal training entry supports `--initialize-from` to preserve a mature
physical checkpoint or `--resume` to continue the full online learning state.
The current global rank1 reference should first receive a variance-reducing
factorization before such a long training run. The current primary evaluation
keeps learning active, scores first-pass experience before updating, and runs
actual continuous A/B/A context-change and revisit measurements. Physical,
eligibility and optimizer state all persist. See `LIFELONG_EVALUATION.md`;
historical frozen warm-local scores keep their original protocol labels.

Numerical report:

```powershell
python scripts/ib/audit_online_credit.py `
  --output results/published/plastic_online_credit_numerical.json
```

References: [UORO, Tallec & Ollivier](https://arxiv.org/abs/1702.05043),
[e-prop, Bellec et al.](https://www.nature.com/articles/s41467-020-17236-y).
UORO supplies this implementation's equations; e-prop motivates the biological
three-factor comparison and is not used as a substitute COBA/medium derivative.

## First local migration

`train_online_plastic.py --credit local-receptors` now selects deterministic
eligibility for the medium's actual receptor closing kinetics, with exact joint
current-event gradients. It retains per-location/content/branch identity,
supports CUDA Graph and complete continuation, and leaves the physical forward
model unchanged. Its narrower historical scope, derivative error against full
history and real-data execution timing are documented in
`LOCAL_MEDIUM_CREDIT.md`. Global UORO remains a reference, while local membrane
history is the next concrete extension of this component-wise migration.
