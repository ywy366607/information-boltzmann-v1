# Short-term pathway plasticity in the learnable 3D medium

The objective is to let a persistent medium change how it transmits repeated
activity while retaining its signals and learnable spatial structure. This
adds the missing pathway history to the existing local E/I receptor history
and continuous heterogeneous material. It does not copy a fixed fly topology
or require a firing-rate target.

## Equation and physical interpretation

Each positive-axis edge stores resource availability `x` and utilization `u`:

\[
\dot x=\rho(1-x)-u r x,\qquad
\dot u=\kappa(U-u)+U r(1-u).
\]

These are the continuous mean-rate form of depression/facilitation in
[Tsodyks, Pawelzik & Markram (1998)](https://pubmed.ncbi.nlm.nih.gov/9573407/).
Replacing discrete spike arrivals by a continuous local activity rate is an
explicit continuum approximation. Sharing two scalars across the feature
channels of an edge is a capacity/compute choice, not a claim that all
biological synapses share release resources.

Activity is the symmetric in-flight fraction of the existing edge signal:

\[
a_e=\frac{\langle j_e^2\rangle}
 {\langle j_e^2\rangle+\tfrac12(\langle f_i^2\rangle+\langle f_j^2\rangle)},
\qquad r_e=\eta_e a_e.
\]

Zero activity is defined at zero signal, using a floating-point denominator
floor. This activity proxy is a model choice, not a measured presynaptic spike
rate. Static stored field energy by itself does not deplete the resource.
Only the endpoints and their stored edge coordinate enter the activity.

The existing Fourier material `m(x)` supplies learnable local coefficients:
`rho`, `kappa`, `eta` are exponential positive fields; `U` is a sigmoid field.
They start at the declared reference time scale and symmetric `U=1/2` prior.
These are trainable initialization choices, not prescribed biological ranges.
No degree-to-time mapping or fixed percentage of active sites is imposed.
Existing learned material also continues to parameterize C, L, resistance,
receptor kinetics, propagation and nonlinear collision at each position.

## Coupling and invariants

The pathway gain is `g_e=x_e*u_e/U_e`. At birth `x=1,u=U`, so gain is exactly
one and enabling STP does not initially shrink every pathway. Increased
utilization can facilitate a pathway; resource depletion can depress it.
This is a reparameterization of effective coupling, not removal of flux.

Transport uses `c_e=base_speed_e * g_e` with exactly the same edge coefficient
at both endpoints and in the edge response equation:

\[
\dot f_i=-c_e j_e/h,\quad
\dot f_j=c_e j_e/h,\quad
\dot j_e=c_e(f_i-f_j)/h.
\]

Pair field sum and field-plus-edge quadratic energy remain conserved for
this internal transport. Frozen material/history yields a linear transport
operator; updating history makes the entire coupled system nonlinear.
Collision remains the content/space-conditioned nonlinear local scattering
operator. STP does not introduce a channel mixing network or replace it.

The invariant interval is explicit: at `x=0` its derivative is positive, at
`x=1` nonpositive; at `u=0` positive, at `u=1` negative. Thus x,u stay in
[0,1] under the continuous equations, and `0<=g<=1/U` for finite trained
`0<U<1`. This certifies resource/gain admissibility and internal energy
closure. Long-time driven field behavior still depends on boundary supply,
electrical source work and dissipation, as in the existing medium.

For constant activity, the stationary values are:

\[
u_*={\kappa U+Ur\over\kappa+Ur},\qquad
x_*={\rho\over\rho+u_*r}.
\]

These allow both gain above and below one; neither is enforced by a loss.
At rest x recovers to one and u relaxes to its own local U.

## Integration and continuation

Exact frozen-activity receptor-like relaxations use a symmetric u-half /
x-full / u-half update. STP takes a half duration on each side of the
existing medium evolution. All substeps share the requested physical
duration: increasing numerical resolution does not multiply depletion or
thinking time. Whole-system refinement remains necessary because activity
changes while the field evolves.

`MediumState.transmission` has shape `[B,X,Y,Z,3,2]`. Writes, collision,
response, detach, clone, CUDA training/evaluation graphs and inference
graphs retain it. Runtime continuation schema 5 records the enabled law
and validates shape/range. Historical checkpoints retain their original
architecture with STP disabled; changing the law requires an explicit
branch rather than silently calling a fresh resource pool a continuation.

The production 8x8x4 grid needs **6 KiB** of additional FP32 runtime state
per individual and **108** added trainable scalars at material width 8.
Training activation/backward memory is measured separately.

`train_plastic_conductance.py` and its execution benchmark enable
`--short-term-plasticity` by default. Use `--no-short-term-plasticity` for
the preceding architecture. Core constructors keep their historical
default (disabled); explicit `short_term_plasticity=True` selects the upgrade.
New runs record resource/utilization, gain min/mean/max and spatial variance
in metrics/progress at the monitoring interval.

## Verification

`tests/test_short_term_plasticity.py` covers the invariant interval, exact
rest recovery, both steady-state regimes, local access, initial unity gain,
transport conservation, the full infinitesimal generator, refinement at a
fixed physical duration, continuous spatial heterogeneity, strict weight
loading on a finer grid, CE gradients, state persistence and resume checks.
CUDA tests exercise accumulated training gradients, evaluation and
timestamped inference with STP enabled. Numerical checks establish the
implemented equations/interfaces; language improvement requires joint
training and independent warm-site validation.

Execution audit: 115 affected CPU checks pass (six opt-in CUDA checks skipped,
one historical compiler check deselected); nine captured CUDA training,
evaluation and inference checks also pass, plus the deployment gradient-mode
rejection check. Updated-parameter inference uncovered and fixed an existing
coefficient-storage lifetime/signature issue in the runtime graph cache.

Matched real OWT execution, one warmup and one measured 128-token update,
FP32 CUDA Graph/BPTT8 at duration0.005 under the independent fly workload:
STP disabled **3.491s**, enabled **3.816s** (about **9.3%** overhead).
Peak tensor allocations were **425.05MiB** and **432.14MiB**, respectively;
new eager/graph accumulated gradients and all state components match exactly
in this audit. These are execution measurements, not language training
results. The full reports are `results/published/local_stp_execution.json`
and `results/published/local_stp_execution_baseline.json`.

## Execution fusion (2026-10-04)

The initial extra cost came from repeatedly evaluating small operations, not
the 108-parameter material map. Each token invokes two STP half updates.
The original activity calculation also recomputed field mean-square energy
across the three axes and rolled full channel tensors.

The optimized expression computes stored energy once, rolls only the scalar
energy map and reuses the facilitation half-step exponential. CUDA FP32 uses
`torch.compile`/AOT autograd to fuse forward and backward; CPU and FP64 retain
the native expression and higher-order derivatives. Equations, duration,
trainable parameters and continuation schema remain identical. Compilation
is performed during graph warmup, then cached, rather than during replay.

An isolated captured forward/backward component check at the production shape
measured **1.225ms original**, **0.913ms shared-energy native**, and **0.159ms
fused**, with maximum reference state/gradient errors **7.45e-9 / 4.77e-7**.
This is about7.7x faster for the component, not for the complete model.
The reproducible script is `scripts/ib/benchmark_stp_execution.py`;
the report is `results/published/local_stp_component_optimized.json`.
Fused native-equivalence tests include zero and nonzero durations, batch2,
every input gradient, complete accumulated CE gradients, evaluation and
timestamped runtime capture. All24 checks in the CUDA-enabled affected set
passed.

The complete real OWT execution check used the same 128-token/BPTT8 cadence
and independent fly contention. After one warmup, two measured updates were
**3.561s / 3.566s**, averaging **3.564s** versus the original STP **3.816s**.
Relative to the previous no-STP **3.491s** reference, added overhead fell from
about9.3% to about2.1%. Peak tensors decreased from432.14MiB to**425.45MiB**;
whole GPU dedicated peak was**3463MiB**, below4GiB. Complete two-chunk eager
versus graph state/gradient maximum errors were zero. Reports are execution
checks under concurrent load, not performance of an isolated card or trained
language quality: `results/published/local_stp_execution_optimized.json`.
