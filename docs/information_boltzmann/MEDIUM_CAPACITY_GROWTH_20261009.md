# Value-driven continuous capacity growth: stage-one implementation

Date: 2026-10-09. The implementation and numerical interface are accepted after
independent review. The running 8³ D768 individual remains the working baseline;
this delivery has not switched its learning rule or started another language run.

## What changes and why

Current task supervision can already reach structural coefficients. Stage one
changes how that credit reallocates finite capacity: three installed propagation
lanes compete with idle resource. Useful sensitivity, inherited structure and
maintenance jointly determine the update. Raw traffic is not used as a reward.
There are zero new trainable parameters. Existing direction learning, physical
evolution, read/write locality and persistent belief remain unchanged. Explicit
new direction adaptation, branching and algebraic recombination are later stages.

## Closing local effects into a continuous material field

Let Φ(x) be the existing continuous Fourier basis and μ its three logit columns.
With fixed window noise ε and frozen posterior standard deviation σ,

\[
 p(x)=\operatorname{softmax}([\Phi(x)(\mu+\sigma\epsilon),0]),
 \qquad c_a(x)=R p_a(x),\quad \sum_{a=0}^3c_a(x)=R.
\]

The objective is unchanged:

\[
 \mathcal F=\overline{\ell}_{\rm task}
 +[D_{KL}(q(\mu)\Vert p_{\rm window})+
 \lambda(M-M_{\rm supply})]/N.
\]

The task derivative includes each retained physical microstep and the existing
cross-event BPTT window. Removing mean from Adam does not detach its task VJP.
There is no new long-history credit algorithm hidden in the growth rule.

Local allocation geometry is pulled back to the continuous coefficient chart:

\[
 F_{ma,nb}=\sum_iw_i\Phi_{im}\Phi_{in}
 [\operatorname{diag}(p_{i,\rm active})-
 p_{i,\rm active}p_{i,\rm active}^T]_{ab},\quad \sum_iw_i=1.
\]

For the accumulated raw gradient g=∇μF, the proposed direction is

\[
 d=-[F+\operatorname{diag}(1/(N\sigma_{\rm prior}^2))]^{-1}g.
\]

The prior term is the curvature of the existing Gaussian KL/N; no additional
regularizer is charged. It makes the solve positive definite even for an
under-observed Fourier chart. Mean uses one independent updater; log_std is
frozen and both old structural Adam entries are removed explicitly.

The step remains inside the continuous chart instead of installing an arbitrary
voxel field and then pretending it is continuously representable. Consequently
different positions are coupled through Φ; this is a chart-constrained natural
step, not independent voxel-wise mirror descent. Local importance enters through
the task chain rule, and the shared chart transports that importance to μ.

Before installation, the candidate is quantized to the actual parameter dtype.
Its mean and reconstructed noisy sample both run the real allocation arithmetic.
Every sampled location must satisfy KL(new || old) ≤ max_capacity_kl, and
g·δμ < 0. A sub-ULP finite step is a counted no-op. These conditions establish a
descending linearization and bounded allocation movement; nonlinear language
benefit will be decided by real-stream evaluation.

## Numerical settings and ownership

The candidate config inherits every physical resource/time scale from the
current baseline. step_size=1 selects the full regularized natural direction;
backtracking enforces max_capacity_kl=0.001 nats per sampled location. This is an
explicit numerical movement allowance, not a biological constant. It bounds
total-variation allocation movement by √(0.001/2) ≈ 2.24% at checked locations.
Off-grid queries remain continuous; the KL guard itself is a runtime-quadrature
guard and is not claimed as a continuum-wide supremum bound.

One completed window records evidence, proposes growth, prevalidates OU/dual
state, steps ordinary Adam, applies growth, commits the existing prior/ledger,
and clears structural credit. Recompute and pre-update scoring never grow
structure. Arbitrary optimizer failures remain fatal and resume from the last
complete checkpoint; no in-place transactional optimizer rollback is claimed.

## Continuation contract

`--adopt-capacity-growth` requires an old completed-window checkpoint, this
explicit structure config and a distinct destination directory. It preserves
physical belief, clocks, prior, RNG, cursors, evaluation, exposure budget and all
other Adam moments. Total parameters must be unchanged; trainable and active
counts must decrease by exactly log_std.numel(). It rejects pending-window rule
migration, a source already using growth, unrelated settings/source changes and
combinations with broad source-compatibility exceptions. Ordinary candidate
checkpoints can resume mid-window with saved noise and pending mean gradients.

Use the inherited run's exact arguments, change output and structure-config,
resume its last complete checkpoint, replace any broad continuation exception
with `--adopt-capacity-growth`. `--check-resume-only` validates the complete
configuration before training; numerical tests cover the migration helper.

## Acceptance evidence

- Related CPU regression: **50 passed, 2 skipped** (CUDA-specific checks skipped).
- Flow, structure and material-view assets: **3 passed**.
- Existing Deslice scatter/gate script: all unit tests passed.
- Full repository pytest collection is blocked by five existing retired fly
  diagnostic/script imports; affected 3D tests execute independently.
- Independent review accepted Fisher indexing, prior scaling, full mean VJP,
  single updater, float32/no-op guards, OU preflight and continuation handling.
- Real checkpoint geometry: 512 Fourier coefficients, 1,536-dimensional solve,
  one CPU64 metric 18 MiB, median **0.23014 s/update**, no CUDA initialization.
  This timing uses explicitly constructed numerical credit, not language credit.
  Solve relative residual 1.17e-15; max allocation-sum error 4.77e-7.

Reproduce the geometry check:

```powershell
python scripts/ib/check_medium_capacity_growth.py --checkpoint results/medium_d768_streaming_pathway_8x8x8_160k/last.pt --output results/published/medium_capacity_growth_geometry_20261009.json --step-size 1 --max-capacity-kl 0.001
```

## Slow material, effective transport and actual flow

The dashboard now separates three views with their real timestamps:

1. Installed material direction: checkpoint posterior mean, excluding window
   noise, fast conduction and STP utilization.
2. Effective transport direction: current factor after fast conduction/STP.
3. Actual energy current: the field-and-flux current, also used by display tracers.

Schematically,

\[
 B_{\rm eff}(x,t)=\operatorname{diag}(u(x,t))B_{\rm installed}(x),
 \quad J(x,t)=\sum_c f_c(x,t)B_{\rm eff}(x,t)q_c(x,t).
\]

Slow structure constrains propagation, fast utilization opens/closes effective
lanes, and f/q phase plus incoming evidence determines the instantaneous current.
A fixed medium under changing input can have strongly varying current through
superposition, standing waves and interference. Nearly degenerate tensor axes
and near-zero currents can also rotate their displayed direction sharply.
Current snapshots show time-dependent wave transport; turbulence would require
separate evidence of nonlinear transfers across spatial scales. Stable directions
are compatible with persistent anisotropic material and fast-changing waves.
