# Fourth pillar: health and information retention of the persistent medium

Primary capability evaluation remains live prediction, real context adaptation,
and actual A-to-B-to-A retention/relearning. The fourth pillar explains the
mechanisms accompanying those outcomes. It adds no optimizer objective,
controller, state reset, artificial noise, or criticality target.

## Actual-event accounting

`EventHealthCapture` observes the same write/evolve/read event that supplies
the real learner's loss. Fixed-address outputs are included in CUDA Graph
capture; the CPU auditor consumes them only after an event successfully
completes. Forward-AD reference executions expose primal measurements without
feeding diagnostic derivatives back into the objective.

The expression interface also records feature RMS before/after the final norm,
vocabulary-logit standard deviation and predictive entropy from the actual
pre-target logits. These are descriptive calibration measurements, separate from
physical entropy or the internal energy budget. See `MEDIUM_DECODER_SCALE.md`.

For unit-torus quadrature volume `v = 1/(X Y Z)`, energy is

    E = v/2 * (sum(field**2) + sum(flux_x**2 + flux_y**2 + flux_z**2)).

The writer's real incident and reflected wave energies close

    E_after_write - E_before = E_incident - E_reflected.

The native quadratic bath returns its actual released energy. The conductance
medium instead returns integrated reversal-source work and Joule heat from
the same electrical transition. The evolution ledger is

    E_after - E_after_write = W_response - D + R_evolution.

Both per-event balances, energy continuity between events, and the telescoping
whole-observation-interval balance are reported. Write work can be negative;
reversal-source work can be signed. Receptors, structural conduction, STP and
precision are dynamical memory variables, but have no assigned energy in this
model. The ledger therefore describes the declared field/edge storage function,
not metabolic ATP or a thermodynamic energy for every state variable.

NLL is predictive surprisal. It is never added to conductance, ALIF state, or
weight decay to manufacture entropy inflow/outflow. Physical entropy production
would require a justified stochastic/environment model, e.g. the trajectory
definitions in [Seifert (2012)](https://arxiv.org/abs/1205.4176).

## Spatial and representation structure

Field and each flux component are decomposed into their spatial mean (DC) and
the remaining spatial component (AC), independently for every content channel.
The DC/AC energies close the full quadrature energy. Zero-energy fractions are
unavailable (`null`), rather than evidence of perfect structure. Actual
transport, collision and bath transitions additionally report changes of total
energy and total field-plus-flux spatial energy, accumulated over the actual
solver substeps. This identifies where structure changed without replacing an
operator or replaying a different learning trajectory. Conservative nonlinear
collision may redistribute DC/AC even while total energy is conserved.

Read features are recorded before the corresponding target's optimizer update.
On centered feature covariance eigenvalues, report spectral entropy, effective
rank `exp(H)`, participation rank, variance, and the finite sample rank bound
`min(window_samples-1, channels)`. Constant features have zero effective rank
and unavailable normalized entropy. A low rank can reflect either compact
useful features or collapse; interpret it together with live NLL and retention.

Temporal roughness is `0.5 * mean(||z[t+1]-z[t]||**2)`. Under stationarity this
is `trace(C0)-trace(C1)`, not an identified noise variance. No SNR or
`noise_expelled` conclusion is generated. Differences are never formed across
stream-role boundaries; each contiguous A/B/revisit segment is reported
separately. The measurement window is bounded history, not a learning horizon.

## Learning drift

Risk trends use complete consecutive NLL blocks within a single experience role.
Replay and first-pass observations stay separate. Report a descriptive linear
NLL slope and heteroskedasticity/autocorrelation-consistent standard error
(Bartlett lag weights, declared cube-root block-count bandwidth). Fewer than
three blocks yield `null` trend estimates. This uncertainty estimate assumes
conditions suitable for local linear regression; a plateau is not certified
Bayes optimality or infinite-time convergence.

At every actual optimizer step, record the accumulated gradient norm before
clipping, the norm of the actual parameter displacement, parameter norm and
relative displacement. A temporary parameter snapshot is released after that
update. These are measurements of the real AdamW update, including decay;
there is no readout-only surrogate DNR or fabricated thermal noise split.

## Conditional complete-state response

At live evaluation boundaries, obtain directional Jacobian-vector products of
one actual write/evolve event at the current parameter snapshot. All persistent
physical degrees of freedom participate: field, three fluxes, structural
conduction, receptor gates, STP resource/utilization and belief precision.
The exogenous elapsed clock is excluded. Eligibility and optimizer variables
are outside this explicitly conditional physical response.

Report directional gains and `log(gain)/event_duration` in a declared equal
per-component mean-square model-coordinate metric. Probe directions use their
own reproducible generator; the individual's RNG, parameters, gradients,
physical state, eligibility and observation cursor remain unchanged. No real
life event or optimizer update is counted for this derivative measurement.

This is a local directional response, not a maximal or long-time Lyapunov
exponent. There is no arbitrarily thresholded SOC/edge-of-chaos verdict, and no
membrane-voltage sum renamed a spike branching ratio. Useful reverberation
need not coincide with strict criticality; see
[Wilting and Priesemann (2018)](https://www.nature.com/articles/s41467-018-04725-4).

## Integration and continuation

`scripts/ib/train_online_plastic.py` enables the fourth pillar by default for
either credit algorithm. `--health-window` (default 512) controls measurement
storage; `--measurement-block` shares the declared block convention with live
adaptation measurements. `--response-directions` (default 2) controls only the
derivative measurement budget, never model time, physics or search branches.
`--no-health` permits a matched execution-overhead comparison.

`metrics.jsonl`/`progress.json` contain `health`. `lifelong_evaluation.jsonl`
additionally preserves before-change, after-B, and after-revisit health
snapshots, plus the conditional state response. Existing shock and savings
remain the evidence of adaptation/retention; health alone declares neither.
All bounded measurement histories and cumulative ledgers continue in the live
checkpoint. Attaching monitoring to a historical unmonitored checkpoint starts
a clearly bounded new observation interval on its existing mature life.

Implementation tests establish exact objective/state/gradient equivalence,
energy closure, continuation and JVP/finite-difference agreement. Short real
OWT timing runs measure execution only. Learning and memory claims require
sufficient joint training under the repository capability policy.

Recorded implementation verification: 66 affected CUDA-enabled checks plus
two additional quadratic-bath/forward-AD-reference checks pass (68 distinct
checks). Real OWT256-event monitoring-on/off runs have identical prequential
scores and field statistics; monitored peak tensor storage is 362.41 MiB.
The complete monitored calibration measured about 3.18 s/128 observations
with an affected test process present, and 6.74 s/128 when a separately owned
GPU training task began. The latter is not an isolated monitoring-overhead
comparison. Setup and summary/file serialization are outside event-loop timing.
Raw provenance and source hashes are in
`results/published/persistent_medium_health_interface.json`; neither calibration
created checkpoint assets or established a learning/convergence result.


## BPTT32 production integration (2026-10-08)

`train_medium_active_stream.py` enables the auditor by default. Both eager and
CUDA Graph execution capture each token's actual energy and read feature in
fixed-size chunk buffers. Vocabulary calibration measurements are computed from
already-produced per-token logits. Observations are transferred after the chunk;
no second physical trajectory supplies the health ledger. Real optimizer
parameter displacements are measured at updates. Auditor history and cumulative
energy totals are checkpointed with the learner, including pending credit.

`progress.json` carries `health`; each live evaluation saves `B_health` before
revisit traffic can evict B's observations and the post-revisit `health` report.
The B boundary additionally records the conditional full-physical-state response
using the existing independent-generator JVP diagnostic. This is a conditional
probe, not an elapsed experience or a learning update.

The shared AG score requires both a confirmed stable tail and observed recovery.
Unfinished recovery remains right-censored and has no AG value. Quality compares
model and reference on the identical tail targets. The time reference is the
explicit B observation budget; scores at different budgets need separate labels.

Full stage accounting selects the native diagnostic arithmetic inside the training
CUDA Graph; the quiet compiled/fused dynamics path currently omits these stage
outputs. The new instrumented throughput has not been benchmarked. Conditional
JVP probes also use native diagnostic arithmetic to preserve forward-AD support.
