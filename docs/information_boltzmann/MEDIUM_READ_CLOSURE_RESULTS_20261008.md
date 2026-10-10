# What is missing: dynamic observation or new temporal state?

Status: read-only CPU mechanism audit complete. Zero optimizer updates, every
parameter unchanged. Four preselected real OWT events0/8/16/24 on the next32
events of the complete96k last individual. Execution6.31s. No new head was
trained/fitted and no production architecture was modified.

Preregistration: MEDIUM_READ_CLOSURE_PREREG_20261008.md.
Script: scripts/ib/audit_medium_read_closure.py.
Measurements: results/published/medium_read_closure_20261008.json.

## Distinction

For fixed parameters/solver and specified future inputs, the medium evolves
from s=(field, three fluxes, conduction, receptors, STP, elapsed). The next write
also needs its existing precision/control. Its short-time physical response is
determined by those current quantities. Reading field alone is a partial
observation. Temporal traces can estimate unobserved dynamics or retain extra
past-input information, but extra history is not logically necessary to define
this existing physical initial-value problem.

The last statement is limited to physical evolution. Current physical state
is not proved sufficient for external next-token statistics. The future of
active learning additionally depends on optimizer, pending credit, input
stream and learner configuration. These distinctions preserve a meaningful
role for CTM-style histories/synchrony without treating them as the already
confirmed missing physical initial condition.

## Existing state contains usable first-order response information

Compute v(s)=d(field)/d(tau) at zero autonomous elapsed time, with the current
complete state and no new input. Evaluate the existing read at field+epsilon*v,
and compare with its actual output after epsilon of autonomous evolution.

First preselected event:

| epsilon | Hold current field: read-feature relative error | Existing-state first-order prediction |
| --- | ---: | ---: |
| .00125 | .051224 | .006866 |
| .000625 | .026758 | .001749 |
| .0003125 | .013675 | .000441 |

Halving epsilon reduces the frozen-field error by about2 and the first-order
error by about4. Across all four events, first-order read-feature error halving
ratios range3.925 to4.085. This verifies a first-order dynamical observable
already available in the current physical state, with an approximately
second-order residual in this tested neighborhood. The last-event correction
is not a learned neural history or improved language score.

The JVP here is an offline numerical reference, not an efficient production
read implementation. A candidate can use the model's local vector field or
compact signed flux/dynamic features, retaining finite read ports; its cost
and predictive gain still require separate acceptance.
The reference uses complete state and real dynamics; frozen-field hold is not
a fitted field-only forecasting competitor. Consequently the entire error
reduction cannot be attributed exclusively to hidden components or advertised
as acceptance of a local-only replacement read.

## Locality needs a boundary condition, not arbitrary global access

The16 current apertures cover192/256 sites in their union. We inspect each
probe separately. The geometry-selected smallest-support probe contains12
sites; one periodic nearest-neighbor halo contains44. Boundary-edge quantities
are counted when either endpoint touches the selected region.

Holding ALL current state in that aperture fixed while scaling external flux
by +/-1% leaves its immediate measurement exactly unchanged. Its one-step
future measurement changes by about6.6e-6 to7.8e-6 relative. Thus strict local
state is not exactly closed against incoming boundary signals.

Keeping the additional one-hop halo fixed reduces this counterfactual response
to1.2e-7..2.6e-7, near FP32 precision. This does not prove exact closure. Across
128 sampled future-measurement VJPs (4 events x16 probes x2 directions), the
largest outside-halo share of squared sensitivity is2.935e-5 for field and
4.391e-6 for flux_x, with lower maxima in the other tested components. All
fractions compare within one physical component, never incompatible units.

These are sampled directional diagnostics of short-time propagation, not an
exhaustive Jacobian, a long-time closure theorem, or a claim of universal
one-hop propagation. The splitting schedule can spread higher-order influence
beyond one halo; the measured residual remains explicit.

## Decision

The supported first candidate is a local dynamic read: retain current field
and expose signed motion/response information already present in the medium,
with an explicit local boundary condition or nearby incoming-edge context.
This makes existing internal time observable without adding a new lifetime
history just to repair physical-state blindness.

CTM-style history is a distinct candidate: it can add external-input memory,
temporal coactivity features, or a partial-observation estimator. Whether any
of those improves next-token NLL beyond an existing-state read remains unknown
under the no-training constraint. No claim that CTM is redundant or that a
particular read design solves the language bottleneck is made.

Stop the diagnosis here: the two roles have been distinguished at the physical
interface level. Do not launch training or accumulate further surrogate
measurements as if they could certify language gain. Next work is a minimal
design for local dynamic observation, followed by an explicitly authorized
matched comparison when learning is allowed.

Review: independent /root/rtc_contract_review approved the closure/derivative
plan. Final implementation and conclusion review are recorded in research_tree.
