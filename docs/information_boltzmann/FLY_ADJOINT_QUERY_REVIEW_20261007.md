# Student-error-guided physical queries

Status: mathematical review of the pasted Gemini proposal; candidate design,
independent review and experiment approval pending. No training or model edits.

## Retain

Actual connectome-derived aggregates, regional embedding SIGReg, explicitly
window-final quiet teachers, and student-generated rollout errors are useful
ingredients. Joint encoding/prediction learning remains an allowed candidate.

## Correct the claimed lifting identity

For an affine encoder E(h)=Wh+b, injecting alpha W^T delta_z changes the encoded
state by alpha W W^T delta_z, not by delta_z in general. Example: W=diag(2,1),
delta_z=(1,1), W^T delta_z=(2,1), W delta_h=(4,1). A scalar alpha cannot align
both coordinates. Transposition supplies a gradient direction, not an inverse.

Actual encoders include LayerNorm. The local operator is the full Jacobian
J=dE/dh, including normalization. Its adjoint J^T delta_z is a valid first-order
steering direction. A minimum-norm local lift uses J^T(JJ^T)^dagger delta_z for
attainable perturbations, with nonlinear re-encoding checks. Rank deficiency
and off-image latent values prevent a universal exact lift. A blended latent
posterior can also be a belief mean rather than the encoding of one physical
state. Therefore neither W^T nor a scalar gain establishes exact DAgger queries.

## A smaller, honest candidate

Use the student's free-rollout discrepancy to select physical perturbation
directions on a COPY of the current teacher carrier. Re-encode the actual
perturbed state:

    S_q = admissible_physical_perturbation(copy(S_ref), student_error)
    z_q = E(S_q)
    label_q = E(F_physical(S_q))
    train G(z_q) against label_q

Aggregate/replay these actual queried pairs with a bounded sample buffer. Do
not label G(z_hat) with a response to some other encoded input while pretending
the two states are identical. The live never-reset individual is untouched by
counterfactual queries. Teacher updates, student rollout, replay and the task
objective must have declared timing/provenance.

Call this student-error-guided physical query aggregation, inspired by DAgger.
Exact DAgger-on-latents needs the stronger attainable-state/closure contract.
The same z may correspond to different conductance/ring histories; either carry
the relevant context or model a conditional distribution. A single membrane
projection is not automatically a complete physical state.

Physical-query supervision may improve a surrogate's local accuracy. It does
not guarantee a restoring contraction, because the queried physical evolution
can amplify perturbations. An imposed membrane impulse is external work; the
carrier's COBA/ALIF equations still apply, but energy bookkeeping includes that
impulse. Avoid claims of conserved closed-system energy or guaranteed inhibition.

## Graph provenance corrections

Use separate E/I matrices for each actual delay tier A_d, rather than erasing
sign and phase with one absolute-weight A and one mean delay D. These small
statistics reuse existing data and add no separate prediction backbone.

1.11+1.36+1.19=3.66 sums weighted regional means; it is not a shortest physical
neuron path or a minimum arrival time. The regional graph can combine edges
belonging to different intermediate neurons. A high internal aggregate weight
does not prove long memory, and motor feedback does not by itself establish
efference-copy content/function.

## Decision and acceptance

Proceed conceptually with graph provenance and regional representation fixes.
Replace the strict-DAgger claim with the explicit re-encoded query contract
above unless exact lifting is demonstrated. No experiment is launched here.

Before capability training, numerical tests should reject mismatched query
inputs and account for perturbation work. Matched real OWT joint training then
decides whether query aggregation improves first-pass and fresh-context NLL.
If physical prediction improves alone, retain that numerical finding without
claiming better language. If it harms NLL, remove the auxiliary from the task
route rather than adding another corrective module.
