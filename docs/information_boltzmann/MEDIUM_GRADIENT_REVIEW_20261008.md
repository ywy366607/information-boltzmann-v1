# 3D medium: scope of the fly surrogate-gradient repair

User goal: preserve useful task credit in a never-reset learner while keeping
the physical forward, local ports and full credit window. This review asks
whether the S14/W32 fly failure also exists in the current3Dmedium.

## Current execution path

`train_medium_active_stream.py` -> `ActiveMediumTrainer` ->
`quiet_training_chunk` / `CapturedPlasticChunk` -> `PlasticMediumPorts3D`.
The current default read is instantaneous; optional dynamic read uses local
field derivatives. The old torus3d evolver and archived compressed online
credit learner are not the running training path.

This graph has no hard spike forward paired with an ATan surrogate backward.
Its local nonlinear rotations and conductance/STP/activity states use ordinary
autograd. Therefore the diagnosed fly silent-pulse proxy feedback path is
absent. This does not make the whole medium Jacobian a contraction.

For a state-dependent rotation F(x)=R(theta(x))x,

    DF = R(theta) + (R'(theta)x) grad(theta)^T.

The second term can amplify perturbations despite exact preservation of
the state norm. As a numerical algebra counterexample, theta=2*x2/||x||
atx=(1,0) gives DF=diag(1,3). This establishes a possible feedback mechanism,
not an observed runaway in the trained3Dmedium.

## Existing real-corpus evidence

Source: `results/published/medium_learning_chain_audit_20261007.json`;
96000freshtrain tokens,3000updates,103296actualevents, then three backward-only
32target windows at offsets96001/96033/96065. This is historical evidence,
not a newly rerun checkpoint audit. Combined total gradient norms computed
from the reported disjoint groups are approximately4.881,4.924,6.101.
Global norm1 clipping leaves read gradients approximately.370,.329,.299.
Thus these windows show usable read credit; they do not exhibit the fly's
1.25e14norm with4.9e-15clippedread gradient.

The trained checkpoints of the two96kmedium runs were deleted by earlier
cleanup (recorded in FLY_CTM_S14_REVIEW_20261008.md). Existing reports remain.
Fresh replay on that individual requires restoring a backup with provenance.
We do not substitute random weights for a trained-state conclusion.

## Implemented numerical protection

Both current active-stream and conductance-training entrypoints now use
the same bounded FP64 norm and common-factor clipping as the fly repair.
This preserves optimizer cadence, norm1 policy and gradient direction.
It prevents a finite FP32 gradient's sum of squares from overflowing during
clipping. Actual nonfinite entries are still rejected explicitly.

Numerical integration test: finite gradients[1e20,2e20] produce the expected
nonzero direction-preserving update through ActiveMediumTrainer. Existing
state continuation and CUDA capture/eager contracts remain covered.

Independent read-only operator reviewer: `/root/medium_operator_review`.
No fly threshold-width proxy is added to the3Dmedium. No new model weights,
long training, or NLL improvement claims result from this review.
