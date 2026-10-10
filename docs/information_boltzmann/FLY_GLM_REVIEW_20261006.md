# Review of centered readout and the proposed DAN level

This review inspects commits 16500b1, a64650a and 4dcc71e, actual artifacts,
and a bounded direct invocation of the skipped DAN interface test. No training,
checkpoint or physical state was changed. GPU was idle; five older Python test
processes remained present. Those unrelated test processes were left intact.

## Centered readout

On the 100k centered continuation's saved 127-target A-revisit head audit:
full NLL 7.861199, own-window common NLL 7.949809, fixed unigram 7.306862.
Conditional variation improves the descriptive saved-head NLL by .088610;
full still exceeds the reference by .554337. Decoder common/covariance
gradient ratios are .5115, .1916, .3990, .4719. Three of four blocks improve
with variation; one worsens. This is concrete local progress.

Actual active first-pass B results are a separate measurement. Across the
20 encounters, first five average excess over the exact same-text train-only
unigram is 1.113226; final five average excess .913524. Final fresh B NLL is
9.170039 versus reference 7.942650, excess 1.227389. Histories/text differ, so
these are descriptive trends. The .554337 A-head result must not substitute
for fresh active risk. Artifact: fly_centered_active_review_20261006.json.

The centered config retains frozen STP parameters and changes decoder LR to
.0002 (earlier diagnostic .0001). Hence the result describes that complete
learning configuration. Claiming every update follows conditional credit,
or that the overall predictive disease is cured, exceeds these measurements.

## Single-update diagnostic

The completed existing artifact fly_one_update_outcome_20261005.json records
old mature-head fitted NLL 9.287054 -> 9.257496 and immediate fresh NLL
8.271411 -> 8.259405, with surrogate first-order predicted reduction .028392.
This update is broadly consistent with its local gradient. Actual R update
common-gradient inner product is -.00441354, covariance -.000159105;
decoder common -.02274633, covariance -.000000632. These support background
dominance on that window. They do not permanently exonerate the optimizer or
surrogate. Edge first-order terms alone cannot establish lasting circuit harm;
the full body-only finite replay slightly improves fitted and fresh losses.

## DAN implementation blockers

1. `_dan_update` reduces h to [N], then indexes h[0] with neuron indices;
   DAN pre/post/weight also take only [0]. Vector indexing is incorrect.
2. COBA inhibitory weights are nonnegative conductance magnitudes. Local
   clamp(-5,0) would remove or negate their magnitudes. Both COBA tensors
   must retain nonnegative amplitude. A signed layout needs a retained
   original sign mask, not a mask obtained after crossing zero.
3. Generic substring promotion/collection includes dan_edge_weight in BPTT,
   while the forward loss has no dependence on that tensor. Its gradient is
   None, conflicting with observe norm checks and CUDA Graph coverage checks.
4. Claimed gate EMA is absent: current rule uses a terminal membrane matvec,
   without persistent EMA/release-delay state or saved gate. Window-end
   coactivity is a different rule from delayed activity eligibility.
5. Generic tick/observe still contain COBA arity and topographic-writer
   assumptions. Non-topographic quiet drive is None; empty projection norm
   and unconditional writer.gate_linear/a_adapt accesses remain.

The direct dedicated test run with OMP/MKL threads1 reaches backward and
raises: `PyTorchDelayedSynapticTransmissionBackward returned an incorrect
number of gradients (expected 8, got 4)`. The empty inhibitory branch fixture
has only one split boundary; the delayed Function requires four ring-gradient
slots. This locates a reproducible interface failure rather than establishing
the cause of every previously stalled process. The fixture also lacks
previous_token for observe and a strict noneligible-edge invariance assertion.

## Decision

Prioritize restoring the shared learner's validated parameter coverage/sign
and optional-state contracts, then test the DAN local update in isolation.
Keep DAN disabled until vector, delay, gate-state and sign invariants pass.
Centered continuation has an evidence-based candidate direction, but its
acceptance criterion is first-pass predictive improvement and conditional
contribution under the actual online learner, not just a changed gradient ratio.
No further training was launched by this review.

Independent diagnosis_review confirmed the code blockers and limited the
centered acceptance interpretation to the actual cached head measurement.
