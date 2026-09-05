# Address-Failure Harness Docket

Status: pre-registered probe plan, 2026-09-06. This document separates the
next diagnostic from its result; no hypothesis below is a conclusion.

## Evidence ledger

- **O:** A free per-pixel RGB field fits the 256px target nearly exactly, while
  the deployed U1 blank-field assignments are near-uniform and low-rank.
- **O:** U1’s 400-step address-alignment run improved paired IoU from 0.0037
  to 0.0266, but digit top-1 remained 0.103. Its final language prior has
  entropy 0.996 and rank 1.045/64.
- **D:** A KL can teach a prior only if the target-conditioned posterior differs
  from it on the variable being aligned. A near-zero KL cannot supply a
  spatially selective teacher.
- **Constraint:** Preserve one full-resolution Slice–MoT–Deslice graph,
  frozen Pythia, and no inference-time target access or recurrent pass.

## Competing hypotheses

| ID | Mechanism | Status | Distinguishing prediction |
|---|---|---|---|
| H1 | The target posterior is uninformative, so KL has no spatial teaching signal. | active | Target versus blank assignments have tiny divergence and no stroke/background separation, even after write sharpening. |
| H2 | The deployed post-sharpened write basis cannot represent a 1px stroke. | active | A target-aware linear oracle over all four actual write maps has low stroke IoU despite using the target during fitting. |
| H3 | The basis is adequate but the seven-task schedule/optimization starves T2I. | active | The oracle fits well and target assignments are selective; a matched T2I-only run improves address metrics before RGB metrics. |
| H4 | The diagnostic measures pre-write routing while gamma=8 creates a usable post-write basis. | active | Post-sharpening changes radius/rank and target oracle fit materially relative to pre-write maps. |
| H5 | A null/background state is required for sparse generation. | possible | The target posterior becomes foreground-selective only when an explicit null mass is enabled; without it all rows remain content-normalized. |

## Prediction registration

**P-20260906-01.** On `_u1_address_alignment_candidate.pt`, capture both
pre-write `W` and exactly the deployed gamma-sharpened `W_write` under blank
and target-observed boundaries, then fit a diagnostic-only four-layer linear
oracle to the known target using `W_write`.

- H1 predicts target/blank divergence remains small and foreground separation
  is absent; H2/H3/H4 are not thereby confirmed.
- H2 predicts the oracle fails to recover the sparse stroke even if `W_write`
  differs from `W`.
- H3 predicts target selectivity plus a high oracle fit; H4 predicts that only
  the post-gamma measures show this change.
- The probe is falsified as a decision tool if it cannot distinguish pre/post
  write weights, target/blank boundaries, and a free per-pixel oracle.

## Decision rule

Run this probe before another training arm. If H1 holds, replace the learned
posterior teacher before tuning KL. If H2 holds, do not claim a loss-only
repair: introduce an explicit sparse/null address state or a higher-rank
write basis, then compare with matched parameters and updates. If H3 holds,
run a matched T2I-only optimization control. If H4 holds, move all address
losses and diagnostics onto post-gamma `W_write`.

**P-20260906-02.** Hold model, source checkpoint, ten-digit groups, optimizer,
400 updates, foreground likelihood, digit contrast, evaluations, and compute
calls fixed. Set only KL, compactness, and diversity coefficients to zero. If
the no-address control reaches comparable paired IoU/color/flood, the earlier
gain is schedule/ordinary RGB optimization (H3), not posterior alignment. If
the alignment arm wins on paired IoU *and* blank post-write locality without
prompt-control regression, retain a restricted address-mechanism claim.

## False-win protections

The oracle sees targets only to test representability; it is never rendered
as generation. All candidate outputs use blank fields. No new backbone,
checkpoint overwrite, external generator, recurrence, or extra inference pass
is allowed. Metrics report digit identity, paired IoU, flood, address
divergence, radius, rank, and oracle fit separately.

## Review status

Hypothesis and experiment plan are pending an independent reviewer. The
workflow runtime is unavailable in this Codex host; repository git hooks are
installed as the harness’s enforceable layer.

## Results and reviewer disposition

The independent reviewer rejected H1: the target branch is selective, with
target/blank assignment changes of 0.018, 0.024, 0.064, and 0.055 across depth
and foreground/background change ratios of 2.2--10.3. It supported H4: fixed
gamma=8 transforms pre-W into a different deployed object.

During verification, however, the deployed power normalizer was found to run
in FP16. With diffuse weights, `w**8` underflowed, so the old post-write maps
were not probability distributions and their row masses collapsed. The
implementation now performs power sharpening and normalisation in FP32 before
returning the original dtype; a half-precision mass-preservation regression
test passes. The old post-write probe and negative-KL smoke result are
invalidated. The corrected probe has post-write entropy 0.187/0.330/0.071/0.330
and radius 0.515/0.558/0.804/0.727; its constrained blank-basis stroke oracle
is 0.278, below pre-write 0.417 and far below the free-pixel oracle 1.0.
Sharpening therefore changes the usable write object, but is not by itself an
addressability repair.

P-20260906-02 completed under the old numerical implementation. Its matched
no-address control reached paired IoU 0.02680, digit top-1 0.111, color
accuracy 0.430 and flood 0.0316; the old pre-W KL arm reached 0.02664, 0.103,
0.397 and 0.0336. This rejects an independent benefit from the old objective
in that runtime, but cannot rank fixed-runtime training arms. A one-step
fixed-runtime post-write-KL smoke test gives a valid positive KL of 0.063.
The next admitted comparison is a clean, matched gamma=1 versus gamma=8
focused-T2I run; only then may a post-write KL arm be compared with no-KL.
