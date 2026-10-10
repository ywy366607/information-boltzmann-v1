# Motor-only persistent fly: repair decision under the research harness

Date: 2026-10-07. Stage: EXPLORE, before new diagnostic results.

## Research state and evidence ledger

- **O (observation)**: the approved 3,000-update joint OWT run has final-32,000 training risk 7.38905 versus the same-target locked frequency reference 7.45343. All six active new-context B checks have positive excess risk. This is a finite training-trajectory gain and a repeated transfer limitation, not convergence.
- **K (constraint)**: sensory-only assimilation, motor-only prediction, no observed-token-to-decoder bypass; complete physical state and optimizer continuation; current target sealed before assimilation; 4 GiB hardware.
- **D (derivation)**: four per-edge delay slots do not limit cumulative path delay. A 14-tick causal route can remain physically active across a BPTT32 boundary while its source parameter derivative is truncated.
- **D**: all analog writer-to-motor dependence crosses transmitted spike events. Holding the complete event pattern fixed makes the motor read locally independent of writer weights. Hard-forward local derivative zero and a nonzero surrogate gradient can coexist.
- **D**: reset read `h_post = v*(1-s)` is non-injective above threshold. Its registered surrogate derivative is `1-s-v*phi(v-theta)`; the ordinary fixed-event derivative is `1-s`. These are different objects.
- **A (assumption)**: useful next-word distinctions reach the chosen motor observable and can be learned by the present objective. Reachability tests establish access, not this semantic assumption.
- **I (interpretation)**: marginal adaptation, context carryover, observation loss, event-credit mismatch, truncated history, and joint code drift are competitive causes.
- **S (speculation)**: reset-only backward detachment may improve finite update quality without changing any forward physiology. It has not been selected or verified.

Known results above were available when hypotheses were written. New predictions are prospective relative to the new diagnostic, not blind predictions of the completed training run.

## Independent reconstruction and necessity review

The hypothesis collaborator separately supplied three problem formulations and seven structurally distinct hypotheses in `FLY_PIPELINE_REPAIR_HYPOTHESES_20261007.md`. The necessity reviewer separately audits the delayed event system in `FLY_PIPELINE_REPAIR_NECESSITY_20261007.md`. No diagnosis follows merely from the presence of a surrogate, a low effective rank, or a long biological delay.

Three questions guide the decision:

1. Which part of the prediction provides same-target conditional benefit rather than a shared marginal shift?
2. Does the actual saved-Adam displacement improve the actual hard forward, separately for body, head and bias?
3. Does the chosen post-reset output retain the causal response already present in the motor event?

H4 (missing cross-window history) remains outside the first diagnostic's identification scope. A single-window negative result cannot invalidate full-history learning or the connectome.

## Minimal diagnostic and authority

The experiment committee prescribed a checkpoint-fork numerical diagnosis: preserve all stored physical fields and both optimizers, use pre-fixed real OWT tokens, perform one actual update on a diagnostic copy, and replay only predeclared parameter displacements. Fixed-weight branches are counterfactual mechanism audits of the saved live individual, not replacements for primary active evaluation.

Implementation, exact target indices, all directions/scales, tick ledger, identity tolerances, source hashes and decision rules must be sealed in a separate protocol and independently approved before execution. No production weight update or long training is authorized by this document alone.

## Clean candidate selected for pre-result consideration

Keep the forward equation, delayed pulses, ALIF/STP and prediction deadline unchanged. The single candidate from the necessity audit is a learning-operator substitution:

\[
h_{t+1}=v_t\,[1-\operatorname{stopgrad}(s_t)],\qquad
s_t=H(v_t-\theta_t).
\]

The output spike still uses the registered surrogate through transmitted pulse, adaptation and resource dynamics. Only the membrane reset branch stops reusing that surrogate as an analog reset derivative. This removes the `-v*phi` reset term; it does not claim to make all event credit exact or to solve cross-window history.

**Mathematical binding**: candidate and baseline must have identical forward fields, pulses, predictions and target deadlines at identical parameters. Its fixed-event reset derivative must equal `1-s`. A numerical forward change would invalidate the intended substitution.

There is also a weight-independent counterexample for the direct passive membrane factor. On a silent inhibitory branch `v<0`, the original registered backward multiplier is

\[
J_{\rm reset}=\alpha\left[1+\frac{|v|}{1+\pi^2(v-\theta)^2}\right].
\]

It can exceed one while the actual fixed-event passive multiplier is just `alpha<1`. For instance, `v=-0.2`, `theta=0.1`, `alpha=exp(-1/36)` gives approximately 1.076 versus 0.973. This is a numerical illustration of the formula, not a measurement or a prescribed biological constant. For general positive theta the maximum extra factor over negative v is `(sqrt(theta**2+1/pi**2)-theta)/2`. The candidate removes this particular amplification term. It does not bound the recurrent network's complete Jacobian, establish the dominance of this term in the saved individual, or prove an expected-loss gradient for hard events.

**Selection binding**: select this candidate only if a pre-registered actual-update diagnostic implicates the reset/event learning direction and an independently approved candidate-direction comparison improves that specific failure. If body updates work while only head calibration fails, this candidate loses priority. If evidence is ambiguous, retain the existing production architecture and report that ambiguity rather than install it.

## Primary research support and limits

[Neftci, Mostafa & Zenke (2019)](https://arxiv.org/abs/1901.09948) describes surrogate-gradient learning for spiking systems. [Gygax & Zenke (2025)](https://direct.mit.edu/neco/article/37/5/886/128506/Elucidating-the-Theoretical-Underpinnings-of) distinguishes stochastic/surrogate derivatives, notes that generic surrogate fields need not be gradients of a scalar surrogate loss, and discusses excluding reset from the backward path for robustness. These works support treating the learning operator as a research object; they do not establish the cause of our OWT deficit.

## False-win boundaries and stop rules

- Use the actual post-clamp displacement from both saved Adam optimizers; gradient norms alone do not establish update magnitude or direction.
- A common-feature decomposition using whole-window statistics is retrospective algebra, not an admissible online predictor.
- New motor features cannot be judged by simply substituting them into a head trained on old features.
- A short diagnostic establishes only the recorded state/window/direction effect. A capability improvement requires a separately reviewed joint real-data run with sufficient updates and all active-evaluation pillars.
- Stop on provenance mismatch, identity/full-state mismatch, nonfinite values, uncovered threshold calls, or device memory above the sealed gate. Do not interpret failed numerical controls as a scientific result.

## Harness execution status

The installed harness's Claude `Workflow` tool is unavailable in this Codex session. We use separate hypothesis, necessity, plan and conclusion review agents; immutable source/hash approval files; and the strict research-tree validator. We do not claim execution of the unavailable workflow engine. The final tree node records actual review status and will remain active unless evidence and independent review warrant confirmation.

## Result and decision

The proposed repair remains **C_reset**, a single backward-only operator change.
The original and proposed reset pass independent-entry CPU interface checks:
forward values agree exactly, both reset partials become `1-s`, transmitted
event gradients remain present, and instrumentation restores production globals
after normal and exceptional exits. Seven focused numerical tests pass in
`tests/test_fly_pipeline_repair_instrument.py`. These are numerical fixtures,
not a capability study or a mature-individual NLL result.

The sealed diagnostic uses 96 unique real OWT positions, 44 window forwards
(1,408 physical ticks), three backwards and two disposable actual-Adam updates.
It records body/head/bias and global-clipping/clamp effects separately. Six
selection parameter points each repeat fit and following-window scores; their
own event/score/state replay floors replace an assumption that the old-parameter
floor bounds every hard-event parameter point. Matched body norms and a common
baseline head displacement constrain the step-size explanation.

The fixed-scale candidate must improve both the joint and raw body-only
following-window risk relative to their original updates, keep body following
risk nonworse than the old predictor, satisfy the fit checks, and retain its
advantage with matched body norms/common head. Raw head gradients must agree;
unstable hard-event replays make selection ambiguous. The smaller original-body
step is descriptive only. No post-result endpoint selection is allowed.

These 44 forwards reuse 96 real token positions for deterministic controls and
replays; they are not 44 independent samples. A passed diagnostic supports only
local usefulness at this checkpoint, these two windows and these saved Adam
moments. "Nonworse within tau" means no numerically distinguishable harm in the
registered comparison. Mixed-coordinate norm matching does not match functional
step size or individual parameter-group norms; it cannot identify a unique root
cause. A radius below the numerical resolution of matching cannot support a
direction-specific conclusion. The final gate requires a radius over ten
absolute matching error units and relative norm error at most `64*float32eps`.
These are numerical identification tolerances, not biological constants or
trainable model settings. At each of the six selection parameter points,
both the hard masks and every continuous physical field must replay within
their registered floors. An unresolved radius, mismatched norm, raw-head
gradient mismatch or unstable replay is explicitly `ambiguous` and cannot
select or scientifically reject the candidate.

**Execution approval:** independent scientific-scope review and final
implementation/source-lock review are complete. The reviewer independently
verified all code, data and checkpoint hashes, seven focused CPU tests, and a
32-event/64-call identity oracle covering all trainable gradients and full
physical fields. Root's combined CPU regression passes 23 tests. Earlier
selection/replay gaps were amended before seeing results; the final approval
also checks relative norm matching, resolved radius and continuous-state
replay. The approval is restricted to this 44-window diagnosis in the fixed
repository/runtime with an idle GPU and a 3,900 MiB device gate.

The GPU became idle after review, and the approved bounded diagnosis has now
been launched. No outcome has been inspected at this document update.
A positive local result permits
only follow-up design review. A failed direction prediction removes C_reset
from this local repair rationale. Production physiology, weights, ongoing
training and checkpoints remain untouched by this diagnosis preparation.

## Executed result: C_reset is not selected

The approved first execution reached the final report write after its full
44-window budget, but NumPy scalar serialization failed and no NLL report was
saved or inspected. An independently approved representation-only output fix
preserved scientific endpoints, targets, saved moments and source. Its one
authorized recovery replay completed. Both attempts together used 88 windows,
2,816 physical ticks, six backwards and four disposable updates. The attempts
are not independent samples; the pre-recovery history remains immutable and
the final execution ledger records both. No production checkpoint was changed.

All numerical identity/replay, raw-head-gradient, resolved-radius and matched-
norm gates passed. The prospective candidate gates did **not** pass. The
registered decision floor is 0.000108099 NLL.

| Following32 comparison | Original update NLL | Reset-only update NLL | Candidate minus original |
| --- | ---: | ---: | ---: |
| Joint body+head+bias | 7.85275261 | 7.85251398 | -0.00023863 |
| Body only, original head | 7.85501869 | 7.85504409 | +0.00002540 |
| Matched body norm, common original head+bias | 7.85275264 | 7.85277513 | +0.00002249 |

Joint improvement clears the floor, while raw-body and matched-body differences
are unresolved and fail their required improvement gates. Raw head gradients
agree, but global clipping differs (original0.843084, candidate1.0), so the
actual head update differs too. The joint gain cannot identify an improved
reset/body learning direction. The candidate is locally unsupported by its
prospective selection rule and is withdrawn as the proposed production repair.
This does not reject reset exclusion in all systems or prove a global cause.

The original body update provides a separate local conflict to retain:
fixed-event fit change is -0.000282541; its own derivative predicts
-0.000282884 (remainder approximately+0.000000343). Hard-event fit change is
+0.000135057, with3,519 changed begin/finish spike decisions and a
hard-minus-fixed difference+0.000417598. Thus the registered smooth branch's
decrease is offset and reversed in hard replay on this fit window. The same
body update improves the following window by0.000551254, so this is not a
general assertion that body learning harms prediction. It does not isolate
which event, path, observable or omitted history causes the population risk.

Current frozen-head algebra is also descriptive only: full-versus-common
logit fit/following differences are -0.000861928/-0.000299640. Common logits use
whole-window retrospective features and are not an admissible online baseline.
These numbers constrain the current head's observed within-window variation;
they do not prove the motor state has no task information.

The next decision is to rebuild the learning/observable hypothesis around the
event-associated discrepancy and its task alignment, keeping cross-window
credit and head/body coadaptation as alternatives. No new mechanism or training
is launched from this local result. Independent result review is recorded
separately; predictions above remain the pre-result ones.

**Independent result review:** `/root/repair_result_review`, who did not generate
the candidate, prediction or implementation, reviewed all36 report fields and
recomputed the decision floor, all12 selection replays, eight identity controls,
source/data/checkpoint/report locks and both execution ledgers. The reviewer
accepts `locally_unsupported` and the local event-associated counterfactual;
rejects an inference to a unique reset cause, global motor-information absence
or long-term capability benefit. The review executed no additional GPU work.
Its record is `results/published/fly_pipeline_repair_result_review_20261007.json`.
