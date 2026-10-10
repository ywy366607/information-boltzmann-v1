# Pipeline learning: necessity audit and falsifiable repair

Status: independent THEORY NECESSITY/AUDIT by /root/diagnosis_review,
2026-10-07. This document was written before inspecting any prospective repair
diagnostic outcome. No model measurement, training or core mutation was
performed. The candidate below requires separate plan and implementation review.

## Evidence and problem reconstruction

The completed single-individual run preserved the approved causal interface and
all physical state. Its fixed primary last1000 fresh-update risk beat the locked
reference by0.06438 NLL. All six active B windows had positive model-minus-reference
risk; removing their first32 targets descriptively leaves all six positive.
These observations establish a finite-trajectory risk advantage and an observed
context-transfer limitation. They do not identify the component responsible.

Three distinct questions must remain separate:

1. **Causal representation:** can past observations reach a predictive motor
   observable at the required deadline?
2. **Task observability:** do differences in that observable distinguish
   distributions of the next real OWT token, beyond the declared adaptive prior?
3. **Learning dynamics:** does an actual update improve this distinction while
   preserving useful continuing state?

The first has numerical acceptance through cumulative delays1..14. The other
two are open. A valid clock does not force semantic discrimination.

## What delay, thresholds and motor read can support

Write at tick t only after sealing its prediction. Positive chemical edge
delays imply a path of total delay d can affect motor prediction at t+d, never
an earlier prediction through that path. Four ring slots per edge support
multi-hop sums beyond four; they do not limit the full recurrent history.
The existing tests exhibit distinctions at each d=1..14 and nonzero within-window
surrogate source/path gradients. This proves possibility under explicit fixture
conditions, rather than sufficiency for arbitrary natural-language history.

Hard thresholds can distinguish inputs by pulse pattern. Membrane, conductance,
ALIF, STP and delayed pulses can retain consequences of those patterns. Thus
hard spiking and motor-only reading do not mathematically forbid causal
next-word use. The motor observable need not be a sufficient statistic of the
complete body or history.

Useful dynamic conditioning requires target-relevant information in the
observable after accounting for a declared baseline information set C.
For example, I(next token; motor observable | C)>0 is necessary for a Bayes
predictor to improve on the corresponding Bayes baseline. Finite learned heads
also require accessible coding and appropriate optimization. A fixed unigram
is not this ideal adaptive baseline, so beating it alone cannot establish
motor-conditioning utility. Adding a decoder cannot manufacture information
absent from its input.

## Exact hard-forward learning contract

Sources: SpikeFn and finish_coba_tick in core/fly_reservoir.py;
begin_fly_prediction in core/fly_pipeline.py; observe and optimizer groups in
core/fly_bptt_learning.py.

Let s=H(v-theta), h'=v(1-s), and phi(q)=1/(1+pi^2 q^2).
Away from an event boundary, at fixed theta and other local inputs,

    actual within-event partial: dh'/dv = 1-s
    current surrogate partial:   d_tilde h'/dv = 1-s-v phi(v-theta)

A firing branch has actual local derivative zero and a negative surrogate reset
term when v>0. A negative silent v can give a membrane multiplier
alpha*(1-v*phi)>1 despite 0<alpha<=1. Unit peak phi does not bound the complete
recurrent Jacobian. These are algebraic properties, not evidence that this
run's relevant gradient is dominated by those terms.

The functional writer sends continuous drive only into disjoint sensory units.
Cross-neuron communication uses transmitted pulses. Holding the initial full
state, all other parameters and the complete finite-window spike/clamp event
pattern fixed, changing writer weights changes sensory voltages/baseline but
not the pulse sequence or the motor feature. There is no continuous writer-to-
motor bypass in this implementation. The hard-forward motor feature is
therefore locally constant in writer parameters in that event region.
Edges/macros can still have smooth within-event effects on conductances and
motor voltages; fixed-feature R/RMSNorm/decoder calibration is smooth.

Consequently, a nonzero surrogate writer gradient is necessarily an approximation
to learning event changes, rather than the ordinary derivative of that locally
constant hard-forward map. It must be judged by finite event-aware updates.
Failure to match an infinitesimal finite difference on a hard plateau is
expected and cannot, by itself, reject surrogate learning.

Gygax and Zenke establish that surrogate updates generally need not be gradients
of a surrogate loss. Their reset discussion supports testing reset exclusion,
but also reports little difference with unit-scaled surrogates in their settings
and retains reset differentiation under its stochastic interpretation.
Our deterministic COBA/ALIF/STP system cannot inherit a cure from that result.
[Primary paper, sections4.2,5,7](https://arxiv.org/html/2404.14964v3).

## Prediction variation, task alignment and actual updates

For fixed-feature logits l_t=c+delta_t and p0=softmax(c), the exact identity is

    CE(c+delta_t,y_t)-CE(c,y_t)
      = (p0-onehot(y_t))^T delta_t + KL(p0 || softmax(c+delta_t)).

Variation helps only when its target-alignment term outweighs the nonnegative
distribution-change cost. Raw motor variance, latent variance or large probability
variation does not guarantee this alignment. Locally the latter cost is
approximately delta^T [diag(p0)-p0 p0^T] delta /2.
A retrospective common-logit comparator is a decomposition, not automatically
an available online baseline.

RMSNorm suppresses radial sensitivity when squared signal dominates epsilon.
With finite epsilon and nonzero affine gains it does not literally delete every
radial difference. Projection and normalization must be audited separately.
No current-arm motor/feature/probability variance result has been inspected for
this necessity review.

For normalized feature z, decoder gradient is
G_D=mean(a)mean(z)^T+Cov(a,z), a=p-onehot(y).
For R it is the analogous expression with the complete RMSNorm/decoder VJP.
Layer-specific covariance and actual update inner products are required;
gradient norms cannot attribute predictive benefit. Bias and constant decoded
features can compensate each other, so bias-only loss is not a gauge-independent
measure of the learned marginal.

The completed run's123 logged mixed-phase snapshots have median writer, synapse,
read and decoder gradient RMS approximately2.87e-6,1.94e-5,2.30e-4,8.67e-5.
The median preclip norm is0.960 and its maximum50.98. These show differing
raw sensitivities and occasional strong clipping. They do not show relative
Adam displacement or its sign. Both optimizers use AdamW; moments, epsilon,
decay and edge projection determine the actual displacement. In general
g_current^T DeltaTheta need not be negative.

At each32-event update the incoming physical state was generated under previous
parameters, whereas later transitions/readouts use updated parameters.
This is a valid nonautonomous online system, not an invalid checkpoint.
It can nonetheless produce code/read mismatch or interference.
Its importance is an empirical hypothesis; forcing a state reset would violate
the continuous-individual contract.

## Four competing hypotheses and binding predictions

| Hypothesis | Predeclared discriminating prediction |
|---|---|
| H1: motor/read task observability is weak or poorly decoded | At fixed parameters and actual real-input states, dynamic logits fail to provide target-alignment benefit over a declared common/prior branch. If useful motor information survives, it is realizable in the tested head class, and calibration is adequate, fixed-feature calibration must improve independent conditional risk over a matched prior/common control. A short failed fit cannot reject observability. High variance alone is insufficient. |
| H2: surrogate event credit, particularly reset feedback, is misleading at the realized step scale | Writer perturbations below every relevant event margin leave hard motor outputs unchanged despite proxy derivatives. At event-changing scales, useful proxy directions must predict hard-loss changes better than matched control directions. Reset-only exclusion must improve finite-step agreement or future direction in registered cases if reset feedback is the claimed cause; large reset terms alone do not satisfy this prediction. |
| H3: credit truncation misses task-relevant delayed sources | With unchanged weights and identical full-state values, moving a detach boundary changes only gradients for causal ancestors that cross it. A14-tick source has full within-window credit at positions0..17, not18..31. Dominant real task dependencies must actually occupy these cut routes for truncation to explain risk; state persistence or long physical tails alone do not prove that. |
| H4: parameter-time drift or Adam history dominates | Compare the same actual terminal old-parameter state under actual body/head displacement masks. A body displacement can harm the next real window even when smooth fixed-feature head calibration is beneficial. If drift is dominant, reducing actual body displacement or already-observed-prefix recalibration reduces the mismatch while source/event-credit quality need not improve. A favorable head-only step alone does not distinguish drift from H2. |

Fit-window replay and following-window continuation are distinct estimands.
Replay starts from the saved pre-window state. Following-window scoring starts
from the physical terminal state actually evolved under old parameters, then
uses new parameters. Future labels cannot select the current repair step.
Any re-evolved-state control must be named a counterfactual and must not replace
the actual continuing state. No toy capability task or extra training is needed
to test these operator identities and finite-step predictions.

## Minimal structurally sound candidate: reset-only gradient exclusion

The single candidate C_reset changes only the reset's backward dependency:

    spike = existing hard SpikeFn(v-theta)
    h_next = v * (1-spike.detach())

Apply the same convention in both the native finish and sealed motor reset;
leave the spike's surrogate gradient active for pulse transmission and
ALIF/STP event learning. This restores the exact within-event reset partial
1-s and removes the direct -v*phi term. Forward values, timing, motor-only read,
hard events, state fields, topology, capacities and optimizer cadence remain
unchanged before learning.

This candidate is structurally admissible and directly isolates a known
operator choice. It is not mathematically necessary for NLL improvement:
event credit remains approximate, boundary derivatives remain truncated,
and global recurrent stability remains unproved. It does not introduce a
probabilistic spike model or claim an exact expected-loss gradient.

Pre-reset motor-voltage reading is a **different axis**: it changes the observable
and its forward predictions while keeping the anatomical read location.
It could preserve voltage discarded by reset, but its usefulness must be
tested separately. Do not bundle it with C_reset and attribute a gain to reset
credit. A bare rescaling or new head also cannot establish missing information.

## Decision and review boundary

Proceed only to an independently approved, minimal real-OWT finite-update
mechanism probe: preserve full state and Adam history, separate smooth head
calibration from surrogate body updates, record actual displacement/inner
products, event changes and the two scoring estimands above.

Keep C_reset as the unique first repair candidate only if those predeclared
checks implicate its reset operator at useful finite-step scales. Reject that
local rationale if reset exclusion fails its direction prediction; do not add
a timing patch or a second mechanism to protect it. If head/prior calibration,
task observability or cross-boundary credit dominates instead, select the
corresponding hypothesis for a new design review.

Theory/necessity review is independent; experiment sufficiency and implementation
approval remain separate. No new production run, NLL victory, unique root cause
or exact long-history credit is approved by this document.
