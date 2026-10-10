# One-tick pipelined predictive interface

Status: implemented numerical interface, independently reviewed for conditional
mathematical validity; final implementation/experiment approval pending. Nine
new numerical tests and44 affected existing tests pass. No training outcome is
claimed.

## Evidence and problem

Fresh anatomy-prior measurements resolved sensory-to-motor input contrast at the
next tick in all36conditions. Response peaks depend on port/body state. This
supports a delayed access contract; it does not select a global settling time.

Three formulations of the problem are: (1) prediction deadline versus physical
access; (2) variable-delay recurrent state versus a single response peak;
(3) the input target's information set versus a likelihood evaluated after the
target has already been assimilated. The proposed repair addresses (1) and (3).
Prediction quality and learning credit remain separate questions.

## Proposed single contract

At tick t, use the complete previous physical state and unchanged parameters:

1. Consume old delayed-ring entries; update conductances and non-sensory units.
2. Read the motor surface and seal p_t(x_t | previously observed inputs).
3. Observe x_t, score the sealed probability, and integrate its drive in the
   sensory units using the same old-state snapshot and newly computed arrivals.
4. Complete all ALIF/STP updates; commit the new full pulse ring and physical
   state. Apply optimizer updates only after the complete tick is committed.

Example: tick0 writes A; tick1 transmits prior pulses, predicts B, then writes B;
tick2 transmits prior pulses, predicts C, then writes C. The long pathways remain
in flight and supply later predictions. This is one physical tick per observed
input at steady state, with finite pipeline latency.

The body is not advanced twice. Arrival conductances and integration coefficients
are shared by both stages. In particular, do not advance/reset sensory units with
zero input and then integrate them a second time with the new input.

## Conditional equivalence

With fixed theta, minimum chemical delay>=1, disjoint injection/read surfaces,
old-ring-only transmission, local state updates and a writer evaluated against
its original old-state snapshot:

    m_t^- = F_motor(S_{t-1}, old_arrivals),
    p_t = Decoder(m_t^-),
    S_t = F_theta(S_{t-1}, sensory_drive(x_t)).

Motor output is independent of x_t at the current tick. The rearrangement changes
when a probability is sealed, while retaining the complete native one-tick
physical values for the same input and parameters. This requires implementation
verification; it does not follow merely from giving a wrapper this name.

The supervised head now predicts the input that has not yet arrived, x_t. The
archived post-write head was trained against x_{t+1}, although the latest input
could not reach its motor surface. This is an explicit objective and deadline
change requiring training; an old checkpoint cannot be relabeled as a trained
instance of the new protocol. Both protocols use causal observed histories, but
their prediction issue times differ and must be recorded for fair comparison.

The initial candidate reads the native motor state with its existing head, without
an additional Gamma observation filter. Membrane, synaptic, ALIF and STP states
continue to retain history. This is an economical candidate, not a conclusion
that Gamma never helps. No new timing estimator, response-peak gate, multi-head
decoder, learned wait loop or input-to-decoder shortcut is introduced.

## Arrival order

Different paths can reorder components: an event at0 with delay4 arrives at4;
an event at1 with delay1 arrives at2. A fixed-delay individual path still preserves
its input order. Thresholds, inhibition and recurrence further affect which
components become visible. Single-event response measurements do not establish
that actual two-event earliest motor responses reverse order.

Relevant primary research: Izhikevich (2006), Polychronization, describes computation
with axonal delays and reproducible nonsynchronous spike patterns. It motivates
keeping varied paths; it does not certify this language interface's performance.
https://www.izhikevich.org/publications/spnet.htm

## Competing explanations and prospective checks

- Deadline mismatch: repairing the target/deadline contract may improve predictive
  use of the most recent accessible input. Improvement is not guaranteed by a
  nonzero physical response.
- Insufficient semantic discrimination at the motor surface: corrected timing
  may remain insufficient even with mathematically valid transmission.
- Harmful or inadequate plastic credit: corrected timing need not cure surrogate
  gradients, short-window credit or optimizer interference.

Before any long training, verify full-state equality against the native no-Gamma
tick, current-target replacement invariance of the sealed prediction, old-ring
commit ordering, source-state continuation and complete time-stamped prequential
scoring. For an arrival-order claim, use predeclared A/B order swaps from the same
initial state and explicit controls; do not infer order from separate norm peaks.

Only after the numerical gates pass should a single sufficient-budget joint OWT
training plan be independently approved. Reinitialization or continuation must
state the provenance and bridge boundary; an archived previous-token target must
not be counted again as a fresh prequential target. Judge held-out predictive
quality, not response magnitude or transport ablation alone.

## Independent review

harness_hypotheses accepted conditional causality and steady-state throughput;
diagnosis_review accepted the predictive interpretation and required same-state
sensory integration, atomic final ring commit, and post-tick parameter updates.
Experiment approval and implementation verification are pending. No new run has
been started.

## Implementation contract (2026-10-07)

`core/fly_pipeline.py` exposes begin_fly_prediction(model,state) without a token
argument and commit_fly_observation(model,pending,token). The shared COBA tick
preparation consumes arrivals once; both motor prediction and final integration
retain the old state. A pending tick is committed once. Sealed motor features may
be batch-decoded in a BPTT window because the decoder and RMSNorm are pointwise
and weights remain unchanged until the complete window is assimilated. Generation
uses actual begin-phase logits before choosing or receiving a token.

The new learner consumes the targets themselves, once each, beginning with the
empty-prefix first target. The archived learner keeps its previous-token shift.
All physical fields, optimizer states and elapsed-event counts continue across
windows and active-validation phase boundaries. Fixed-state motor blindness is
distinct from an online predictor's parameter history; the latter can also carry
information about previously observed tokens.

A fresh-initialization defect was corrected: the independent decoder weight now
clones the embedding after its existing std=.02 initialization, instead of
cloning Embedding's default random values. Saved decoder values loaded from a
checkpoint remain authoritative. This repair is explicitly part of the fresh
candidate and must not be treated as proof of a past checkpoint's NLL failure.

## Multi-hop acceptance through 14 ticks (2026-10-07)

One-tick throughput means a new observation enters each tick; the complete body
continues evolving while older signals are in flight. Four ring slots represent
each chemical edge's delay of 1..4. A path with edges 4+4+4+2 and a path with
fourteen delay-one edges both have cumulative delay14. Recurrent paths can last
longer;14 is the user's requested acceptance range, not a truncation constant.

At tick t, a component from input x_(t-d), for d=1..14, can enter the sealed
prediction of the still-unobserved x_t. It is not used to retroactively score
x_(t-d+1) after subsequent observations have entered the body. All motor paths
contribute to one conditional distribution; no lag-specific target duplication,
14-tick settling loop or new observation filter is added.

Numerical acceptance now checks exact earliest arrival for disjoint known-delay
paths1..14, absence of earlier motor contrast, native full-state equality,
writer/all-edge surrogate gradient reachability at14, full checkpoint continuation
with in-flight pulses, and current/future-observation invariance of actual CE
scores. These verify implementation, not learned language capacity.

State continuation and credit continuation differ. BPTT32 preserves every
physical state field at each window boundary while detaching historical
derivatives. For a14-tick path, only source positions0..17 in a32-event window
reach their loss within that same window (18/32). The last14 source positions
retain later physical influence, but their source-to-later-loss derivative
crosses the boundary. An explicit detach-at7 test preserves the complete
tick14 response and removes the earlier token's derivative. The model therefore
represents14-tick paths; this protocol does not certify every event's14-tick
credit or eliminate derivative decay.

## Registered production arm

The motor-only pipeline now matches Gemini's candidate's externally supplied
GPT-2 input table and independent decoder initialization, training-only static
frequency bias, d768, and initial read-norm gain .1. Exact local artifact hashes
and the reference's actual10,940,858 training tokens are recorded. The input
table remains frozen; the body, writer and motor decoder train jointly.
The external prior's offline exposure is distinct from new96,000 train tokens.

The single prospective budget is3000 new optimizer updates of32 events, with
six active B/revisit validations. All98304 events,3072 actual updates and98304
physical ticks include the actual phase bridges. Calibration is the first real
update and, if used, resumes at event32 without reset or repeated exposure.
CUDA capture setup work is counted separately. The primary endpoint is the
last1000 fresh-training updates; final active B is a separate direction check.
Best B is a monitoring
artifact. Compare each model to the same locked unigram reference and compare
common fresh-data prefixes explicitly. Different prediction deadlines,
auxiliary losses and direct token-conditioned paths remain route differences,
so this is a comparison of complete candidates, not isolated causal attribution.

If the corrected pipeline improves its own locked predictive risk, retain the
interface. If the last1000 fresh updates and final active B remain worse than
the reference, report that this budget did not establish useful learned motor
conditioning; do not infer a unique credit failure or add a timing mechanism
from that result alone. Source-hash-locked independent review precedes execution.

The registered risk decision uses per-target model-minus-reference differences
over all32000 primary tokens. A95% paired percentile block interval uses256-token
blocks and predeclared128/512 sensitivity,5000 resamples,seed0. Blocks terminate
at each active-evaluation insertion; segment-end128-token partial blocks remain
in the estimand and resampling with actual token weights. Interval upper bounds
below0 at all three sizes support a budget-limited complete-candidate risk
advantage, conditional on these blocks capturing relevant dependence. This is
finite evolving-trajectory evidence, not an infinite-stream guarantee. Final B
has only one256-token block: retain its complete curve and reference differences,
report direction only. Even a risk advantage may include decoder-bias adaptation;
it does not by itself prove motor-conditioning use or timing-specific benefit.
Compare fixed first1000 and last1000 update NLL descriptively and retain the full
curve; continued improvement means convergence remains unestablished.
