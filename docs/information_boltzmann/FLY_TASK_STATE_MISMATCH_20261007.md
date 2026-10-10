# Task-state mismatch: competing explanations, not a new training arm

Status: provisional diagnosis; independent review pending. Existing registered
training continues unchanged. No new experiment or architecture is authorized
by this note.

## Target

Improve matched-target first-pass language NLL in a continuing individual with
sensory-only physical injection, motor-only physical readout, and less than
3900 MiB dedicated CUDA allocation/reservation. Separate actual language
benefit from numerical correctness, internal forecasting and circuit activity.

## Three problem formulations

1. Information availability: what information about recent inputs is available
   to the motor decision at its prediction deadline?
2. Representation/objective: which state distinctions help predict language,
   rather than merely predict another physical state?
3. Learning: can the task objective preserve useful input representations while
   changing the very dynamics that generate those representations?

## Observed implementation and interim evidence

- The current observer uses fixed random local projections of sampled h/ge/gi,
  adaptation, STP and delay-ring values. Its arrival target is detached.
- Its auxiliary objective is mean squared error on that physical projection;
  the task objective is next-token cross entropy. Both are optimized jointly.
- It predicts one future regional tick. Incoming features follow the fixed
  aggregated E/I-delay graph. It does not execute a fourteen-hop computation
  on the current event. Coarse regional edges are not individual axonal paths.
- At 5k, 10k, 15k and 20k, fresh active evaluation was respectively about
  0.170, 0.094, 0.037 and 0.070 nats worse than its matched unigram reference.
  Fresh contexts differ, so raw NLL decreases alone are not a learning curve.
- At about 23k, recent training traffic beat the reference by 0.184 nats;
  the original physical forecast MSE was slightly worse than persistence.
  These are interim observations, not a full-budget capability verdict.

## Competing hypotheses

H1 -- Deadline/information path: one-step state estimation still leaves valuable
current input on slower paths. A continuing observer improves estimation but
does not automatically compensate arbitrary multi-hop latency. Prediction:
current-input sensitivity is limited by the actual observer computation graph.
Reject H1 as dominant if existing readable features already retain and support
the requisite current-input distinctions with adequate task training.

H2 -- Objective mismatch: projected physical-state MSE rewards prediction of
large/persistent background, which can be irrelevant to conditional language.
Prediction: internal MSE improvement can coexist with no matched NLL benefit;
task-trained forecasting/representation can outperform physical-MSE supervision.
Reject dominance if the auxiliary demonstrably improves matched task NLL over
an otherwise identical jointly trained baseline.

H3 -- Optimization/representation destruction: unfreezing changes the substrate
and the read representation concurrently, with truncated surrogate gradients.
Prediction: comparable-budget, identical-interface frozen and plastic models
diverge through learning rather than an initial information-path change.
No conclusion follows solely from a gradient being nonzero or locally correct.

H4 -- Measurement: earlier good scores involved other injection/read surfaces,
initialization, data budgets or protocols. The research tree explicitly records
that the level-0 6.9662 broadcast result survived zero-transmission probes.
That historical result therefore is not itself proof that the connectome once
performed the desired routed calculation. Other historical results retain their
own provenance; this does not invalidate all earlier positive results.

## Mathematical checks

For fixed readable representation R and target X, the minimum expected log loss
over unrestricted decoders is H(X | R). Better calibration approaches that
ceiling; additional task-relevant information in R can lower the ceiling.

Under squared error, the Bayes-optimal physical forecast is E[Y_next | R].
There is no general implication from minimizing its MSE to minimizing
H(X_next | R). A high-amplitude irrelevant coordinate and a low-amplitude
task-relevant coordinate provide a direct counterexample to such an implication.

Likewise, for projection P and physical evolution F, a closed reduced transition
F_bar exists only if P F(s1,x) = P F(s2,x) whenever P s1 = P s2 for the represented
state/history and input. Random sampling/projection alone provides no such
closure. Missing information can create an irreducible forecast error.

These statements identify missing assumptions, not a proof that all predictive
models or the physical substrate must fail.

## Decision

Stop treating physical realism or physical forecast accuracy as sufficient
justification for a language architecture. Before the next candidate, declare
the task-relevant state, its causal availability at the output deadline, and the
objective that trains it. Keep sensory/motor locality explicit.

The most economical future comparison would isolate the physical-MSE auxiliary
from the same streaming architecture and joint task training, with matched data,
initialization, budget and active evaluation. This is a proposed discriminator,
not an approved or launched experiment. Use the existing run as its own
registered endpoint first; no rescue modules are appended to this arm.
