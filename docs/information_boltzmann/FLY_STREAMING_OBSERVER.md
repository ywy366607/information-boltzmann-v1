# Continuing predictive observer: independent route

## Goal and execution contract

Improve first-pass language prediction in a never-reset COBA-ALIF-STP individual
without waiting for a per-token 14-step rollout. The physical connectome advances
one existing tick per arriving token. A separate reduced observer advances once
on that same event and immediately supplies a motor-only predictive read feature.
Its inference interface carries the same state used in training.

This route has its own modules and CLI. HX-0, HX-1 and their checkpoints remain
comparison artifacts. The physical integration equations are reused unchanged.

## What is predicted

Let `y_t` be a **fixed** seeded local projection of samples of membrane potential,
E/I conductance, ALIF adaptation, STP resources/release, and four in-flight pulse
slots. Existing superclass labels are split at the exact motor read surface;
there is no all-region flattening into the vocabulary readout.

The observer carries a regional posterior history and a next-tick prior:

\[
z_t=p_t+K(y_t,p_t)(y_t-p_t),\qquad
p_{t+1}=F(z_t,\sum_d A_d^{E\top}z_{t+1-d},
                 \sum_d A_d^{I\top}z_{t+1-d}).
\]

`A_d` is constructed from the actual initial physical E/I weights and cumulative
delay splits. Receiving weight is normalized jointly over signs and delays;
zero edges stay zero. These aggregates are fixed in the first implementation.
Physical synapses continue learning, so this is a reduced model approximation,
not an assertion of exact evolving single-cell dynamics or exact arrival times.

`K` is a content-dependent innovation gate, not a Kalman covariance update.
The forecast is a conditional-mean prediction under the observed input stream;
future incoming words are unknown. It is not a quiet-input teacher trajectory.

Motor readout is:

\[
r_t=W_{\rm read}h_{\rm motor,t}
       +W_{\rm adapter}p_{\rm motor,t+1},\qquad
\ell_t=W_{\rm dec}\,\mathrm{RMSNorm}(r_t)+b_{\rm dec}.
\]

The adapter starts at zero, preserving the original physical readout and its
initial normalization gain 0.1. The additional one-step forecast path is explicit.
Current sensory input can affect that motor forecast only through a real
one-delay inter-region connection; longer paths arrive through continuing history.
There are no 14 independent forecast heads or 14 per-event serial iterations.

## Arrival supervision and credit

At event `t`, save the issued forecast, its transition source and its deadline
`t+1`. At the next event, score the original forecast against the newly arrived
fixed observation before adapting the parameters. Keep that score separate from
the prediction recomputed for training.

The auxiliary objective recomputes `F(saved_source)` using current parameters and
the arrived target. Within a learning window the source retains its autograd
graph. Across windows it is detached and retained as bounded online replay. This
updates the predictor after the deadline while maintaining constant history
storage; it does not restore physical credit beyond BPTT32. The target branch is
detached for this loss. CE still jointly trains the physical brain, sensory
writer, observer, motor adapter and decoder.

The observer's projection buffers are fixed, preventing a learned target encoder
from shrinking its output to lower MSE. Representation quality and missing sampled
information remain empirical questions. Report original forecast MSE alongside
the previous-observation persistence baseline, not MSE alone.

## Complete continuing state

Checkpoints contain physical state, observer history/prior, pending source,
issued forecast, previous observation, tick, all weights and optimizer moments,
input/validation cursors, preceding token, recent A exposure, score meter and RNG.
Generation requires an explicit observer state and returns it. A window boundary
detaches graphs while preserving every state value.

`observer_dim=128`, `sample_per_region=64` and `window=32` are named computational
or learning budgets; they are not biological constants. Physical delays retain
the existing model's declared tick units.

## Falsifiable predictions and acceptance

1. Removing one-tick sensory-to-motor support makes the current sensory input's
   derivative on the immediate motor forecast vanish. A three-tick message is
   absent from the earlier forecast arrivals.
2. With fixed parameters, whole-sequence, partitioned and restored execution have
   the same forward values; the generated logits equal the training interface.
3. Detached pending sources still give the current predictor a gradient when
   their evidence arrives. Changing later targets cannot change earlier logits.
4. At initialization, the zero adapter makes the decoder path exactly equal to
   the physical motor-read baseline.
5. If forecasting is useful for language, sufficient joint real-OWT training
   improves matched-target first-pass NLL; better internal MSE alone is insufficient.

The first four are numerical/interface tests. They do not establish reasoning or
memory capacity. Independent review approved these checks and one 32-target real
GPU update for memory calibration. Formal evaluation uses 100,000 new OWT targets
(3,125 fresh updates), active context changes and scored A-B-A returns. Actual
revisit intervals are reported, including bridges. Complete only the registered
budget; do not equate budget completion with convergence.

## Decision rule

Primary full-budget endpoint: all first-pass training targets paired with the
locked train-only unigram reference; report the last 1,000 fresh-update mean and
every active Fresh B window paired with that window's reference.

Secondary comparison: match the HX-0 common 55,040 training-target prefix and its
11 identical Fresh B contexts. HX-0 has no completed 100k artifact, so a full-budget
architecture superiority claim requires a future matched-budget baseline.

If forecast MSE beats persistence but NLL has no benefit, retain the valid
forecaster finding and reject language-use benefit for that budget. If forecast
MSE also fails, revisit observation sufficiency/transition fit. No new modules
are added to rescue a failed result without a new registered hypothesis.

## Biological motivation

Ongoing distributed and local activity is measured across the fly brain:
https://www.nature.com/articles/s41467-023-41261-2

Motor-related inputs with appropriate timing/sign for visual compensation:
https://pubmed.ncbi.nlm.nih.gov/26237362/

Continuous angular integration in E-PG/P-EN circuits:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6320684/

These support continuous coupling and specific predictive compensation. They do
not certify this reduced observer as a whole-brain biological predictive coder.
