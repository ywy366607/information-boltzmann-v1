# Online fly plateau: mechanism audit at 34,190 train tokens

The user stopped the 100k run for diagnosis. `last.pt` preserves the complete
individual at 34,190 train predictions, 35,810 total learning events and 8,952
Adam updates, including the two pending gradient events. No production state,
anatomical input/output surface or graph was changed by the audit.

Audit entry point: `scripts/ib/inspect_fly_online_plateau.py`. Detailed output:
`results/q8_fly_infinite_stream_three_factor_100k/plateau_mechanism_audit.json`.
These are real-stream, implementation-level measurements, not a new short-run
capability benchmark. Centered-rank and update measurements use 128 continuing
events on a checkpoint fork; the impulse response conditions on fixed current
weights and uses identical subsequent real tokens.

## Anatomical circuits do transmit

Write/read overlap is zero, as required. Replacing the current real input
changes sensory voltage by norm 114.98, while same-event motor voltage/logits
remain exactly unchanged. At lags 1, 2, 3 and 31, the motor voltage differences
relative to the first branch's motor voltage norm are 0.18%, 7.63%, 9.02% and
10.44%. This tests causal transmission; it does not measure semantic accuracy.
The graph is not inert. The instantaneous delay follows its implementation.
Any future protocol repair must retain bounded sensory and motor surfaces.

## Learning is severely asymmetric

Across the whole 34k run, sampled excitatory and inhibitory edge magnitudes
changed relative to graph initialization by 2.46e-5 and 1.68e-6 respectively.
During the 128-event fork, projection relative changes are roughly 1e-8,
edge changes 1e-8/1e-9, while the output matrix changes by 5.86%. Physical
parameters and modality gates also change: the entire learner is not frozen.

Sampled raw updates show a numerical bottleneck. For three sensory matrices,
97.3%, 98.4% and 98.2% of proposed updates are below half their weight's
float32 ULP. For excitatory/inhibitory edges, this is 99.37% and 99.97%.
Sub-ULP changes are repeatedly rounded off when added directly to weights.
The current two-event edge accumulation does not retain rounding residuals
after applying an update. Exact fusion/reference agreement did not test whether
the intended update was representable at the existing weight magnitude.

## Activity and readout

Whole-brain firing is 1.48%, sensory firing 14.55%, motor firing 0.272%.
DAN firing and release are zero in this window; its highest mean threshold
margin is still negative. Receptor occupancy is effectively zero, leaving
the configured 0.05 baseline plasticity floor. This is an observed inactive
modulatory circuit, not evidence that the biological mechanism is unsuitable.

Centered rank: sampled brain 26.1, sensory sample 38.9, full motor surface
27.75, projected latent 2.32, normalized decoder latent 1.52. The normalized
latent's temporal mean accounts for 99.91% of squared norm. Rich motor
variation exists, but the current decoder representation largely follows a
common direction. This localizes the compression; rank alone does not prove
that recovering every variation would improve prediction.

## Credit approximation audit and next repair

Current recurrent credit uses a presynaptic trace decayed by source-neuron
membrane alpha, multiplied by an instantaneous postsynaptic factor. A COBA
synaptic conductance sensitivity instead depends on the destination synaptic
decay, and its membrane/adaptation sensitivity depends on destination history.
With heterogeneous time constants these recurrences do not factor exactly.
The implementation is a more aggressive approximation than full local e-prop.
Random low-rank directional feedback introduces a further approximation.

Priority is to make small updates accumulate without loss, then validate
destination-conditioned local synaptic eligibility against the actual COBA/
reset/ALIF equations. Per-edge storage remains constant in stream duration;
O(1) in duration does not require O(number of neurons) rather than O(edges).
The anatomical surfaces remain fixed. Output latency should be evaluated
with the physical response times before altering the language event protocol.

Revisit NLL improved at all six recorded evaluations after 230 intervening
events. That measures retention plus active relearning at one lag. A forgetting
curve requires comparable earlier exposures at several actual lags and
pre-update revisit scoring; the current report does not identify its shape.
Spaced replay is a possible consolidation policy after the learning update
path is operational, not a substitute for repairing it.
