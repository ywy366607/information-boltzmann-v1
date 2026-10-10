# Lossless finite-horizon response and fused transmission

## The repair

The preceding hybrid retained exact motor-local physical states but forecast unknown incoming pulses from a lossy regional code. Its heldout response error was 0.1915 times a hold-current baseline. The new `CausalMotorForecaster` preserves the physical upstream state needed by the output horizon, rather than relying on the code to reconstruct information it discards.

It computes directed shortest delay distances to the original motor surface. A node belongs to the conservative physical domain when its shortest path has total delay at most the requested forecast horizon. For every retained node, all original h/ge/gi/ALIF/STP states and four pulse slots are retained. Origin pulses outside the domain are aggregated into their known arrivals into **every retained node**, not only motor nodes, for the next four ticks. Future updates then use the same local equations and original internal edges.

There is no new clock, bath, readout head or neural fitting. The original output surface remains the only readout. The current individual is untouched: predictions operate on selected copies and return motor physical states for each tick.

## Why this closes the missing path

Let \(D(i,M)\) be the shortest sum of edge delays from neuron \(i\) to the motor surface. A newly emitted outside-domain pulse cannot occur before the first predicted update. If \(D(i,M)>H\), its effects cannot reach the motor surface during the next \(H\) ticks. Already emitted outside pulses can arrive sooner because part of their delay has elapsed; these are retained separately in the origin-arrival queue.

Path cost is \(\sum d\), **not** \(\sum(d+1)\): an intermediate neuron can receive and emit a pulse in the same physical update. The first new emission incurs the additional update, so using \(D\le H\) is conservative. Some boundary-domain neurons may differ from the full simulation after omitted new pulses arrive, while their effects remain causally unable to reach the motor surface within the horizon. Exactness is claimed for the motor surface through H.

The proof assumes local COBA/ALIF/STP equations, fixed graph/parameter revision and the declared zero future external drive. A new sensory event revises the forecast origin; a synapse/delay/output-surface revision requires rebuilding the compiled operator. This is a known-boundary physical response forecast, not a forecast of unknown future stimuli.

## Numerical and speed verification

The approved CPU assay uses the same induced real-connectome graph and stimulus lanes as the prior tests: 256 neurons, 5,901 edges, seed 11, 8 continuous excitation lanes, 8 origin snapshots, heldout lanes 4–7. Future branches have declared zero drive. Future oracle states are comparison-only. No optimizer updates or checkpoints were used.

| Horizon | Retained neurons | Scatter motor response error | Fused max motor-voltage error | Pulse decision mismatches |
| --- | --- | --- | --- | --- |
| 1 | 230/256 | 0 | 7.45e-9 | 0 |
| 2 | 246/256 | 0 | 7.45e-9 | 0 |
| 4 | 255/256 | 0 | 7.45e-9 | 0 |
| 7 | 255/256 | 0 | 2.98e-8 | 0 |
| 14 | 255/256 | 0 | 3.73e-8 | 0 |

The scatter implementation reproduces all motor-voltage/readout trajectories exactly. A 2.33e-10 conductance rounding difference occurs at horizon1. All measured motor states meet the preregistered 2e-6 tolerance; b/x/u and all motor pulse slots match exactly in both implementations.

The fused implementation compiles excitation/inhibition and four delay tiers into one sparse block matrix, with row `sign*N+post` and column `(delay-1)*N+pre`. Each tick performs one sparse matrix multiplication against the four pulse slots. Duplicate edges add their actual weights; no dense N-by-N matrix is allocated. Different accumulation order explains the fused floating-point differences.

For H14, the matched CPU batch8 measurement (one warmup, median of7) returned:

- Original complete integrator, saving all tick motor state fields: **30.706 ms**.
- Causal scatter implementation, same output contract: **37.385 ms**.
- Causal fused implementation, same output contract: **11.849 ms**.
- Fused compiler: **4.513 ms**, separate from warmed forecast time.
- Sparse matrix payload: **74,876 bytes**, separate from origin/state/output storage.

This is approximately 2.59 times faster than the measured complete CPU integrator and 3.15 times faster than the causal scatter implementation in this assay. CPU timing is machine-specific and does not establish GPU throughput. At H14, preserving 255/256 neurons removes very little state; the speed benefit comes from transmission fusion rather than a drastically smaller causal domain.

The earlier scatter report used a full-integrator timing that returned only its final state; it is retained as historical provenance. The fused report corrects this by collecting the same per-tick motor state fields in all implementations. Empty boundary groups were subsequently skipped as a semantics-preserving overhead reduction; published timing above belongs to the measured implementation before that minor change.

## Interface

```python
predictor = CausalMotorForecaster(model, horizon=14, fused_transmission=True)
coefficients = predictor.coefficients(model)
motor_states = predictor.rollout(current_physical_state, coefficients)
```

`motor_states[0]` is the actual current motor state; entries1–14 are autonomous responses. Decode through the current original `output_read` using the returned h and mean. The predictor accepts no teacher-future input. Rebuild after anatomical weight/delay/surface changes; refresh coefficients after physical parameter changes.

This interface is experimental and production training/runtime has not been switched. It provides a reliable response oracle and an optimized physical forecast implementation. A cheap learned surrogate remains a distinct option and must approximate this target on independent streams before promotion.

## Tests and review

44 affected RTC and streaming tests pass. Coverage includes multiple horizons, both scatter and fused variants, all local physical state fields, delayed outside pulses entering intermediates, nested causal domains, horizon guards, and preservation of the preceding student contracts.

`rtc_contract_review` independently reviewed the corrected delay proof, domain boundary queues, teacher-input separation, block-matrix layout and timing fairness. Final result review is recorded in the research tree. This work supports the numerical response claim; language NLL and full-brain cost remain separate acceptance questions.
