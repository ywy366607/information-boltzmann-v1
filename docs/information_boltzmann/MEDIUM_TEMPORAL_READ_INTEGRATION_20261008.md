# Persistent local temporal readout: integration acceptance

The accepted local complex filter bank is now part of `PlasticMediumPorts3D`,
the joint BPTT learner, CUDA training capture and the continuous event runtime.
Enable it in the primary live-learning entry point with `--temporal-read`.
Existing instantaneous/dynamic constructors and old state payloads retain their
previous behavior. This is an optional architecture branch, not an implicit
conversion of an existing individual.

## Measurement and state contract

Each compact probe measures signed physical coordinates of the field and of
`response_time_reference * field_rhs`. Its observer state obeys

\[
\dot z=-(\alpha+i\omega)z+\alpha r,\qquad \alpha=\exp(\log\alpha)>0.
\]

One call to `advance` executes the physical interval, measures the available
endpoint and integrates the held sample with the exact complex-exponential
recurrence. This is endpoint quadrature for a varying physical signal, rather
than an exact continuous observation of that signal. The real and imaginary
components retain their signs. Their learned mixture is added to the existing
dynamic local read feature before the shared final RMSNorm and vocabulary head.

`PlasticBelief.temporal` retains complex `[B,P,M,2D]` history and an FP64 clock.
History updates on advance only. Assimilation, repeated reads and diagnostic
snapshots leave it unchanged. Detach changes the credit boundary while retaining
all values. Clone, checkpoint packing, graph replay and schema-7 continuous
runtime continuation include both history and clock. Clock mismatch fails fast.

**Moving sensor contract:** history belongs to a probe identity and records the
signals encountered along its learned trajectory. Moving a probe changes future
samples; it does not reinterpret old samples as observations at the new position.
No interpolation of history onto a different spatial location is performed.
The same causal convention applies when rates change during online learning.

Sampling cadence is the cadence of physical advance calls, not the number of
internal solver substeps. For matched deployment, train at the same observation,
advance and read timestamps. Refining a continuous runtime's `max_step` also
refines its endpoint sampling; its quadrature then has a different resolution.

## Learning and initialization

Rates, frequencies, temporal mixing, projection, probe coordinates, writer and
physical medium receive the joint task gradient. The rate/frequency parameters
have zero generic weight decay. They are not fixed biological constants.

The CLI exposes the acceptance bank's initial coverage explicitly:

- `--temporal-half-life-events 1 4 16 64`;
- `--temporal-frequency-fractions 0 0.0625 0.125 0.25`.

Half-lives are converted using the declared event duration. Frequency fractions
are converted using `pi / event_duration`. These are configurable coverage
initializations; their optimality has not been established. Direct constructors
require explicit `temporal_rates` and `temporal_frequencies` in model-time units.

The additional branch has 32,904 parameters for the standard D128/16-probe,
four-mode model and 131,080 bytes of persistent state at batch one / FP32.
Full model parameters: 14,280,639. BPTT activation storage still depends on its
window. The branch starts trainable rather than zeroing all its learning paths.

Legacy UORO/local-receptor learner constructors explicitly reject this new mode;
their existing state-factor definitions do not include complex observer history.
Use the integrated joint BPTT learner for this branch. Existing legacy modes
continue to pass their tests.

## Accepted checks

- Public likelihood, quiet chunks and timestamped runtime agree on state and
  gradients at matched sampling timestamps.
- Reads are idempotent; observation alone does not advance history.
- Temporal-only task credit reaches rates, frequencies, positions, medium and
  both writer components, with the original read path detached.
- Live context changes retain history and the optimizer cadence. Checkpoint
  restoration in the middle of gradient accumulation exactly reproduces
  subsequent scores, parameters and complete state.
- CUDA training replay matches eager scores, all gradients and complete state
  across successive chunks and parameter changes. Returned history owns its
  storage. The deployment physical graph also preserves the observer history.
- Old payloads without temporal fields load in their original architecture.

Affected tests: **94 CPU passed**, **13 CUDA opt-in cases skipped** in CPU runs;
the two new temporal CUDA tests were enabled separately and **both passed**.
The repository-required Deslice standalone checks also passed.

The full repository run stopped at its first three unrelated failures after
410 passes: two tests require removed historical checkpoints, and the legacy
fly one-update diagnostic receives an event tuple where it expects a physical
state. Their sources are outside this integration and were left intact.

## One full-size real-text execution check

Actual primary entry point, actual OWT train stream, 8x8x4 / D128 / BPTT32,
fused medium and captured forward/backward, one AdamW update:

- 32 first-pass targets, one optimizer update, no corpus repetition;
- 0.3803656 seconds for the measured 32-token group (84.13 tokens/s);
- whole-board dedicated memory sample: 3,059 MiB (2.99 GiB);
- allocator peak allocated: 1,038.87 MiB;
- finite pre-clip gradient norm: 5.98453;
- physical elapsed time: 0.16000000000000006;
- calibration mode produced configuration/metrics/progress only, zero `.pt` files.

This timing excludes compilation/capture startup and is one group, not a
long-run throughput estimate. Its NLL is not used to judge capability. The
captured training graph covers native temporal/port arithmetic; the medium
retains its existing fused execution. CUDA graph correctness is verified even
though a separate compiled temporal arithmetic kernel is not provided.

The saved calibration source hashes identify the measured code. Subsequent
entry-point changes only expose temporal clocks/rates/powers in telemetry and
clarify configuration metadata; final source hashes are recorded separately in
the integration report.

Artifacts:
`results/published/medium_temporal_read_integration_20261008.json` and
`results/medium_temporal_read_calibration_20261008/{config,metrics,progress}`.

Next capability decision: sufficiently trained joint real-stream NLL with a
matched-budget baseline, preserving active prequential evaluation and the
individual's complete state. Long training has not been launched by this task.
