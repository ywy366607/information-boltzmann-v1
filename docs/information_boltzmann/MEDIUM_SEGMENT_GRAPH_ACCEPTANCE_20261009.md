# Bounded CUDA segments on the continuous D768 individual

Execution change only, 2026-10-09. Retains 8³ full state, intrinsic event time,
all solver refinements, moving finite ports, structural posterior, temporal
history, writer objective, AdamW, BPTT32 and the active four-pillar learner.

## Adopted execution

`--medium-segment-graph` captures only forward physical microsteps and derivative
measurements run inside event checkpointing. Backward recomputes the complete
event and uses the existing reverse-mode VJP. Graph outputs are cloned into
owned storage: the next replay cannot overwrite any prior token's state.
Dynamic state, prepared material, structural sample and duration are copied to
graph inputs. Module parameter addresses remain live across Adam updates.
FP64 elapsed-time storage is retained. Graph caches are bounded at two advance
variants and one RHS variant; extra shapes fall back to the ordinary execution.
Forward-mode diagnostics retain the native operator.

The analytic field RHS now also has fused forward/backward kernels. On Windows,
both first-call routes configure the dynamic CUDA launcher before compilation;
this includes a dashboard snapshot before the first evolution step.

## Real-checkpoint numerical acceptance

Saved step211: 6752 fresh tokens, 7136 total events, 223 optimizer updates. The
same next32 real OWT targets were unfolded with the complete credit window.
Comparing original fused evolution/native RHS to the accepted execution:

| Quantity | Maximum absolute error |
| --- | ---: |
| 11 outgoing state tensors | 1.4305115e-6 |
| Incoming-state gradients | 2.9802322e-8 |
| 101 named parameter gradients | 1.2516975e-6 |

None-gradient semantics and finite gradients match; acceptance uses FP32
`atol=5e-6, rtol=5e-4`. Complete-state forward AD succeeds. This verifies
execution equivalence within floating-point tolerance, not a capability claim.

## Matched complete-update timing and resource budget

Both independent arms restore the same checkpoint/RNG and process the same next
128 targets, including four-pillar observations and parameter-update snapshots.
Compile/capture-containing first updates are excluded. Remaining three:

| Execution | Seconds / 32 targets | Mean |
| --- | --- | ---: |
| Original | 12.1500, 10.8645, 10.6990 | 11.2378 |
| Segments + fused RHS | 10.1905, 10.0374, 10.2065 | 10.1448 |

Measured throughput ratio 1.1077×; wall-time reduction 9.73%. Median ratio is
1.0661×, so this is a modest gain, with no basis for claiming a several-fold
speedup. Matched four-window NLL differs at most 1.9372e-7 nats.
Sampled dedicated peak is 3344.65MiB, below the declared 3900MiB total stop
threshold and 4096MiB hardware capacity. Allocator budget is 3708MiB, leaving
192MiB for context/driver overhead. Ordinary Adam moment CPU staging remains.

Keeping Adam moments on GPU was rejected: the second window hit the allocator
cap through fragmentation when requesting a 148MiB contiguous workspace.
It produced no production checkpoint. The accepted mode uses CPU staging and
the previously verified idle-cache release boundaries.

## Continuation and validation

Production resumed from the complete saved step211, preserving the same
individual and evaluation cursors. The dashboard remains on port8085. Progress
publishes graph counts/replays separately from physical/learning clocks. Existing
last/best filenames are overwritten; this change adds no checkpoint series.

Focused CPU regressions: 63 passed, seven opt-in CUDA cases skipped. CUDA buffer
ownership/current-parameter tests: two passed. Deslice/scatter/gate checks pass.
The first CPU invocation hid CUDA devices, provoking two compiler-environment
failures; repeating with normal visibility passes. Real acceptance reports are
in `results/published/medium_segment_graph_acceptance_20261009.json` and its linked
identity, matched timing and live-memory reports.
