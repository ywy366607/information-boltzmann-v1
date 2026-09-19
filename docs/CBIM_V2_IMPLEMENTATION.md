# CBIM v2

Implementation: scripts/ib_local/cbim_field.py. This is a new architecture;
v1 checkpoints must not be silently loaded with strict=False and evaluated as v2.

- Four learned state-only queries, four attention heads, normalized field K/V
  projections, learned periodic-grid slot positions, concatenated retrievals.
- Writer conditions content correction and local write rate on token, current
  slot, and three-slot neighborhood. Token still proposes address and width.
- Local scattering reads both members and learns rotation angles directly from
  CE. Alternate channel planes between spatial brickwork layers. Sum and energy
  remain conserved; no likelihood-ratio estimator is used.
- Diagnostics remain detached device tensors; callers convert to host values
  once per logging window. Clamp frequency counts individual batch samples.
- Sequence vocabulary projection is batched after state evolution, avoiding one
  large vocabulary matrix multiplication per token.
- step(disable_scattering=True) supports frozen causal interventions.
- rFFT DC/Nyquist derivative entries are exactly zero.

Validation: tests/test_cbim_field.py, seven tests passed (invariants, causal
behavior, direct nonzero scattering CE gradient, state-writer gradients,
readout response and sequence/streaming equality).

CUDA numerical throughput smoke: B=1,T=128,L=64,d=128,vocab=50257, FP32,
forward+backward+gradient clipping+AdamW. Two updates: 2.1920s and 1.8163s;
peak allocated 278.19 MiB, reserved 292 MiB. Random inputs are numerical test
data, not a learning experiment. No weights saved. This uncompiled measurement
does not yet meet the user's 1.5-second target or prove trained collision benefit.

Next: real-text training integration with persistent state detached only at
window boundaries, then matched frozen no-scattering evaluation. Preserve v1
and old particle results as separate baselines. Norm preservation does not imply
bounded state Jacobians; retain gradient clipping.

## Whole-update CUDA Graph acceleration

`scripts/ib_local/cbim_cuda_graph.py` provides `CBIMGraphTrainer(model,
tokens=128, batch_size=1, lr=3e-4)`. Call `runner.step(ids, targets)` with the
captured shape. The runner retains the field across updates and performs full
window BPTT, clipping and AdamW. Warmup/capture changes are restored before use.
Outputs use static device storage: clone if retaining them across updates.
Diagnostics remain device tensors; convert only at logging boundaries. Runner
owns its optimizer; save model, optimizer and `runner.state` for continuation.

GTX 1650, same FP32 B=1/T=128/L=64/d=128/vocab=50257 numerical benchmark:
capture 9.38s once; six synchronized update times 0.49751, 0.24896, 0.24629,
0.24600, 0.24626, 0.24623 seconds. Peak allocated 321.61 MiB, reserved 394 MiB.
This includes input copying, forward/backward, clipping, AdamW and state carry;
excludes data loading, checkpointing and host logging. No trained weights saved.

Profiler of the eager update recorded 60,097 cudaLaunchKernel calls, 816.6ms
self CPU launch time, 2.950s total self CPU time and 240.1ms self CUDA time.
These instrumented timings are diagnostic, not wall-time benchmark numbers.
Graph replay removes launch overhead without changing the model equations.
Eight tests pass including two consecutive graph/eager parameter-update and
state comparisons and exact restoration of initialization after capture.
