# D768 event fusion execution audit

The continuous individual was saved after 32 updates / 1024 fresh OWT targets.
The optimization retains D768, 8x8x4 geometry, pretrained frozen input vocabulary,
BPTT32, AdamW, the W4 objective, physical cadence and all four evaluation pillars.

`runtime/training.py` exposes one shared event for eager and compiled execution.
Unused read-port geometry diagnostics are omitted. Audit-only event outputs are
detached; the write free-energy objective and all physical transition/read paths
retain their gradients. `--compile-event` compiles this event's forward and
backward with Inductor/Triton. Non-reentrant activation recomputation remains.

Validation uses numerical interface checks and the actual saved individual with
its next OWT window. The benchmark checks per-target NLL, complete continuation
state, each token's audit outputs, and joint parameter-gradient error before
measuring complete updates. It leaves the live checkpoint unchanged.

Execution-only resume requires `--allow-execution-change`; only the entrypoint,
training runtime and active-trainer source hashes may differ. Physical model,
data budget, evaluation contract and optimizer configuration must still match.
Both source-hash snapshots are written in `execution_continuation`.

Initial validation: existing affected checks 18 passed / 7 skipped; AOT event
fusion + recomputation gradient/state/audit equivalence check passed.

Real D768 GPU validation passed: joint gradient relative L2 error 7.48e-7;
per-target scores, full continuation state and audit outputs met declared
tolerances. Complete fused updates measured 4.42 and 3.39 seconds per 32 targets
(8.20 tokens/s). Eager updates in the same benchmark measured 17.73 and 16.98
seconds, slower than the preceding live run (8-9 seconds), so the 4.45x paired
benchmark ratio must not be advertised as the live speedup. Dedicated memory
was 2341 MiB, peak PyTorch allocation 1968 MiB. First compile/backward took
1076 seconds. Compilation is cached for continuation.

The original FP32 dot-product cosine reduction returned a value above one;
that diagnostic is invalid due to reduction precision and is excluded. The
gradient relative-error check remains the equivalence criterion. Future cosine
reductions use FP64. No model capability conclusion follows from this benchmark.

Live continuation succeeded with PID 23772, preserving the physical field,
optimizer, cursor and cumulative health ledger. Before fusion, updates 16-32
averaged 8.171 s/32 targets (3.916 tokens/s). Resumed updates 45-73 averaged
1.646 s/32 targets (19.443 tokens/s), a 4.96x observed live speedup. This is a
sequential wall-clock comparison, rather than a simultaneous hardware-controlled
comparison. Observed GPU dedicated memory remained below the 3900 MiB guard.
