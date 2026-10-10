# Conductance candidate: forward/backward CUDA Graph

The execution bottleneck was the eager sequence of small PyTorch operations,
especially backward. Capturing the complete differentiable chunk makes the same
physical model practical at substantially lower launch overhead.

`information_boltzmann/runtime/training.py` shares one quiet likelihood path for
eager execution and capture. Every chunk includes the W4 port objective,
conduction adaptation, local transport, conservative collision, conductance and
receptor evolution, physical read agent, and full vocabulary decoder.
Parameter-dependent coefficients are rebuilt inside each replay. Static gradient
storage is initialized before capture so successive BPTT chunks **accumulate**
gradients. It is zeroed once per optimizer update. Returned belief tensors are
owned copies and include field, all three currents, elapsed time, conduction,
receptors and precision. Optimizer updates preserve parameter addresses; graph
replay reads their new values.

## Matched measurement

GTX 1650 4 GiB, with the independent fly job still running at 100% aggregate GPU
utilization. Actual OpenWebText GPT-2 tokens, 50257 vocabulary, 14354610 parameters,
8x8x4/D128, FP32, batch1,128 tokens/update, BPTT8. Each variant runs one warm-up
optimizer update and one measured update on identical corpus offsets. All fast
state continues across updates.

Provenance: these measurements precede the local-material audit revision adding
3072 metric parameters (new total14357682). CUDA capture regressions also pass
after that revision; its latency has not been remeasured. Keep the timings
above associated with the measured14354610-parameter version.

Duration0.005 with one solver step per event is an explicit execution-test
cadence. It does not establish a biological cadence, solver convergence at an
arbitrary input rate, or task capability. Neither the token budget nor the
physical equations were changed to obtain the speedup.

| Metric | Eager | Forward/backward graph |
| --- | ---: | ---: |
| Measured update execution | 13.174922 s | 3.067339 s |
| Forward | 4.230976 s | Captured with backward |
| Backward | 8.900492 s | Captured with forward |
| Combined graph | — | 3.023259 s |
| AdamW update | 0.043455 s | 0.044080 s |
| PyTorch peak allocated | 469.60 MiB | 403.88 MiB |
| PyTorch peak reserved | 576 MiB | 932 MiB |
| Process dedicated peak, sampled WDDM | 654.63 MiB | 1124.64 MiB |
| Total card dedicated peak, sampled | 2463 MiB | 2937 MiB |

Matched speedup:4.295229x. Capture preparation costs3.315022 seconds once.
Timing uses CUDA events and synchronized execution intervals; external CIM and
nvidia-smi counter-query durations are excluded. Stage synchronization remains
in the wall measurement. These are bounded execution measurements under a
competing job, not a long-run median or an exclusive-GPU throughput claim.

Two different8-token chunks were also compared at the real shape/vocabulary:
maximum gradient difference0, maximum complete-belief difference0, equal
losses. Final measured eager/graph update gradient norms and chunk NLL agree.
CPU numerical regressions:63 passed,1 opt-in check skipped. Explicit CUDA
deployment/training checks:5 passed, including accumulated gradients and replay
after changing learned coefficients.

## Memory policy and reproduction

The user's revised2026-10-03 policy permits shared-counter growth and requires
total dedicated memory below4 GiB. Record both; stop at the dedicated limit or
allocator-budget OOM. The allocator fraction0.40 caps this process's PyTorch
allocator at1.6 GiB while the competing process remains in place. Sampled shared
usage grew4 MiB in both variants. Separate no-model controls locate2 MiB at
first-kernel initialization and2 MiB at first GPU-to-CPU scalar monitoring.

```powershell
python scripts/ib/benchmark_conductance_training.py --tokens 128 --chunk-tokens 8 --event-duration 0.005 --substeps 1 --allocator-fraction 0.40 --backend eager --warmup-updates 1 --updates 1 --output results/published/conductance_training_eager_warm.json
python scripts/ib/benchmark_conductance_training.py --tokens 128 --chunk-tokens 8 --event-duration 0.005 --substeps 1 --allocator-fraction 0.40 --backend cuda_graph --warmup-updates 1 --updates 1 --output results/published/conductance_training_graph.json
```

The captured update remains above1.5 seconds in this concurrent measurement.
The whole-medium and port-fusion follow-up preserves the BPTT window and physical
law. Current matched native/fused measurements are3.582/1.815s per128-token
update with local activity and STP enabled. Details and native-gradient checks
are in `PLASTIC_EXECUTION_FUSION.md`. Increasing graph chunk size changes the
credit-assignment window and remains a separately declared training choice.
