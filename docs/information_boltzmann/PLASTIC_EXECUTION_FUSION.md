# Plastic-medium execution fusion

The execution backend preserves the W4 policy, compact ports, local collision,
conductance response, adaptive conduction and STP equations. It changes kernel
scheduling only. The model still carries field, flux, conduction, receptor,
resource/utilization and precision states across observations and BPTT chunks.

## Captured bottleneck

`scripts/ib/profile_plastic_training.py` instruments actual CUDA Graph module
forward intervals and the entire backward. It does not estimate backward cost
from eager component ratios. Production throughput is measured separately:
external timing events add instrumentation overhead.

At 8x8x4/D128, vocabulary50257, FP32, BPTT8, real OWT, with the independent
fly job running concurrently, the captured native execution measured:

| Eight-token chunk | Native | Medium fused |
| --- | ---: | ---: |
| Medium forward | 33.877ms | 8.215ms |
| Entire backward | 160.660ms | 105.584ms |
| Write forward | 15.818ms | 15.897ms |

These stage profiles identify the medium as the first target and writing as
the next forward bottleneck. They are not optimizer-update throughput.
Reports: `results/published/plastic_training_captured_profile.json` and
`results/published/plastic_training_captured_profile_fused.json`.

## Complete optimizer-update measurement

Whole-medium and write/read fusion together measured **1.815s/update** versus
the matched native **3.582s/update**, approximately **1.97x throughput** and
**49.3% less execution time**. Both used real OWT, 128 tokens/update, BPTT8,
8x8x4/D128, vocabulary50257, 14067383 parameters, FP32 and CUDA Graph,
duration0.005/substeps1, with the same independent ALIF/STP fly job running.
Each runs one warmup optimizer update followed by two measured updates.
Compilation and external memory-counter query time are excluded.

At the production grid/vocabulary, the fused graph was compared against native
medium and native write/read over two different BPTT chunks. Full accumulated
parameter gradients and all continuation tensors passed: maximum state error
**1.43e-6**, maximum gradient error **8.11e-7**, equal losses at reported FP32
precision. Optimizer updates continue through the same captured parameter
storage; the objective remains token CE plus the existing W4 objective.

Final affected CPU set:31 passed,10 opt-in CUDA checks skipped. Five distinct
CUDA checks passed across native/fused full-state/full-gradient comparison and
training, evaluation, variable-time deployment replay. The final three replay
checks also update write/read/medium parameters and preserve old snapshots.
Warm cached production capture setup was15.17s; cold compilation took minutes.

Peak tensors were425.45MiB native and417.87MiB fused, with the latter including
the native-reference audit alongside the captured graph. Sampled whole-card
dedicated peaks were3477MiB and3275MiB, both below4GiB. The fused training-only
profile has lower allocation than this conservative audit-inclusive peak.
These timings describe concurrent-load execution, not exclusive-card throughput
or a language-quality improvement.

Reports: `results/published/plastic_native_execution_matched.json` and
`results/published/plastic_all_fused_execution.json`.

## Execution contract

`PlasticMedium3D.advance` can fuse the complete quiet CUDA FP32 native flow
with `torch.compile` and AOT autograd. `PlasticMediumPorts3D` similarly supports
separate quiet write/read fusion. CPU, FP64 and diagnostic calls keep the
native reference. Compilation occurs during capture warmup and is cached.
The nested STP implementation is inlined into full-medium compilation.

`--medium-execution native|fused` and `--port-execution native|fused` select
execution in the trainer and benchmark. Parameters/state dictionaries and
architecture identity are unchanged. Both flags are recorded in checkpoints;
resume may change these execution flags while retaining strict physical-law,
token-budget, cadence and BPTT compatibility checks.

The command-line tools place disposable compiler artifacts under
`scratch/compiler_cache` on the workspace drive. `--compile-cache-dir` and
explicit `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` environment settings
override that location. This avoids consuming the Windows system temp disk;
it has no effect on model equations or checkpoints.

The training graph captures both forward and backward. Every replay reads
updated parameters at stable addresses. FP32 kernel fusion can change rounding;
compare native/fused complete state and all active parameter gradients, and
separately check replay after parameter updates. There is no gradient detach
inside an event or additional state reset.

## Reproduction

```powershell
python scripts/ib/profile_plastic_training.py --fusion all --output results/published/plastic_training_captured_profile_all_fused.json
python scripts/ib/benchmark_conductance_training.py --backend cuda_graph --medium-execution fused --port-execution fused --tokens 128 --chunk-tokens 8 --event-duration 0.005 --allocator-fraction 0.40 --warmup-updates 1 --updates 2 --output results/published/plastic_all_fused_execution.json
```

The 0.005 duration is the unchanged execution-test cadence in nondimensional
model time. Runtime equivalence and speed do not establish a trained-language
improvement; the joint learning objective and data protocol stay unchanged.
