# Language execution audit — 2026-10-01

The current language workload is K=16, dt=4, d=128, 8×8×4 sites,
32-token BPTT chunks and 128 observed tokens per AdamW update. The tested
checkpoint is `results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt`;
its saved step is 2500. These are execution measurements, with the existing
unified bath, rather than a new capability or convergence comparison.

## What actually costs time

The original component estimate apportioned a captured total using eager
component proportions. Direct CUDA events inside the captured graph give a
different priority. The following intervals bracket both forward execution
and each compiled module's backward node, after batching the decoder and
sharing the innovation chart, before adjacent transport fusion:

| Component | Forward + backward per 32-token captured chunk |
| --- | ---: |
| Unified bath | 157.32 ms |
| Collision | 107.99 ms |
| Write agent | 73.21 ms |
| Transport | 66.07 ms |
| Read agent | 12.42 ms |
| Batched decoder | 0.84 ms |
| Unassigned work and instrumentation | 60.53 ms |
| Instrumented graph total | 478.38 ms |

These measured intervals exclude clipping and AdamW. The instrumented graph
has event overhead; whole-update benchmarks below run without those events.
No eager-to-production proportional scaling is used. The cost ranking is
specific to this checkpoint/configuration and the unified bath.

The full vocabulary is not exceptionally large. One `[1,128] @ [128,50257]`
projection costs approximately 12.87 million FLOPs. The two writer vocabulary
products and the decoder total approximately 4.94 GFLOPs over 128 tokens in
forward execution, or approximately 14.82 GFLOPs including their two standard
matrix-product derivatives. This excludes all other operators. The earlier
claim of 1.65 GFLOPs **per token per projection** confused a token with an
entire 128-token update.

The write agent performs a full categorical prior and an expected feature
at every causal event. Conventional language-model input encoding is usually
an embedding lookup. Here the interior also has 2048 microsteps per update,
each operating on a 256-site field. The unified bath evaluates a learned thin
QR basis at each microstep, in addition to spectral diffusion. Thus this
implementation has substantial extra serial work beyond a vocabulary head.

## Equivalent execution changes implemented

1. **Batch output decoding.** Form each causal read feature sequentially,
   then decode `[B,L,D]` with one vocabulary GEMM. The decoder never writes
   into the next belief. The mean CE and its complete parameter gradient
   are the same as averaging the per-event CEs.
2. **Share the innovation chart.** For a fixed pre-event belief, the packet
   chart P is linear in feature φ, so `P(φ_observed)-P(Eφ)` becomes
   `P(φ_observed-Eφ)`. The categorical prior, posterior precision and write
   action retain their original definitions.
3. **Skip discarded writer monitoring.** Every event retains its complete
   differentiable port objective. Full public write diagnostics are evaluated
   for the final event of a training chunk, which is the row the trainer
   already logs. Event inference keeps full diagnostics by default. This
   gave little additional speed on this GPU.
4. **Fuse adjacent fixed-symbol transports.** In the palindromic sequence
   `T-C-B / B-C-T`, adjacent T maps compose into one FFT pair. For K=16,
   there are 9 transport FFT pairs instead of 16. The multiplier is the
   composition of the original maps, not Cayley at a changed time step.
   The rFFT boundary planes are first projected onto their effective real
   multiplier, including conjugate pairing, so composition also preserves
   the learned Nyquist residual and its gradient. Adaptive clocks/directions
   continue through their ordinary separate transport path.

## Complete update results

All arms start from the same saved model and posterior field/precision and
the same OWT offset. Each uses two warmup updates and eight timed updates,
with a continuing state and the same fresh AdamW setup. No checkpoint is
produced by these performance runs.

| Execution | Median update | Tokens/s | Peak reserved memory |
| --- | ---: | ---: | ---: |
| Original | 2.0000 s | 64.00 | 1206 MiB |
| Shared chart + batched decoding | 1.8783 s | 68.15 | 1216 MiB |
| All retained execution changes | 1.7675 s | 72.42 | 1194 MiB |

Latency falls 11.6% and throughput rises 13.2%. The largest joint-loss
difference along these eight matched updates is below 2e-6. Deterministic
FP64 tests separately compare belief continuation and all parameter gradients,
including a nonzero learned transport residual. These checks concern numerical
equivalence; no new language-performance claim is inferred from ten updates.

An isolated, captured FP16 vocabulary test reduced the two vocabulary GEMMs
plus probability/derivative work from 2.0074 to 1.8227 ms, with relative gradient
error about 0.00050 after calibrated loss scaling. This is a component result,
not a measured FP16 production update. FP16 has not been enabled in the trainer.

The full repository suite finished with 640 passes, one skip and one failure:
the historical adaptive-force compatibility test references an absent untracked
`results/ib_local_bpe_256_3000_v2/age_003000.pt`. The active checkpoint strictly
loads and executes the CUDA graphs; affected numerical tests pass.

## Direction after this audit

The largest execution target is the **unified bath's basis construction and
repeated transforms**. Before investing in equivalent kernel optimization,
its spatial smoothing should be assessed against the simpler quadratic bath:
preserving a costly map is worthwhile only if that map serves the task.
The writer's vocabulary prior is a separate architectural question: its cost
buys an additional token generative model. A coherent future simplification
would share the prediction likelihood used for output and for the next input's
prior, with one likelihood supervision per observed event. Such sharing changes
the model/gradient paths and makes that decoder causal again; it must be assessed
as a jointly trained model, not advertised as an equivalent kernel optimization.

The measured 1.77 seconds is still above the 1.5-second target. Profiling now
identifies a concrete next target rather than attributing that gap to vocabulary
size or deleting transport on the basis of an eager timing estimate.

## Reproduce

```powershell
python scripts/ib/benchmark_port_execution.py --checkpoint results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt --output results/published/port_execution_fft_fused.json
python scripts/ib/profile_captured_ports.py --checkpoint results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt --output results/port_execution_audit/components_current.json
python scripts/ib/benchmark_vocabulary_gemm.py --checkpoint results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt --output results/published/vocabulary_gemm_precision.json
pytest tests/test_port_execution_equivalence.py tests/test_predictive_impedance_write_agent.py -q
```

The archived component report `results/published/port_captured_components.json`
belongs to the intermediate execution before adjacent-transport fusion. A
fresh profile of the final implementation will therefore show fewer transport
calls, while the whole-update report is from the final implementation.

## Bath comparison and spatial damping

`benchmark_bath_execution.py` compares compiled, captured forward/backward
calls on the same checkpoint field, shape, dtype, probe gradient and dt=4.
The unified bath loads its actual trained weights; the quadratic bath uses
its standard initialization. This is a computation comparison, not a language
evaluation after replacing the bath.

| Bath | Median forward + backward per call |
| --- | ---: |
| Unified | 0.4292 ms |
| Quadratic | 0.1013 ms |

The isolated ratio is 4.24. Unified samples range from 0.3675 to 0.6693 ms,
so this ratio should not be extrapolated into a precise whole-update speedup.
The production component profile separately places bath work at 157.3 ms
per 32-token chunk, above write work at 73.2 ms.

Quadratic damping is local and elementwise: with local radius R,
E=||f||^2/2 and rho=2E/R^2, it applies exp(-kappa*rho*dt).
For fixed positive kappa and dt its outgoing energy obeys
E_out=E*exp(-4*kappa*dt*E/R^2) <= R^2/(4*e*kappa*dt).
This is a one-bath-step bound, not a statement about all trained parameters
or the tangent dynamics. Complementary cold-bath output records the removed
energy. All channels at a site share this attenuation, so the bath offers
overload control rather than demonstrated semantic selection.

Unified damping adds a context MLP, a fresh QR basis and its derivatives,
low-rank content projections, and a 3D FFT/IFFT for spatial diffusion in each
microstep. In the saved step-2500 checkpoint, nu=0.020451674 and
gamma0=0.013322956. On the unit torus the first nonzero Laplacian eigenvalue
is (2*pi)^2. The diffusion-only attenuation over K*dt=64 is therefore
exp(-64*nu*(2*pi)^2)=3.62e-23. This factor describes the homogeneous
diffusion part; interleaved write/collision can generate new spatial modes.
It nevertheless exposes a very strong spatial homogenization bias. A learned
content basis does not by itself restore modes erased by that spatial term.

Separately retained evaluations give 7.21536 NLL for the historical quadratic
champion and 7.42028 for the W4 unified-bath run. These also differ in writer
and evolution duration, so the complete gap is not isolated bath causality.
The constructive baseline is to retain the agent architecture while using
the quadratic bath to preserve spatial signals; semantic selective forgetting
can then be judged independently. Unifying the state generator does not require
this particular three-layer dissipation implementation.

## Champion readout inspection

The saved step-3000 `q8_ness_reference_3000/BBest.pt` loads
`CharacteristicKernelReadout`, with four heads and four queries per head on
the 8x8x4 torus. Probe coordinates are trainable parameters; their periodic
Fourier position features enter attention scores without detachment. Queries
also depend on the observed token embedding. Keys and values are separate
learned projections. Each probe's weighted mean and channelwise variance are
kept separate before learned merging. Coverage of the whole grid does not
reduce this operator to uniform mean pooling.

The periodic coordinate displacement from initialization is 0.04822 on
average and 0.07156 at maximum. Learned scales retain a sharp-to-broad
hierarchy: approximately 14.13, 8.29, 4.50 and 2.35 by head.
On one saved terminal field queried by validation token 8192, mean head
attention entropies are 1.6801, 2.5150, 3.9141 and 4.7112, versus
log(256)=5.5452 for uniform attention. Mean peak node weights by head are
0.5073, 0.3012, 0.1472 and 0.0412. All node weights are positive.
An arbitrary categorical CE gradient through this saved readout gives finite
nonzero coordinate, scale and query gradients. This inspection verifies the
position-learning path and spatially differentiated measurements; it is not
a new validation-NLL experiment or a new inference-capability claim.

The previously recorded single-site coordinate-binding interventions cost
0.061 to 0.068 nats. Together with the differentiated attention, this supports
a useful learned position/processing association. Small coordinate movement
alone does not establish weak use: a covering initialization can already
provide good locations while query, key, value and merge weights learn their
roles. This champion is evidence that the kernel reader can use spatial
structure in its trained W2 system; it does not establish equivalence to the
distinct W4 belief reader. The immediate W4 repair remains the quadratic bath,
after which readout learning should be evaluated on surviving spatial signals.

## Complete quadratic-bath update timing

The explicit `--bath quadratic` execution benchmark replaces only bath weights
with the standard quadratic initialization, preserving every other checkpoint
weight and the saved field/precision. It forms a short continuing stream with
fresh AdamW, two warmups and eight measured updates; it saves no model weights.
This establishes execution cost for the repaired map, not its validation NLL.

With 128 tokens/update, chunk32, d128, 8x8x4 sites and K16/dt4, median complete
update time is **1.1291 s** (113.37 tokens/s). Peak allocated memory is 181.9 MiB
and peak reserved memory is 1114 MiB. The measured interval includes backward,
four captured chunk replays, state/precision continuation, gradient clipping,
and AdamW. Compared with the prior 1.7675 s unified-bath execution, latency
falls approximately 36 percent. This timing does not apply to K64 or d256.

Fresh instrumented 32-token graph intervals give approximately:

| Component | Forward + backward per chunk |
| --- | ---: |
| Collision | 107.4 ms |
| Write agent | 73.3 ms |
| Transport | 36.6 ms |
| Quadratic bath | 12.7 ms |
| Read agent | 12.4 ms |
| Batched vocabulary decoder | 0.84 ms |

The instrumented graph is about 293.6 ms, including remaining gradient,
loss and event instrumentation work; these are not a disjoint accounting of
the complete optimizer update. Collision is now the largest measured module.

CUDA Graph and module compilation are already enabled. Independent sites,
channels and read heads execute in parallel, and decoding is batched across
the chunk. Persistent-token and state-dependent microstep dependencies should
be preserved when pursuing further fusion. At this audit's baseline, the
hand-written Givens adjoint lacked angle gradients and training used the
native differentiable expression. This has now been replaced with a fused
forward/backward including both derivatives: see
[Triton collision kernel](TRITON_COLLISION_KERNEL.md). Against the compiled
native baseline it measures 1.1277 to 1.1022 s/update, with reserved memory
falling from 1114 to 1028 MiB.

The isolated vocabulary FP16 gain remains about 9.2 percent, not a full-update
AMP gain. Local vocabulary GEMM precision is a candidate; field energy,
precision, normalizations and FFT should retain FP32 during initial evaluation.
This GPU reports no native BF16 support (`including_emulation=False`), so the
default emulation-inclusive capability flag should not be used to promise
native BF16 acceleration. The 1.5-second target is already met for this K16
configuration; no numerical-resolution reduction or architecture deletion is
needed to claim that measured milestone.

Reproduce:

```powershell
python scripts/ib/benchmark_port_execution.py --checkpoint results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt --bath quadratic --output results/published/port_execution_quadratic.json
python scripts/ib/profile_captured_ports.py --checkpoint results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt --bath quadratic --repeats 4 --output results/published/port_captured_components_quadratic.json
```
