# Fused training collision rotation

The two-layer Givens part of `LocalInvariantCollision3D` now uses one Triton
forward kernel and one backward kernel. Each program owns one site and both
rotation layers. It handles all batches and even nullspace widths up to 512;
CPU/FP64, noncontiguous tensors and other layer counts retain native autograd.
The collision network, nullspace basis, dt scaling, invariants, bath and
optimizer objective are unchanged. The rotation is fused; the MLP retains
its compiled implementation. A subsequent exact structured projection path
is described below.

## Derivative

For one pair, y_left=c*x_left-s*x_right and y_right=s*x_left+c*x_right.
The backward applies g_x_left=c*g_left+s*g_right,
g_x_right=-s*g_left+c*g_right, and
g_theta=g_right*y_left-g_left*y_right.

The second layer is differentiated first and its adjoint feeds the first.
Both angle derivatives return to the angle network. Local intermediate values
are recomputed in registers, avoiding full-field scratch copies and per-layer
launches. The fused primitive supplies first-order BPTT; the native path
retains higher derivatives. Runtime kernel failures surface instead of silently
disabling the path mid-run.

## Numerical validation

On a two-batch d128 collision projection, the direct audit reports zero maximum
forward/state-gradient difference against native eager evaluation, maximum
angle-gradient difference 1.91e-6, relative angle-gradient error 7.28e-8, and
maximum relative site-energy error 1.47e-7. The integration test compares every
angle-network parameter gradient and checks mass/momentum, energy, fullgraph
compilation and CUDA Graph replay. Tests also cover d256 and Q27-width rotation,
multiple batches and native layer-count fallback.

Regression: 651 passed, one skipped, one pre-existing failure caused by the
absent results/ib_local_bpe_256_3000_v2/age_003000.pt fixture. All 11 new
collision tests pass, along with the direct Deslice/gate script.

## Production timing

Matched checkpoint/data offset, fresh AdamW, quadratic bath, K16/dt4/d128,
128 tokens/update, chunk32, two warmups and eight measured updates:

| Rotation backend | Median full update | Tokens/s | Peak reserved |
| --- | ---: | ---: | ---: |
| Original compiled native schedule | 1.1277 s | 113.51 | 1114 MiB |
| Fused Triton, complete angle/state gradients | 1.1022 s | 116.13 | 1028 MiB |

Latency falls 2.26 percent and reserved memory falls 86 MiB. Instrumented
collision work falls from 107.4 to 102.6 ms per 32-token chunk. The modest
whole-update gain locates remaining work in the angle network and nullspace
matrix operations. FP32 operation/derivative rounding differs, so continuing
optimizer trajectories are not bitwise identical. Short-run joint losses are
timing diagnostics, not evidence of better language NLL or convergence.

```powershell
python scripts/ib/benchmark_port_execution.py --checkpoint results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt --bath quadratic --collision-kernel native --output results/published/port_execution_quadratic_native.json
python scripts/ib/benchmark_port_execution.py --checkpoint results/q8_predictive_ports_k16_unified_bath_3000/BBest.pt --bath quadratic --output results/published/port_execution_quadratic_triton.json
pytest tests/test_triton_collision.py -q
```

## Exact structured projection and multiscale cost

The actual four-invariant SVD completion satisfies N = J + L R^T to FP64
error below 1e-12. J selects coordinates 4:D; L has shape [D,4] and R has
shape [D-4,4]. Consequently

```
N^T f = f[...,4:] + (f L) R^T
N c   = pad(c, left=4) + (c R) L^T
```

This factors the existing basis, not the field, and retains D-4 scattering
degrees of freedom and the unchanged checkpoint parameters. Constructor-time
factorization is verified before enabling it. One Triton kernel applies each
projection/adjoint. CPU/FP64 retains dense autograd; CUDA supports both
contiguous sites and the channel-major layout actually produced by FFT.
Supporting this latter layout is essential: a contiguous-only dispatcher
silently leaves production on the dense path. Static compilation avoids the
Torch 2.9 symbolic-stride tracing failure across alternating physical layouts.

Matched complete-update timings with fused Givens and quadratic bath:
1.10105 s dense versus 1.08712 s structured, a 1.27 percent latency reduction;
reserved memory 1028 versus 998 MiB. The kernel improves projection execution
but the angle network, writer, FFT and gradient accumulation still cost time.
These are computation-only bath-replacement timings, not trained-model NLL.

For a multiscale stack, count sites, channel widths and steps separately:
collision work is approximately sum_l K_l N_l (4 D_l + D_l H_l + layers_l D_l),
plus launches/reductions/weight gradients and layer coupling. With constant
width and step count, 3D halving gives ideal site-work ratio
1 + 1/8 + 1/64 + ... = 8/7. Doubling channel and hidden widths at each
level instead gives MLP ratios 1 + 1/2 + 1/4 + ... = 2. Small-grid launch
overhead makes actual timings larger than these arithmetic estimates.

The present RG proposal keeps all fine-grid work and adds a coarse integrator;
it is an overhead-bearing capacity extension. A compute-saving design must
move suitable work to the coarse scale and retain fine detail through a skip
path. Preserve state-dependent microstep dependencies; only independent sites
and shared-network batches may be grouped. Combine multiscale read features
before the existing batched vocabulary decoder, avoiding duplicated logits.

Reports: `port_execution_quadratic_dense_projection_matched.json`,
`port_execution_quadratic_structured_projection.json`, and
`collision_stages_structured.json` under results/published. Isolated stage
timings include full derivatives and are not additive production percentages.

Captured isolated stage medians (milliseconds, channel-major FFT layout,
1000 graph warmups and five timed samples):

| Width/sites | Angle network | Dense full collision | Structured full collision |
| --- | ---: | ---: | ---: |
| d128 / 256 | 0.12507 | 0.20137 | 0.19569 |
| d128 / 32 | 0.06372 | 0.09935 | 0.08566 |
| d256 / 256 | 0.18123 | 0.35602 | 0.31631 |

The d256 full collision improves 11.15 percent in this isolated workload;
its complete language update has not been benchmarked. The 32-site angle
network costs about half the 256-site network, despite one eighth the sites.
Fusion of normalization, MLP activation/linear work and their derivatives is
the next execution target. Reusing a computed angle across dependent steps
would change the state-conditioned model; batch independent sites instead.

After the projection extension: 37 focused tests pass; full regression has
660 passes, one skip and only the existing absent historical checkpoint
fixture failure. The compiled kernel test patches the live Windows launcher
configuration as well as its environment variable, so earlier suite imports
cannot restore the broken static launcher.
