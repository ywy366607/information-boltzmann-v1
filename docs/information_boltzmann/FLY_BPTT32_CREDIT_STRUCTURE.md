# BPTT32 credit structure on a continuing OWT individual

Date: 2026-10-04. This is a gradient/mechanism audit, not a capability benchmark.
Production weights, optimizer, physical state, input/output surfaces and training
cadence were not changed. The live 100,000-target continuation completed during
the audit.

## Method and provenance

- Snapshot A: 80,000 additional training targets, 2,692 total BPTT optimizer
  updates including active evaluations, train cursor 180,000. Six complete
  consecutive 32-target windows (192 real OWT targets).
- Snapshot B: 95,008 additional training targets, 3,197 total BPTT updates,
  train cursor 195,008. Two complete consecutive windows (64 real targets).
- Each fork restores all physical states and the actual preceding token.
  Weights stay fixed for differential measurement; forks never update the live
  learner. These NLLs are not primary active-evaluation NLLs.
- Window-mean-loss parameter credit is distinguished from a single endpoint
  prediction's state adjoints at every lag. Endpoint positions 16 and 32 are
  measured separately.
- Recompute each full event and compose its VJP backwards through all 32 events:
  membrane, E/I conductance, ALIF, STP, sensory baseline, delayed pulses and all
  cross-neuron wiring participate. This is checkpointed BPTT, not conditional
  local eligibility. The existing ATan threshold backward is retained.
- A deterministic numerical test matches native whole-window autograd for
  boundary state, physical parameters and E/I edge-weight gradients. Five
  affected tests pass. Real replay state error is at most 7.45e-9.
- Gradients are unscaled/unclipped. An edge's full weight gradient is reconstructed
  from its full downstream current adjoint and actual delayed transmitted pulse.
  Current-adjoint-to-state propagation still covers all wiring.
- Bounded replay allocated 711 MiB on CUDA, and combined dedicated GPU usage was
  observed at 3,119 MiB. Snapshot A's seventh-window diagnostic hit Windows host
  commit-memory pressure; its six complete reports are retained. Reports now
  stream to disk rather than accumulating spatial dictionaries in memory.
- Snapshot A's original full-field drive adjoints include hypothetical injection
  outside sensory surfaces and are explicitly excluded from write-credit claims.
  Snapshot B restricts drive credit to actual sensory injection indices.

## Findings (within-snapshot medians)

| Measurement | Snapshot A | Snapshot B |
| --- | ---: | ---: |
| E edge gradients exactly zero | 93.06% | 94.40% |
| I edge gradients exactly zero | 98.07% | 98.31% |
| E edges carrying 99% squared-gradient mass | 0.413% | 0.382% |
| I edges carrying 99% squared-gradient mass | 0.0496% | 0.0389% |
| Neurons carrying 99% h-adjoint mass at lag 16 | 2.677% | 2.459% |
| h-adjoint norm at lag 32 / lag 1 | 1.821 | 2.481 |
| h-adjoint mass on currently firing neurons at lag 16 | 0.000295% | 0.000676% |

At lag 8 and longer, approximately 98.8% of neurons have a nonzero h adjoint,
while its squared mass remains strongly concentrated. This distinguishes exact
sparsity from approximate concentration. Sparse firing is not a state-adjoint
mask: silent membrane and conductance histories carry substantial credit.

The 32-event boundary is not a point of vanished credit in these trajectories.
At snapshot A the median h-adjoint norm ratio to lag 1 is 0.891 at lag 8,
0.949 at lag 16, 1.244 at lag 24 and 1.821 at lag 32. All six lag-32 ratios exceed
one (range 1.422 to 2.544). Snapshot B corroborates a surviving/amplifying tail.
This is finite-trajectory sensitivity, not an infinite-horizon Lyapunov claim
or proof that increasing the window will improve predictive performance.

Spatial credit has cell-type structure. At lag 16, descending neurons carry
approximately 47–50% of h-adjoint mass; VNC intrinsic/motor and central-brain
intrinsic neurons carry much of the remainder. These are superclass aggregates,
not a claim of newly discovered anatomical neuropil specialization.

ALIF parameter credit is relatively weak at these snapshots: log adaptation
time-constant gradient norms are about 9.5e-6 / 1.2e-5, log adaptation coupling
about 1.9e-4 / 1.6e-4, versus log threshold about 0.74 / 0.90. Parameter groups
have different effects and Adam normalization, so these norms identify weak
supervision, not dispensable mechanisms. STP x/u credit is spatially sparse and
increases toward earlier events; its long physical time constant does not alone
establish an important predictive credit direction.

Snapshot B's actual sensory-drive adjoint norm is zero at lag 0 (disjoint
input/output with delayed transmission), about 4.4e-5 at lag 1 and 1.64e-2 at
lag 31. The input-credit path therefore has a real delay and a surviving tail.
The restricted sensory/motor interfaces remain intact.

## Consequence for implementation

The supported first optimization is event-sparse transmission and edge-weight
gradient accumulation, retaining complete within-window state credit. A second,
approximate optimization could target concentrated adjoint support, but a
99%-mass set alone is not an error bound on future learning. Its selection must
include delayed dependencies and silent state. Temporal compression requires
retaining slow boundary credit; uniform earlier truncation is not justified by
these measurements. No pruning or learning-rule change was made in this audit.

## Artifacts

- `scripts/ib/inspect_fly_bptt_credit.py`
- `scripts/ib/summarize_fly_bptt_credit.py`
- `tests/test_fly_bptt_credit.py`
- `results/fly_bptt32_credit_structure/report.json` and `summary.json`
- `results/fly_bptt32_credit_structure_late/report.json` and `summary.json`

Reproduce with a complete continuing checkpoint:

```powershell
python scripts/ib/inspect_fly_bptt_credit.py --run results/q8_fly_bptt32_continuous_100k --windows 8
python scripts/ib/summarize_fly_bptt_credit.py
pytest tests/test_fly_bptt_credit.py tests/test_fly_bptt_learning.py -q
```
