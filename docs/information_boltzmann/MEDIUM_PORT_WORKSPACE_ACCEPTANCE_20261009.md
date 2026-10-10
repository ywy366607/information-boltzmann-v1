# Finite read workspace and live dashboard acceptance

Date: 2026-10-09. Scope: execution equivalence and continuity of the existing
D768, 8x8x8 OWT individual. This is not a new language-capability experiment.

## Implemented and accepted

- `read_key_execution=support`: rebuild each head's union of live finite query
  supports, gather full-channel inputs, evaluate only that head's original
  weight rows, scatter keys into the original attention. The original parameter
  names, value path, footprint derivatives and hard-zero mask remain intact.
- Omit the final expression feature in writer-only auxiliary replay. All event
  timing, physical evolution, endpoint RHS and temporal-history updates remain.
- Complete-state forward AD diagnostics use native port operations. This avoids
  creating reverse-mode compiler variants at the first active evaluation.
- Continuation retains birth characteristic-scale metadata rather than comparing
  it against scales measured from already learned material. No restored material
  or physical-time value is replaced by a birth value.
- The 8085 dashboard selects the active run through
  `present/medium_dashboard_runs.json`. Run identity includes its resolved path;
  changing runs clears stale scene and curve caches. Actual evaluation progress
  is published between the same 32-event learner windows.

## Numerical and resource acceptance

Two affected CPU regression groups: **108 passed, 13 optional CUDA tests skipped**.
Independent reader/auxiliary review and dedicated tests cover all gradients,
finite boundaries, moving/overlapping/periodic ports, strict checkpoint keys,
forward AD, 32-event replay and pending-gradient continuation.

Actual saved step157 / fresh5024, real next32 OWT targets, full BPTT32:

| Compared quantity | Maximum absolute difference |
| --- | ---: |
| All 11 persistent state tensors | 0 |
| Incoming-state gradients | 3.3813194e-8 |
| All 101 named parameter gradient entries, including None semantics | 1.1138618e-6 |

Joint loss, task NLL and per-token scores pass atol5e-6/rtol5e-4. Native
complete-state JVP succeeds with fused ports enabled. Sampled dedicated-memory
peak is 2952.65MiB, below the declared3072MiB limit. Observed shared usage is80MiB;
the audit does not separately isolate its WDDM baseline. This audit writes no
production checkpoint or cursor.

## Complete-update timing

Separate disposable snapshot executions restore the same model, optimizer,
belief, carry token and CPU/CUDA RNG. Each executes two real32-target AdamW
updates. The first contains compilation and is excluded from warm comparison.
CPU threads=2; health update-copy monitoring is excluded from these timing
processes, while ordinary production keeps it enabled.

| Execution | Warm second complete update | Sampled dedicated peak |
| --- | ---: | ---: |
| Dense keys, auxiliary final read retained | 12.0728s | 2924.65MiB |
| Support keys, auxiliary final read omitted | 11.5765s | 2924.65MiB |
| Dense keys, auxiliary final read omitted | 11.5987s | 2924.65MiB |

The two optimized variants differ by0.2%, within ordinary single-window timing
variation. Both are about4% below the reference in this check; this does not
establish a stable broad throughput gain. The key-projection MAC ratio7.11x
does not translate into a sevenfold complete-update gain. In the measured case,
omitting the unused auxiliary read accounts for essentially all visible saving.
Support execution is numerically correct and remains an explicit option. The
two-window resource check alone was insufficient to accept a long continuation;
the extended production acceptance below governs that claim.

The saved raw read radii(.125,.1875,.25) are scaled by the saved aperture
budget.25 to runtime radii(.08667015,.13000523,.17334031). Fixed padding is72
sites/head. Two key projections use84.93M rather than603.98M MAC. The16x figure
from effective birth points is not the implemented padded arithmetic ratio.

## Live continuation

The old process exited at step157 during its first active-evaluation JVP with
`FailOnRecompileLimitHit`; its complete pre-evaluation checkpoint remained valid.
Continue from that exact checkpoint with the audited execution flag. Retain the
same5000 fresh groups /160k token budget, physical state and history, learned
structure, all optimizer moments, RNG, stream cursors and active evaluation.
Numerical calibration targets are not committed to this production individual.

The first two resumed production attempts subsequently exhausted the CUDA
allocator after four fresh-B windows. At the failure,1.48GiB was live and1.22GiB
was idle reserved; a148MiB request failed under the declared cap. Repeating with
`max_split_size_mb:128` failed at the same point. `expandable_segments` is not
supported on this Windows runtime. Neither incomplete evaluation was recorded as
retained B experience: both attempts restore their complete pre-evaluation save.

The exact storage repair releases unused allocator segments after all Adam
moments have moved to CPU, and again before moments are restored following the
complete backward. Completed-window temporaries are released explicitly. This
does not release any live field, temporal state, pending gradient or moment
value, and does not shorten BPTT32. Affected staging/Adam/decoder-lifetime CPU
checks pass; independent weak-reference review confirms old CUDA-moment objects
are not retained at the cache-release boundary. Production now publishes actual
allocated/reserved/inactive-split memory at every evaluation window.

The retained individual is resumed from step159 / fresh5088 /159updates. The
new full fresh-B256 + bridge/revisit128 chain, subsequent warm updates and sampled
dedicated-memory peak are recorded in the final continuation JSON below. The
original short GPU audit remains a numerical reference, not evidence that the
earlier multi-window allocator problem did not occur.

Final resumed progress and first completed active evaluation are recorded in
`results/published/medium_port_workspace_resume_20261009.json`.

Actual continuation acceptance: full fresh-B256 and bridge/revisit128 completed
at fresh5120 /172actual updates, followed by five warm fresh-training updates
through step165 /177actual updates. Those five complete32-token groups took
12.5347,13.4690,13.3551,12.0385 and12.0332seconds, including ordinary production
health monitoring. Dedicated-memory sampled peak was3003MiB under3072MiB.
The browser verified progressing training steps, field snapshots, learned port
displacements and live evaluation counters; its captured console had no errors.
The same-target active-B meanNLL was6.954048 versus fixed prior7.512127. The
256-token assessment does not establish a stable adapted platform: the recovery
estimator correctly reports `plateau_unconfirmed`. Continue the registered run.

Browser proof: `results/published/medium_live_dashboard_acceptance_20261009.jpg`.

Artifacts:

- `results/published/medium_port_workspace_gpu_acceptance_20261009.json`
- `results/published/medium_port_workspace_dense_matched_20261009.json`
- `results/published/medium_port_workspace_support_matched_20261009.json`
- `results/published/medium_port_workspace_dense_aux_omitted_matched_20261009.json`
- `scratch/medium_port_final_review_20261009.md`
- `scripts/ib/audit_medium_port_workspace.py`
- `results/published/medium_boundary_cache_memory_20261009.json`
