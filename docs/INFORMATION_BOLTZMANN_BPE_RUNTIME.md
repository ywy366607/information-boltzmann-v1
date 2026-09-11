# GPT-2 BPE / 512-particle runtime candidate

2026-09-11. Target: 512 continuous particles, 256 BPE targets per optimizer
update, approximately one second per update on the existing GTX 1650 (4 GiB).

## Implementation and tradeoffs

`prepare_ib_gpt2_bpe.py` reconstructs the existing 10,000-document OWT subset,
verifies source hashes, preserves split/document order, and uses GPT-2
`encode_ordinary` plus EOT 50256, as in
[nanoGPT's OWT preparation](https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py).
Vocabulary: 50,257. Training split: 10,940,858 BPE tokens. This is not full OWT.

The new candidate has 6,698,826 parameters, width 128, shared embedding/readout
weights, 512 particles, four-dimensional position and velocity, 16 Slice
channels and four Strang substeps per token. It uses learned local gamma.
This is a new model, not a continuation of the width-2048 byte checkpoint.
Width reduction and vocabulary tying must not be described as kernel speedup.

Bulk token projections/readout, deterministic activation recomputation,
Triton OU fusion, device-only Slice solves and whole-update CUDA Graph replay
reduce host launches and synchronization. States remain sequential and causal;
there is no detach inside the 256-token window. Initial density learns on the
first update, followed by persistent detached state between optimizer windows.
Slice Cayley is a new deterministic low-rank collision, not an equivalent LBM
or original stochastic binary collision operator.

## Verified measurement

`results/ib_bpe256_graph_verified`: 1 warmup update plus 10 measured updates,
2,816 actual tokens consumed without skipped windows. Median **1.340789 s**,
range **1.313624–1.343907 s**, approximately **191 tokens/s**.
PyTorch peak allocated memory **746.78 MiB**; this excludes driver/context and
is not total device usage. First update 15.59 s; graph capture 13.13 s.
Timing includes input/noise preparation, transfers, forward, backward,
gradient clipping, AdamW and state propagation. Checkpoint writes, validation
and detailed energy accounting are excluded. This is near the requested
target, not a claim of <=1.00 s or a long-run throughput guarantee.

At 16 tokens, matched eager/graph runs both consume 48 events with identical
generator state. Maximum parameter difference 4.77e-7; x 5.07e-7; v 3.58e-7.
Initialization-only parameters have one Adam step; recurrent parameters have
three. All 47 affected numerical tests pass, including whole-window versus
causal single-event loss/state/gradient equivalence and long-lifetime clocks.

Earlier `ib_bpe256_graph` and `ib_bpe16_graph_v3` were preliminary: their
capture bookkeeping skipped one data window and initialization was frozen.
They are excluded from continuous-trajectory evidence. Corrected capture does
not advance the data cursor; initialization is learned on the warmup stream.

## Reproduction and research boundary

Run from this worktree with a new output directory:

```powershell
python scripts/benchmark_ib_bpe_window.py --data data/ib_owt_gpt2 --output results/bpe_runtime_new --particles 512 --tokens 256 --hidden 128 --updates 10 --graph
```

`--updates` counts timed updates after the one real warmup update. `last.pt`
contains model, optimizer, persistent state, event cursor and RNG state.
This entry is a runtime calibration tool, not yet a resume/validation/energy
instrumented long-run trainer. The short run establishes execution only;
no capability, convergence, memory or criticality conclusion follows.
Before a new 5,000-update study, integrate this path into the lifetime trainer
with audited resume, independent validation and sampled dynamics accounting.

## Authorized 5,000-update run

`scripts/train_ib_bpe_lifetime.py` now implements that training path. Run
`results/ib_bpe_512_5000` uses 512 particles, H128, 256 BPE tokens per update,
constant AdamW learning rate 3e-4 and 5,000 total updates (including the first
learned-initialization update): 1,280,000 training targets with no stream wrap.
Every 250 updates and at birth, frozen validation uses 256 burn-in tokens and
4,096 scored tokens from the independent validation split, seed 99173. The
test split is untouched. `BBest.pt` minimizes this fixed validation NLL;
`last.pt` and `age_*.pt` retain optimizer, persistent phase, cursor, RNG and
source/data provenance. Selection can retain birth if no trained weight wins.

Every update records kinetic-plus-harmonic energy and local gamma distribution.
These boundary observables are not a full work/heat ledger or a criticality
measurement. Saved phase checkpoints permit separate response diagnostics.
The stopping rule is the authorized update budget, not asserted convergence.

The numerical smoke check consumed real held-out/train OWT tokens only to
verify execution. Resuming age 2 to age 3 exactly matched uninterrupted model
parameters, phase and random generator state (maximum observed error zero).
The runtime-only entry above remains available for isolated calibration.

The user also requested detailed live monitoring and GPU/memory calibration.
`serve_ib_bpe_monitor.py` serves `present/ib_bpe_live.html` on localhost:8080
from the run directory, with hardware polling in its own CPU process. Training
records per-axis population variances, energy components and gradient norms
each update. Every ten updates it saves all 512 x/v/gamma samples, the 8x8
covariance and eigenvalues. These are sparse training-boundary snapshots,
not sufficiently sampled trajectories for wave spectra. The full snapshot
history remains on disk; browser polling is every three seconds.

Memory tradeoff calibration: disabling all recomputation exceeded the 2.72
GiB allocator safety limit. Recomputing half the drive calls used 1786.78 MiB
peak allocated memory but slowed to median 1.4887 seconds/update. The formal
run retains full recomputation (~746.78 MiB in the earlier benchmark) because
it was faster. Total-device memory and utilization are recorded separately;
neither is promised to equal a target percentage. The pre-monitor startup was
stopped before a reported training update and retained separately at
`results/ib_bpe_512_startup_before_monitor`; the formal monitored run restarts
with the same seed, not from a partially instrumented checkpoint.
