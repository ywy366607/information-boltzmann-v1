# Information Boltzmann entry points

New runnable entry points live here. Their shared library code belongs in
`information_boltzmann/`, and each run must have a JSON configuration in
`configs/information_boltzmann/`.

The fly online-credit exploration is archived at the user's 2026-10-04
decision. Its replacement trial is `train_fly_bptt_stream.py`: 32-event
truncated BPTT, continuing sensory/motor COBA/ALIF/STP state and active
evaluation. See [continuation protocol](../../docs/information_boltzmann/FLY_BPTT32_CONTINUATION.md).

The current fly continuation uses one sensory pulse and one zero-source physical
tick before motor-only reading. BPTT32 covers all 64 physical ticks; the complete
individual and AdamW history persist. See [event timing and execution](../../docs/information_boltzmann/FLY_PULSE_QUIET_TIMING.md).
`summarize_fly_live_timing.py` reads its actual four-pillar evaluation ledger
without invoking a model. It separates timing phases, scores a fixed train-only
unigram reference on the same fresh targets, excludes the unmatched return
bridge from revisit gains, and checks actual event/update counters.

Online learning without a BPTT window is available as a numerical reference in
`train_online_plastic.py` / `audit_online_credit.py`. The full physical state and
UORO forward eligibility persist across updates. Current global rank1 variance
is too high for production training; see
[`ONLINE_CREDIT.md`](../../docs/information_boltzmann/ONLINE_CREDIT.md).

The destination line is **M8 complete fields with path-adaptive physical time
T at K=64**.  The current joint training stage is its required M=1 port-law
closure: W4 predictive writing, a field-only kinetic read agent, and one
persistent posterior precision state.  It uses the same K=64 interior flow;
M8 may begin only after this shared boundary law has a registered language
result. Do not add one-off mechanism sweeps or task-private solvers here.
Historical scripts remain under `scripts/ib_local/` for reproducibility.

## Historical frozen language comparisons

`evaluate_continuous_owt.py` reproduces historical frozen-checkpoint comparisons;
the ongoing individual uses the active-learning ledger above for primary results.
For a kinetic checkpoint it derives an energy-aligned NESS initializer from
the saved terminal field, randomizes only its Fourier phase, assimilates 256
observed local tokens, then scores the immediate 128-token continuation. The
GDN-2 reference uses its saved mature recurrence state and the same local text
window; each report names the model-specific state regime.

```powershell
D:\conda_envs\vox\python.exe scripts\ib\evaluate_continuous_owt.py `
  --checkpoint results\<run>\best.pt `
  --output results\published\<run>_warm_local_sites.json
```

## Predictive-port joint training

```powershell
D:\conda_envs\vox\python.exe scripts\ib\train_q8_port_agents.py `
  --output results\q8_predictive_ports_k16_quadratic_3000 `
  --micro-steps 16 --tau-0 4.0 --dissipation-type quadratic --compile-operators
```

K is the quadrature-resolution axis (`--micro-steps`) and the event
duration is `micro-steps * tau-0` (`--tau-0`); the registered line keeps
the duration at 64 tau0 while integrating it with 16 steps of 4 tau0,
based on the two-axis probe on the parent checkpoint (+0.010 nats at
dt=4 tau0 versus +0.473 nats for shortening the evolution time itself).

Add `--resume results\<run>\last.pt` to continue a stopped run from its
checkpoint (field, precision, optimizer, stream offset and best score).
`--compile-operators` fuses the per-microstep transport/collision/bath
operators with torch.compile before the CUDA graph capture; with the
event-norm port law this measured 4.4 s per 128-token update versus 11.3 s
eager on a GTX 1650, at roughly half the reserved GPU memory.
The default bath is `quadratic`: local energy-dependent cold outflow,
without a QR basis or spectral diffusion. The three-agent write/state/read
architecture remains in place. By user decision on 2026-10-01, the
three-layer `unified` bath is retired from the main line; its explicit option
is retained to load historical checkpoints. `selective` is an experimental
option and is not the default. Historical run configurations retain their
original bath selection for reproducibility.

The write port angle is driven by the precision-weighted event-total
innovation norm `\|\delta\|_\Pi` (site-averaged, hence grid-resolution
independent) and the incident mode is its unit direction; a null innovation
is the exact identity rotation and an unpredictable token exchanges a finite
fraction of the incident mode at initial admittance.  The first 3000-update
attempt (`results\q8_predictive_ports_k64_3000`) used a pointwise angle law
whose grid-averaged magnitude locked the port near total reflection and was
abandoned at step 440.

The entry point captures an 8-token BPTT chunk in CUDA Graph and carries both
the field and its channelwise posterior precision across all chunks.  It saves
only `BBest.pt` and `last.pt`; both include the posterior precision required
by warm-local evaluation.


As of 2026-10-02, physical duration is declared independently of K:
`--event-duration 3 --micro-steps 16` gives dt=0.1875. The reference duration
3 follows the historical K3 champion scale; increasing K refines the same
event. `--tau-0` remains an explicit legacy per-step option. The active
quadratic joint run is `results/q8_w4_quadratic_t3_k16_3000`, with 128 tokens
per update, BPTT32, compiled operators and CUDA Graph. Monitor raw and
measured post-clipping gradients separately. See
`docs/information_boltzmann/PHYSICAL_EVENT_TIME.md` for the diagnostic evidence.


New belief-reader runs default to `--readout-aperture learned_probes`:
continuous trainable torus coordinates, variance-calibrated QK/position
scales and equal initial contributions from all queries. Historical resume
inherits `atlas` when the checkpoint has no aperture key; it never silently
upgrades an existing run. The current T3 quadratic run remains the atlas
control. The matched new run is registered in
`configs/information_boltzmann/q8_w4_quadratic_t3_learned_read.json`.
See `docs/information_boltzmann/LEARNED_READ_PROBES.md` for the equation,
initialization derivation and complete launch command.

The independent plastic-medium candidate now has a timestamped continuous runtime
in `information_boltzmann/runtime/`. Inputs and read requests use a nonblocking
background owner; the differentiable training engine shares the same causal
timestamp semantics. `benchmark_continuous_execution.py` compares actual dynamics
with cached coefficients and optional deployment CUDA Graph, and monitors total
dedicated GPU memory. Shared counters are recorded and growth is allowed under
the user's revised 4 GiB dedicated-memory limit. See `docs/information_boltzmann/CONTINUOUS_RUNTIME.md` for
the interfaces, CPU result and current GPU guard outcome. This candidate has not
replaced the paused Q8 language checkpoint or started a new training run.

The plastic candidate also exposes `bath_type="conductance"`: a persistent
two-state receptor and conductance-based local response with explicit electrical
source/Joule accounting. `audit_conductance_response.py` verifies the equation
ledger and an actual local E/I Jacobian without training. Use
`benchmark_continuous_execution.py --bath-type conductance` for matched execution.
Equation provenance, learned time scales and approximations are documented in
`docs/information_boltzmann/CONDUCTANCE_RESPONSE.md`.

`benchmark_conductance_training.py --backend cuda_graph` captures full chunk
forward and backward for that candidate. It checks accumulated gradients and
every continuation tensor against eager before timing real OWT updates. Matched
128-token updates improved13.175 to3.067 seconds with the competing fly job still
running; sampled total dedicated memory stayed below2.869 GiB. See
`docs/information_boltzmann/CONDUCTANCE_TRAINING_EXECUTION.md` for scope and commands.

The local-material audit in
`docs/information_boltzmann/LEARNABLE_MEDIUM_AUDIT.md` distinguishes trainable
physics from fixed capacity budgets. The conductance candidate now has a direct
initial electrical-coefficient gradient to material and a material-conditioned
edge content metric; the quadratic control retains its historical metric.

`train_plastic_conductance.py` is the real-OWT joint-training entry point for
this candidate. It uses FP32 forward/backward CUDA Graph chunks, preserves the
complete belief across updates, saves only `best.pt` and `last.pt`, and publishes
`metrics.jsonl`, `config.json` and `progress.json`. Validation clones mature
training belief per local site, warms256 context tokens and scores128 tokens;
it does not initialize a zero validation state. A `STOP` file in the output
directory requests checkpointed stopping at the next completed update.

```powershell
python scripts/ib/train_plastic_conductance.py --output results/plastic_conductance_d128_3000 --steps 3000 --tokens 128 --chunk-tokens 8 --event-duration 0.005 --substeps 1 --allocator-fraction 0.40 --validate-every 500
```

Duration0.005 is the explicit first-launch cadence inherited from the execution
check, not an optimized biological clock or a claim of solver convergence.
The run uses a100-update warmup, stable learning rate1e-4, and300-update decay.
Task claims require the registered independent validation and learning curve;
the first updates establish execution and memory use only.

`audit_trained_plastic_medium.py` inspects a saved trained medium on real warm
OWT contexts without optimization. It records material heterogeneity, per-head
spatial sampling, stage changes, energy ledgers, direct CE gradients and current
inference reliance. The update-703 findings and the boundary/internal-timescale
mismatch are documented in
`docs/information_boltzmann/TRAINED_MEDIUM_MECHANISMS.md`.

The plastic medium now defaults to `--write-exchange contact_mode`: the learned
incident packet exchanges only its contacted field mode with the environment.
The shared historical W4 path remains `global`. Training and continuation
record this choice; old checkpoints require global exchange for exact resume.
See `docs/information_boltzmann/CONTACT_MODE_BOUNDARY.md` for the energy proof,
untouched-mode gradient preservation and the explicit old-weight diagnostic.

New plastic-medium runs also default to `--port-scope compact`: persistent,
trainable centers with finite write/read footprints and locally conditioned
read policies. Write/read overlap is allowed and measured; no anti-overlap
penalty or forced anatomical separation is added. `metrics.jsonl` records
geometric overlaps, active policy use and spatial energy at monitoring intervals.
Both physical radii are saved in the constructor; set `--write-port-radius RX RY RZ`
and `--read-port-radius RX RY RZ` to declare a different communication budget.
Historical exact resume requires `--port-scope global` along with its saved
exchange law. Choose a fresh output directory for the compact architecture.
See `docs/information_boltzmann/COMPACT_SPATIAL_PORTS.md`.

Local activity adaptation reuses persistent inhibitory receptors to adjust
excitatory response, compact write-port competition and reciprocal conduction.
The conductance trainer defaults to `--activity-adaptation`; exact historical
resume requires `--no-activity-adaptation`. There is no added fatigue state,
target activity rate, or extra evolution loop. Its equations, contracts and
references are in `docs/information_boltzmann/LOCAL_ACTIVITY_ADAPTATION.md`.

Short-term pathway plasticity now adds local Tsodyks--Markram resource and
utilization states. The conductance trainer defaults to
`--short-term-plasticity`; historical runs use `--no-short-term-plasticity`.
The two states modulate reciprocal conservative coupling and remain part of
the full continuation checkpoint and every captured chunk. Learned local
material sets recovery/facilitation rates without a degree-to-time mapping.
The 8x8x4 runtime adds 6 KiB of FP32 state and 108 parameters (material width8).
See `docs/information_boltzmann/LOCAL_SHORT_TERM_PLASTICITY.md`.

Plastic-medium training and execution benchmarks support whole-medium and
write/read AOT fusion through `--medium-execution fused --port-execution fused`.
These execution choices preserve physical equations, trainable weights and
all persistent state. CPU/FP64/diagnostic calls retain native references.
Captured stage profiling and production update timing are separate tools;
see `docs/information_boltzmann/PLASTIC_EXECUTION_FUSION.md`.

For deterministic local online credit, use `train_online_plastic.py --credit
local-receptors`. Physical state and receptor eligibility persist across events
and optimizer updates; current-event gradients jointly train the whole model.
Historical credit currently covers receptor closing kinetics and their spatial
material map. CUDA Graph and full continuation are supported. See
`docs/information_boltzmann/LOCAL_MEDIUM_CREDIT.md` for the exact scope,
derivative audit, memory and real-OWT execution calibration. The `--credit uoro`
path remains a numerical reference with global stochastic compression.

Both online credit modes now evaluate the same live learner through first-pass
prequential prediction, fresh context-change adaptation and actual A/B/A return.
Learning, eligibility, physical state and optimizer cadence continue across all
phases. B's cursor and pending gradients are saved for exact resume. See
`docs/information_boltzmann/LIFELONG_EVALUATION.md`; the older frozen-validation
reports remain historical comparisons with their original meaning.
# Fourth evaluation pillar

The online 3D entry `train_online_plastic.py` also emits `health` measurements
on its actual learning trajectory: energy balance, field/flux DC and spatial
structure, feature covariance rank and temporal roughness, risk trends and real
optimizer displacement. Live A-to-B-to-A reports include phase snapshots and
conditional complete-physical-state response. Monitoring has bounded persistent
checkpoint history and adds no training objective. See
`docs/information_boltzmann/PERSISTENT_MEDIUM_HEALTH.md`.

New plastic-medium runs use final pre-decoder RMSNorm and explicit AdamW
prediction/physical parameter groups (`--weight-decay` defaults to the prior
0.01 prediction-matrix rate). All read, chunk and graph paths share `decode`.
Exact resumes preserve historical architecture and optimizer policy; the online
entry's `--initialize-from` explicitly adopts the new head with fresh optimizer
and eligibility while keeping checkpoint weights, physical state and data cursor.
See `docs/information_boltzmann/MEDIUM_DECODER_SCALE.md`.


### Continuous fly three-factor learning

`train_fly_infinite_stream.py --total-tokens 100000` uses one ongoing learner
for training and active A/B/A evaluation. `last.pt` is a complete lifecycle
resume; `best.pt` is a weights-only branch asset. Implementation and credit
approximation scope: [FLY_THREE_FACTOR_ONLINE](../../docs/information_boltzmann/FLY_THREE_FACTOR_ONLINE.md).
