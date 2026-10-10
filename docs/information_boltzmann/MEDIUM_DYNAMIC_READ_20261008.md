# Local dynamic read: implementation and acceptance contract

User goal: improve persistent-medium language/thinking through a read that can
observe ongoing motion. The read-closure audit established instantaneous hidden
state aliasing and useful first-order physical response. This branch implements
that information path; it does not establish improved language likelihood.

## One change

Read the existing local field and its analytic autonomous derivative together:
`R(f, tau_ref * df/dtau)`. The derivative adds transport, collision and electrical
response (or quadratic bath) generators at the same current state. It is the
zero-duration derivative of the native splitting integrator. Existing learned
material, conduction offsets, STP and receptor state remain differentiable.
No finite difference, extra physical tick, history queue or new clock is used.
`tau_ref` is the existing `response_time_reference`, not the training cadence.

The original field branch stays intact. Signed motion, normalized per site by
the joint field/motion RMS, enters three additive bias-free paths: local policy,
keys and weighted mean/variance measurement. All heads retain their compact
finite apertures. Each site's derivative reads its local state plus incident
edge stores, so the read's causal receptive support includes the boundary edges.
It does not acquire a global precision/control shortcut.

## Alternatives and scope

The selected candidate addresses omitted current dynamic information. Temporal
history/synchrony would add memory to a partial observer; autonomous timing
would change input/output scheduling; loss unification would change supervision.
Those remain separate candidates. This patch leaves evolution and loss intact.
Additional motion projections add parameters, so later NLL attribution needs a
matched-size field-only control as well as the original baseline.

## Compatibility

`read_mode='instantaneous'` remains the default. `read_mode='dynamic'` is explicit
and requires compact ports. Fresh models use ordinary fan-in initialization.
`initialize_dynamic_read_branch` copies an exact compatible instantaneous
weight dictionary and zeroes only the three new additive projections, preserving
the old output. This is weights-only architecture migration, not exact optimizer
resume. Copy the physical belief explicitly and create an optimizer containing
the added parameters. Continuous runtime payload schema 6 guards read mode and
physical time reference. Existing schema 1–5 payloads remain instantaneous.
Trainer option `--dynamic-read` selects a fresh joint branch. Resume retains its
existing architecture/source/optimizer checks.

## Predicted numerical/interface outcomes (registered before execution)

1. Analytic field derivative and its parameter VJPs match zero-time native JVP,
   under heterogeneous material, conduction, STP and receptor state.
2. Equal fields with different incident flux can produce different dynamic reads.
   Holding aperture-local state and incident edges fixed makes outside changes
   invisible to a selected probe. This is an exact structural locality check.
3. Explicit zero-projection migration preserves old logits while each new
   projection receives a nonzero task gradient on a nondegenerate real state.
4. Read calls preserve all state, elapsed time and continuation. Native chunk,
   timestamped execution and inference use the same read semantics. Targets enter
   only the likelihood.
5. Measure added runtime and memory with no optimizer step. A training speed or
   NLL claim requires a separate approved real-data joint run.

Independent design review: `/root/rtc_contract_review`, accepted with required
checks on transport scale/sign, same-state collision generators, preserved legacy
path, gradients to zero-initialized additions, locality and checkpoint metadata.
On a failed identity/gradient/continuation test, repair the implementation before
considering a training run. This request authorizes implementation and read-only
calibration; no training is launched.

## Acceptance results

All 9 new numerical/interface tests pass, including double-precision RHS and
parameter-VJP identities, strict selected-probe state locality, hidden-flux
observability, exact migration with nonzero gradients for each new projection,
chunk/target causality, continuation metadata, full-graph tracing parity and cache
invalidation. Final affected-suite run gives 68 passed and 5 optional CUDA
skips; the Deslice/gate script also passes.

On the completed D128 individual's full real physical state, explicit migration
has bit-identical logits. Correct pending-carry CE gives new projection gradient
norms 9.033 / 2.208 / 25.168 (policy / keys / merge). All parameters remain
unchanged; optimizer updates = 0. Parameters: 14,067,511 -> 14,247,735 (+180,224).

Initial interleaved native CUDA-graph measurements, with another job using
2605MiB, give read+decoder forward 1.072 -> 2.910ms, including backward
4.501 -> 8.913ms. Native/graph outputs and gradients match. The calibration
process peaked at 374.9MiB allocated and 608MiB reserved; total device usage
after measurement was 3065MiB. These timings include coefficient preparation
but exclude write, physical advance, full BPTT32 capture and optimizer updates.
They do not establish complete training speed. The initial timing's arbitrary
CE label preceded the pending-carry correction; current CPU acceptance uses the
correct issued-prediction label. Both evidence artifacts retain that provenance.

Further GPU calibration was stopped by the pre-allocation headroom guard when
the other job increased its memory use. Corrected CPU read+decoder measurements
were 5.08 -> 17.64ms; source fingerprints cover the complete changed read path.
Production source review by `/root/rtc_contract_review` accepts the numerical
and interface conclusions. A language improvement remains the next separate
joint-training decision, with a matched-size field-only control needed for
mechanism attribution.

Full repository CPU run: 1019 passed, 85 skipped, 12 failed. The failures are
outside the medium read implementation: one removed historical checkpoint,
one Fly CTM test forcing CUDA in CPU mode, and Fly learning/diagnostic API and
checkpoint-hook failures (including `None` training loss and tuple/state
mismatch). They remain recorded rather than presented as an all-green suite.
This change's numerical/port/runtime tests and source review pass; Fly code is
owned by the concurrent route and was not modified here.
