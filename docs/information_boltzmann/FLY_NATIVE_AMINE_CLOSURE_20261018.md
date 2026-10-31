# Native amine closure (2026-10-18)

## Current integration and kinetics repair

The original four interface tests did not check the physical time scale. The
parameters named `amine_log_clearance_*` and `amine_log_off_*` store log time
constants; the original transition incorrectly used `exp(-tau)` instead of
`exp(-1/tau)`. Kinetics version 2 fixes this, uses `expm1` for accurate slow
integration, and advances both transmitter contact graphs in one scatter
without per-tick channel-mask synchronization. Time constants and affinity
remain learnable; they are initial conditions, not fixed biological limits.

Analytic tests now check an 80-tick concentration e-fold, a 12-tick receptor
off-time, the constant-exposure mass-action solution, persistence, gradients,
zero-response neural equivalence and named optimizer registration. Previously
enabled version-1 checkpoints require explicit repair; the loader refuses to
silently change their learned dynamics. The 32904 source had this module off
and contained no amine parameters, so its activation does not reinterpret an
existing learned chemical state.

The complete 32904 state was migrated in a separate directory with no new
exposure. Every previous model parameter, named Adam moment, other learner
field, world queue, pending credit, meter and RNG was independently checked
equal from the written files. In the zero-response 10-arm/54-real-tick matrix,
chemical delivery reached OA and 5-HT target populations while sensory and
action log probabilities were exactly unchanged by amine removal. Sham,
restored-state and source/RNG controls passed. This is a safe initialization
control, not evidence that an untrained response already changes behaviour.

Contact amplitudes are normalized relative delivery. Receptor gene/subtype
identities, extrasynaptic volumes and their effector biochemistry are not
recovered from EM contacts. The learned class response is an explicit model
assumption. Slow physical state persists across windows, but exact carried
forward sensitivities for these new amine kinetics are not yet implemented;
their current learning derivatives cover the actual BPTT window. These gaps
remain on the system list rather than being called a complete biological
reconstruction.

See `results/published/fly_native_amine_integration_20261018.json` for the
subsequent bounded real-input continuation, current checkpoint and controls.

This change closes one anatomical gap in the whole-connectome fly model. The
connectome audit had pinned octopamine and serotonin fast edges to zero, but
had no slow chemical route for those cells. The implementation restores that
missing physical role without reopening the fast edges or adding a reward
current.

The route is:

`native source spikes -> immutable anatomical contacts -> normalized local
exposure -> concentration clearance -> mass-action receptor occupancy ->
learnable signed cell-class response -> existing membrane threshold -> native
network, plasticity and readout`.

Contact weights are reconstructed from the original graph before the fast-edge
pinning and normalized per target and transmitter. They represent relative
local exposure, not measured molecule counts. Concentration and occupancy are
persistent fields of `FlyPhysicalState`, so they cross BPTT windows and are
included in generic state save/restore. Clearance, affinity, receptor off-rate,
and response are trainable. The response is signed and starts at zero; no
assumption that octopamine or serotonin is globally excitatory or inhibitory is
encoded.

The switch is opt-in as `--amine-modulation`; legacy checkpoints and runs keep
the exact disabled route. A complete migration must attach the module after
the final cell-class split, register its named optimizer parameters explicitly,
and preserve the fast-edge pin masks. A zero response gives the frozen
network-equivalence control. The first tests cover bounded occupancy, contact
delivery, gradients, disabled identity, and state persistence. A real language
run and a matched off/sham intervention are still required before claiming a
learning or behavioural gain.
