# Continuing fly learning: archive online-credit exploration, adopt BPTT32

User decision (2026-10-04): archive the O(1)-in-duration local-credit learning
research and try 32-event BPTT. The old code, configuration, completed run,
first-revisit curve and final checkpoint stay in place as reproducible assets.
They are not deleted or reinterpreted as BPTT evidence.

The archived individual completed 100,000 training targets and 113,592 total
observed events. Its final progress reports approximately 21.35 training
tokens/s, peak allocated 3450.9 MiB, and last EMA training loss 8.5434. The
reported best live NLL 7.1352 was inherited from the pre-repair lineage; it is
not a capability achievement of the new local-credit recurrence. Numerical
agreement of local sensitivities did not establish correct global credit.

## Unchanged physical model and explicit learning migration

- MaleCNS topology, sensory input, motor/descending output, COBA conductance,
  ATan spike surrogate, ALIF and STP are retained.
- Continue final learned weights, physical voltage, conductances, adaptation,
  transmitted-pulse delay queue, writer baseline, token identity and stream
  cursors. Restore existing AdamW moments by the original parameter ordering.
- Online eligibility, random feedback, compensated local updates and their
  diagnostic/revisit queues are archived. The new temporal gradient starts
  at the migration boundary. BPTT has one joint optimizer update per 32 real
  observed events, compared with the old 4-event/2-event update cadence.
- Input embeddings and STP parameter values remain frozen as in the prior
  learner; STP physical state and its causal dependency are differentiated.
  Writer projections, gates, decoder, readout, eight physical parameter groups
  and all excitatory/inhibitory connection weights receive BPTT gradients.
- Edge and writer SGD retain the old rates/bounds; existing prediction and
  physical groups retain AdamW and their separate decay settings.

## Gradient and execution repairs

The delayed synaptic backward now returns both pulse and edge-weight VJPs.
CUDA gathers/scatters and edge gradients are fused; only node-sized delayed
pulses are saved, not an edge-sized activation per time step. The functional
writer carries its adaptation dependency through the window. Physical values
are detached, never zeroed, at the window boundary.

CUDA Graph captures forward/backward only. Warm-up makes no parameter updates
and does not advance the continuing state. Actual replay copies all physical
inputs and scores 32 observations before their joint update. Optimizers operate
outside capture. Dedicated allocation/reservation must stay below 3900 MiB.

## Real-stream trial and evaluation

Entry: `scripts/ib/train_fly_bptt_stream.py`. Resume the archived final individual
and continue 100,000 additional fresh OWT targets: at least 3125 joint training
updates plus active evaluation updates. A one-window calibration is a real
learning window and saves full continuation, so it can resume without replaying
or discarding that experience. Speed/memory checks are implementation checks,
not capability verdicts.

Validation is the same active learner, with fresh B experience and an A revisit.
All actual context bridges are scored; the first A target is excluded from the
matched revisit delta. No state reset, cold start or frozen evaluation is used.
Field energy, firing activity and centered representation rank are descriptive
health observations; no criticality/entropy-expulsion certification is claimed.
Every run records data cursors, optimizer updates, per-target pre-update scores,
allocation peaks and learning curves. Different update cadence/data exposure
must be reported when comparing with the archived run.

The new run writes only `last.pt` and best weights to its own directory. The
full checkpoint includes physical state, Adam/SGD state, token/cursor, recent
revisit episode, RNG state and counters. No intermediate checkpoint ladder.

## Execution verification

Sixteen affected numerical/interface tests passed, including delayed pulse and
edge-weight gradients, functional writer adaptation, complete state detach,
and CUDA Graph/eager parameter and physical-state agreement over two windows.
Capture warm-up was verified to leave parameters and counters unchanged.

First actual 32-event update: 0.703 s, 45.5 token/s excluding checkpoint I/O.
After resuming the saved calibration, the continuing run reached 1024 fresh
training targets (32 joint updates): about 0.669 s/window, 44.8 training
token/s including startup checkpoint overhead. Peak allocation including
warm-up was 1914 MiB, reservation 2144 MiB; nvidia-smi showed approximately
2293 MiB dedicated use. No active-validation score exists at that early point;
these measurements establish execution feasibility, not predictive superiority.
