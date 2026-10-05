# Persistent local interfaces to a learnable medium

The contact-mode boundary previously selected a spatial/content mode while
retaining a global Fourier packet and globally conditioned read policy. The
new `port_scope=compact` limits both physical exchange and observation to
persistent, trainable port locations. Content controls actions at these ports;
port coordinates are parameters, not token-generated teleporting addresses.

## Geometric communication budget

For port p and periodic displacement delta_j in [-1/2,1/2), use the C1 footprint

    K_p(x) = product_j max(0, 1 - (delta_j / r_j)^2)^2.

It is exactly zero outside its compact support and has zero first derivative
at the boundary. Centers learn through the likelihood. No forced anatomical
roles, disjoint masks or overlap penalty are introduced. Port count is the
existing write chart count / read head-query count; radius is an explicit
physical aperture budget. Defaults use half a balanced port cell minus a
half-sampling-cell guard gap, with a one-sampling-cell minimum radius and a
half-period maximum. Mid-sample initialization keeps all coordinate axes
differentiable. On 8x8x4 with 8 write ports the radius is
(0.1875,0.1875,0.25); 16 read ports use (0.125,0.1875,0.25).
Both initial banks leave genuinely uncontacted grid sites. These are declared geometry choices, not biological
constants or a proof of optimal capacity. The CLI accepts three-axis radii.

The radii are saved explicitly in the constructor. Pass these same physical
radii when refining the grid; recomputing sampling defaults would change the
physical communication budget. Existing learned coordinates load across grids.

## Write, process and read

Write prior and action controllers observe only the write bank. Each chart
mode has its own persistent center, compact envelope and local state-conditioned
channel modulation. Its chart remains exactly linear in token feature, so the
predicted packet is the categorical expectation of the observed packet.
Local normalized basis amplitudes retain the previous event energy convention;
there is no additional per-grid-cell amplitude division. Contact-mode energy
exchange is unchanged. Uncontacted cells remain identical through assimilation.
The boundary controller combines its finite bank observations; this is an
explicit shared environmental interface, not an independent neural graph hop.

Each read head/query constructs prior, posterior, Q and measurement from its
own footprint. Semantic QK attention and aperture mixtures are masked within
that footprint. Writer precision is excluded from the compact read controller:
it can contain distant write-bank evidence. Local second moments supply read
evidence instead. The decoder combines the resulting finite local measurements.
Readout measures the local field directly, preserving visibility of the current
token when learned write/read footprints overlap. Flux, receptors and elapsed
time continue as before; no new integration loop is added.

## Overlap and differentiation

Overlapping and separated ports are both allowed. Overlap permits immediate
local response; separation gives spatial transfer a role. Even coincident ports
can use local nonlinear collision: overlap alone cannot establish that collision
is idle. If distant state must influence a port, its influence must reach the
observed region through the medium or another explicitly accessible port.

Training records every memory interval:

- Write/read normalized-footprint cosine overlap and nearest read distance.
- Within-write and within-read pair overlap (coincident banks tend to one).
- Write, read and shared union support coverage.
- Port centers and field spatial energy fraction.
- Current write policy's effective port count, actual read attention overlap,
  per-head entropy and overlap between the current write policy and read.

Support coverage and head entropy should be interpreted alongside NLL, local
field diversity and actual operator effects. Complete union coverage does not
make each controller global; controller locality is enforced per query. Low
entropy alone does not prove useful specialization. No anti-collapse loss is
added until the measured failure and its task consequence are identified.

## Compatibility and validation

Historical configurations default explicitly to global access when loaded by
the audit. New models/training default to compact contact-mode interfaces.
Global exchange plus compact scope is rejected: global old-field refresh would
violate the geographical access contract. Continuation schema3 records scope,
exchange and both physical radii and rejects silent changes. New parameters and
read policy dimensions mean old checkpoints require an explicit migration and
joint learning, rather than pretending strict-loading is exact continuation.

`tests/test_compact_ports.py` checks finite support, periodicity, moving-center
gradients, exact zero influence outside an individual read port (including its
controller), untouched write cells, energy closure, overlap/separation and
observable co-location, joint CE gradients and continuation rejection. These
are numerical/interface validations, not language capability experiments.

Production execution is checked separately with real OWT128-token updates,
BPTT8 and FP32 CUDA Graph. No long training is launched by this implementation.

The completed production check measured 3.215s/update after one warmup update,
under concurrent independent fly training. Peak allocated tensors were409MiB;
the complete graph matched eager accumulated gradients and state exactly in
this check. This is one execution sample, not a trained language result. See
`results/published/compact_ports_execution.json`.
