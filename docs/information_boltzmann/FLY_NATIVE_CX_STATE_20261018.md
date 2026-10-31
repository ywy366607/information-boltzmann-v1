# Native CX persistent state (2026-10-18)

The first system-level mechanism from the unified registry is the central
complex. The MaleCNS graph already contains 2,950 annotated `CX` neurons and
their native recurrent and downstream connections. This change does not add a
heading label, navigation target, action reward or decoder shortcut.

With `--cx-state`, CX spike output writes a persistent per-CX activity trace
with a learnable positive decay time. The trace changes the existing CX cell
threshold through a signed learnable feedback parameter; all downstream effects
therefore travel through the fixed connectome into MBON/association/descending
and VNC pathways. Both parameters start with zero feedback for the frozen
equivalence control. The trace is part of `FlyPhysicalState`, so it crosses
BPTT windows and full checkpoint/resume.

This is an implementation closure, not yet a navigation or action capability
claim. The next evidence is a real-language continuation from a preserved
checkpoint and a same-state, same-clock `--no-cx-state`/sham comparison. A
successful causal path must change native downstream state and action
consequences, not merely CX activity.
