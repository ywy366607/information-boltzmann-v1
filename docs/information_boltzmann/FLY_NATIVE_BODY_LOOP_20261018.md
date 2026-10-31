# Native ascending/descending body loop (2026-10-18)

This continuation adds the first body-interface state to the connectome path. Annotated descending neurons provide a graded action trace; annotated ascending neurons provide a graded body-feedback trace. The action trace is injected back only at the existing ascending cells, after which the original static connectome carries the signal onward.

The loop has two learnable quantities: a logit decay (initialized to a 32-tick time constant) and one feedback gain per ascending cell (initialized at zero, so attachment is behavior-preserving at birth). The persistent traces are part of `FlyPhysicalState`, therefore they survive BPTT windows and full checkpoints. They are included in native policy credit and the named Adam migration path.

This is a reachability implementation, not a claim that proprioception, motor control, or self-maintenance has emerged. Acceptance requires matched-clock paired evidence that a changed output action changes ascending feedback and that the resulting sensory consequence changes later prediction or native plasticity. A short continuation only verifies state persistence, finite gradients, and checkpoint provenance.
