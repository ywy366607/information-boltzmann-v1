# Causal local temporal probe acceptance — results

The optional complex-exponential history path passes numerical and real-data interface acceptance. Production defaults, physical equations, write/read positions and the existing training/evaluation entry points were not changed. No GPU, optimizer step or checkpoint creation was used.

Pre-registration: `MEDIUM_TEMPORAL_PROBES_PREREG_20261008.md`.
Implementation: `information_boltzmann/core/temporal_probes.py`.
Audit: `scripts/ib/audit_medium_temporal_probes.py`.
Measurements: `results/published/medium_temporal_probes_20261008.json`.

## What passed

- Held-input exact ZOH agrees with the augmented matrix-exponential reference. Constant-input interval refinement preserves the state. This is a solver identity, not a claim that splitting a changing physical signal leaves its observed history identical.
- The ZOH-held oscillatory reference gives absolute complex-response error 2.38956e-15 and measured/reference phase -0.223048918245168 / -0.223048918245170 rad. These frequencies/time increments are numerical test conditions, not biologically assigned frequencies.
- Two histories with the same current signal have distinct signed bank states. Their difference contracts by exp(-alpha*t) under identical future drive and fixed finite positive alpha. Ongoing driven activity need not decay to zero.
- Chunk replay error is zero. Serialized continuation preserves values and elapsed time. In-memory split tapes preserve gradients; disk serialization explicitly does not retain a Python autograd tape.
- Double-precision gradcheck covers input, initial complex state, duration, decay rate and frequency. Instantaneous sampled-probe gradients are exactly zero outside its finite footprint. Historical dependency is on earlier aperture supports and information delivered there.
- Float32 history uses a double-precision elapsed clock; a 1000-interval clock contract matches double-precision physical timestamp accumulation exactly. Invalid durations, underflowed rates, nonfinite inputs/history/time and nonrepresentable rate-duration products fail immediately.
- Final affected CPU suite: 49 passed (temporal probes, dynamic read, compact ports and port-event timing).

## Real-text interface

Existing full 8x8x4, D128, compact ports, conductance response, activity adaptation and STP; local dynamic read enabled. Fresh seed990 weights use the saved constructor settings; the old matured checkpoint is unavailable. There are 64 real OWT burn-in inputs, followed by 32 actual next-token targets. Both medium and filter states persist. Burn-in is no-grad; score32 is one connected tape.

There are exactly96 physical advances, no additional settle/ponder steps. Physical and filter elapsed times are both 0.4800000000000003. Parameters stay unchanged and optimizer updates are0. The banks ingest the already-available endpoint field/motion samples. Their recurrence is an endpoint-held discretization of observed histories, not exact integration of the varying continuous physical waveform.

To prove that old readout was not the sole gradient route, detach the original base read feature only and compute CE through the new temporal correction:

| Parameter group | Temporal-only gradient norm |
| --- | ---: |
| Log decay rates | 0.00242010 |
| Frequencies | 0.000176910 |
| Temporal mode mixing | 0.292445 |
| Temporal projection | 0.608405 |
| Probe coordinates | 0.775999 |
| Physical medium | 0.0884826 |
| Write agent | 0.315437 |
| Source writer | 0.0114278 |

The last three groups therefore receive credit through the sampled temporal history itself. This establishes a viable learning path within score32. It does not extend computational credit beyond the explicit burn-in boundary or establish infinite-horizon credit.

The bank has16probes,4modes and256 input channels (field plus motion). Its persistent inference state is131080bytes, approximately128KiB, independent of stream length. Shared mixing/projection and filter parameters add32904parameters. Training-tape storage still depends on the chosen credit window; it is not reduced to constant history-independent BPTT memory.

Final CPU observation: bank update0.994ms/event, field/RHS/probe pooling7.089ms/event, write+physical advance28.872ms/event. These are instrumented component measurements on this CPU execution, not GPU/full-training throughput. An integrated dynamic reader could share its already-computed RHS; that potential saving is not claimed as measured.

## Decision

The candidate can preserve signed temporal/phase distinctions, remain causal and finite, and receive real language supervision through the medium. It is ready for an optional production integration and adequately trained comparison. Predictive advantage over the current dynamic reader remains open; all NLL fields in the JSON are fresh-random-weight derivative provenance, not capability results.

Keep rates and probe locations fixed during this acceptance. A future moving-probe learner needs a transport/coordinate-history contract, while online parameter updates must preserve the interpretation of state accumulated under older parameters. Four modes, dyadic half-lives and Nyquist-relative frequencies are declared coverage choices, not claims of optimality. Any later NLL attribution needs a matched-size nonoscillatory historical-read control.

Independent /root/rtc_contract_review accepted the plan and the exact recurrence, phase reference, causal ordering, isolated gradient path and numerical scope. Subsequent hardening adds finite-input guards and exact clock alignment; source fingerprints and the final acceptance are recorded in the evidence/tree.
