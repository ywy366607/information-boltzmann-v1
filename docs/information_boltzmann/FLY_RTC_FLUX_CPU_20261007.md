# RTC response error: physical state closure

## Target and scope

The user requested lower error and near-complete coupling on the authorized small CPU numerical response assay. This is a reproducibility/interface experiment on the existing real-connectome induced graph: 256 neurons, 5,901 edges, 66 motor neurons, horizon 14. The graph drops 128,055 edges crossing its boundary, so the results describe this induced boundary-value system. They establish neither whole-brain response nor language capability.

Eight continuous 64-tick excitation lanes supply eight origin states per lane. Lanes 0–3 train the student and lanes 4–7 are held out. Future oracle branches have declared zero external drive. The original graph, oracle parameters, seed and response labels are unchanged from the surface assay. No production weights or GPU training were changed.

## Competing explanations and mathematical closure

1. **Read observation loss:** preserving complete motor membrane voltage repairs instantaneous native readout but may leave its future dynamics underdetermined.
2. **Missing physical state:** motor conductances, adaptation, synaptic resource/release states and in-flight pulses determine the next voltage. Voltage alone is insufficient.
3. **Unknown upstream arrivals:** even complete motor-local state is an open subsystem. Its future requires the incoming flux from outside the motor domain.
4. **Optimization/generalization:** a lower training loss may fail to improve independent response trajectories.

For the registered COBA/ALIF/STP equations, split motor arrivals into

\[
a_{E/I,M}(t)=W_{MM,E/I}p_M(t-d)+a_{E/I,\mathrm{outside}}(t).
\]

The local state is \(h,g_E,g_I,b,x,u,\bar h\), together with all four transmitted-pulse slots. Original motor-to-motor anatomical edges are integrated exactly. At the origin, outside pulses already in flight determine a known future arrival prefix. The first future tick is completely determined by this existing queue. Later ticks also include newly emitted outside pulses, which must be modeled.

A constructive numerical counterexample holds both the current random regional codec and all motor voltages fixed, then changes an unobserved motor conductance by 0.005. The next motor voltage changes by 0.00446665. These are legal numerical initial states; reachability of the pair along one original stream has not been established. Thus membrane preservation alone cannot certify autonomous exact prediction on the ambient state space.

## Registered candidate

`MotorFluxForecaster` preserves the exact motor-local physical state and motor-to-motor integration. A small network predicts only unresolved outside arrivals using regional delay history and the student's own current motor state. Existing queues use origin data only. Teacher future states appear only in losses and comparisons. The candidate has 7,906 trainable parameters.

The 120-update fit uses training-normalized regional latent, raw motor and external flux squared errors. This candidate adds state, known physics and boundary supervision; comparison with the old 18,310-parameter surface forecaster is a structural comparison, not an isolated parameter-matched learning ablation.

## Results

| Check | Outcome |
| --- | --- |
| Local split supplied actual boundary flux, 14 ticks | Max voltage error 3.17e-8; every local state field within numerical tolerance |
| Autonomous hybrid before training | Heldout response MSE / hold-current MSE = 0.189243 |
| Autonomous hybrid after 120 updates | Ratio = 0.191515; raw motor ratio = 0.192635 |
| Earlier full membrane surface model | Ratio = 1.30937 on the same heldout lanes |
| First future tick of the hybrid | Numerically exact |
| Zero-tick read | Exact |
| Normalized training loss | 1.92917 → 0.68950 |

The hybrid after training reduces response MSE by about 85.4% relative to the preceding surface model and by 80.8% relative to holding the current response. Most of this benefit exists before learning. The learned 120-update component **does not improve heldout response error**; the slight deterioration must remain visible. Training flux MSE falls while heldout motor accuracy fails to improve, so fitting flux alone does not certify improved response.

## Autonomous exact numerical ceiling

A separate deterministic reference expands the exact physical subdomain to all 256 neurons. Every graph edge becomes internal; no unknown boundary flux remains. It starts from the complete origin state and applies the independently implemented local update 14 times with zero external drive. Future teacher states never enter prediction.

All 14 motor maximum errors and the maximum errors of all retained state fields are **exactly zero** in this assay. This proves complete coupling is achievable for the declared initial-value problem. It retains full simulation state and arithmetic, so it provides a numerical ceiling, not a fast learned surrogate or an acceleration result.

## Review, tests and decision

`rtc_contract_review` approved the bounded candidate and checked origin-queue indexing, exact local equations, autonomous feedback and teacher-label separation. It recommended ending this learned arm at 120 updates and adding the complete-state numerical reference. The final reference audit is recorded in the research tree.

32 focused RTC/streaming regression tests pass. New checks independently compare the origin queue against the original delayed transmission function, compare 14-tick local split state updates, validate finite flux gradients, demonstrate voltage insufficiency and check the complete-state autonomous reference.

**Decision:** preserve this physical-state/arrival split as an experimental candidate. The exact reference establishes feasibility; the hybrid identifies unresolved upstream state compression as the next engineering target. A compressed fast student earns an exact-coupling claim only after its autonomous per-tick heldout errors pass the same numerical criterion. No additional bath/readout patches or production long run were introduced.

## Artifacts

- `information_boltzmann/core/fly_rtc_motor_flux.py`
- `scripts/ib/check_fly_rtc_flux_cpu.py`
- `scripts/ib/check_fly_rtc_closed_reference_cpu.py`
- `tests/test_fly_rtc_motor_flux.py`
- `results/published/fly_rtc_flux_cpu_20261007.json`
- `results/published/fly_rtc_closed_reference_cpu_20261007.json`

Each assay has a corresponding preregistration JSON. No checkpoints were written.
