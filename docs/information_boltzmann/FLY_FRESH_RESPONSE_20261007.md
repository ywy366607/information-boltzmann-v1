# Fresh connectome material-response identification

The user requested response identification before architectural selection, using
new weights rather than a known-deficient trained checkpoint. The protocol was
independently reviewed and registered before measurement. This is a numerical
mechanism study, not a short-budget language capability study.

## What was initialized and what was held fixed

Three seeds (0, 1, 2) initialize every neural-network parameter from scratch,
including the token embedding and sensory writer. No checkpoint or pretrained
embedding is loaded. The original COBA graph and sensory partitions are held
fixed. Input supports only 15,912 sensory neurons; the observed motor/output
surface comprises 2,333 neurons, disjoint from the input surface.

The substrate contains 165,122 neurons and the existing delayed COBA, ALIF and
STP dynamics. Gamma is disabled because the observed material is the physical
state, before an additional read filter. Random decoder scores are not measured.
Random embeddings mean that real OWT token IDs identify distinct current packets;
they do not yet encode learned word meanings.

The graph carries anatomical topology and model-assumed gains/delays. Initial
thresholds, reversal potentials and superclass time constants are model priors,
not physiological measurements recovered from EM anatomy. Their actual parameter
values, graph/partition hashes and initial parameter hashes are in the result.
This probes three sensory initializations of this specified prior, rather than
the entire admissible physiological parameter space.

## Registered protocol

- The first 128 tokens of `data/ib_owt_gpt2_31m/train.npy` are fixed in the registry.
  The first three distinct IDs are pulse probes; the first eight form the repeated
  phrase. No target or response is used to select a stimulus.
- Two complete initial states: rest, and the state obtained after the same 128
  native one-tick token events with zero parameter updates.
- Two pulse amplitudes: native 1, and an initial-state analytic calibration.
  For positive sensory coordinates, set the median first-tick voltage/threshold
  ratio to one. With zero initial conductances,
  `beta=(1-exp(-g_L))/g_L`, so
  `A=1/median(beta*packet/theta)`. The first reference token determines one A per
  seed: 1.18064, 1.11881 and 1.08697. No response-based amplitude adjustment occurs.
- Pulse at tick 0, followed by 127 explicitly zero-drive ticks. Each input branch
  and its matching no-pulse branch are replayed. Resolution uses the sum of the
  two replay error norms and a numerical absolute tolerance.
- At rest and amplitude 1, repeat the eight-token phrase for six cycles at
  intervals 1/2/4/8/16 ticks. Compare the native adaptive writer to repeated raw
  zero-baseline packets. Both trajectories are replayed, followed by 128 quiet
  ticks. Record actual input magnitudes and signed means; this is an open-loop
  source intervention, not an equal-energy ablation.
- Measure actual driven motor trajectories, not driven-minus-quiet periodicity.
  Compare complete physical fields at cycle boundaries. Full-state replay error
  is measured at the last boundary; earlier boundary changes are descriptive.

## Outcomes

All 36 pulse conditions have motor contrast within numerical noise at the input tick, and a resolved
motor response at tick 1. Across nine native resting pulses, the initial sensory
spike fraction is 21.38–23.38%; inadequate first-tick excitation does not explain
absence of same-tick motor access here. Different pulse tokens produce motor
contrasts at tick 1 with norms 0.0232–0.0305.

| Native pulse observation | Rest | After 128 text events |
|---|---:|---:|
| First resolved motor tick, all nine cases | 1 | 1 |
| Peak motor contrast tick | 43–53 | 5–31 |
| Peak contrast norm | 0.949–1.187 | 0.142–0.202 |
| Contrast norm at tick 127 | 0.267–0.379 | 0.024–0.112 |

The contrast is against a same-initial-state no-pulse trajectory. It characterizes
the total port-plus-substrate response. The text-conditioned state also changes
writer baseline; the resting/conditioned difference cannot be attributed solely
to neuronal adaptation. A late contrast maximum is not an axonal transit time,
optimal read deadline or proof of useful semantic memory.

For native repeated input, cycle-6 squared drive is 88.45–89.95% of cycle-1 drive.
The ratio is identical across spacings within each seed, as expected from the
writer's input-event-clock baseline. Fixed-packet drive energy is exactly constant.

| Token interval | Native motor cycle-6 cosine range | Native relative waveform change | Fixed-packet relative waveform change |
|---|---:|---:|---:|
| 1 | 0.766–0.769 | 0.701–0.706 | 0.696–0.704 |
| 2 | 0.894–0.909 | 0.421–0.457 | 0.382–0.427 |
| 4 | 0.940–0.941 | 0.341–0.345 | 0.339–0.343 |
| 8 | 0.956–0.960 | 0.282–0.296 | 0.247–0.281 |
| 16 | 0.962–0.966 | 0.261–0.279 | 0.237–0.259 |

The change compares the actual cycle-6 motor waveform with cycle 5, normalized by
the cycle-5 norm. High cosine can coexist with substantial magnitude or waveform
change. Full physical state also changes: at interval 16 with fixed packets,
cycle-boundary membrane differences are 18.5–19.9% of the previous membrane norm.
Continued changes cannot be fully explained by decreasing writer input, since
they persist under fixed packets. These observations alone do not identify which
of ALIF, STP, membrane integration and recurrent routing is responsible.

Most identical replays differ near floating-point precision. At fixed-packet
interval 16, seed 1 has visibly amplified replay differences: the final membrane
replay difference is 0.001744 and motor cycle replay
relative error 0.000353. These are recorded, and remain well below the measured
cycle-to-cycle waveform change of 0.259. This is numerical sensitivity in the
nonlinear substrate; individual threshold-event flips were not separately logged.

Longer spacings also allow more elapsed physical time and less frequent input;
the increasing cosine is not evidence of a resonant optimum at interval 16.
Six cycles show incomplete recurrence within this finite budget, while leaving
slower periodic attractors an open possibility.

## Design implications

The delayed sensory-to-motor interface already carries stimulus differences
without task training. Any prediction intended to depend on newly arriving input
must respect that physical access delay. First access is robust in these cases;
the largest total response is state-dependent, so its peak cannot be substituted
for one universal fixed settling time.

An architecture may use a time-stamped predictive read/feedback contract while
the state continues to evolve. Whether such a contract supplies better word
probabilities remains a separate learning question. This report chooses no new
readout, Gamma filter, predictive-coding objective or training run.

## Execution and provenance

32,838 completed physical ticks, 148.105 seconds, 536 MiB peak reserved CUDA
memory. Zero optimizer updates and zero checkpoint reads/writes. Every seed's
parameters hash identically before and after. Sources and graph hashes pass;
complete finite-state and native single-event interface gates pass. Related
interface tests: 21 passed.

The first attempt failed to allocate CUDA memory at the initial interface gate,
with zero completed response trajectories. The reviewed execution-only revision
kept unused vocabulary/read/decoder modules on CPU and preserved exact embedding
lookup values through device-copy hooks. Initialization, stimulus, physics and
the registered tick budget were unchanged. The failed source lock is preserved
in the amended registry.

Artifacts:
- `scripts/ib/measure_fly_fresh_response.py`
- `results/published/fly_fresh_response_preregistered_20261007.json`
- `results/published/fly_fresh_response_20261007.json`
- `results/published/fly_fresh_response_summary_20261007.json`

Hypotheses: independently reviewed by `harness_hypotheses`. Plan and execution
amendment: approved by `harness_plan`. Implementation: approved by
`diagnosis_review`. Outcome review is recorded in the research tree.
