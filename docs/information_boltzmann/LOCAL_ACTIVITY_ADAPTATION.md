# Local activity adaptation in the learnable medium

The aim is to reduce repeated processing at the same locations while preserving
the option to specialize, overlap write/read ports, and transmit stored signals.
This is a continuum adaptation law inspired by ALIF and receptor kinetics. It
uses the existing inhibitory receptor history; it adds no persistent fatigue
tensor, firing-rate target, assigned anatomical region, or internal time loop.

## State and equation

Let `z_i` be field energy coordinates and `j_i,a` the three locally stored edge-response
coordinates. The dimensionless processing proxy is

```
u_i = sum_a ||j_i,a||^2 / (||z_i||^2 + sum_a ||j_i,a||^2).
```

The numerical denominator floor is dtype epsilon. Away from that floor, a
common rescaling of energy coordinates leaves the proxy unchanged. A static
stored field with zero edge response gives zero additional activity drive.
This proxy measures in-flight energy occupancy, not biological firing rate,
semantic usefulness, or a directly measured amount of computation.

Native opening rates `a_E,a_I`, closing rates `b_E,b_I`, and spatial gains
`g_E,g_I` are positive and learned from the local material. Existing native
opening rates also depend on instantaneous local voltage and edge currents.
Define the native inhibitory equilibrium `r_eq_native=a_I/(a_I+b_I)`. The added law is

```
alpha_I = a_I + g_I*u/t_reference
alpha_E = a_E / (1 + g_E*max(r_I-r_eq_native, 0))
dr_s/dt = alpha_s*(1-r_s) - b_s*r_s,  s in {E,I}.
```

For fixed native rates, increasing activity increases inhibitory opening, and
excess inhibitory history reduces excitatory opening. As activity subsides,
inhibitory history returns to the native equilibrium. The recovery timescale
comes from learned rates, rather than a prescribed number of tokens or ms.
Positive gains start at one in dimensionless units; this is an initialization,
not an asserted biological constant or target level of activity.

For frozen nonnegative opening and positive closing rates,

```
r_next = r_eq + exp(-(alpha+b)*dt)*(r-r_eq),
r_eq = alpha/(alpha+b).
```

Thus `[0,1]` is invariant at every positive step size. The additional kinetics
leave the frozen electrical work/Joule-heat ledger unchanged. This is a local
kinetic certificate; coupled dynamics still require timestep refinement and
the existing source/heat stability assumptions.

## Spatial competition and conduction

Each compact write port observes only its own footprint of inhibitory history.
The spatial policy is `softmax(existing_logits - exp(log_sensitivity)*local_I)`.
Equal history across all ports cancels exactly in softmax. Relative local use
therefore changes where an event is written without uniformly shutting input
off. The observation/prediction packet chart remains linear in token features
for a fixed pre-event belief. Readout keeps direct access to local stored fields;
there is no added fatigue multiplier on readout.

Existing reciprocal conduction adaptation can also sense local inhibitory
contrast. If its original bounded evidence is `s` and its positive gain is `g`,
use `eta=abs(I_i-I_j)*g/(1+g)` and `s_new=(1-eta)*s+eta`.
Since inhibitory fractions are in `[0,1]`, `eta` is in `[0,1]` and `s_new` stays
in `[-1,1]`. Equal history recovers the original rule. One shared edge value
maintains reciprocity. This is an engineered preference for connecting regions
with different activity histories; energy flow follows the existing field/current
equations. It guarantees neither a downhill current nor useful task routing.

## Compatibility and evidence

Shared module defaults preserve historical behavior (`activity_adaptation=False`).
The conductance trainer/benchmark expose `--activity-adaptation` and
`--no-activity-adaptation`, with the new candidate enabled by default. Checkpoints
record the flag, full receptor/conduction state, optimizer and stream position.
Continuous-state schema4 rejects an implicit change of this law. Use an explicit
new branch for the additional learned parameters.

`tests/test_local_activity_adaptation.py` checks monotone feedback, recovery,
bounded fractions/evidence, electrical energy closure, local write influence,
chart linearity, end-to-end likelihood gradients and continuation. Training and
evaluation CUDA Graph tests run both historical and adapted paths. Real OWT
execution records feedback engagement, total dedicated VRAM and timing. These
checks establish implementation, rather than language improvement. Assess NLL
and spatial participation together after sufficient joint training; useful
specialization can include repeatedly active regions.

2026-10-04 validation: 100 affected CPU tests passed; 8 CUDA-enabled training/
evaluation checks passed, including feedback on/off graph equivalence. Matched
OWT execution at D128, 8x8x4, 128 tokens/update, BPTT8, dt0.005 and one evolution
step used one warmup update plus one measured update per arm. Historical compact
ports measured 3.221s; feedback measured 3.511s under simultaneous independent
fly training. The candidate added 2305 parameters and no state tensor. Peak
allocated memory was 425.05MiB and total dedicated VRAM reached 3193MiB.
Eager/graph complete-state and gradient maximum absolute differences were zero.
After 256 real corpus tokens, inhibitory opening increment averaged 0.4977,
excitatory opening/native ratio 0.9298, spatial inhibitory std0.0263 and routing
evidence change0.00360. These are engagement diagnostics of an initial model,
not a finding about improved NLL or trained regional specialization. Reports:
`results/published/local_activity_execution_baseline.json` and
`results/published/local_activity_execution_adapted.json`.

## Primary references

- [Bellec et al., 2018, LSNN](https://papers.nips.cc/paper/7359-long-short-term-memory-and-learning-to-learn-in-networks-of-spiking-neurons): slowly adapting neuronal thresholds as persistent state.
- [Brette and Gerstner, 2005](https://pubmed.ncbi.nlm.nih.gov/16014787/): adaptation coupled to a dynamical neuronal state.
- [Destexhe, Mainen and Sejnowski, 1994](https://direct.mit.edu/neco/article/6/1/14/5768/An-Efficient-Method-for-Computing-Synaptic): opening/closing receptor kinetics.

The flow proxy, write competition and conduction contrast above are explicitly
our model choices. These references motivate stateful local adaptation; they
do not assert this exact continuum coupling or guarantee a language benefit.
