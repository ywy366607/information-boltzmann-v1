# Distinguishing source resources, physical timing and readout

This continues the conditional-readout diagnosis. The observation is weak
task use of core dynamics, not a prescription to add a phase oscillator or
more solver ticks. Protocol was registered before measurement in
`results/published/fly_clock_readout_preregistered.json` and independently
reviewed. Production weights, state, optimizer, timing and cursors were not
modified. The old process PID 22884 was absent when inspected; its progress
file still said running at 1144832 targets. No training was launched by this
diagnosis, and no cause of that process exit has been established.

## New measured chain

CPU forks of the exact 1.1M complete state differ in one real OWT input. After
the initial pulse, both receive only zero-source physical ticks for a fixed
eight-tick horizon. The observation is amplitude/functional response, not
semantic information or capability.

- 4186 sensory spike decisions differ. The STP pulse difference has 0.13138
  times their unweighted spike-contrast norm. On changed sensory neurons,
  pre-input resource x averages 0.05355 and release state u averages 0.81147.
- First quiet tick: motor arrival contrast E/I norms 0.428716 / 0.059576,
  filtered conductance contrasts 0.041520 / 0.003521. Norm ratios 0.09685 /
  0.05910. Exact delta_g=(1-rho_s)*delta_arrival holds with maximum residuals
  1.56e-8 / 2.79e-9, since previous motor conductances agree.
- At the normal read tick (first quiet tick), motor relative contrast is
  3.8485%, projected 0.13495%, normalized 0.12784%. 97.4% of the projected
  difference is tangent to the reference latent: this particular difference
  was already small before RMSNorm; deleting purely radial variation cannot
  account for most of it.
- Continuing quiet increases motor relative contrast to 9.4497% at tick 6;
  projected/normalized contrasts are 0.9651% / 0.7095%. At tick 7 motor
  contrast is 9.8965%, but projected contrast falls to 0.3355%. Probability
  Fisher squared contrast is 7.90e-6 at tick 1, 3.46e-4 at tick 6, then
  1.48e-5 at tick 7. Delayed responses exist; waiting longer does not
  monotonically recover readable differences. One contrast cannot prove
  whether a later read improves the target prediction.

One additional real 32-event continuation uses the same mature head for every
event. Targets are used only after collecting forward states; no updates or
cold starts occur. Motor variation fraction 0.081378 becomes 0.00043731 under
projection, then 0.00024698 after normalization. The complete head has 0.00716
worse NLL than its common-background-only descriptive counterpart on this
window. This is a mechanism probe, not a new active evaluation score.

The actual motor projection's gradient is decomposed with the full saved
RMSNorm/decoder VJP: common norm 0.106078, covariance norm 0.0238825, ratio
4.44165, cosine -0.00405. Identity reconstruction relative error 1.10e-16.
This is before clip/Adam, not actual parameter displacement. The earlier
33--42 ratio concerned the decoder gradient on a different cached window;
it must not be presented as the motor projection's ratio.

## Confirmed missing adaptation dimension

All 27 entries of each of `logit_u0`, `log_tau_fac`, `log_tau_rec` are excluded
from `adam_names`. Saved U0=0.25, tau_fac=100, tau_rec=200 are exactly equal
between the legacy 100k and current 1.1M checkpoints. There are 81 fixed STP
parameters. State x/u evolves and is differentiable within BPTT, but the
release/recovery parameter family cannot adapt in this training configuration.
Earlier configuration did record these groups as fixed: the problem is a
limiting freeze policy, not an unverified claim of accidental omission.

## Exact recovery-limited release bound

Write r_t=U_active,t * x_t * s_t for the unnormalized released resource. The
implemented recovery equation is

    x_(t+1) = (1-rho) + rho*x_t - rho*r_t.

Summing for L physical ticks gives, without independence assumptions:

    sum(r_t)/L = [(1-rho)/rho] * [1-mean(x_t)]
                 - [x_L-x_0]/(rho*L).

Since 0<=x<=1, the boundary contribution vanishes in a long-time average.
Mean unnormalized release is bounded by exp(1/tau_rec)-1. Current U0=0.25
gives normalization U0*(2-U0)=0.4375; the transmitted-pulse clamp at 3 is
inactive on permitted states because U_active*x/normalization<=2.285715.
Thus at tau_rec=200:

    long-time mean normalized pulse <= [exp(1/200)-1]/0.4375
                                    = approximately 0.01146 per physical tick.

With two physical ticks per input event this is about 0.02291 per event.
This is a throughput bound, not a maximum single spike amplitude, a semantic
information bound, or a claim that every sensory neuron is saturated.
Spatial/spike-time patterns can remain informative even at low amplitudes.

If U and firing can additionally be approximated by constants, the usual
steady-state approximation is x*=(1-rho)/(1-rho+rho*U*f). It explains why
the input cadence and recovery time should be considered together; the exact
bound above does not need that approximation.

Primary related literature: Tsodyks & Markram (1997),
https://pmc.ncbi.nlm.nih.gov/articles/PMC19580/, demonstrates that release
probability/depression shapes rate versus temporally coherent transmission.
It motivates examining STP coding rather than treating its constants as
task-independent. The numerical limits above are derived from our code,
not biological measurements or an assertion that weakening STP is beneficial.

## Decision and falsification

The first repair candidate is to restore the missing STP adaptation dimensions
under joint training, preserving topology, input/output surfaces and current
clock. Retain all existing state/moments; only newly trainable parameters get
new optimizer moments. This is a candidate and has not been applied here.
Avoid simultaneously changing R, delays, bath or K so the resulting evidence
can identify which defect contributes to predictive loss.

Accept that repair as a predictive contribution only if actual STP parameter
updates, resource/time response changes, positive conditional readout value
and first-pass NLL gains agree across a sufficient registered real-stream
budget. A larger pulse alone is insufficient. If transmission improves but
conditional task value remains absent, prioritize the independently observed
read geometry/learning path; do not add waiting steps to protect the hypothesis.

Current evidence supports input/recovery/integration scale mismatch plus read
imbalance. It does not certify phase-locking failure, task-relevant motor
information, or STP freezing as the sole causal source of NLL deficit.

Artifacts: scripts/ib/diagnose_fly_clock_readout.py;
results/published/fly_clock_readout_diagnosis_1100k.json;
results/published/fly_stp_learning_coverage_1100k.json.

Independent reviewer agrees that the search-space omission and resource
restriction are confirmed; requires prediction gains to establish NLL causation.
