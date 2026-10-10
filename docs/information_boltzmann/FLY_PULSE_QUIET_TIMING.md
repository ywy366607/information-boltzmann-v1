# Persistent fly: input pulse, quiet propagation, next-token read

Date: 2026-10-05. Implementation, numerical tests and resource calibration are
complete. The user's continuation goal authorized resuming the repaired long
run; PID 11852 is executing it. Capability progress awaits active evaluation.

## Event ordering

The minimal repaired event has one token pulse followed by one quiet physical
tick, then an output-surface read. The quiet tick has zero external sensory
drive; it advances the complete membrane, four-slot delayed pulses, E/I
conductances, ALIF and STP state. Sensory baseline relaxes toward zero by its
existing one-tick retention coefficient. It does not run the token embedding,
gate or projection networks a second time.

Existing physical tick duration, transmission delay buckets, thresholds and
time constants are retained. External token spacing is now two graph-delay
ticks. Thus relaxation per token also changes: this is an explicit timing change,
not an otherwise identical extra read. OWT supplies token order, not measured
physiological timestamps. The graph-delay unit is not asserted to be a validated
millisecond. One quiet tick is the measured earliest nonzero output response;
it opens the immediate causal path but does not guarantee that all longer
pathways have arrived or establish an optimal token interval.

The shared interface is `advance_fly_input_event`; training and active evaluation
use it through `FlyBPTTLearner.forward_window`. Label-free generation uses
`predict_fly_next(model, physical_state, token, settle_ticks=...)`, which returns
logits and the complete next state. The caller samples/assimilates the next token
only after that return. Generation must use the saved timing explicitly.

Legacy `model.step`, `forward_chunk`, online/e-prop scripts and historical
one-tick experiments retain their original physical-tick behavior. They are not
alternative inference entry points for a repaired checkpoint. The historical
manual credit inspector rejects pulse/quiet checkpoints because its edge-credit
reconstruction assumes one physical tick per token; native BPTT includes every
physical tick. The learning-rate, update-scale and learned-brain diagnostics
read the saved timing rather than silently using the old event ordering.

## Continuation and supervision

BPTT32 still means 32 input tokens and one joint AdamW update. It now covers 64
physical ticks, with no within-event detach. Full state and optimizer moments
persist across windows and resumes. Input `x_t` is processed before scoring
target `x_(t+1)`; the target does not enter the writer or physical evolution.
The next observed token is assimilated at the next input event.

Checkpoints retain `settle_ticks` and an independent `physical_ticks` counter.
For old checkpoints, elapsed physical ticks migrate from the old input-event
counter, since each event had one tick. Active reports include input and
physical-tick exposure intervals; A/B/A revisit gaps include both units.

The CLI default inherits checkpoint timing. First activation requires
`--settle-ticks 1`; subsequent resumes inherit that timing. Timing migration
retains all parameter/physical values and Adam moments, archives the previous
phase's config/progress and moves its best weights into a stage directory.
The new timing has its own validation best. Only one rolling full `last.pt` is
kept, avoiding a second full optimizer checkpoint. The archive move is checked
to stay within the output directory.

## Verification

21 focused numerical/interface tests passed, including both one-tick and
pulse/quiet CUDA Graph agreement, zero-source quiet propagation, one writer
call per token, current-input sensitivity, target/future isolation, generation
agreement, window-split continuity, and optimizer moment migration.

Matched mature checkpoint: BPTT training count 600,928; fresh train cursor
700,928; physical/input events 730,264; actual optimizer updates 19,271. Two
forks changed only the real OWT input token (345 versus 460):

| Timing | Current motor difference norm | Maximum current logit difference |
| --- | ---: | ---: |
| Original one tick | 0 | 0 |
| Input pulse + quiet tick | 0.146458 | 0.034845 |

Input and output surfaces remain disjoint (15,912 input neurons; 2,333 output
neurons). These response magnitudes belong to this matched pair; the absent
original path is structural, while response size depends on state and input.

Three matched actual-update calibration windows per timing, each 32 real OWT
targets and CUDA Graph forward/backward with AdamW updates:

| Timing | Median seconds / 32 tokens | Tokens / second | Peak allocated MiB | Reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.68355 | 46.81 | 2203.44 | 2390 |
| Pulse + quiet | 1.21157 | 26.41 | 2748.29 | 2934 |

The initial repaired execution cost was 1.77 times the original. Two exact
finite-input kernel optimizations then recovered the requested throughput:

| Repaired execution | Median seconds / 32 targets | Targets / second |
| --- | ---: | ---: |
| Original kernels | 1.21157 | 26.41 |
| Skip zero forward atomic contributions | 0.97556 | 32.80 |
| Also reduce contiguous presynaptic backward sums | 0.80105 | 39.95 |

The forward mask saves only exact-zero pulse traffic. Silent neurons retain
their complete pulse adjoint, including ATan surrogate credit. Backward reduces
each contiguous run within a block before atomically adding its sum, preserving
all edges across block and delay boundaries. A reset-flag segmented scan also
handles unsorted input and repeated keys separated by other keys. Float32
summation order changes; CPU/native reference tests check the outputs and VJPs
within rounding tolerance. Weight gradients retain every connection.

CUDA profiler measurements on the repaired 32-target update identify forward
synaptic kernels decreasing from 308.77 to 73.70 ms, and backward from 519.24 to
344.92 ms. These are measured captured kernel durations rather than inferred
shares of an eager timing. Peak/reserved calibration memory remained
2748.29/2934 MiB. Production's first window was 0.826 s; its next logged window
was 0.801 s, with a 2731 MiB peak allocation.

Calibration fork updates were discarded and did not advance the primary
learner's cursor or optimizer. No additional `.pt` was produced by calibration.
These measurements verify causality, gradients and execution resources;
NLL improvement and convergence require the continuing active learning run.

## Continuing long training

The old long run has been cooperatively saved and stopped at 600,928 cumulative
BPTT targets. The remaining registered budget is 29,581,216 fresh targets, ending
at 30,182,144. Preserve that target by omitting `--extend-budget`.

The run resumed from this boundary with the following command (the stop marker
was removed before launch):

```powershell
Remove-Item -LiteralPath results/q8_fly_bptt32_adamw_continuous_100k/STOP
python -u scripts/ib/train_fly_bptt_stream.py --resume results/q8_fly_bptt32_adamw_continuous_100k/last.pt --output results/q8_fly_bptt32_adamw_continuous_100k --data data/ib_owt_gpt2_31m --additional-tokens 30000000 --window 32 --settle-ticks 1 --lr 0.0002 --lr-decoder 0.0001 --lr-synapse 0.0002 --lr-sensory 0.0002 --plasticity-optimizer adamw --validate-every-tokens 100000 --log-every-tokens 1024 --vram-limit-mib 3900
```

The first migration preserves the historical best under
`segments/timing_before_600928_settle0/best.pt`; a new rolling best is written
when a repaired-timing active validation encounter completes. This yields at
most three assets: one stage best, one current best, one rolling full last.

Artifacts: `results/published/fly_current_token_timing_repair.json`,
`results/published/fly_current_token_timing_repair_control.json`, and
`results/published/fly_event_timing_execution.json`. Optimized captured profiles
are in `fly_event_timing_sparse_forward.json` and
`fly_event_timing_segmented_backward.json`; production launch and verified
progress are recorded in `fly_pulse_quiet_30m_launch.json` in the same directory.

## Read-only four-pillar interpretation

`python scripts/ib/summarize_fly_live_timing.py` reads the actual active-learning
ledger while production runs. It executes no model, training, freeze or reset.
It reports the original train-only add-one unigram reference on the exact fresh
B targets, with repaired and historical timing separated. New timing's fresh
training updates are counted from the migration boundary, rather than crediting
the candidate with all earlier optimizer updates. The user's 3000-update minimum
is a prerequisite for capability judgment; convergence requires additional
learning-curve and validation evidence.

Matched revisit excludes A2[0], whose input is the B-to-A bridge. The remaining
A pairs are identical to A1. The first return window's 31 matched scores precede
the return update, whereas the full 127 matched-pair scores also include later
return learning. The real exposure before the first matched pair is 256 B
events plus the return bridge: 257 input events, or 514 repaired physical ticks.
The producer's historical `actual_intervening_events=256` field names B exposure
only; this reader explicitly reports both durations without rewriting old logs.

The adaptation trajectory additionally reports marginal-reference-relative
block changes. This controls word-frequency difficulty only; the sequential
text and persistent history remain different across encounters. Health fields
remain descriptive activity/energy/rank records. Neither a lower sampled EMA
nor these health values certify predictive improvement, thermodynamic criticality
or NESS. Three numerical bookkeeping tests verify bridge exclusion, phase
separation and the actual input/physical/update counters.

## First repaired-timing active encounter (700,000 targets)

The repair has received 99,072 fresh training targets / 3,096 joint updates.
Production is still running at approximately 39.5 targets/s. The real active
B encounter processes 256 fresh targets; the return processes 128 A targets.
Its recorded exposure is 384 input events / 768 physical ticks / 12 updates.

| Pillar | Recorded result |
| --- | --- |
| Fresh prediction | NLL 8.93702; fixed train-only unigram 8.25307; deficit 0.68395 |
| Context adaptation | Descriptive half-recovery crossing at 112 input events; marginal-reference-relative opening-to-late drop 0.37339 |
| First return | 31 matched pairs before the return update: NLL saving 0.12884; all 127 matched replay pairs: saving 0.10184 |
| Internal health | Field mean-square 1.44735; last pulse activity 0.01318; centered effective rank 5.10913; peak allocated 2731 MiB |

The previous old-timing encounter's deficit to the same fixed reference was
0.72520; it is now smaller by 0.04125. These are different text/history
encounters, so this is a descriptive reference-relative change. The new
candidate remains worse than the marginal reference on all eight 32-target
B blocks. A predictive plateau breakthrough is not established; further
fresh learning and active encounters are still required.

Exact positive-delay min-plus bounds on the retained connectome give earliest
possible anatomical arrival at 1,424/2,333 output nodes after one quiet tick
(61.04%), 2,319 after two (99.40%), and 2,329 after three (99.83%). This counts
every directed path fitting the delay horizon, handles parallel edges without
summing their delays, and uses the existing maximum single-edge delay as its
reporting horizon. Two numerical tests verify direction, delay and parallel
edge handling. Actual functional arrival additionally depends on learned
weights, thresholds, synaptic filtering and ongoing state. The minimal timing
repair guarantees an immediate path, not arrival through every circuit. This
structural bound does not establish that adding ticks improves NLL; the
current training interval remains unchanged.

The complete encounter and interpretation are saved in
`results/published/fly_pulse_quiet_700k_evaluation.json`; the structural audit is
`results/published/fly_port_arrival_bounds.json`.

## Second active encounter (800,000 targets)

The unchanged repaired run has now received 199,072 fresh training targets /
6,221 training updates. Fresh B NLL is 8.58572, compared with 8.07454 for the
fixed unigram reference. Its deficit is 0.51118, narrower by 0.17277 than at
700,000. Raw NLL improved 0.35130; marginal-reference NLL improved 0.17853.
Different texts and histories remain, so the residual change is descriptive
evidence rather than a paired architecture effect. Both repaired encounters
underperform the reference on each of their eight recorded 32-target blocks.

The first return window's 31 matched pre-return-update pairs save 0.11136 NLL;
all 127 matched replay pairs save 0.12162. This encounter has no confirmed
raw within-B recovery crossing. Effective rank is 3.130 and mean-square field
value 0.78737, showing that the first encounter's higher rank and recovery
were trajectory-dependent observations rather than monotonic capability
markers. Activity is 0.01934. Production remains about 39.5 targets/s, with
2731 MiB peak allocation and an empty error log.

Across the two repaired encounters, the actual 512 fresh B targets have weighted
NLL 8.76137 versus reference 8.16380. The long run continues under the same
configuration, with the original 30M fresh-data budget. Complete state and
AdamW moments are checkpointed; the second encounter replaces the current
rolling best. See `results/published/fly_pulse_quiet_800k_evaluation.json`.

## Complete first-pass training loss accounting

The historical EMA mixes fresh training, fresh B encounters and A replay;
sampled 32-target log rows do not reconstruct the mean over every training
target. A measurement-only meter begins at 827,392 cumulative training targets
(train cursor 927,392). Earlier targets remain explicitly outside its coverage.
Each real learner score is accumulated before its target's optimizer update;
only ordinary fresh training enters this meter. Actual stream bridges remain
included. B and replay stay in the separate four-pillar records.

Full double-precision sums/counts and a bounded tail of 128 optimizer-window
aggregates are saved in the complete lifecycle checkpoint. The tail therefore
covers the latest 4,096 training targets once filled. Scores are compared with
the same fixed, add-one, original-train-only unigram reporting reference used
by the four-pillar reader; its count digest is checked on resume. This reference
is neither a learner nor a complete sequential task-difficulty control.

Twelve accounting/encounter tests verify target-weighted means, bounded storage,
complete resume, invalid-score rejection, cursor coverage, bridge exclusion and
physical/update cadence. Installation preserves state, AdamW moments, physical
timing, fresh cursors and the original 30M budget. The subsequent training PID
is 5392; model and learning equations remain unchanged. Predictive breakthrough
requires actual first-pass and active-encounter evidence.

## Third active encounter (900,000 targets)

At 900,000 cumulative training targets, the timing repair has received 299,072
fresh training targets / 9,346 training updates. The new B encounter's NLL is
8.06890 versus fixed unigram 7.74012: deficit 0.32877, smaller by 0.18241 than
at 800,000. Raw NLL improved 0.51683; the reference improved 0.33442. Texts and
history differ, so the residual comparison remains descriptive. One of the
eight optimizer windows now beats the fixed reference by 0.01896; the complete
encounter remains worse. Across three encounters, weighted fresh B NLL is
8.53054 versus reference 8.02258 (deficit 0.50797).

First return has 31 matching pairs scored before the return update, with NLL
saving 0.16562; all 127 matched replay pairs save 0.17258. The recorded raw
half-recovery crossing is 48 input events, while the reference-relative
opening-to-late residual drops 0.42563. These are trajectory descriptions;
the full revisit includes relearning and its benefit does not isolate physical
memory from the preceding updates. Actual B+return exposure is 384 input
events / 768 physical ticks / 12 optimizer updates, with no state reset.

Complete first-pass training measurement now covers 72,608 fresh targets since
installation: NLL 7.83705 versus reference 7.49468, deficit 0.34236. The recent
4,096-target mean is 7.87086 versus reference 7.49350. This corrects the sampled
EMA interpretation; a mean below the earlier roughly-seven platform has yet
to be established. Production remains about 39.3--39.6 targets/s including
periodic lifecycle saves; peak allocated 2731 MiB, reserved 2912 MiB.

The post-return centered rank is 2.479, activity 0.01698 and whole-field
mean-square 5.193. A CPU-only read of the exact 900,000-target lifecycle snapshot
separates the latter: sensory nodes account for 99.9385% of summed squares;
non-sensory mean-square is 0.003535 and output mean-square is 0.016305.
Non-sensory voltages span [-0.19352, 0.23220], whereas the sensory current
interface reaches -36.50. All physical arrays inspected are finite; ge/gi are
nonnegative and ALIF/STP states retain their permitted ranges. Global
mean-square is dominated by sensory current integration in this snapshot,
rather than measuring the internal circuit's excitation alone.

For a non-sensory node, source current is zero. With nonnegative conductances,
the implemented exponential COBA step is a convex combination of its previous
voltage and the conductance-weighted equilibrium of fixed reversals -0.2, 0,
and 1. The reset sends voltage to 0. Thus [-0.2, 1] is forward-invariant when
the starting internal voltages are in that interval; this follows from the
existing equations, independently of the learned positive threshold. Sensory
nodes additionally receive signed current and have a different amplitude bound.
The snapshot supports this partition; it is neither a NESS nor criticality
certificate. Learning equations and timing remain unchanged.

See `results/published/fly_pulse_quiet_900k_evaluation.json` and
`results/published/fly_900k_physical_health.json`. Prediction and pre-return
scores show progress, while the plateau-breakthrough objective remains active.

## Input prediction clock correction at 943,712 training targets

The added quiet tick incorrectly decayed the packet baseline toward a fictitious
zero observation while injecting exactly zero current. For decay `lambda`,
one quiet tick gave the event recurrence
`a_next = lambda^2 * a + lambda * (1-lambda) * packet`. A constant packet
therefore retained innovation `packet/(1+lambda)`: 51.2505% at the implementation's
lambda=0.9512. This is an exact filter defect, not a measured explanation of the
entire NLL difference.

The corrected `writer_baseline_clock=input` holds the baseline during quiet
propagation and updates it once per observed token. Its stationary constant
packet innovation is zero. Physical membrane, synapse, ALIF, STP and delay-ring
evolution still execute both ticks; physical time constants are unchanged.
The archived `physical` clock is explicit and inherited when loading checkpoints
without a clock key. Learning, generation and checkpoint diagnostics share this
choice. Baseline and all other physical values retain their existing history.

Production continues from 943,712 targets, cursor 1,043,712, under the unchanged
30,182,144 cumulative target. The old phase best is moved into
`segments/writer_clock_before_943712_physical/best.pt`; no full lifecycle
checkpoint duplicate is created. Complete first-pass accounting continues, with
an additional bounded current-phase meter starting at this migration boundary.
Active validation summaries split clock phases and assess minimum training
updates within the current implementation. All 34 affected numerical/interface
tests pass, including both clocks' CUDA Graph/eager gradients and updates.

Historical anchors remain: 3D champion 7.10975 (frozen 4096-token validation),
historical quadratic NESS 7.21536, and the first fly BPTT run's best fresh
256-target active encounter 7.03032. The latest pre-correction fresh encounter
is 8.06890; its three timing-repaired encounters average 8.53054. These protocols
and encounter texts differ. Historical performance restoration and predictive
platform breakthrough are pending; startup scores establish execution only.
