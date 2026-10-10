# Latest plastic medium: first joint live-stream training

## Goal and evidence

Goal: useful next-token prediction by the learned 3D medium, measured by actual
first-pass, test-before-learn OWT NLL against the fixed prior on identical targets.
The architecture remains PlasticMediumPorts3D, 8x8x4, D128, 14,067,511 trainable
parameters: contact-mode local write, heterogeneous conductance, local transport,
nonlinear collision, activity feedback, STP, compact learned read and RMS decoder.
There is no added observer or latent response simulator.

The earlier conductance run stopped at 703 updates / 89,984 tokens. Its frozen
warm-local validation at update 500 was 7.59481; a subsequent mature-state audit
was 7.51761. Those are historical protocols, not today's live B benchmark.
The latest combined architecture has no sufficiently trained result.

CPU readiness on 32 real OWT events verifies finite full state and CE gradients
into the medium, writer and readout. State storage is 796,168 bytes. Spatial
energy share is 91.87% in that untrained trajectory; this establishes execution,
not learned spatial computation. The three-update CPU execution check has no
checkpoint and no capability conclusion.

## Competing explanations and accessible search space

1. The existing medium has a usable task path, and sufficient joint training
   will develop conditional prediction. The next action is its first full
   joint training, rather than another architecture change.
2. Task gradients can reach the medium but its cadence, local access or writer
   dynamics do not sustain useful conditional information. If learning remains
   confined to the read head, a positive spatial-energy metric alone cannot
   validate the internal computation.
3. The apparent NLL plateau is partly a measurement/calibration issue: different
   texts have different unigram NLL; training and revisit gains need not transfer
   to fresh context changes. Matched fresh B targets distinguish this case.

There is no mathematical NLL=7 boundary here. With decoder weight zero and bias
equal to log training frequencies, the model expresses the fixed unigram exactly.
Lower conditional entropy is accessible only to the extent its state carries
predictive information and the learner uses it. Nonzero CE-to-medium derivatives
establish a current training path; orthogonal state-dependent rotations do not
guarantee a stable overall Jacobian or unlimited historical credit.

## Registered run

- Real dataset: data/ib_owt_gpt2, revision and split in its manifest.
- One newborn individual; random embeddings and decoder weights, decoder bias
  initialized from the add-one training-only unigram. Every parameter remains
  trainable. This differs from the fly's frozen GPT-2 embedding transfer.
- At least 3000 fresh optimizer groups, 32 targets/group = 96,000 fresh training
  targets; additional actual B/revisit learning updates recorded separately.
- BPTT32, AdamW 2e-4, prediction weights decay .01, physical/norm/bias scales
  decay zero. WSD: 100 warmup groups, 2600 stable, 300 decay to 1e-6.
- Event duration .005, one solver substep, inherited execution cadence. This is
  dimensionless model time, not a biological measurement or optimized clock.
- CUDA Graph forward/backward with fused existing medium/port execution.
- Every 5000 fresh training targets: 256 genuinely new B targets, a scored return
  bridge, then the exact preceding 127 A targets. The return budget is 128 real
  events including its bridge; total B/return traffic is384, keeping 32-event
  captures aligned without erasing pending credit. No reset, rollback, freeze or
  evaluation-specific optimizer. Phase boundaries keep pending gradients and
  elapsed time. Bridge/revisit phase labels cross the same 32-event tape. No
  padding or optimizer flush is introduced to restore alignment.
- Capture warmups use owned belief copies and do not count as events/updates.
  Per-token losses come from the same production forward used for backward.
- Complete last.pt and best.pt only; calibration produces no model weights.
  Best is selected by pooled matched B gain, not the easiest single segment.
- Stop on explicit STOP file, requested fresh budget, nonfinite loss/gradient,
  or dedicated memory >3900 MiB. GPU calibration precedes the full run.

## Predictions and decisions, registered before training

Useful conditional learning must improve paired first-pass NLL; spatial structure
and conservation alone do not pass. Report train first-pass and fresh B separately,
and show every B segment with its own prior, pooled gain and learning curves.
Revisit savings describe the actual 257-event exposure gap and subsequent learning,
not arbitrary long-term retention.

- Positive pooled B gain with improving curves: retain the architecture; compare
  against fly on matched target IDs, data exposure, updates and initialization.
- Train gains with fresh B losses: prioritize generalization and readout/source
  calibration rather than claiming the material has a universal capacity ceiling.
- No conditional gains after the registered budget: inspect the task path in this
  trained medium; do not add a biological mechanism merely to improve proxies.
- Unresolved drift/no convergence: report the budget-limited outcome and choose
  continuation only from its actual trajectory. 3000 updates are the agreed
  minimum learning check, not a mathematical convergence theorem.

Independent plan review: /root/rtc_contract_review accepts first joint training,
conditioned on per-token scores, full state/gradient continuation, eager/capture
score-gradient-update parity and actual GPU calibration. CPU numerical tests,
real-data readiness and the small CUDA parity test passed. GPU speed/peak and
capability results are recorded in the run artifacts when executed.

Production calibration completed: 32-token training groups took0.1680-0.1685s
after warmup (about190tokens/s); a separate live context-change/revisit check
completed160 actual events and5 actual updates, pending0, with the following
training group at0.1671s. Recorded dedicated GPU usage1191MiB, allocated peak
681.5MiB, reserved1000MiB. Three CPU/CUDA prequential group NLLs differed by at
most2.31e-7. These are execution measurements; calibration is not a learning result.

Independent source re-review approved launch after the bridge alignment and
dependency/data hashing repairs. Affected CPU tests and CUDA score/gradient/Adam
parity passed. Full suite:1079passed,24skipped; one legacy compatibility test
requires the absent, previously cleaned age_003000.pt artifact. Deslice/gate
checks passed. No architecture edits were introduced to repair that old fixture.
