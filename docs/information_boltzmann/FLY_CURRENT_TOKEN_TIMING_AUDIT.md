# Current-token timing audit of the continuing fly

Date: 2026-10-05. This is a CPU-only interface diagnostic on the complete
400,000-BPTT-target checkpoint (528,568 physical events), not a new capability
experiment. The live 30M-target learner continued throughout the audit.

## Confirmed causal gap

The topographic writer injects 15,912 sensory neurons. The output surface reads
2,333 output neurons. Their intersection contains zero neurons.

For one synchronous COBA event, incoming conductances are computed from the
**previous** four-slot spike ring. The current token changes sensory drive, but
does not enter incoming motor conductances. The motor membrane is computed,
reset on a spike, and read immediately. Current sensory spikes are inserted into
the ring only for later events.

With weights and pre-event physical state fixed, therefore:

\[
\frac{\partial h^{\mathrm{motor}}_{t+1}}{\partial e(x_t)}=0,
\qquad
\frac{\partial \mathrm{logits}_{t+1}}{\partial e(x_t)}=0.
\]

This is an absent computational path, not a small-gradient observation. Longer
BPTT can train how an earlier token influences later outputs; it cannot create
the missing current-event path.

Two forks of the same mature physical state differed in one actual OWT input
(token IDs 24166 and 1301), with subsequent inputs identical:

| Event lag | Full membrane difference norm | Motor difference norm | Maximum logit difference |
| --- | ---: | ---: | ---: |
| 0 | 121.6134 | **0** | **0** |
| 1 | 82.1196 | 0.004835 | 0.004435 |
| 2 | 57.7483 | 0.218941 | 0.007494 |
| 3 | 39.8795 | 0.137626 | 0.027422 |

The initial sensory-drive difference norm was 174.2209. The first nonzero motor
response was one event later. This confirms that the physical wiring does carry
input influence, but the language prediction deadline precedes its first arrival.
The numerical response magnitudes describe these two forks only.

## Validation difficulty versus learning

Each active validation encounter is a different 256-token text. Its scores are
pre-update scores from the actual learner. A fixed train-only add-one unigram
reference (the original 10,940,858-token training array used for decoder-bias
initialization) was scored on the exact same target tokens:

| Cumulative BPTT training targets | Model NLL | Fixed unigram NLL | Model minus unigram |
| --- | ---: | ---: | ---: |
| 200,000 | 7.6440 | 7.1759 | +0.4681 |
| 300,000 | 7.7524 | 7.2440 | +0.5085 |
| 400,000 | 7.9075 | 7.3256 | +0.5819 |
| 500,000 | 8.1369 | 7.4115 | +0.7253 |

Across these encounters, raw NLL increased 0.4928 while fixed-unigram surprisal
increased 0.2356. The remaining reference-relative gap increased 0.2572. The
unigram comparison controls marginal token frequencies only; contextual
difficulty and the learner's continuous physical history still differ. These
four encounters do not establish an architecture's asymptotic capability.

The model has yet to beat this fixed marginal reference on these particular
encounters. This motivates restoring current-token conditional access before
attributing the performance plateau to rank, parameter count or learning rate.

## Repair direction

Keep sensory-only writing, output-only reading, the biological delayed wiring,
and the complete never-reset state. Separate external token arrival events from
physical solver ticks: a next-token score must follow actual signal arrival at
the output surface, while remaining strictly before the next target token is
revealed. Counting one token as one delay tick currently compresses all available
propagation time into one synchronous update.

The next implementation must specify the event/score ordering explicitly and
verify current-token output sensitivity, future-token exclusion, continuing
state and wall-clock cost. It should not simply open a global read/write shortcut,
shift targets into the future while consuming their tokens, or add an arbitrary
extra optimization mechanism. No production architecture or training schedule
was changed in this audit.

Artifacts:

- `scripts/ib/audit_fly_current_token_causality.py`
- `results/published/fly_current_token_timing_audit.json`
