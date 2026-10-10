# Time, read and supervision: read-only findings

Question: are neural internal time, read representation, and temporal supervision
blocking use of the evolving3D medium? The three axes were audited before any
new architecture or training. CPU native reference; zero optimizer updates;
every parameter verified unchanged. Final execution21.57s. Complete checkpoint
state was forked and the next32 real OWT targets were used at cursor96001.
This is a mechanism audit, not primary lifelong active evaluation.

Plan: MEDIUM_TIME_READ_CREDIT_PREREG_20261007.md.
Data: results/published/medium_time_read_credit_20261007.json.
Script: scripts/ib/audit_medium_time_read_credit.py.

## Timing has two distinct effects

Same already-written event, unseen next label, no future tokens injected:

| Read duration / registered .005 | Mean next-token NLL |
| --- | ---: |
| 0 | 6.758108 |
| .5 | 6.752333 |
| 1 | 6.755676 |
| 2 | 6.817062 |
| 4 | 7.201668 |

The offline main trajectory always advances .005; branches with later reads
imagine a delayed next arrival. More waiting at2/4 times the default has no
average benefit on this checkpoint/window. This rejects a simple immediate
'increase waiting' explanation here, not learned temporal organization or
retraining at another cadence. The same current decoder bias gives7.282259;
the default conditional read gains .526583 on these32 targets.

Fixed total .005 with1/2/4 numerical substeps gives6.755676/6.756617/6.757050.
Mean relative read-feature difference from1 step is .004024/.005686 for2/4.
The small refinement effect in this local test does not support a severe
integration-error explanation at this cadence or certify all longer durations.

Changing EVERY interval in the32-event sequence by +/-1% produces
6.758924/6.752816. The exact local task elasticity dL/dlog(duration) is-.306253;
finite difference is-.305390. Corresponding auxiliary derivatives are+.001980
and+.001931. Thus cadence has a differentiable, locally useful direction, and
fixed .005 is not a demonstrated timing optimum. This differs from prolonging
one isolated event: a common cadence changes the phase/history of every later
write. These observations motivate time calibration, not a fitted new constant.

The descriptive label-selected best among the five read instants is6.650066;
entropy-selected read is6.748214. The former uses labels and is an oracle, not a
deployable selector or CTM training result. Confidence selection alone recovers
little of that local selection gap. This motivates examining temporal output
calibration; it does not prove CTM's loss necessary or predict learning gain.

## Instantaneous read has a proven observational blind spot

Read receives field only. Independent flux, conduction, receptors and STP
components have zero direct CE derivative. After one .005 evolution, their
next-label derivatives are nonzero. The first observed carry token is written
before this audit's target is scored; independent review repaired an earlier
misaligned diagnostic label before publication of the final report.

Scale the three fluxes together by .99 or1.01, preserving the field and other
state. Current read features and logits remain exactly equal. After evolution,
feature relative changes are .002345/.002347 and next-label CE changes
+ .000934 / - .000928. Thus equal field snapshots can have different future
responses. This is a structural inability to observe the complete dynamic state
directly, not a proof that those distinctions improve language prediction.

A local motion-aware read, a temporal-relation read, and the possibility that
the field already contains sufficient task information are competing candidates.
Current evidence does not pick CTM over exposing existing local flux/state.

## Within-tape task credit is connected

The task CE reaches all156854 medium parameters, norm3.322838. Writer task norm
.913387 and auxiliary norm1.872778 have cosine-.035859: a mild opposition in
this window, not cancellation or evidence the auxiliary loss is harmful.
Medium task/auxiliary cosine is+.108958. Read/decoder task paths are connected.

For the terminal token, independent suffix replays reproduce the same
NLL13.548386 at every selected state slice. Its field gradient norm is.854540
at the current slice and.471241 at the32-event-old slice. Hidden-state credit
also remains nonzero at lag32. This single difficult terminal token proves
connectivity and contradicts an absolute within32-step credit collapse; it is
not a lifetime gradient profile or evidence of universal nonvanishing gradients.

Raw graph-node gradients are retained separately: components returned inside
one physical step can depend on each other. Cloned independent full-state
suffix replays are the component-wise Markov partials, avoiding that alias.

An actual detach yields no credit to prior-state leaves while the field values
remain exactly unchanged. That establishes the hard32-step training boundary,
and explicitly distinguishes gradient truncation from forward memory reset.

## Decision

Confirmed: fixed scheduler duration has no demonstrated optimum; instantaneous
field-only read has hidden-state observational aliasing; finite tape cuts earlier
computational credit. Confirmed favorable facts: the existing medium has dynamic
responses, differentiable cadence, nonzero32-step task credit, and conditional
predictive gain in this real window.

Unresolved: which limitation causes the remaining fresh-stream NLL gap, whether
neural history filters help beyond existing Markov memory, and whether CTM
synchrony/trajectory loss outperforms a simpler local motion-aware read.
Do not launch training or retrofit all CTM mechanisms as a presumed fix.
Next design must discriminate local full-state observability from temporal
relation necessity, and respect target timestamps before any additional data.

Review: /root/rtc_contract_review accepted read-only plan and audited execution;
flagged label alignment and the graph-node/Markov-partial distinction, both
corrected. Final conclusion audit recorded in the research tree.
