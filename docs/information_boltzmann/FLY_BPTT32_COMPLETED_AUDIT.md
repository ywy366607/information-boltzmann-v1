# Completed fly BPTT32 learning audit (2026-10-04)

The run completed 100,000 additional OWT training targets and 3,365 joint
optimizer updates including active evaluation. It continued the archived
local-credit individual at train cursor 100,000 with its physical state,
weights and Adam moments, reaching train cursor 200,000. Sensory-only writing
and motor-only reading were preserved. No new production training was started
for this audit.

## Primary active-learning record

Twenty fresh validation segments of 256 targets gave mean pre-update NLL
7.984413; the best segment was 7.030319 and the last segment 8.263542. The
first five, middle ten and last five segment means were 8.663685, 7.582523
and 8.108919. These segments contain different text: their trend mixes
learning and context difficulty. The run ended at its budget, not at a
demonstrated convergence criterion.

All twenty matched A revisits improved their mean NLL over the original
encounter, excluding the changed first bridge target. Mean improvement was
0.148111 nats after 256 actual intervening events. Replay included ongoing
updates; this measures retention plus relearning rather than unupdated
first-revisit recall. Nine fresh segments met the descriptive recovery rule;
eleven did not. Centered readout effective rank had median 1.109759. This is
a descriptive measure rather than a diagnosis of capacity failure.

Runtime was 42.062 training targets/s including validation and concurrent
credit diagnostics; allocation setup peak 1,914 MiB, final reservation
2,144 MiB. Full energy ledger and full-state conditional FTLE were not logged
by this branch; field energy, activity and representation structure were.

The old frozen-internal-edge run's twenty reported validation plateau means
average 8.087508. It used a different stream range, scoring budget and update
cadence. Its 6.772387 best segment and this run's 7.030319 best segment do not
provide a matched causal comparison. No equal-data frozen-brain training
control exists for this continuation.

## Actual accumulated parameter changes

Relative L2 displacement from the continuation's starting checkpoint:

| Parameter group | Relative displacement |
| --- | ---: |
| Excitatory edge weights | 0.0005905% |
| Inhibitory edge weights | 0.00003987% |
| Visual sensory projection | 1.18e-12% |
| Chemical sensory projection | 0.000007423% |
| Mechanical sensory projection | 0.00002474% |
| Motor readout matrix | 56.35% |
| Vocabulary decoder matrix | 49.28% |

7.18% of excitatory and 2.72% of inhibitory edges changed at least one
float32 representable value. Eight groups of physical parameters and the
sensory gates also changed. STP parameters and embeddings remained fixed by
the registered learner. Relative displacement in logarithmic parameters is
not a relative change of their positive physical values.

The implementation differentiated all edges but used plain edge SGD at
1e-5, sensory projection SGD at 1e-4, and AdamW at 3e-4 for readout,
decoder, physical parameters and sensory gates. This explains a concrete
optimization-scale difference to investigate; increasing any learning rate
is not justified solely by a desire for larger displacement.

## Paired accumulated-learning intervention

`scripts/ib/audit_fly_bptt_learned_brain.py` forks the final complete checkpoint.
Each arm keeps final writer/readout/decoder weights, starts from identical
saved full physical state, executes 256 real burn-in events and scores the
same next 1,024 fresh validation targets. Physical state is never zeroed.
Weights stay fixed solely to isolate accumulated parameter changes; these
NLLs are mechanism diagnostics, not replacement active-evaluation scores.

| Arm | Diagnostic NLL | Increase on rollback |
| --- | ---: | ---: |
| Final learned parameters | 8.213205 | 0 |
| Restore starting E/I edge weights | 8.213198 | -0.0000083 |
| Restore starting internal physical parameters | 8.212747 | -0.0004593 |
| Restore both | 8.213352 | +0.0001473 |

Restoring physical parameters changes readout features substantially
(relative L2 difference 86.5%); paired 128-event block NLL effects have mixed
signs. Thus internal dynamics changed, while a consistently positive average
predictive benefit is not demonstrated by this segment. Restoring edge
weights alone changes features by 0.0972% and yields a negligible mean
predictive difference. The diagnostic allocates less than the 4 GiB budget.

This establishes a narrow finding: the accumulated internal learning in
this completed continuation has little mean benefit on this paired segment.
It does not establish an architecture ceiling or the result of training a
frozen-brain control from the same origin. The actionable issue is the
disproportionate learning scale of interfaces versus synaptic/projection
plasticity. Preserve the architecture and credit window when investigating
that optimization mismatch.

Artifacts: `results/q8_fly_bptt32_continuous_100k/completed_learning_audit.json`,
`learned_brain_intervention.json`, `lifelong_evaluation.jsonl`, and
`results/published/fly_bptt32_completed_audit.json`.

Reproduce the intervention:

```powershell
python scripts/ib/audit_fly_bptt_learned_brain.py --run results/q8_fly_bptt32_continuous_100k --burn-in 256 --score-tokens 1024
```
