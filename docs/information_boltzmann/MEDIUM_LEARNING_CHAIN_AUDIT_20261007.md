# Persistent medium: end-to-end learning-chain audit

This audit covers the production 3D-medium constructor, all fourteen registered
source dependencies, the OWT data/split/target path, CUDA capture, optimizer,
complete-state continuation, and active evaluation. Historical experimental
entry points are separate architectures; conclusions below concern the actual
`medium_v8_joint_bptt32_96k_recovered` run. No model or learning rule was changed
during this audit, and no additional training was launched.

## Outcome of the registered first joint run

The run completed 96,000 unique fresh training targets, 103,296 actual events,
and 3,228 optimizer updates. The nineteen active B encounters scored 4,864 new
targets before their respective updates, with continued state and learning:

| Quantity | Model | Same-target fixed training unigram | Gain |
| --- | ---: | ---: | ---: |
| Fresh B, pooled | 7.378615 | 7.460323 | 0.081708 |
| Fresh training, pooled | 7.244141 | 7.477571 | 0.233429 |
| A revisit, pooled | 6.961694 | 7.370734 | 0.409041 |

Fourteen of nineteen B encounters had positive paired gains. The first five B
encounters averaged +0.029294; the last five averaged +0.132081. The first and
last hundred training groups had paired gains +0.011192 and +0.208670. B
segments are contiguous and span few documents, so these counts are descriptive,
not independent statistical replication. A revisit versus unigram is also not
the A1-to-A2 savings metric, and its 257-event exposure gap is not a claim of
long-term retention. Budget completion is not convergence.

Final spatial energy share is 91.93%. Recorded GPU allocated peak is 778.56 MiB;
dedicated usage was 1,171 MiB. Final group time was 0.1641 seconds/32 targets.
These support execution and spatial-state claims. Predictive quality is judged
by the matched scores above, not by spatial share or throughput.

## Verified execution and data findings

- OWT train/validation/test hashes match the manifest. Token ranges, document
  EOT counts, unique document hashes and split assignments are valid; the 31M
  extension preserves earlier train/validation prefixes. No shared data bug
  imposing an NLL=7 ceiling was found. Historical GDN1/GDN2 reached 6.5011/6.5604
  on this corpus with 384k targets under their older fixed-window evaluation.
- Observations are `[carry_token, targets[:-1]]`. Each observed word is written,
  evolved and read before scoring the corresponding next target. There is no
  current-target injection before its own prediction.
- Task and write objectives each use one mean. CUDA Graph's 32/32 scale is one;
  short eager segments scale by their actual share of the optimizer window.
  Capture preserves parameter/gradient addresses, clears warmup gradients and
  recomputes parameter-dependent coefficients on replay. Scores are retained
  from the production forward before the optimizer step.
- Full field, flux, receptor, transmission, conduction, precision and physical
  clock persist after tape truncation. Recovery retained optimizer moments,
  pending gradients, carry token, cursors and RNG; uncommitted interrupted tail
  was excluded. The Windows telemetry lock repair was independently tested.
- CPU/CUDA score/gradient/Adam parity and affected numerical/interface tests
  passed. The previous full-suite result was 1,079 passed, 24 skipped and one
  legacy test failing for a previously removed checkpoint fixture, rather than
  a medium forward/backward failure.

## Concrete discrepancies and limits

### 1. The nominal FullRank source is not the active write implementation

Compact W4 uses `PredictiveImpedanceWriteAgent._packet_chart`. It bypasses
`FullRankTorusWrite.synthesize_packet`; fourteen parameters in source address,
width, content, state_content, neighbor_content and angle modules are unused.
They total 149,382 parameters, or 1.06% of the nominal 14,067,511 trainable count.
The active graph has 13,918,129 connected parameters in the audited windows.

The current chart is still state conditioned: eight compact spatial bases,
channel permutations and local-state modulation produce the packet. Calling it
rank-one or unconditioned would be incorrect. Nevertheless, claims that the old
additive neighborhood correction is preserved do not describe this graph.
Unused modules are a configuration/reporting discrepancy, not evidence that
their removal would improve prediction.

### 2. Auxiliary credit is larger at the language input interface

`quiet_training_chunk` optimizes next-token CE plus port observed-token CE plus
action KL, with unit weights and a shared gradient clip. A CPU copy of the final
checkpoint was evaluated on the next three real 32-token windows, without an
optimizer update or GPU work. Reproduction:

```text
python scripts/ib/audit_medium_learning_chain.py --checkpoint results/medium_v8_joint_bptt32_96k_recovered/last.pt --output results/published/medium_learning_chain_audit_20261007.json
```

| Parameter group | Task gradient norm | Write auxiliary norm |
| --- | ---: | ---: |
| Medium | 2.95–3.32 | 0.77–4.11 |
| Write agent | 0.85–0.91 | 1.87–2.00 |
| Source, active embedding/scale | 0.064–0.077 | 0.206–0.242 |
| Readout | 1.62–1.82 | 0 |
| Decoder | 1.89–2.02 | 0 |

Thus the auxiliary has roughly 2.05–2.33 times the write-agent task norm and
3.16–3.39 times the active source task norm. Directions vary: medium cosines
are +0.109, +0.281 and -0.093; write-agent cosines -0.036, +0.294 and -0.017.
This establishes competing optimization demands, not that removing the port
objective must improve NLL. The full task gradient reaches every medium
parameter; a disconnected brain is inconsistent with this trace.

The same three forward windows were also compared with the checkpoint's own
current decoder bias alone. Full output NLLs were 6.755675, 7.281693 and 7.691560;
bias-only NLLs were 7.282259, 7.848755 and 8.046759. Gains of 0.526584, 0.567062
and 0.355199 establish actual conditional-output use on these windows. This is
a local decomposition without learning, not a replacement for live evaluation.
The pooled live B gain against a locked prior includes both online bias changes
and conditional computation; its two components were not logged separately.

### 3. Forward memory and historical credit have different horizons

At a step1750 snapshot, conduction adaptation's median rate was 1.0046. With
event duration .005 this gives about 199 tokens per time constant. BPTT32 spans
.16 physical time, about 14.8% of one frozen-target relaxation. Older states
remain in the forward trajectory while their originating event gradients are
truncated. This is an explicit credit limit, not state reset.

Small dt does not make all dynamics identity: at that actual snapshot, one
transport changed field norm-relatively by 23.40%, whole advance by 23.33%,
collision by 1.18%, and conductance response by 1.42%. Larger dt is therefore
not justified as a general "wake up the brain" fix.

### 4. Readout sees only part of the persistent state immediately

The compact readout receives field; flux/receptors/conduction/STP influence it
through subsequent evolution. A numerical check at step2750 multiplied the
three flux fields by .99 while keeping field fixed. Immediate read feature
changed by exactly zero; after one .005 advance it differed relatively by
0.001832. This proves delayed observability and an actual return path. It does
not establish that directly reading all hidden states would improve language.

### 5. Prior comparisons mixed exposure, initialization and measurement

Old 3,000-update language runs used 128 targets/update (384k); this run uses 32
(96k). Fly imports GPT2 input and decoder embeddings and freezes its input
embedding; this medium learns both from random weights. Only 12,802 of 50,257
token types occur in the first96k corpus tokens, and 6,041 occur once. The fixed
prior uses the entire training split, legally supplying offline frequency
knowledge at birth. These initialization conditions must be stated.

Old champion scores repeatedly measured a fixed 4,096-token validation segment
after NESS construction, whereas current active B scores follow new texts after
real context changes. In addition, old champion step0 and later evaluations
used slightly different text offsets. Paired gains repair the raw-NLL ambiguity;
a new same-protocol comparison is required to establish superiority to GDN.

## Additional recording defects

Independent execution review found three narrower metadata issues:

1. The source fingerprint omits imported `train_plastic_conductance.py`, which
   supplies learning-rate and state packing helpers. Their actual implementations
   were reviewed and are correct; future source manifests must include this
   dependency. The existing checkpoint's recorded manifest is preserved.
2. The saved 257-event `actual_intervening_events` is the A1-end to A2-start
   boundary gap. Corresponding A1/A2 target scores are 384 events apart, with
   383 intervening events. Forgetting reports must distinguish boundary gap
   from each pair's repeat lag; no long-term forgetting conclusion follows from
   the earlier label.
3. `best.pt` is chosen by pooled historical B gain. This ranks cumulative
   individual experience, not the isolated ability of each checkpoint. Use
   complete `last.pt` for exact continuation; the selection criterion should be
   recorded explicitly in future files.

These defects affect reproducibility/interpretation. They do not establish a
cause of weak task performance. This audit leaves completed training artifacts
unchanged and documents the corrections rather than rewriting past scores.

## Decision

Retain the trained medium and completed artifacts. The registered prediction of
positive pooled fresh-B gain is supported at this budget; beating GDN and
long-credit superiority remain unestablished. No automatic second run or new
biological module follows from this audit.

For the next quality-improvement candidate, prioritize how the port auxiliary
shares task credit, while holding the medium graph fixed. Any changed loss is
a separate experiment: require a matched baseline, identical initialization,
data exposure and live evaluation, and register predictions before launch.
Simple continuation to the historical384k exposure is also a meaningful
baseline; it answers maturation, not the auxiliary's causal contribution.

Independent reviews: data/loss by `/root/data_loss_review`, core operators by
`/root/medium_operator_review`, execution/continuation by
`/root/rtc_contract_review`. Claims are scoped to what their audits and the real
gradient trace establish.
