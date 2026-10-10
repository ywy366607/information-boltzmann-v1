# Read-only time, read and supervision audit

Registered before execution. No optimizer, parameter update, training launch,
new checkpoint, or change to the production architecture is permitted.

Target: determine whether current event timing, instantaneous read interface,
and finite supervision tape exhibit specific obstacles to using the evolving
3D medium. Capability improvement remains a separate question.

Snapshot: complete last.pt of medium_v8_joint_bptt32_96k_recovered. Preserve all
physical components and consume the next32 real OWT training targets at its
stored cursor. This is an offline mechanism audit of owned state forks, not
the primary active-learning evaluation. CPU native reference, one thread.

Competing explanations:

1. The current .005 event cadence misses useful response; later predetermined
   times improve the same-target output. An opposite result rejects a simple
   current-checkpoint 'just wait longer' repair.
2. Differences attributed to elapsed time are solver splitting errors. Compare
   identical .005 duration at1/2/4 substeps separately from additional time.
3. The field-only read is blind to hidden variables that alter future responses.
   This is an observability limitation, not proof of their linguistic utility.
4. Task supervision cannot reach local dynamics within32 events, or writer
   auxiliary gradients systematically oppose it. Nonzero state/parameter credit
   and mixed directions reject an absolute wiring failure. Long memory outside
   the tape remains forward-persistent, while its earlier computation is cut.

Checks and decisions:

- For each of32 consecutive actual events, write the previous observed token
  once. Read at0,.5,1,2,4 times the registered duration; later branches evolve
  without receiving any future token. The main offline trajectory always
  continues at the original duration. Report predetermined-time average CE,
  same-current-bias gain and paired signs. Label-selected minimum and
  entropy-selected time are descriptive only: neither is a deployed policy.
  All extra time is a hypothetical delayed-arrival branch, not free compute.
- At the same duration, report1/2/4-substep field/feature discrepancies and CE.
  A large discrepancy prompts a numerical-time audit before a timing redesign.
- On full state after the first observed carry token is assimilated, scale all
  three fluxes by .99 or1.01, retaining
  field and other state. These small admissible state perturbations are numerical
  sensitivity checks. Current outputs must coincide by interface construction;
  after one evolution, record actual feature and CE changes. Check direct
  and evolved read gradients to every hidden state separately.
- On one real32-event graph, measure terminal CE gradients to every stored
  full-state slice, per-event elapsed-duration sensitivities, and mean task
  versus writer auxiliary parameter gradients. Use norms and cosine together;
  magnitude alone cannot establish destructive supervision. Explicitly test
  detach as a zero-credit interface without resetting forward state.
  Selected lag0/1/2/4/8/16/24/31/32 suffix replays independently clone every state
  component to separate Markov partials from dependencies inside stored graph
  nodes. A +/-1% common duration finite difference checks the elapsed-time VJP.

Amendment before final interpretation: independent review caught that the first
observability call used a pre-carry snapshot with a post-carry label. Re-run from
the correctly assimilated state. Immediate feature blindness does not depend on
the label; all reported CE sensitivities must use the aligned label.

No verdict that CTM is necessary, that longer thought improves a retrained model,
that synchrony is biological phase locking, or that BPTT32 measures lifetime
credit is admissible from these checks. A structural blind spot plus future
task sensitivity motivates a candidate; task acceptance requires matched learning.

Plan review: /root/rtc_contract_review accepted CPU/read-only execution and the
three-check grouping, with the causal and conclusion limits above.
