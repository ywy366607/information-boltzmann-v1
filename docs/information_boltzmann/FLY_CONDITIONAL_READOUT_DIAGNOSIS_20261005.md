# Conditional readout diagnosis after the timing repair

## Decision

The next repair target is the task readout/learning interface, not more physical
ticks or a new bath. Current-token causal access has been restored, but the
observed decoder output still relies almost entirely on a common background.
Sensory-only input, motor-only output and the complete never-reset state remain
mandatory. Production training was neither stopped nor reconfigured by this
audit; its registered 30M additional unique-token budget is unchanged.

This is a mechanism diagnosis, not a convergence claim or a new capability
benchmark. It does not yet establish that every motor variation carries useful
next-token information.

## Evidence and scope

- Primary active B at 1.1M cumulative fresh training targets: NLL 8.035377 versus
  fixed train-only unigram 7.650820. The corrected input-clock phase has 156288
  fresh targets / 4884 joint training updates by this encounter.
- Retrospective saved-head decomposition of 127 matched targets in the actual
  128-event A revisit: fixed unigram 7.440678; learned bias alone 8.319743;
  complete saved head 7.709641; common read background plus bias 7.709649.
  Cached features used event-time read weights; the diagnostic uses the saved
  end-of-encounter decoder. The actual prequential replay mean was 7.755529.
  The common feature mean is a retrospective descriptive statistic, not an
  online inference policy or a held-out performance claim.
- The legacy 100k checkpoint also gives essentially equal complete/common
  retrospective predictions (7.589847 / 7.589443) on its own actual revisit.
  It is a different text/history, not a matched causal comparison.
- On 16 real-token events from the continuing 1.1M physical state, temporal
  variation energy fractions are motor h 0.077297, projected latent 0.000344064,
  RMS-normalized latent 0.000203063. Relative variation is much smaller after
  projection, by about 225 times. Absolute centered RMS rises from 0.03261 to
  0.20846 under projection: this is common-background amplification / relative
  suppression, not a proof that physical information has been erased.
- One real input substitution changes 4186 sensory spike decisions at the input
  tick. At the first quiet tick motor v_pre difference norm is 0.027391,
  post-reset h difference 0.219400, and two motor spike decisions differ.
  This contrast argues against reset erasure as the first repair target.
  Current max-logit difference is 0.017288; larger responses appear at later
  events. One contrast does not certify all-input reachability or task value.
- Four actual cached 32-event blocks have decoder-gradient common/covariance
  norm ratios 40.22, 40.36, 42.23, 32.75. Exact gradient norm reconstruction
  errors are below 2.2e-15. This is before clipping and Adam, and does not
  attribute the same ratios to actual parameter displacement or to the motor
  projection gradient.
- The earlier 34k audit already observed motor centered rank 27.75, projected
  latent 2.32, normalized latent 1.52 and 99.91% common-direction energy. Later
  update/timing repairs did not make conditional predictive contribution an
  acceptance criterion. That unresolved dependency is now explicit.

## Three competing formulations

1. Propagation: does sensory coding reach the motor surface at a useful time
   and with task-relevant differences? Restored nonzero access is necessary,
   while filtering/STP/delay can still attenuate timely information.
2. Observability: does the task readout select useful differences or mostly
   amplify a shared background? The current read decomposition identifies the
   latter behavior on the inspected streams. Raw motor task relevance remains
   to be established rather than inferred from activity or rank.
3. Optimization: do marginal-frequency fitting and conditional learning compete
   through the same feature directions? The decoder gradient decomposition
   supports strong common-background credit; accumulated Adam/preconditioning
   effects need separate accounting.

## Mathematics

Let q_t = RMSNorm(R h_t) and logits l_t = b + W q_t. On any fixed sample:

    l_t = (b + W mean(q)) + W (q_t - mean(q)).

The first term is the effective marginal background, the second the varying
conditional term. Removing b alone removes half of a learned compensation;
adding a fixed unigram while retaining W mean(q) double-counts a background.
Merely centering and compensating the intercept is an equivalent function and
cannot improve NLL by itself.

For a linear layer with inputs q_i, CE errors e_i = p_i - one_hot(y_i):

    grad_W = mean(e) mean(q)^T + mean[(e-mean(e))(q-mean(q))^T].

The two terms are common-background and conditional-covariance gradients. The
same identity holds for R using h_i and the full upstream derivative a_i,
including normalization and decoder VJP. A common gradient is valid learning;
the question is whether it dominates the pathway intended to learn conditions.
Norms alone do not show destructive cancellation or Adam update contributions.

The synaptic filter g_next = rho*g + (1-rho)*A has transfer function
H(exp(i omega)) = (1-rho)/(1-rho exp(-i omega)). It preserves DC and attenuates
rapid variation. This provides a separate propagation hypothesis, not a proof
that the entire nonlinear spiking brain is an inappropriate low-pass filter.

## Bound decision and stopping rules

The useful repair direction is to give marginal statistics and conditional
readout distinct learning responsibilities while retaining the anatomical
surfaces. Before choosing an implementation, decompose an actual continuing
window's motor-projection gradient into its common and covariance terms and
account for the actual optimizer step. A purely equivalent centering change
will be rejected as a performance fix.

Accept progress only when (a) actual first-pass NLL improves relative to the
fixed reference and (b) conditional variation supplies positive same-history
predictive value after sufficient joint training. If motor differences have
no task relevance, move the diagnosis upstream to sensory coding/propagation;
do not add readout capacity or ticks to protect a failed hypothesis. Do not
infer semantic information, criticality or capacity from rank/energy alone.

Independent review: diagnosis_review examined code and both decompositions.
It agrees with the observed readout imbalance, rejects reset erasure for the
tested contrast, and distinguishes gradient dominance from optimizer causation.

Artifacts: scripts/ib/audit_fly_prediction_decomposition.py;
scripts/ib/audit_fly_current_token_causality.py;
results/published/fly_prediction_decomposition_1100k.json;
results/published/fly_prediction_decomposition_100k_legacy.json;
results/published/fly_current_token_biophysics_1100k.json;
results/published/fly_current_token_read_chain_1100k.json.
