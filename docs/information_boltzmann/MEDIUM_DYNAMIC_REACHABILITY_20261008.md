# Dynamic information: expressive and local task reachability

## Target and hypotheses

The user asks whether existing dynamic information can be used by the actual
model. The prior test established response and gradients, not useful prediction.
Distinguish three possibilities: a blocked expression/supervision path; an open
path dominated by shared background/calibration; or a dynamic conditional signal
that transfers to subsequent text. Joint adaptation of medium and ports is a
fourth possibility when current frozen representations have weak signal.

The question here is local reachability around the completed individual, not
capability/convergence. Use its next 1024 actual OWT targets, continuously from
the complete saved state. The first512 specify directions, the last512 evaluate
them. No optimizer, weight update, additional microsteps, target input, reset or
history feature is introduced. State advances on the real input sequence.

## Production subspace and derivation

In the explicit zero-migrated reader, fix new policy/key weights at zero and let
only the actual `motion_merge.weight` W vary. Existing geometry, attention,
correction, RMSNorm and decoder stay fixed:

    b_t = merge(p_t) + W m_t
    logits_t = decoder(read_norm(b_t + correction(b_t)))

At W=0 this is exactly the old function. The current raw motion measurement m
has1024 components at D128; W is128x1024. Let v_t = d(CE_t)/d(b_t). The gradient
is G=mean(v_t m_t^T). If G is nonzero, the production function contains a smooth
direction -G with derivative -||G||_F^2 < 0. It can therefore exploit a task
error locally; this alone is not generalization or linguistic information proof.

Remove the training-feature mean response without adding a bias or changing
architecture: mu=mean(m_train), P=I-mu mu^T/||mu||^2, D=-G P. Then D mu=0,
training derivative is -||G P||_F^2, and held-out derivative is <G_hold,D>.
This is a permissible production weight direction. All state variables and
labels used at inference are causal. RMS normalization of each direction uses
its train-only base-response RMS to compare equal realized perturbation budgets.

## Predeclared diagnostics and controls

- Four contiguous256-target blocks; split exactly512/512. Per-token CE gradients
  are evaluated at the actual pre-correction base insertion point in batches32.
- Raw and mean-zero-response directions for motion; same-size field measurement
  p control (equivalent to a legal change of the existing merge.weight); and a
  fixed-seed712 permutation of motion within each partition as a negative control.
  Permutation is an offline diagnostic, not a deployable input path; no feature
  moves from held-out into direction selection. No permutation selection.
- Report gradient norms, feature variation, train/held-out directional derivative,
  projected-train versus raw-held-out gradient cosine and each held-out256 block
  separately. Shuffled-feature directions are explicitly offline controls,
  outside the production parameter subspace with causal measurements fixed.
- Verify analytic held-out derivative by symmetric finite perturbations with
  base-response RMS fractions2^-8 and2^-9 of train base RMS. Parameters stay
  unchanged: logits are evaluated using cached base+epsilon*D*measurement.
  These finite-difference sizes are numerical diagnostic resolutions, not
  model hyperparameters or training steps.

Training derivative <0 and finite-difference agreement accept the local
optimization path. A mean-free motion direction with negative subsequent
derivative provides local transfer evidence; field/permutation controls locate
its specificity. Flat/positive held-out derivative means no positive transfer
was detected here, not that joint learned representations cannot use motion.
Do not tune directions on the held-out labels, claim training progress, select a
best horizon/seed, or turn the constrained test into an architecture ranking.
Independent plan review: `/root/rtc_contract_review`, accepted with the actual
insertion-point, matched controls, production-valid mean projection and scope.

## Expressive reachability and its boundary

For two admissible states with the same old field measurement and different
motion measurements, let delta=m_A-m_B. If delta is nonzero, the real production
matrix can create any chosen base difference a via W=a delta^T/||delta||^2.
That base difference changes prediction when the downstream correction/norm/
decoder Jacobian does not annihilate a (a common logit shift also leaves the
probabilities unchanged). Thus motion-dependent distinctions lie inside the
reader's function class, subject to actual downstream sensitivity.

For many states, a single shared W cannot independently prescribe arbitrary
base corrections: the attainable correction matrix is X W^T, where X contains
the measured motion vectors. Its columns lie in col(X), and its rank is bounded
by min(rank(X),128). Spatial pooling, moment measurement and normalization can
alias distinct physical states; this proof covers measured distinctions, not
every hidden state or arbitrary trajectory. The medium and policy/key branches
can learn additional distinctions during joint training.

## Executed evidence and current availability

The deterministic production contract
`test_mean_zero_motion_direction_is_in_production_subspace_and_lowers_local_ce`
passes in double precision. It verifies the exact weight gradient v^T m, the
legal mean-response-zero direction, the negative analytic derivative, symmetric
finite differences through actual reads, and a lower CE for the positive small
perturbation. The entire dynamic-read test file has10 passing tests. This is an
algebra/implementation result, not a synthetic capability study or real-text
transfer result.

The planned1024-target OWT diagnostic has not executed: the full
`results/medium_v8_joint_bptt32_96k_recovered/last.pt` used for preceding audits
is now absent, while configuration and reports remain. No replacement individual,
random checkpoint or old score was used. The same complete state is required to
continue that diagnostic; its moved/archived path has been requested from the
user. Therefore local expression/optimization reachability is established under
the stated conditions, and actual next-text transfer remains pending.
