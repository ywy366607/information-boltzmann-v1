import CGBAxioms.GenerationOptimizer
import CGBAxioms.LanguageLifeKernel
import CGBAxioms.GenerationLinearMemory
import CGBAxioms.GenerationGradientFilter
import Mathlib.Algebra.BigOperators.Field

/-! Scope checks for the ongoing CLO6 branch. No production imports or new axioms.
These are algebraic interface checks, not optimizer performance results. -/
namespace CGB.OptimizerHandoffReview

/-- The column-budget WTA witness cannot be the squared entries of a 2x2
orthogonal matrix. Dropping orthogonality before constructing a counterexample
does not refute Muon itself. -/
theorem orthogonal_columns_exclude_one_row_wta (a b c d : ℝ)
    (h0 : a ^ 2 + c ^ 2 = 1) (h1 : b ^ 2 + d ^ 2 = 1)
    (hcross : a * b + c * d = 0) (hc : c = 0) (hd : d = 0) : False := by
  subst c
  subst d
  have ha : a ≠ 0 := by intro hz; subst a; norm_num at h0
  have hab : a * b = 0 := by simpa using hcross
  have hb : b = 0 := (mul_eq_zero.mp hab).resolve_left ha
  subst b
  norm_num at h1

/-- Algebraic specialization of a scalar conjugate Gaussian calculation:
prior variance S, observation noise R, posterior mean variance S^2/(S+R),
posterior variance S*R/(S+R). Subtracting the posterior variance does NOT
recover S. The probability-model interpretation is external to this lemma. -/
theorem posterior_mean_subtracted_power (S R : ℝ) :
    S ^ 2 / (S + R) - S * R / (S + R) = S * (S - R) / (S + R) := by
  rw [← sub_div]
  congr 1
  ring

/-- At unit signal/noise variance this proposed subtraction gives zero,
although the signal variance is one; adding posterior variance recovers it. -/
theorem unit_posterior_power_counterexample :
    (1 / 2 : ℝ) - 1 / 2 = 0 ∧ (1 / 2 : ℝ) + 1 / 2 = 1 := by norm_num

/-- Total variance of a two-component mixture. Probability interpretation
requires 0 <= w <= 1 and nonnegative component variances. -/
theorem mixture_total_variance (w mo mj po pj : ℝ) :
    w * (po + mo ^ 2) + (1 - w) * (pj + mj ^ 2) -
      (w * mo + (1 - w) * mj) ^ 2 =
    w * po + (1 - w) * pj + w * (1 - w) * (mo - mj) ^ 2 := by ring

/-- If soft hold mixes an ordinary Kalman write with an unchanged branch,
its mean matches CLO5/6, but its variance includes a between-branch term.
This identifies the missing term under that interpretation, not a complete
generative model for jump acceptance or correlated observations. -/
theorem soft_hold_variance_gap (w k p mu innovation : ℝ) :
    w * ((1 - k) * p + (mu + k * innovation) ^ 2) +
      (1 - w) * (p + mu ^ 2) - (mu + w * k * innovation) ^ 2 -
      (p - w * k * p) = w * (1 - w) * (k * innovation) ^ 2 := by ring

/-- Conditional noisy-error contract: once the actual update satisfies this
recurrence, a forward-invariant error tube follows. Noise need not contract on
every tick. This does not assert that CLO6 satisfies the recurrence. -/
theorem disturbed_error_stays_in_tube (err : ℕ → ℝ) (q noise radius : ℝ)
    (hq : 0 ≤ q) (hbudget : q * radius + noise ≤ radius)
    (hinit : err 0 ≤ radius)
    (hstep : ∀ n, err (n + 1) ≤ q * err n + noise) (n : ℕ) : err n ≤ radius := by
  induction n with
  | zero => exact hinit
  | succ n ih =>
    have hmul := mul_le_mul_of_nonneg_left ih hq
    have hs := hstep n
    linarith

end CGB.OptimizerHandoffReview

/-! Admission checks for the proposed single-decision reference. These are
conditional local facts and counterexamples, not admission of a full learner.
In particular the world/evidence recursion is not supplied by this file. -/
namespace CGB.OptimizerAdmissionReview

noncomputable def objective {n : ℕ} (g h err u : Fin n → ℝ) (cost : ℝ) : ℝ :=
  -(∑ i, g i * u i) + (∑ i, err i * |u i|) +
    (∑ i, h i * u i ^ 2) / 2 + cost

/-- The final executed step is certified only with a valid whole-step model,
valid error envelopes, and a nonpositive final objective. This neither assumes
nor proves that the Python solver or online estimates meet these premises. -/
theorem complete_step_certificate {n : ℕ} (g actual h err u : Fin n → ℝ)
    (cost oldLoss newLoss : ℝ)
    (herr : ∀ i, |actual i - g i| ≤ err i)
    (hmodel : newLoss - oldLoss ≤
      -(∑ i, actual i * u i) + (∑ i, h i * u i ^ 2) / 2)
    (hchosen : objective g h err u cost ≤ 0) : newLoss + cost ≤ oldLoss := by
  have hnoise : (∑ i, (g i - actual i) * u i) ≤ ∑ i, err i * |u i| := by
    apply Finset.sum_le_sum
    intro i _
    calc
      (g i - actual i) * u i ≤ |(g i - actual i) * u i| := le_abs_self _
      _ = |actual i - g i| * |u i| := by rw [abs_mul, abs_sub_comm]
      _ ≤ err i * |u i| := mul_le_mul_of_nonneg_right (herr i) (abs_nonneg _)
  have hsum : (∑ i, (g i - actual i) * u i) =
      (∑ i, g i * u i) - ∑ i, actual i * u i := by
    simp only [sub_mul, Finset.sum_sub_distrib]
  rw [hsum] at hnoise
  unfold objective at hchosen
  linarith

/-- The explicit coordinate proposal, with a supplied resource price. No age
or historical-evidence recursion is hidden in this definition. -/
noncomputable def thresholdMove (g threshold denom radius : ℝ) : ℝ :=
  if threshold < g then min radius ((g - threshold) / denom)
  else if g < -threshold then -min radius ((-g - threshold) / denom)
  else 0

/-- This proposal permits exact holding and both signs of plastic response
under positive available radius/denominator. This is a gate property, not yet
a healthy or beneficial historical latch. -/
theorem threshold_hold_and_plasticity (g threshold denom radius : ℝ)
    (ht : 0 ≤ threshold) (hd : 0 < denom) (hr : 0 < radius) :
    (|g| ≤ threshold → thresholdMove g threshold denom radius = 0) ∧
    (threshold < g → 0 < thresholdMove g threshold denom radius) ∧
    (g < -threshold → thresholdMove g threshold denom radius < 0) := by
  refine ⟨?_, ?_, ?_⟩
  · intro hg
    obtain ⟨hlo, hhi⟩ := abs_le.mp hg
    simp [thresholdMove, not_lt.mpr hhi, not_lt.mpr hlo]
  · intro hg
    simp only [thresholdMove, if_pos hg]
    exact lt_min hr (div_pos (sub_pos.mpr hg) hd)
  · intro hg
    have hnot : ¬threshold < g := by linarith
    simp only [thresholdMove, if_neg hnot, if_pos hg]
    have hp : 0 < min radius ((-g - threshold) / denom) :=
      lt_min hr (div_pos (by linarith) hd)
    linarith

/-- With the same quadratic update objective and one upper budget, both
directions can have positive signal, yet the weaker one's squared update
share can be arbitrarily small at the exact optimum. Thus a shared budget
alone gives no uniform anti-concentration margin. This concerns update
shares, and is not a claim about functional diversity. -/
theorem shared_budget_has_no_uniform_tail_share (ε : ℝ) (hε : 0 < ε) :
    ∃ t : ℝ, 0 < t ∧ t < 1 ∧ t ^ 2 / (1 + t ^ 2) < ε ∧
      ∀ u v : ℝ,
        (1 + t ^ 2) / 2 - 1 - t * t ≤ (u ^ 2 + v ^ 2) / 2 - u - t * v := by
  let t := ε / (ε + 1)
  have hp : 0 < ε + 1 := by linarith
  have ht0 : 0 < t := div_pos hε hp
  have ht1 : t < 1 := by
    dsimp [t]
    rw [div_lt_iff₀ hp]
    linarith
  have hte : t < ε := by
    dsimp [t]
    rw [div_lt_iff₀ hp]
    nlinarith [sq_pos_of_pos hε]
  have hsq : t ^ 2 < t := by nlinarith
  have hfrac : t ^ 2 / (1 + t ^ 2) < t := by
    rw [div_lt_iff₀ (by positivity : 0 < 1 + t ^ 2)]
    nlinarith [mul_nonneg ht0.le (sq_nonneg t)]
  refine ⟨t, ht0, ht1, hfrac.trans hte, ?_⟩
  intro u v
  nlinarith [sq_nonneg (u - 1), sq_nonneg (v - t)]

/-- If an identical visible prefix must support the same scalar decision a
in two worlds with incompatible desired values 0 and delta, at least one
squared error is delta^2/4. Future return/nonreturn cannot be inferred from
that identical prefix. Extra delivered evidence changes this premise. -/
theorem identical_prefix_retention_adaptation_bound (a delta : ℝ) :
    delta ^ 2 / 4 ≤ max (a ^ 2) ((a - delta) ^ 2) := by
  have h0 := le_max_left (a ^ 2) ((a - delta) ^ 2)
  have h1 := le_max_right (a ^ 2) ((a - delta) ^ 2)
  nlinarith [sq_nonneg (a - delta / 2)]

/-- A temporal resource condition, using the existing G20 exchange law. A
single use can be zero, but replenishment and two persistent eligible demands
give a positive two-opportunity service bound. For a stochastic two-state
pool this is a conditional-mean statement, not a pathwise deadline. Nothing
here proves that demand represents useful learning or that opportunities
are supplied by the actual organism. -/
theorem replenished_pool_temporal_service (supply demand nextDemand available η : ℝ)
    (hs0 : 0 < supply) (hs1 : supply ≤ 1)
    (ha0 : 0 ≤ available) (ha1 : available ≤ 1)
    (hη : 0 < η) (hd : η ≤ demand) (hd1 : demand ≤ 1)
    (hn : η ≤ nextDemand) :
    let next := CGB.Generation.twoStateStep supply demand available
    0 ≤ next ∧ next ≤ 1 ∧
      η * supply ≤ demand * available + nextDemand * next := by
  dsimp [CGB.Generation.twoStateStep]
  have hd0 : 0 ≤ demand := by linarith
  have hnext0 : 0 ≤ available + supply * (1 - available) - demand * available := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hd1) ha0,
      mul_nonneg hs0.le (sub_nonneg.mpr ha1)]
  have hnext1 : available + supply * (1 - available) - demand * available ≤ 1 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hs1) (sub_nonneg.mpr ha1),
      mul_nonneg hd0 ha0]
  have hsum : supply ≤ available +
      (available + supply * (1 - available) - demand * available) := by
    nlinarith [mul_nonneg (show 0 ≤ 2 - supply - demand by linarith) ha0]
  have hfirst := mul_le_mul_of_nonneg_right hd ha0
  have hsecond := mul_le_mul_of_nonneg_right hn hnext0
  have htotal := mul_le_mul_of_nonneg_left hsum hη.le
  exact ⟨hnext0, hnext1, by nlinarith⟩

end CGB.OptimizerAdmissionReview

/-! Finite-write refinement of the EXISTING conserved carrier. The additional
premise is an explicitly registered minimum write quantum, not a new life
axiom. This is an integer execution core, not a general optimizer admission.
Neither gradient normalization nor beneficial memory follows from this ledger.
The reference uses Int as mathematics; hardware input/range bounds remain due. -/
namespace CGB.ConservedWriteReview

def pending (halfQuantum old incoming : ℤ) : ℤ :=
  (old + incoming + halfQuantum) % (2 * halfQuantum) - halfQuantum

def committed (halfQuantum old incoming : ℤ) : ℤ :=
  old + incoming - pending halfQuantum old incoming

/-- One bounded carrier retains uncommitted requests exactly; committed writes
are multiples of the declared quantum. This neither stores a new history tape
nor establishes the usefulness of the requests themselves. -/
theorem conserved_write_spec (k old incoming : ℤ) (hk : 0 < k) :
    pending k old incoming + committed k old incoming = old + incoming ∧
    (-k ≤ pending k old incoming ∧ pending k old incoming < k) ∧
    ∃ packets : ℤ, committed k old incoming = (2 * k) * packets := by
  have hk2 : 0 < 2 * k := by omega
  have hlo := Int.emod_nonneg (old + incoming + k) (ne_of_gt hk2)
  have hhi := Int.emod_lt_of_pos (old + incoming + k) hk2
  have hdiv := Int.emod_add_mul_ediv (old + incoming + k) (2 * k)
  refine ⟨?_, ?_, ?_⟩
  · unfold committed
    ring
  · unfold pending
    omega
  · refine ⟨(old + incoming + k) / (2 * k), ?_⟩
    unfold committed pending
    linarith

/-- Holding depends on accumulated evidence, not solely on the newest input;
crossing either boundary releases a signed write using the SAME carrier. -/
theorem accumulated_hold_and_release (k old incoming : ℤ) (hk : 0 < k) :
    ((-k ≤ old + incoming ∧ old + incoming < k) → committed k old incoming = 0) ∧
    (k ≤ old + incoming → 0 < committed k old incoming) ∧
    (old + incoming < -k → committed k old incoming < 0) := by
  have spec := conserved_write_spec k old incoming hk
  refine ⟨?_, ?_, ?_⟩
  · rintro ⟨hlo, hhi⟩
    have hm : (old + incoming + k) % (2 * k) = old + incoming + k :=
      Int.emod_eq_of_lt (by omega) (by omega)
    simp only [committed, pending, hm]
    omega
  · intro h
    omega
  · intro h
    omega

/-- Reuse the existing carrier-path theorem: bounded pending state cannot
discard indefinitely a persistent integer request. Scheduling, resource
payment and input quantization must still be proved for an actual learner. -/
theorem persistent_request_cannot_freeze
    (p : CGB.LanguageLifeKernel.ConservedCarrierPath) (ceiling : ℤ)
    (hbound : ∀ n, p.value n < ceiling) (hdemand : ∀ n, 1 ≤ p.input n) :
    ∃ n, p.output n ≠ 0 := by
  by_contra h
  push_neg at h
  obtain ⟨N, hN⟩ := exists_nat_gt (ceiling - p.value 0)
  have balance := CGB.LanguageLifeKernel.conserved_carrier_path_no_output p N h
  have hsum : (N : ℤ) ≤ Finset.sum (Finset.range N) p.input := by
    calc
      (N : ℤ) = Finset.sum (Finset.range N) (fun _ => (1 : ℤ)) := by simp
      _ ≤ Finset.sum (Finset.range N) p.input :=
        Finset.sum_le_sum (fun n _ => hdemand n)
  have hb := hbound N
  omega

-- A finite legal write can be harmful despite a persistent nonzero gradient.
-- Requests must encode the benefit of the ACTUAL finite edit, not raw sign.
example : (1 / 2 : ℝ) * (1 - 1 / 10) ^ 2 > (1 / 2 : ℝ) * (0 - 1 / 10) ^ 2 := by
  norm_num

-- Adam's bias-corrected scalar first-step normalization with epsilon=1 is
-- already nonhomogeneous. A linear-memory theorem cannot classify it as linear.
example : (2 : ℝ) / (|2| + 1) ≠ 2 * ((1 : ℝ) / (|1| + 1)) := by norm_num

end CGB.ConservedWriteReview

-- Original latch obligation, before health/release: two histories cannot remain
-- positively distinguishable if their complete-state gap contracts uniformly
-- under the same future input. Reuse the existing contraction theorem; this
-- does not select a unique regenerative mechanism or classify Adam/Muon.
example (gap : ℕ → ℝ) {q ε : ℝ} (hq0 : 0 ≤ q) (hq1 : q < 1)
    (hε : 0 < ε) (hstep : ∀ t, gap (t + 1) ≤ q * gap t) :
    ¬ (∀ t, ε ≤ gap t) := by
  intro hretain
  have hgpos : 0 < gap 0 := lt_of_lt_of_le hε (hretain 0)
  obtain ⟨n, hn⟩ := exists_pow_lt_of_lt_one (div_pos hε hgpos) hq1
  have hb := CGB.Generation.histories_merge_uniform gap (fun _ => q)
    (fun _ => hq0) (fun _ => le_rfl) hgpos.le hstep n
  have hlt : q ^ n * gap 0 < ε := (lt_div_iff₀ hgpos).mp hn
  exact (not_lt_of_ge (hretain n)) (hb.trans_lt hlt)

#print axioms CGB.OptimizerHandoffReview.orthogonal_columns_exclude_one_row_wta
#print axioms CGB.OptimizerHandoffReview.posterior_mean_subtracted_power
#print axioms CGB.OptimizerHandoffReview.unit_posterior_power_counterexample
#print axioms CGB.OptimizerHandoffReview.mixture_total_variance
#print axioms CGB.OptimizerHandoffReview.soft_hold_variance_gap
#print axioms CGB.OptimizerHandoffReview.disturbed_error_stays_in_tube
#print axioms CGB.OptimizerAdmissionReview.complete_step_certificate
#print axioms CGB.OptimizerAdmissionReview.threshold_hold_and_plasticity
#print axioms CGB.OptimizerAdmissionReview.shared_budget_has_no_uniform_tail_share
#print axioms CGB.OptimizerAdmissionReview.identical_prefix_retention_adaptation_bound
#print axioms CGB.OptimizerAdmissionReview.replenished_pool_temporal_service
#print axioms CGB.ConservedWriteReview.conserved_write_spec
#print axioms CGB.ConservedWriteReview.accumulated_hold_and_release
#print axioms CGB.ConservedWriteReview.persistent_request_cannot_freeze

/-! Intrinsic optimizer boundary: supplied gradients are the input port. No
feedback through a loss/model is attributed to the optimizer. Adam's m and v
use the common-drive recurrence below (drives (1-beta1)*g and (1-beta2)*g^2);
Muon's momentum uses beta*M + g or beta*M + (1-beta)*g, entry by entry.
Bias correction and nonlinear normalization/orthogonalization are readouts,
not writes back into those moment carriers. Exact-real arithmetic is modeled;
weight decay, state resets and rounding-induced fixed points are not included.
-/
namespace CGB.OptimizerIntrinsicReview

/-- With the same future drive, each moment's old-history difference decays
exactly by beta^t, independently of how its output will be normalized. -/
theorem ema_history_gap (a b drive : ℕ → ℝ) (β : ℝ)
    (ha : ∀ t, a (t + 1) = β * a t + drive t)
    (hb : ∀ t, b (t + 1) = β * b t + drive t) (t : ℕ) :
    a t - b t = β ^ t * (a 0 - b 0) := by
  induction t with
  | zero => simp
  | succ t ih =>
    calc a (t + 1) - b (t + 1) = β * (a t - b t) := by rw [ha, hb]; ring
      _ = β * (β ^ t * (a 0 - b 0)) := by rw [ih]
      _ = β ^ (t + 1) * (a 0 - b 0) := by rw [pow_succ]; ring

/-- A moment with beta != 1 has only one fixed value under a fixed drive;
there are no two self-maintaining moment codes selected by past history. -/
theorem ema_fixed_point_unique (β drive a b : ℝ) (hβ : β ≠ 1)
    (ha : a = β * a + drive) (hb : b = β * b + drive) : a = b := by
  have h : (1 - β) * (a - b) = 0 := by nlinarith
  rcases mul_eq_zero.mp h with h | h
  · exact False.elim (hβ (by linarith))
  · linarith

/-- Full optimizer interface, including an arbitrary nonlinear readout and
time-dependent bias correction. The gradient port is supplied externally;
advance/read do not inspect parameters. This matches the dependency boundary
of ordinary Adam and momentum-orthogonalized Muon without weight decay. -/
def suppliedPath {P S G : Type*} [AddCommGroup P]
    (advance : ℕ → S → G → S) (read : ℕ → S → G → P)
    (gradient : ℕ → G) (parameter : P) (state : S) : ℕ → P × S
  | 0 => (parameter, state)
  | t + 1 =>
    let old := suppliedPath advance read gradient parameter state t
    let next := advance t old.2 (gradient t)
    (old.1 + read t next (gradient t), next)

/-- Nonlinear output processing cannot repair a parameter displacement when
the moment state and supplied future gradients agree: the displacement is
preserved exactly at every future step. No isolated restorative parameter
latch is provided by this optimizer interface alone. -/
theorem supplied_path_preserves_offset {P S G : Type*} [AddCommGroup P]
    (advance : ℕ → S → G → S) (read : ℕ → S → G → P)
    (gradient : ℕ → G) (parameter offset : P) (state : S) (t : ℕ) :
    (suppliedPath advance read gradient (parameter + offset) state t).1 =
      (suppliedPath advance read gradient parameter state t).1 + offset ∧
    (suppliedPath advance read gradient (parameter + offset) state t).2 =
      (suppliedPath advance read gradient parameter state t).2 := by
  induction t with
  | zero => simp [suppliedPath]
  | succ t ih =>
    constructor
    · simp only [suppliedPath, ih.1, ih.2]
      abel
    · simp only [suppliedPath, ih.2]

-- Explicit Algorithm-1 Adam substitution (one coordinate, arbitrary common
-- age). This keeps both moments, bias correction, square root and epsilon.
example (β₁ β₂ η ε : ℝ) (age : ℕ) (gradient : ℕ → ℝ)
    (parameter offset m v : ℝ) (t : ℕ) :
    let advance := fun (_ : ℕ) (s : ℝ × ℝ) (g : ℝ) =>
      (β₁ * s.1 + (1 - β₁) * g, β₂ * s.2 + (1 - β₂) * g ^ 2)
    let read := fun (n : ℕ) (s : ℝ × ℝ) (_ : ℝ) =>
      -η * (s.1 / (1 - β₁ ^ (age + n + 1))) /
        (Real.sqrt (s.2 / (1 - β₂ ^ (age + n + 1))) + ε)
    (suppliedPath advance read gradient (parameter + offset) (m, v) t).1 =
      (suppliedPath advance read gradient parameter (m, v) t).1 + offset := by
  dsimp only
  exact (supplied_path_preserves_offset _ _ _ _ _ _ _).1

-- Matrix/vector-indexed Muon substitution: orthogonalization is arbitrary,
-- so this includes finite Newton-Schulz and idealized polar readouts. mix=1
-- reads momentum directly; other mix values include Nesterov-style readout.
example {ι : Type*} (β driveScale mix η : ℝ)
    (ortho : (ι → ℝ) → (ι → ℝ)) (gradient : ℕ → ι → ℝ)
    (parameter offset momentum : ι → ℝ) (t : ℕ) :
    let advance := fun (_ : ℕ) (m g : ι → ℝ) =>
      fun i => β * m i + driveScale * g i
    let read := fun (_ : ℕ) (m g : ι → ℝ) =>
      fun i => -η * ortho (fun j => mix * m j + (1 - mix) * g j) i
    (suppliedPath advance read gradient (parameter + offset) momentum t).1 =
      (suppliedPath advance read gradient parameter momentum t).1 + offset := by
  dsimp only
  exact (supplied_path_preserves_offset _ _ _ _ _ _ _).1

/-- Regeneration has a cumulative obligation: preserving a positive history
gap against uniform passive contraction requires continuing regenerative
contribution, up to the initial/final stored gap. The units are gap units;
converting this bound to energy requires a separate material/cost contract. -/
theorem maintained_gap_requires_regeneration (gap boost : ℕ → ℝ) (q ε : ℝ)
    (hq : q ≤ 1) (hkeep : ∀ t, ε ≤ gap t)
    (hstep : ∀ t, gap (t + 1) ≤ q * gap t + boost t) (T : ℕ) :
    (1 - q) * ((T : ℝ) * ε) ≤
      gap 0 - gap T + ∑ t ∈ Finset.range T, boost t := by
  have balance : ∀ n, gap n + (1 - q) * (∑ t ∈ Finset.range n, gap t) ≤
      gap 0 + ∑ t ∈ Finset.range n, boost t := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      simp only [Finset.sum_range_succ]
      have hs := hstep n
      nlinarith
  have hf : (T : ℝ) * ε ≤ ∑ t ∈ Finset.range T, gap t := by
    calc (T : ℝ) * ε = ∑ _t ∈ Finset.range T, ε := by simp
      _ ≤ ∑ t ∈ Finset.range T, gap t := Finset.sum_le_sum (fun t _ => hkeep t)
  have hm := mul_le_mul_of_nonneg_left hf (sub_nonneg.mpr hq)
  have hb := balance T
  linarith

-- Even a state-dependent multiplicative learning-rate gate cannot regenerate
-- a perturbed weight from zero supplied gradient. It must change more than
-- the gain if active weight restoration is the required behavior.
example (gain : ℝ → ℝ) (w perturbation : ℝ) :
    ((w + perturbation) - gain (w + perturbation) * 0) -
      (w - gain w * 0) = perturbation := by ring

-- Existing LocalMedium energy, restricted to fixed p and conserved q+w=s.
-- Its unit-energy contribution has nonnegative discrete curvature when J,p
-- are nonnegative. Mixed coupling alone does not prove a local double well;
-- driven/joint dynamics and compartment potentials still need analysis.
example (ε J p s w : ℝ) (hJ : 0 ≤ J) (hp : 0 ≤ p) :
    let E := fun z : ℝ => ε * z - J * z * p * (s - z)
    0 ≤ E (w + 1) - 2 * E w + E (w - 1) := by
  dsimp only
  have h : 0 ≤ 2 * J * p := by positivity
  nlinarith

end CGB.OptimizerIntrinsicReview

-- Exact two-row/one-column allocation counterexample. A=(7/5,1/5)^2
-- has total mass 2. Equal row targets (1,1) yield the Sinkhorn-scaled A=(1,1),
-- preserving the column total and both signs. For g=(7/5,20), H=diag(1,100),
-- the Newton proposal decreases the quadratic by 149/50, whereas the scaled
-- proposal increases it by 291/10. This refutes descent preservation by
-- post-hoc equal-marginal scaling, not the use of Sinkhorn in a joint problem.
example :
    (7 / 5 : ℝ) ^ 2 + (1 / 5 : ℝ) ^ 2 = 1 ^ 2 + 1 ^ 2 ∧
    ((7 / 5 : ℝ) * (7 / 5) + 20 * (1 / 5) -
      ((7 / 5) ^ 2 + 100 * (1 / 5) ^ 2) / 2 = 149 / 50) ∧
    ((7 / 5 : ℝ) + 20 - (1 + 100) / 2 = -(291 / 10)) := by
  norm_num

#print axioms CGB.OptimizerIntrinsicReview.ema_history_gap
#print axioms CGB.OptimizerIntrinsicReview.ema_fixed_point_unique
#print axioms CGB.OptimizerIntrinsicReview.supplied_path_preserves_offset
#print axioms CGB.OptimizerIntrinsicReview.maintained_gap_requires_regeneration

namespace CGB.OptimizerAdmissionReview

/-! Online observer construction, not a claim of Bayes calibration. Hypothesis H:
one compartment's observation clock advances in proportion to its available
resource. Direction and energy use the same normalized observation weights.
This reuses the local mobility instead of adding dense noise/age/curvature
arrays. It does not supply the parameter-dependent restorative port required
by supplied_path_preserves_offset, or establish task usefulness. -/

/-- A bounded exposure mass replaces a growing update counter. The effective
gain is derived from normalized accumulated observations, including startup.
The old mean is already normalized. No beta^t or stationary-prior shortcut. -/
theorem normalized_observation_clock (rate available mass old observed : ℝ)
    (hα : 0 < rate * available) (hα1 : rate * available ≤ 1)
    (hm0 : 0 ≤ mass) (hm1 : mass ≤ 1) :
    let α := rate * available
    let nextMass := (1 - α) * mass + α
    let gain := α / nextMass
    (0 < nextMass ∧ nextMass ≤ 1) ∧ (0 < gain ∧ gain ≤ 1) ∧
    ((1 - α) * mass * old + α * observed) / nextMass =
      (1 - gain) * old + gain * observed := by
  dsimp only
  have hb : 0 ≤ (1 - rate * available) * mass := mul_nonneg (by linarith) hm0
  have hn : 0 < (1 - rate * available) * mass + rate * available := by linarith
  refine ⟨⟨hn, ?_⟩, ⟨div_pos hα hn, (div_le_one hn).mpr (by linarith)⟩, ?_⟩
  · nlinarith [mul_nonneg (show 0 ≤ 1 - rate * available by linarith)
      (show 0 ≤ 1 - mass by linarith)]
  · field_simp
    ring

/-- Exact cost of sharing one gain: heterogeneity loss plus the chosen
gain's mismatch to the pooled optimum. P and R are declared prior/noise
risks with zero within-coordinate cross term, NOT quantities inferred by
an EMA identity. Correlation across coordinates is allowed for this additive
risk. A resource-clock gain is admissible, but not asserted Bayes optimal. -/
theorem pooled_gain_risk {n : ℕ} (P R : Fin n → ℝ) (gain : ℝ)
    (hlocal : ∀ i, 0 < P i + R i)
    (htotal : 0 < (∑ i, P i) + ∑ i, R i) :
    let total := (∑ i, P i) + ∑ i, R i
    let pooled := (∑ i, P i) / total
    (∑ i, CGB.GradientFilter.gainErr (P i) (R i) gain) -
      (∑ i, CGB.GradientFilter.gainErr (P i) (R i) (P i / (P i + R i))) =
    (∑ i, (P i + R i) * (pooled - P i / (P i + R i)) ^ 2) +
      total * (gain - pooled) ^ 2 := by
  let risk := fun k => ∑ i, CGB.GradientFilter.gainErr (P i) (R i) k
  have aggregate : ∀ k, risk k =
      CGB.GradientFilter.gainErr (∑ i, P i) (∑ i, R i) k := by
    intro k
    simp only [risk, CGB.GradientFilter.gainErr, Finset.sum_add_distrib,
      ← Finset.mul_sum]
  let pooled := (∑ i, P i) / ((∑ i, P i) + ∑ i, R i)
  have hbetween : risk gain - risk pooled =
      ((∑ i, P i) + ∑ i, R i) * (gain - pooled) ^ 2 := by
    rw [aggregate, aggregate,
      CGB.GradientFilter.gain_error_square _ _ gain htotal,
      CGB.GradientFilter.gain_error_square _ _ pooled htotal]
    simp only [pooled, sub_self, zero_pow (by decide : 2 ≠ 0), mul_zero, add_zero]
    ring
  have hwithin : risk pooled -
      (∑ i, CGB.GradientFilter.gainErr (P i) (R i) (P i / (P i + R i))) =
      ∑ i, (P i + R i) * (pooled - P i / (P i + R i)) ^ 2 := by
    rw [show risk pooled = ∑ i, CGB.GradientFilter.gainErr (P i) (R i) pooled from rfl,
      ← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro i _
    rw [CGB.GradientFilter.gain_error_square _ _ pooled (hlocal i),
      CGB.GradientFilter.gain_error_square _ _ (P i / (P i + R i)) (hlocal i)]
    ring
  dsimp only
  change risk gain - _ = _
  linarith only [hbetween, hwithin]

/-- Same-weight direction and observed energy generate a bounded normalized
update. This is empirical energy, not a Hessian/noise estimate. No artificial
epsilon or hard clipping is needed: zero energy gives zero update. The caller
can absorb sqrt(compartment size) into eta to use RMS instead of sum energy.
All allocation/reserve mobilities remain inside the actual final update. -/
theorem observer_energy_and_step {n : ℕ} (mean observed mobility : Fin n → ℝ)
    (gain energy η : ℝ) (hg0 : 0 ≤ gain) (hg1 : gain ≤ 1)
    (he : (∑ i, (mean i)^2) ≤ energy)
    (ha : ∀ i, 0 ≤ mobility i ∧ mobility i ≤ 1) :
    let nextMean := fun i => (1 - gain) * mean i + gain * observed i
    let nextEnergy := (1 - gain) * energy + gain * ∑ i, (observed i)^2
    let u := fun i => if nextEnergy = 0 then 0 else
      η * mobility i * nextMean i / Real.sqrt nextEnergy
    (0 ≤ nextEnergy ∧ (∑ i, (nextMean i)^2) ≤ nextEnergy) ∧
      (∑ i, (u i)^2) ≤ η^2 := by
  dsimp only
  let m := fun i => (1 - gain) * mean i + gain * observed i
  let e := (1 - gain) * energy + gain * ∑ i, (observed i)^2
  have he0 : 0 ≤ energy := (Finset.sum_nonneg (fun i _ => sq_nonneg (mean i))).trans he
  have enonneg : 0 ≤ e := by dsimp [e]; positivity
  have mbound : (∑ i, (m i)^2) ≤ e := by
    calc
      _ ≤ ∑ i, ((1 - gain) * (mean i)^2 + gain * (observed i)^2) := by
        apply Finset.sum_le_sum
        intro i _
        have hp := mul_nonneg (mul_nonneg hg0 (show 0 ≤ 1 - gain by linarith))
          (sq_nonneg (mean i - observed i))
        dsimp [m]
        nlinarith
      _ ≤ e := by
        simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
        exact add_le_add (mul_le_mul_of_nonneg_left he (by linarith)) le_rfl
  refine ⟨⟨enonneg, mbound⟩, ?_⟩
  change (∑ i, (if e = 0 then 0 else η * mobility i * m i / Real.sqrt e)^2) ≤ η^2
  by_cases hz : e = 0
  · simp only [hz, ite_true, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
      Finset.sum_const_zero]
    exact sq_nonneg η
  · have ep : 0 < e := lt_of_le_of_ne enonneg (Ne.symm hz)
    have hweighted : (∑ i, (mobility i)^2 * (m i)^2) ≤ e := by
      apply le_trans (Finset.sum_le_sum (fun i _ => ?_)) mbound
      have hs : (mobility i)^2 ≤ 1 := by nlinarith [(ha i).1, (ha i).2]
      nlinarith [sq_nonneg (m i), mul_nonneg (show 0 ≤ 1 - (mobility i)^2 by linarith)
        (sq_nonneg (m i))]
    simp only [ite_eq_right hz, div_pow, mul_pow, Real.sq_sqrt enonneg, mul_assoc]
    rw [← Finset.sum_div, ← Finset.mul_sum]
    apply (div_le_iff₀ ep).mpr
    exact mul_le_mul_of_nonneg_left hweighted (sq_nonneg η)

end CGB.OptimizerAdmissionReview

#print axioms CGB.OptimizerAdmissionReview.normalized_observation_clock
#print axioms CGB.OptimizerAdmissionReview.pooled_gain_risk
#print axioms CGB.OptimizerAdmissionReview.observer_energy_and_step

/-- Executable normalized observation/energy reference. Energy is mean
squared magnitude here; sqrt is kept outside the rational oracle. No model
or optimizer is trained. Inputs in main have matching nonempty lists. -/
def onlineObserverReference (rate available mass energy : ℚ) (mean observed : List ℚ) :
    ℚ × ℚ × ℚ × List ℚ :=
  let α := rate * available
  let nextMass := (1 - α) * mass + α
  let gain := α / nextMass
  let nextMean := mean.zipWith (fun m g => (1 - gain) * m + gain * g) observed
  let sampleEnergy := (observed.map (fun g => g^2)).sum / (observed.length : ℚ)
  (gain, nextMass, (1 - gain) * energy + gain * sampleEnergy, nextMean)

def main : IO Unit := do
  for reserve in [(8 : ℤ), 12, 16] do
    for sign in [(-1 : ℤ), 1] do
      for rate in [(1 / 8 : ℚ), 1 / 2, 1] do
        for mass in [(0 : ℚ), 1 / 2, 1] do
          let mean := (List.range 8).map (fun i =>
            if mass = 0 then (0 : ℚ) else (sign : ℚ) * (((i % 4 : ℕ) : ℚ) + 1) / 2)
          let energy := 2 * (mean.map (fun m => m^2)).sum / 8
          let z := onlineObserverReference rate (((reserve : ℚ) + 16) / 32) mass energy mean
            [1, 2, -3, -4, 2, 1, -4, -3]
          let frac := fun x : ℚ => s!"{x.num}/{x.den}"
          let fields := [toString reserve, toString sign, frac rate, frac mass,
            frac z.1, frac z.2.1, frac z.2.2.1] ++ z.2.2.2.map frac
          IO.println ("observer\t" ++ String.intercalate "\t" fields)

-- Positive endpoint derivative alignment is not an observed loss reduction.
example : let f := fun x : ℝ => (x ^ 2 - 1) ^ 2 + x / 10
    let df := fun x : ℝ => 4 * x ^ 3 - 4 * x + 1 / 10
    0 < df 1 * df 0 ∧ f 1 < f 0 := by
  norm_num

-- A doubly stochastic 2x2 allocation can reverse the aggregate sign.
example : (0 : ℝ) < -1 + 2 ∧
    (9 / 5 : ℝ) + 1 / 5 = 2 ∧
    (9 / 5 : ℝ) * (-1) + (1 / 5) * 2 < 0 := by
  norm_num
