import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import CGBAxioms.GenerationSelfProduction
import CGBAxioms.GenerationReinvestment
import CGBAxioms.GenerationContactCompetition

/-! V1 implementation admission: signed content itself consumes row/column
capacity. This is a capacity retraction, NOT the equal-marginal allocation of
GenerationContinualOptimizer, a bistability proof, or an organism certificate.
No extra persistent allocation array is assumed. Exact-real reference only. -/
namespace BudgetedAssociationWriteReview
open Finset

noncomputable def mass {n : ℕ} (u : Fin n → ℝ) : ℝ := ∑ j, |u j|

noncomputable def admit {r c : ℕ} (b : ℝ) (u : Fin r → Fin c → ℝ)
    (i : Fin r) (j : Fin c) : ℝ :=
  u i j * (b / max b (max (mass (u i)) (mass (fun k => u k j))))

private theorem slice_bound {n : ℕ} (b : ℝ) (hb : 0 < b)
    (u d : Fin n → ℝ) (hd : ∀ j, max b (mass u) ≤ d j) :
    ∑ j, |u j * (b / d j)| ≤ b := by
  have hm : 0 < max b (mass u) := lt_of_lt_of_le hb (le_max_left _ _)
  calc
    ∑ j, |u j * (b / d j)| = ∑ j, |u j| * (b / d j) := by
      apply sum_congr rfl
      intro j _
      rw [abs_mul, abs_of_nonneg (div_nonneg (le_of_lt hb)
        (le_trans (le_of_lt hm) (hd j)))]
    _ ≤ ∑ j, |u j| * (b / max b (mass u)) := by
      apply sum_le_sum
      intro j _
      exact mul_le_mul_of_nonneg_left
        (div_le_div_of_nonneg_left (le_of_lt hb) hm (hd j)) (abs_nonneg _)
    _ = mass u * (b / max b (mass u)) := by rw [← sum_mul]; rfl
    _ ≤ max b (mass u) * (b / max b (mass u)) :=
      mul_le_mul_of_nonneg_right (le_max_right _ _) (by positivity)
    _ = b := by field_simp

theorem budgets {r c : ℕ} (b : ℝ) (hb : 0 < b) (u : Fin r → Fin c → ℝ) :
    (∀ i, mass (admit b u i) ≤ b) ∧
    (∀ j, mass (fun i => admit b u i j) ≤ b) := by
  constructor
  · intro i
    apply slice_bound b hb (u i)
    intro j
    exact max_le_max_left b (le_max_left _ _)
  · intro j
    apply slice_bound b hb (fun i => u i j)
    intro i
    exact max_le_max_left b (le_max_right _ _)

theorem feasible_fixed {r c : ℕ} (b : ℝ) (hb : 0 < b)
    (u : Fin r → Fin c → ℝ)
    (hr : ∀ i, mass (u i) ≤ b) (hc : ∀ j, mass (fun i => u i j) ≤ b) :
    admit b u = u := by
  funext i j
  simp [admit, max_eq_left (max_le (hr i) (hc j)), ne_of_gt hb]

/-- Slack is determined by actual content; it is not a second food account.
Both endpoint budgets count the same contacts, so they must not be added as
two independent stores of physical mass. -/
theorem implicit_slack {n : ℕ} (b : ℝ) (u : Fin n → ℝ) (h : mass u ≤ b) :
    0 ≤ b - mass u ∧ mass u + (b - mass u) = b := by
  constructor <;> linarith

/-- Actual post-admission read equals the proposed read plus its correction.
Consequently the old delta-only cross-query formula must include this term. -/
theorem read_correction {r c : ℕ} (b : ℝ) (u : Fin r → Fin c → ℝ)
    (q : Fin c → ℝ) (i : Fin r) :
    (∑ j, admit b u i j * q j) =
      (∑ j, u i j * q j) + ∑ j, (admit b u i j - u i j) * q j := by
  simp only [sub_mul, sum_sub_distrib]
  ring

#print axioms budgets
#print axioms feasible_fixed
#print axioms implicit_slack
#print axioms read_correction

/-! Joint weighted-content exchange. a is part of memory, M_i=a_i*c_i.
The posterior may be state- and boundary-dependent; its normalization is an
explicit premise here, implemented by softmax. Gaussian latent likelihood and
nonlinear bifurcation are not claimed as Lean results by this reference. -/
def nextShare {r : ℕ} (a p : Fin r → ℝ) (γ : ℝ) (i : Fin r) : ℝ :=
  (1-γ) * a i + γ * p i

def nextContent {r c : ℕ} (m : Fin r → Fin c → ℝ) (p : Fin r → ℝ)
    (v : Fin c → ℝ) (γ : ℝ) (i : Fin r) (j : Fin c) : ℝ :=
  (1-γ) * m i j + γ * p i * v j

theorem joint_exchange_budget {r c : ℕ} (a p : Fin r → ℝ)
    (m : Fin r → Fin c → ℝ) (v : Fin c → ℝ) (γ b : ℝ)
    (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (ha : ∀ i, 0 ≤ a i) (hp : ∀ i, 0 ≤ p i)
    (ha1 : ∑ i, a i = 1) (hp1 : ∑ i, p i = 1)
    (hm : ∀ i, mass (m i) ≤ a i * b) (hv : mass v ≤ b) :
    (∑ i, nextShare a p γ i = 1) ∧
    (∀ i, 0 ≤ nextShare a p γ i) ∧
    (∀ i, mass (nextContent m p v γ i) ≤ nextShare a p γ i * b) := by
  have hretain : 0 ≤ 1-γ := by linarith
  refine ⟨?_, ?_, ?_⟩
  · simp [nextShare, sum_add_distrib, ← mul_sum, ha1, hp1]
  · intro i
    exact add_nonneg (mul_nonneg hretain (ha i)) (mul_nonneg hγ0 (hp i))
  · intro i
    calc
      mass (nextContent m p v γ i) ≤
          ∑ j, ((1-γ) * |m i j| + γ * p i * |v j|) := by
        apply sum_le_sum
        intro j _
        calc
          |nextContent m p v γ i j| ≤ |(1-γ)*m i j| + |γ*p i*v j| := abs_add_le _ _
          _ = _ := by rw [abs_mul, abs_mul, abs_of_nonneg hretain,
            abs_of_nonneg (mul_nonneg hγ0 (hp i))]
      _ = (1-γ) * mass (m i) + γ*p i*mass v := by
        simp [mass, sum_add_distrib, mul_sum]
      _ ≤ (1-γ)*(a i*b) + γ*p i*b := add_le_add
        (mul_le_mul_of_nonneg_left (hm i) hretain)
        (mul_le_mul_of_nonneg_left hv (mul_nonneg hγ0 (hp i)))
      _ = nextShare a p γ i * b := by unfold nextShare; ring

theorem actual_weighted_read {r c : ℕ} (m : Fin r → Fin c → ℝ)
    (p q : Fin r → ℝ) (v : Fin c → ℝ) (γ : ℝ) (j : Fin c) :
    (∑ i, q i * nextContent m p v γ i j) =
      (1-γ)*(∑ i, q i*m i j) + γ*(∑ i, q i*p i)*v j := by
  simp only [nextContent, mul_add, sum_add_distrib]
  simp only [mul_sum, sum_mul]
  congr 1 <;> apply sum_congr rfl <;> intro i _ <;> ring

/-- A fixed allocation with matching content is a fixed *joint* memory.
Existence/stability of a particular nonlinear posterior must be checked too. -/
theorem joint_fixed_lift {r c : ℕ} (a : Fin r → ℝ) (v : Fin c → ℝ) (γ : ℝ) :
    nextShare a a γ = a ∧
    nextContent (fun i j => a i*v j) a v γ = (fun i j => a i*v j) := by
  constructor
  · funext i; unfold nextShare; ring
  · funext i j; unfold nextContent; ring

theorem exchanged_gains_equal_losses {r : ℕ} (a p : Fin r → ℝ) (γ : ℝ)
    (ha : ∑ i, a i=1) (hp : ∑ i, p i=1) :
    (∑ i, max (nextShare a p γ i-a i) 0) =
      ∑ i, max (a i-nextShare a p γ i) 0 := by
  apply CGB.Reinvestment.gains_eq_losses
  simp [nextShare, sum_add_distrib, ← mul_sum, ha, hp]

#print axioms joint_exchange_budget
#print axioms actual_weighted_read
#print axioms joint_fixed_lift
#print axioms exchanged_gains_equal_losses

/-! V1.9: allocation retention is not content retention.  These statements
diagnose the executed V1.7 rule and admit a conditional contact-local repair.
No stability, finite-precision retrieval, or language advantage is assumed. -/

/-- Even with exactly fixed allocation, differences between conditional row
contents decay at the same global rate. Positive shares allow decoding M/a. -/
theorem fixed_allocation_content_contrast (a b x y v γ : ℝ)
    (ha : a ≠ 0) (hb : b ≠ 0) :
    ((1-γ)*(a*x)+γ*a*v)/a - ((1-γ)*(b*y)+γ*b*v)/b =
      (1-γ)*(x-y) := by
  field_simp
  ring

/-- Contact-local innovation; the local rate is computed from the existing
posterior, not stored in another history or resource page. -/
def contactWrite {c : ℕ} (x v : Fin c → ℝ) (rate : ℝ) (j : Fin c) : ℝ :=
  (1-rate)*x j + rate*v j

/-- The same posterior distributes a finite total write rate. Each conditional
content stays in its old L1 ball. This constrains write capacity, not information
bits, and is a construction choice rather than a universally forced equation. -/
theorem selective_contact_budget {r c : ℕ} (p : Fin r → ℝ)
    (x : Fin r → Fin c → ℝ) (v : Fin c → ℝ) (γ b : ℝ)
    (hp : ∀ i, 0 ≤ p i) (hp1 : ∑ i, p i = 1)
    (hγ : 0 ≤ γ ∧ γ ≤ 1) (hx : ∀ i, mass (x i) ≤ b) (hv : mass v ≤ b) :
    (∑ i, γ*p i = γ) ∧
    (∀ i, 0 ≤ γ*p i ∧ γ*p i ≤ 1) ∧
    (∀ i, mass (contactWrite (x i) v (γ*p i)) ≤ b) := by
  have hrate : ∀ i, 0 ≤ γ*p i ∧ γ*p i ≤ 1 := by
    intro i
    have hi : p i ≤ 1 := by
      calc p i ≤ ∑ k, p k := single_le_sum (fun k _ => hp k) (mem_univ i)
           _ = 1 := hp1
    exact ⟨mul_nonneg hγ.1 (hp i),
      (mul_le_mul hγ.2 hi (hp i) (by norm_num)).trans_eq (one_mul 1)⟩
  refine ⟨by rw [← mul_sum, hp1, mul_one], hrate, ?_⟩
  intro i
  have hr := hrate i
  calc
    mass (contactWrite (x i) v (γ*p i)) ≤
        ∑ j, ((1-γ*p i)*|x i j| + (γ*p i)*|v j|) := by
      apply sum_le_sum
      intro j _
      calc
        _ ≤ |(1-γ*p i)*x i j| + |(γ*p i)*v j| := abs_add_le _ _
        _ = _ := by rw [abs_mul, abs_mul,
          abs_of_nonneg (sub_nonneg.mpr hr.2), abs_of_nonneg hr.1]
    _ = (1-γ*p i)*mass (x i) + (γ*p i)*mass v := by
      simp [mass, sum_add_distrib, mul_sum]
    _ ≤ (1-γ*p i)*b + (γ*p i)*b := add_le_add
      (mul_le_mul_of_nonneg_left (hx i) (sub_nonneg.mpr hr.2))
      (mul_le_mul_of_nonneg_left hv hr.1)
    _ = b := by ring

/-- Inactive contacts hold, matching content does not move, and histories
under equal rates and equal incoming values retain their difference locally. -/
theorem contact_write_retention {c : ℕ} (x y v : Fin c → ℝ) (rate : ℝ) :
    contactWrite x v 0 = x ∧ contactWrite v v rate = v ∧
    (∀ j, contactWrite x v rate j - contactWrite y v rate j =
      (1-rate)*(x j-y j)) := by
  refine ⟨?_, ?_, ?_⟩
  · funext j; simp [contactWrite]
  · funext j; unfold contactWrite; ring
  · intro j; unfold contactWrite; ring

/-- Keeping conditional content does not itself keep the actual weighted read:
changing shares contributes an additional term, even when content holds. -/
theorem selective_actual_read {r c : ℕ} (a a' q rate : Fin r → ℝ)
    (x : Fin r → Fin c → ℝ) (v : Fin c → ℝ) (j : Fin c) :
    (∑ i, q i*a' i*contactWrite (x i) v (rate i) j) -
      (∑ i, q i*a i*x i j) =
    (∑ i, q i*(a' i-a i)*x i j) +
      ∑ i, q i*a' i*rate i*(v j-x i j) := by
  rw [← sum_sub_distrib, ← sum_add_distrib]
  apply sum_congr rfl
  intro i _
  unfold contactWrite
  ring

#print axioms fixed_allocation_content_contrast
#print axioms selective_contact_budget
#print axioms contact_write_retention
#print axioms selective_actual_read

/-- Store conditional content directly, decode the actual weighted content at
read time. This avoids division by tiny shares without changing the weighted
content budget. Conditional content at zero share is retained, so converting
an old weighted checkpoint there is not losslessly invertible. -/
theorem decoded_content_budget {r c : ℕ} (a : Fin r → ℝ)
    (x : Fin r → Fin c → ℝ) (b : ℝ)
    (ha : ∀ i, 0 ≤ a i) (ha1 : ∑ i, a i = 1)
    (hx : ∀ i, mass (x i) ≤ b) :
    (∀ i, mass (fun j => a i*x i j) ≤ a i*b) ∧
    (∑ i, mass (fun j => a i*x i j)) ≤ b := by
  have hm : ∀ i, mass (fun j => a i*x i j) ≤ a i*b := by
    intro i
    simp only [mass, abs_mul, abs_of_nonneg (ha i), ← mul_sum]
    exact mul_le_mul_of_nonneg_left (hx i) (ha i)
  refine ⟨hm, ?_⟩
  calc
    (∑ i, mass (fun j => a i*x i j)) ≤ ∑ i, a i*b := sum_le_sum (fun i _ => hm i)
    _ = b := by rw [← sum_mul, ha1, one_mul]

#print axioms decoded_content_budget

/-! The SAME share supplies availability. These are contact-level constraints,
not a claim that all complete learned trajectories avoid concentration. -/
def availableWeight {r : ℕ} (a score : Fin r → ℝ) (i : Fin r) : ℝ :=
  score i * (1-a i)

noncomputable def availablePosterior {r : ℕ} (a score : Fin r → ℝ) (i : Fin r) : ℝ :=
  availableWeight a score i / ∑ k, availableWeight a score k

theorem simplex_has_available_contact {r : ℕ} (hr : 2 ≤ r)
    (a : Fin r → ℝ) (ha1 : ∑ i, a i = 1) : ∃ i, a i < 1 := by
  by_contra hn
  push Not at hn
  have hsum : (r : ℝ) ≤ ∑ i, a i := by
    calc (r : ℝ) = ∑ _i : Fin r, (1 : ℝ) := by simp
         _ ≤ ∑ i, a i := sum_le_sum (fun i _ => hn i)
  rw [ha1] at hsum
  have hr' : (2 : ℝ) ≤ r := by exact_mod_cast hr
  linarith

/-- At fixed incoming evidence, an occupied contact has less admission room.
The existing shared-pool result supplies this inequality. State-dependent
evidence can oppose it; local anti-dominance is not global convergence. -/
theorem available_weight_anti (score a b : ℝ) (hs : 0 < score) (hab : a < b) :
    score*(1-b) < score*(1-a) := by
  simpa using CGB.Generation.shared_pool_marginal_admission_is_anti_dominant
    1 score 1 a b (by norm_num) hs (by norm_num) hab

theorem available_posterior_contract {r : ℕ} (a score : Fin r → ℝ)
    (ha : ∀ i, a i ≤ 1) (hs : ∀ i, 0 < score i) (hroom : ∃ i, a i < 1) :
    (∀ i, 0 ≤ availablePosterior a score i) ∧
    (∑ i, availablePosterior a score i = 1) ∧
    (∀ i, a i = 1 → availablePosterior a score i = 0) := by
  have hw : ∀ i, 0 ≤ availableWeight a score i := fun i =>
    mul_nonneg (le_of_lt (hs i)) (sub_nonneg.mpr (ha i))
  have hz : 0 < ∑ i, availableWeight a score i := by
    obtain ⟨i, hi⟩ := hroom
    exact (mul_pos (hs i) (sub_pos.mpr hi)).trans_le
      (single_le_sum (fun j _ => hw j) (mem_univ i))
  refine ⟨fun i => div_nonneg (hw i) (le_of_lt hz), ?_, ?_⟩
  · simp only [availablePosterior, ← sum_div]
    exact div_self (ne_of_gt hz)
  · intro i hi
    simp [availablePosterior, availableWeight, hi]

/-- Joint admission for the implemented availability-controlled update:
allocation conservation, local write budget, bounded conditional/actual
content, and release of a fully occupied contact all refer to one posterior.
This is not a healthy-language-latch or microscopic-Medium certificate. -/
theorem available_joint_contract {r c : ℕ} (hr : 2 ≤ r)
    (a score : Fin r → ℝ) (x : Fin r → Fin c → ℝ) (v : Fin c → ℝ) (γ b : ℝ)
    (ha : ∀ i, 0 ≤ a i) (ha1 : ∑ i, a i = 1) (hs : ∀ i, 0 < score i)
    (hγ : 0 ≤ γ ∧ γ ≤ 1) (hx : ∀ i, mass (x i) ≤ b) (hv : mass v ≤ b) :
    let p := availablePosterior a score
    let a' := nextShare a p γ
    let x' := fun i => contactWrite (x i) v (γ*p i)
    (∑ i, a' i = 1) ∧ (∀ i, 0 ≤ a' i) ∧ (∑ i, γ*p i = γ) ∧
    (∀ i, mass (x' i) ≤ b) ∧ (∑ i, mass (fun j => a' i*x' i j)) ≤ b ∧
    (∀ i, a i = 1 → a' i = 1-γ) := by
  dsimp only
  have hau : ∀ i, a i ≤ 1 := fun i =>
    (single_le_sum (fun j _ => ha j) (mem_univ i)).trans_eq ha1
  obtain ⟨hp, hp1, hfull⟩ := available_posterior_contract a score hau hs
    (simplex_has_available_contact hr a ha1)
  have hb := selective_contact_budget (availablePosterior a score) x v γ b hp hp1 hγ hx hv
  have han : ∀ i, 0 ≤ nextShare a (availablePosterior a score) γ i := fun i =>
    add_nonneg (mul_nonneg (sub_nonneg.mpr hγ.2) (ha i)) (mul_nonneg hγ.1 (hp i))
  have han1 : ∑ i, nextShare a (availablePosterior a score) γ i = 1 := by
    simp [nextShare, sum_add_distrib, ← mul_sum, ha1, hp1]
  refine ⟨han1, han, hb.1, hb.2.2,
    (decoded_content_budget _ _ b han han1 hb.2.2).2, ?_⟩
  intro i hi
  simp [nextShare, hi, hfull i hi]

#print axioms simplex_has_available_contact
#print axioms available_weight_anti
#print axioms available_posterior_contract
#print axioms available_joint_contract

/-! The actual memory can mediate a CURRENT learning signal after the old
autograd graph is gone. `signal` is the arrived loss's covector at the readout,
evaluated at the current point, not a reconstructed historical parameter
gradient. No new memory or eligibility array is introduced. -/
noncomputable def memoryScore {r c : ℕ}
    (m : Fin r → Fin c → ℝ) (q : Fin r → ℝ) (signal : Fin c → ℝ) : ℝ :=
  ∑ i, q i * (∑ j, m i j * signal j)

/-- Same stored content used for prediction is the coefficient of current
query learning. A changing query parameter must still apply its own chain rule. -/
theorem memory_read_query_direction {r c : ℕ}
    (m : Fin r → Fin c → ℝ) (q direction : Fin r → ℝ) (signal : Fin c → ℝ) :
    HasDerivAt (fun ε => memoryScore m (fun i => q i + ε * direction i) signal)
      (memoryScore m direction signal) 0 := by
  have hlinear : HasDerivAt
      (fun ε : ℝ => memoryScore m q signal + ε * memoryScore m direction signal)
      (memoryScore m direction signal) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const
      (memoryScore m direction signal)).const_add (memoryScore m q signal)
  convert hlinear using 1
  funext ε
  simp [memoryScore, add_mul, sum_add_distrib, mul_sum, mul_assoc]

/-- A current readout alone can lose a distinction which the same full
memory still supplies to current query learning. This is an algebraic
countermodel, not a language-capability study or a claim about all past credit. -/
theorem equal_read_different_current_teaching :
    ∃ (m n : Fin 2 → Fin 1 → ℝ) (q direction : Fin 2 → ℝ) (signal : Fin 1 → ℝ),
      memoryScore m q signal = memoryScore n q signal ∧
      memoryScore m direction signal = 1 ∧
      memoryScore n direction signal = -1 := by
  refine ⟨![![0], ![1]], ![![0], ![-1]], ![1, 0], ![0, 1], ![1], ?_⟩
  norm_num [memoryScore, Fin.sum_univ_two]

#print axioms memory_read_query_direction
#print axioms equal_read_different_current_teaching

/-- One actual decay-then-delta update simultaneously rewrites its selected
key and accounts for retention at a possibly MOVING query. Apply columnwise
to the runtime key-by-value matrix. The last bound separates global decay,
overlapping write, and retrieval drift; it assumes no semantic memory labels.
This is passive selective storage, not regenerative ignition or a learned
guarantee that future queries remain suitable. -/
theorem decayed_write_joint_retention_retrieval {n : ℕ}
    (k q q' x : Fin n → ℝ) (α β v : ℝ)
    (hk : CGB.Generation.dot k k = 1) :
    let y := CGB.Generation.tiedWrite k β v (fun i => α * x i)
    let drift := CGB.Generation.dot (fun i => q' i - q i) y
    (CGB.Generation.dot k y - v =
      (1 - β) * (α * CGB.Generation.dot k x - v)) ∧
    (CGB.Generation.dot q' y - CGB.Generation.dot q x =
      (α - 1) * CGB.Generation.dot q x +
      β * (v - α * CGB.Generation.dot k x) * CGB.Generation.dot q k + drift) ∧
    |CGB.Generation.dot q' y - CGB.Generation.dot q x| ≤
      |α - 1| * |CGB.Generation.dot q x| +
      |β| * |v - α * CGB.Generation.dot k x| * |CGB.Generation.dot q k| + |drift| := by
  dsimp only
  have scale (u : Fin n → ℝ) :
      CGB.Generation.dot u (fun i => α * x i) = α * CGB.Generation.dot u x := by
    simp only [CGB.Generation.dot, mul_sum]
    exact sum_congr rfl fun i _ => by ring
  have change := CGB.Generation.tied_write_cross_read k q (fun i => α * x i) β v
  rw [scale q, scale k] at change
  have move (y : Fin n → ℝ) :
      CGB.Generation.dot (fun i => q' i - q i) y =
      CGB.Generation.dot q' y - CGB.Generation.dot q y := by
    simp [CGB.Generation.dot, sub_mul, sum_sub_distrib]
  have total :
      CGB.Generation.dot q' (CGB.Generation.tiedWrite k β v (fun i => α * x i)) -
      CGB.Generation.dot q x =
      (α - 1) * CGB.Generation.dot q x +
      β * (v - α * CGB.Generation.dot k x) * CGB.Generation.dot q k +
      CGB.Generation.dot (fun i => q' i - q i)
        (CGB.Generation.tiedWrite k β v (fun i => α * x i)) := by
    rw [move]
    nlinarith [change]
  refine ⟨?_, total, ?_⟩
  · simpa only [scale k] using
      CGB.Generation.tied_write_error_shrinks k (fun i => α * x i) hk β v
  · rw [total]
    have triangle (a b c : ℝ) : |a + b + c| ≤ |a| + |b| + |c| :=
      (abs_add_le _ _).trans (add_le_add (abs_add_le _ _) (le_refl _))
    simpa only [abs_mul] using triangle
      ((α - 1) * CGB.Generation.dot q x)
      (β * (v - α * CGB.Generation.dot k x) * CGB.Generation.dot q k)
      (CGB.Generation.dot (fun i => q' i - q i)
        (CGB.Generation.tiedWrite k β v (fun i => α * x i)))

#print axioms decayed_write_joint_retention_retrieval

/-! A CONDITIONAL read-back construction, not a new installed language body.
The same matrix supplies both the current read and its renewal target. The
rational saturation below is a finite algebraic witness of joint feasibility,
not a necessity claim or a recommendation to fix learned gates at these values.
It is not an extra food, allocation, or latch state. -/
noncomputable def softResponse (z : ℝ) : ℝ := z / (1 + |z|)

noncomputable def feedbackRead (α β g u z : ℝ) : ℝ :=
  α * (1 - β) * z + β * softResponse (g * z + u)

/-- Read-back is a value choice INSIDE the same delta write. With a unit key,
the selected coordinate follows the nonlinear renewal map exactly. This says
nothing about unselected content capacity, learned key alignment, or P2a. -/
theorem feedback_write_read {n : ℕ} (k x : Fin n → ℝ) (α β g u : ℝ)
    (hk : CGB.Generation.dot k k = 1) :
    CGB.Generation.dot k (CGB.Generation.tiedWrite k β
      (softResponse (g * CGB.Generation.dot k x + u)) (fun i => α * x i)) =
    feedbackRead α β g u (CGB.Generation.dot k x) := by
  have h := CGB.Generation.tied_write_error_shrinks k (fun i => α * x i) hk β
    (softResponse (g * CGB.Generation.dot k x + u))
  have scale : CGB.Generation.dot k (fun i => α * x i) =
      α * CGB.Generation.dot k x := by
    simp only [CGB.Generation.dot, mul_sum]
    exact sum_congr rfl fun i _ => by ring
  rw [scale] at h
  unfold feedbackRead
  nlinarith [h]

/-- One common nonlinear update has bounded state, two restoring basins,
finite-cue release in both directions, and an observable response while held.
The nonzero rests survive alpha=1/2 passive decay. Input of either sign, magnitude 8 is an explicit
cue reachability condition, not proof that language supplies such evidence.
No claim of language usefulness or full-medium admission follows. -/
theorem feedback_renewal_joint_contract :
    let f := feedbackRead (1/2) (1/2) 4
    let m : ℝ := 5/12
    (∀ u x, |x| ≤ 1 → |f u x| ≤ 1) ∧
    (f 0 m = m ∧ f 0 (-m) = -m) ∧
    (∀ x, |x-m| ≤ 1/4 → |f 0 x-m| ≤ (7/10)*|x-m|) ∧
    (∀ x, |x+m| ≤ 1/4 → |f 0 x+m| ≤ (7/10)*|x+m|) ∧
    (∀ x, |x-m| ≤ 1/4 → |f (-8) x+m| ≤ 1/4) ∧
    (∀ x, |x+m| ≤ 1/4 → |f 8 x-m| ≤ 1/4) ∧
    (f (1/8) m ≠ f 0 m ∧ |f (1/8) m-m| ≤ 1/4) := by
  dsimp only
  let f := feedbackRead (1/2) (1/2) 4
  have form (u x : ℝ) : f u x = x/4 + softResponse (4*x+u)/2 := by
    dsimp [f, feedbackRead]; ring
  have sat (z : ℝ) : |softResponse z| ≤ 1 := by
    rw [softResponse, abs_div, abs_of_pos (by positivity : 0 < 1+|z|)]
    exact (div_le_iff₀ (by positivity : 0 < 1+|z|)).mpr (by linarith)
  have odd (u x : ℝ) : f (-u) (-x) = -f u x := by
    simp only [f, feedbackRead, softResponse]
    rw [show 4 * -x + -u = -(4*x+u) by ring, abs_neg]
    ring
  have restore (x : ℝ) (hx : |x-5/12| ≤ 1/4) :
      |f 0 x-5/12| ≤ (7/10)*|x-5/12| := by
    have bounds := abs_le.mp hx
    have xp : 0 ≤ 4*x := by linarith
    have den : 0 < 1+4*x := by linarith
    have ident : f 0 x-5/12 = (x-5/12)*(1/4+(3/4)/(1+4*x)) := by
      rw [form]
      simp only [add_zero, softResponse, abs_of_nonneg xp]
      field_simp [ne_of_gt den, show (1+x*4 : ℝ) ≠ 0 by nlinarith]
      ring
    have lo : 0 ≤ 1/4+(3/4)/(1+4*x) := by positivity
    have hi : 1/4+(3/4)/(1+4*x) ≤ (7/10 : ℝ) := by
      have h : (3/4 : ℝ)/(1+4*x) ≤ 9/20 := by
        apply (div_le_iff₀ den).mpr; linarith
      linarith
    rw [ident, abs_mul, abs_of_nonneg lo]
    nlinarith [abs_nonneg (x-5/12)]
  have release (x : ℝ) (hx : |x-5/12| ≤ 1/4) : |f (-8) x+5/12| ≤ 1/4 := by
    have bounds := abs_le.mp hx
    have drive : 4*x + (-8) ≤ -2 := by linarith
    have den : 0 < 1+|4*x+(-8)| := by positivity
    have hi : softResponse (4*x+(-8)) ≤ -2/3 := by
      apply (div_le_iff₀ den).mpr
      rw [abs_of_nonpos (by linarith : 4*x+(-8) ≤ 0)]
      linarith
    have lo := (abs_le.mp (sat (4*x+(-8)))).1
    rw [form]
    apply abs_le.mpr
    constructor <;> linarith
  refine ⟨?_, ?_, restore, ?_, release, ?_, ?_⟩
  · intro u x hx
    change |f u x| ≤ 1
    rw [form]
    calc |x/4+softResponse (4*x+u)/2| ≤ |x/4|+|softResponse (4*x+u)/2| := abs_add_le _ _
      _ ≤ 1 := by
        rw [abs_div, abs_div]
        norm_num
        nlinarith [sat (4*x+u)]
  · norm_num [feedbackRead, softResponse]
  · intro x hx
    have hn : |-x-5/12| ≤ 1/4 := by
      simpa only [show -x-5/12 = -(x+5/12) by ring, abs_neg] using hx
    have h := restore (-x) hn
    have ho := odd 0 x
    simp only [neg_zero] at ho
    rw [ho, show -f 0 x-5/12 = -(f 0 x+5/12) by ring,
      show -x-5/12 = -(x+5/12) by ring, abs_neg, abs_neg] at h
    exact h
  · intro x hx
    have hn : |-x-5/12| ≤ 1/4 := by
      simpa only [show -x-5/12 = -(x+5/12) by ring, abs_neg] using hx
    have h := release (-x) hn
    have ho := odd 8 x
    rw [ho, show -f 8 x+5/12 = -(f 8 x-5/12) by ring, abs_neg] at h
    exact h
  · norm_num [feedbackRead, softResponse]

#print axioms feedback_write_read
#print axioms feedback_renewal_joint_contract

/-- An endogenous squared-error target has a chain-rule term. Freezing that
 target is a different update; neither choice is silently identified with TTT. -/
theorem endogenous_target_error_derivative (f : ℝ → ℝ) (z d : ℝ)
    (hf : HasDerivAt f d z) :
    HasDerivAt (fun x => (x - f x)^2 / 2) ((z - f z) * (1 - d)) z := by
  convert (((hasDerivAt_id z).sub hf).pow 2).div_const 2 using 1
  · rfl
  · dsimp; ring

/-- A mathematical potential, not another stored state or runtime integral. -/
noncomputable def softPrimitive (x : ℝ) : ℝ :=
  ∫ t in (0:ℝ)..x, softResponse t

/-- The response integral generates exactly the selected-coordinate regenerative
 update. Its derivative follows from continuity and FTC, without a new premise.
 Fixed coefficients/cue; no temporal gradient or language-benefit claim. -/
theorem feedback_potential_step (α β g u η v z : ℝ)
    (hβ : β ≠ 0) (hg : g ≠ 0) :
    let c := (1 - α*(1-β)) / β
    let d := c*z - (1-η)*v - η*softResponse (g*z+u)
    HasDerivAt (fun x => c*x^2/2 - (1-η)*v*x - (η/g)*softPrimitive (g*x+u)) d z ∧
    z - β*d = α*(1-β)*z + β*((1-η)*v + η*softResponse (g*z+u)) := by
  dsimp only
  constructor
  · have hc : Continuous softResponse := by
      unfold softResponse
      exact continuous_id.div (continuous_const.add continuous_abs) (fun x => by positivity)
    have hΦ : HasDerivAt softPrimitive (softResponse (g*z+u)) (g*z+u) :=
      (hc.integral_hasStrictDerivAt 0 (g*z+u)).hasDerivAt
    have hp := hΦ.comp z (((hasDerivAt_id z).const_mul g).add_const u)
    have hq := (((hasDerivAt_id z).pow 2).const_mul
      ((1 - α*(1-β))/β)).div_const 2
    convert (hq.sub ((hasDerivAt_id z).const_mul ((1-η)*v))).sub
      (hp.const_mul (η/g)) using 1
    · rfl
    · dsimp; field_simp [hg]
  · field_simp [hβ]
    ring

#print axioms endogenous_target_error_derivative
#print axioms feedback_potential_step

end BudgetedAssociationWriteReview
