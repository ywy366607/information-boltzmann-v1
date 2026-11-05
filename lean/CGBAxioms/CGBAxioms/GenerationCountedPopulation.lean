import CGBAxioms.GenerationReinvestment
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

/-!
# Reinvestment by counted units (language organism, 2026-11-03)

`GenerationReinvestment` treats the shares as real numbers. A body holds countable units, so one
exchange moves whole units. This file reads what follows from the counted form, and states the two
limits that the plain step has on an infinite stream.

* `moran_expected_change`, `exchange_moves_a_unit`: one unit born in proportion to reinvestment and
  one recycled in proportion to share gives the reinvestment step in expectation, and any change of
  counts moves at least one whole unit.
* `extinct_stays_extinct`, `concentrated_absorbing`, `reinvest_ratio`, `ratio_after_le`: an address
  at zero never returns, a population on one address never leaves it, and a losing address falls
  geometrically. The plain step cannot recover on a stream that changes.
* `mix_total`, `mix_floor`, `mixWealth_ge_address`, `recovers_from_floor`: a small transport keeps a
  floor under every address, so any address recovers at a bounded cost after any history.
* `price_equation`: the change of the mean of any carried trait equals the covariance of the trait
  with the multiplier minus one. The multiplier minus one is the innovation at the same event.
* `coarse_predicted`, `coarse_reinvest`: a group of addresses reinvests as one address whose bet is
  its internal prediction. `local_keeps_group_total`, `local_ignores_outside`: when the exchange is
  normalised inside each group, each group keeps its total and reads only its own bets.
* `tag_fraction_eq_posterior`: a tag carried forward with the mass gives the exact posterior of the
  tag given every later byte, with no backward pass.
* `tag_is_exact_credit`: that same fraction is the derivative of the log likelihood of every later
  byte with respect to mass added at the tagged event.
* `bracket_after`: mixing shrinks what a tag can still learn at the rate `ε / n`, so the transport
  that gives recovery also bounds the credit horizon.

Nothing here is about a body. The bets are fixed, the transport is the uniform one except in the
tag theorem, and how a table realises these steps is a separate obligation.
-/

namespace CGB.CountedPopulation

open Finset CGB.Reinvestment

variable {A : Type*} [Fintype A]

section Counted

variable [DecidableEq A]

/-- One birth in proportion to reinvestment and one recycling in proportion to share: the mean
change of the share of `i` is the reinvestment step. -/
theorem moran_expected_change (a w : A → ℝ) (ha : ∑ i, a i = 1) (h : predicted a w ≠ 0) (i : A) :
    ∑ j, ∑ k, reinvest a w j * a k *
        ((if j = i then (1 : ℝ) else 0) - (if k = i then (1 : ℝ) else 0)) =
      reinvest a w i - a i := by
  have hterm : ∀ j k, reinvest a w j * a k *
      ((if j = i then (1 : ℝ) else 0) - (if k = i then (1 : ℝ) else 0)) =
      (if j = i then reinvest a w j else 0) * a k -
        reinvest a w j * (if k = i then a k else 0) := fun j k => by
    by_cases hj : j = i <;> by_cases hk : k = i <;> simp [hj, hk]
  simp_rw [hterm, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.sum_mul, ha]
  simp [reinvest_total a w h]

end Counted

/-- Counts of a fixed total change only by moving at least one whole unit. -/
theorem exchange_moves_a_unit (n n' : A → ℕ) (ht : ∑ i, n i = ∑ i, n' i) (hne : n ≠ n') :
    1 ≤ ∑ i, (n' i - n i) := by
  have hex : ∃ i, n i < n' i := by
    by_contra hno
    simp only [not_exists, not_lt] at hno
    obtain ⟨i, hi⟩ := Function.ne_iff.mp hne
    have hlt : n' i < n i := lt_of_le_of_ne (hno i) (Ne.symm hi)
    have := Finset.sum_lt_sum (s := Finset.univ) (fun j _ => hno j) ⟨i, Finset.mem_univ i, hlt⟩
    omega
  obtain ⟨i, hi⟩ := hex
  calc 1 ≤ n' i - n i := by omega
    _ ≤ ∑ j, (n' j - n j) :=
      Finset.single_le_sum (f := fun j => n' j - n j) (fun _ _ => Nat.zero_le _) (Finset.mem_univ i)

/-- An address at zero stays at zero. New addresses cannot come from reinvestment. -/
theorem extinct_stays_extinct (a w : A → ℝ) (i : A) (hi : a i = 0) : reinvest a w i = 0 := by
  simp [reinvest, hi]

/-- A population on one address never leaves it. -/
theorem concentrated_absorbing [DecidableEq A] (w : A → ℝ) (c : A) (hc : w c ≠ 0) (i : A) :
    reinvest (Pi.single c 1) w i = (Pi.single c 1 : A → ℝ) i := by
  have hp : predicted (Pi.single c 1) w = w c := by
    unfold predicted
    rw [Finset.sum_eq_single c]
    · simp
    · intro j _ hj; simp [hj]
    · simp
  have hne : predicted (Pi.single c 1) w ≠ 0 := by rw [hp]; exact hc
  refine ((no_exchange_iff _ w hne).mpr ?_) i
  intro j hj
  have : j = c := by
    by_contra hjc
    exact hj (by simp [hjc])
  rw [this, hp]

/-- The ratio of two shares moves by the ratio of the bets. -/
theorem reinvest_ratio (a w : A → ℝ) (i j : A) (h : predicted a w ≠ 0) (hj : w j ≠ 0) :
    reinvest a w i / reinvest a w j = (a i / a j) * (w i / w j) := by
  unfold reinvest
  by_cases haj : a j = 0
  · simp [haj]
  · field_simp

variable {B : Type*}

/-- The shares after a stream of plain reinvestment. -/
noncomputable def after (w : B → A → ℝ) : (A → ℝ) → List B → (A → ℝ)
  | a, [] => a
  | a, y :: ys => after w (reinvest a (w y)) ys

/-- **Lock-in**: when each byte pays `i` at most `ρ` times what it pays `j`, the ratio of `i` to `j`
falls by at least `ρ` per byte, however good `i` was before. -/
theorem ratio_after_le [Nonempty A] (w : B → A → ℝ) (hw : ∀ y i, 0 < w y i) (i j : A) (ρ : ℝ)
    (hρ : ∀ y, w y i ≤ ρ * w y j) (ys : List B) (a : A → ℝ) (ha : ∀ k, 0 < a k) :
    after w a ys i / after w a ys j ≤ a i / a j * ρ ^ ys.length := by
  induction ys generalizing a with
  | nil => simp [after]
  | cons y ys ih =>
    have hp := predicted_pos a (w y) ha (hw y)
    have hr := reinvest_pos a (w y) ha (hw y)
    have hrat : w y i / w y j ≤ ρ := by
      rw [div_le_iff₀ (hw y j)]; exact hρ y
    have hρpos : 0 ≤ ρ := le_trans (div_pos (hw y i) (hw y j)).le hrat
    have hstep := ih (reinvest a (w y)) hr
    rw [reinvest_ratio a (w y) i j hp.ne' (hw y j).ne'] at hstep
    simp only [after, List.length_cons, pow_succ]
    calc after w (reinvest a (w y)) ys i / after w (reinvest a (w y)) ys j
        ≤ a i / a j * (w y i / w y j) * ρ ^ ys.length := hstep
      _ ≤ a i / a j * ρ * ρ ^ ys.length := by
          have h1 : 0 ≤ a i / a j := (div_pos (ha i) (ha j)).le
          gcongr
      _ = a i / a j * (ρ ^ ys.length * ρ) := by ring

/-- A small uniform transport of shares. -/
noncomputable def mix (ε : ℝ) (a : A → ℝ) (i : A) : ℝ :=
  (1 - ε) * a i + ε / (Fintype.card A : ℝ)

theorem mix_total [Nonempty A] (ε : ℝ) (a : A → ℝ) (ha : ∑ i, a i = 1) : ∑ i, mix ε a i = 1 := by
  have hn : (Fintype.card A : ℝ) ≠ 0 := by
    exact_mod_cast Fintype.card_ne_zero
  unfold mix
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ha, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul]
  field_simp
  ring

/-- Every address keeps a floor after one transport. -/
theorem mix_floor (ε : ℝ) (hε : ε ≤ 1) (a : A → ℝ) (ha : ∀ i, 0 ≤ a i) (i : A) :
    ε / (Fintype.card A : ℝ) ≤ mix ε a i := by
  unfold mix
  have := mul_nonneg (sub_nonneg.mpr hε) (ha i)
  linarith

theorem mix_pos [Nonempty A] (ε : ℝ) (h0 : 0 < ε) (h1 : ε ≤ 1) (a : A → ℝ) (ha : ∀ i, 0 ≤ a i)
    (i : A) : 0 < mix ε a i := by
  have hn : (0 : ℝ) < Fintype.card A := by exact_mod_cast Fintype.card_pos
  exact lt_of_lt_of_le (div_pos h0 hn) (mix_floor ε h1 a ha i)

/-- Cumulative wealth of reinvestment with transport: a bookkeeping number. -/
noncomputable def mixWealth (ε : ℝ) (w : B → A → ℝ) : (A → ℝ) → List B → ℝ
  | _, [] => 1
  | a, y :: ys => predicted a (w y) * mixWealth ε w (mix ε (reinvest a (w y))) ys

/-- Wealth with transport tracks any one address, paying `1 - ε` per byte. -/
theorem mixWealth_ge_address [Nonempty A] (ε : ℝ) (h0 : 0 < ε) (h1 : ε ≤ 1) (w : B → A → ℝ)
    (hw : ∀ y i, 0 < w y i) (ys : List B) (a : A → ℝ) (ha : ∀ i, 0 < a i) (hs : ∑ i, a i = 1)
    (i : A) :
    a i * (1 - ε) ^ ys.length * (ys.map fun y => w y i).prod ≤ mixWealth ε w a ys := by
  induction ys generalizing a with
  | nil =>
    have hle := Finset.single_le_sum (f := a) (fun j _ => (ha j).le) (Finset.mem_univ i)
    rw [hs] at hle
    simpa [mixWealth] using hle
  | cons y ys ih =>
    have hp := predicted_pos a (w y) ha (hw y)
    have hr := reinvest_pos a (w y) ha (hw y)
    have hm : ∀ k, 0 < mix ε (reinvest a (w y)) k := fun k =>
      mix_pos ε h0 h1 _ (fun k => (hr k).le) k
    have hind := ih _ hm (mix_total ε _ (reinvest_total a (w y) hp.ne'))
    have hmi : (1 - ε) * reinvest a (w y) i ≤ mix ε (reinvest a (w y)) i := by
      unfold mix
      have hn : (0 : ℝ) < Fintype.card A := by exact_mod_cast Fintype.card_pos
      have := div_pos h0 hn
      linarith
    have hprod : 0 ≤ (ys.map fun y => w y i).prod :=
      (List.prod_pos (by simp only [List.mem_map]; rintro _ ⟨z, _, rfl⟩; exact hw z i)).le
    have hpow : 0 ≤ (1 - ε) ^ ys.length := pow_nonneg (sub_nonneg.mpr h1) _
    have hkey : (1 - ε) * reinvest a (w y) i * (1 - ε) ^ ys.length * (ys.map fun y => w y i).prod
        ≤ mixWealth ε w (mix ε (reinvest a (w y))) ys := by
      refine le_trans ?_ hind
      gcongr
    simp only [mixWealth, List.map_cons, List.prod_cons, List.length_cons, pow_succ]
    have heq : a i * ((1 - ε) ^ ys.length * (1 - ε)) * (w y i * (ys.map fun y => w y i).prod) =
        predicted a (w y) * ((1 - ε) * reinvest a (w y) i * (1 - ε) ^ ys.length *
          (ys.map fun y => w y i).prod) := by
      unfold reinvest
      field_simp
    rw [heq]
    exact mul_le_mul_of_nonneg_left hkey hp.le

/-- **Recovery after any history**: from any state with the floor, the wealth over a later stretch
is at least the floor times what the address earns there, minus the transport toll. The cost does
not depend on how long or how badly the address lost before. -/
theorem recovers_from_floor [Nonempty A] (ε : ℝ) (h0 : 0 < ε) (h1 : ε ≤ 1) (w : B → A → ℝ)
    (hw : ∀ y i, 0 < w y i) (ys : List B) (a : A → ℝ) (ha : ∀ i, 0 ≤ a i) (hs : ∑ i, a i = 1)
    (i : A) :
    ε / (Fintype.card A : ℝ) * (1 - ε) ^ ys.length * (ys.map fun y => w y i).prod ≤
      mixWealth ε w (mix ε a) ys := by
  refine le_trans ?_ (mixWealth_ge_address ε h0 h1 w hw ys (mix ε a) (mix_pos ε h0 h1 a ha)
    (mix_total ε a hs) i)
  have hprod : 0 ≤ (ys.map fun y => w y i).prod :=
    (List.prod_pos (by simp only [List.mem_map]; rintro _ ⟨z, _, rfl⟩; exact hw z i)).le
  have hpow : 0 ≤ (1 - ε) ^ ys.length := pow_nonneg (sub_nonneg.mpr h1) _
  have := mix_floor ε h1 a ha i
  gcongr

/-- **Price equation**: the mean of any carried trait changes by the covariance of the trait with
the multiplier minus one. -/
theorem price_equation (a w z : A → ℝ) (h : predicted a w ≠ 0) :
    ∑ i, reinvest a w i * z i - ∑ i, a i * z i =
      ∑ i, a i * (w i / predicted a w - 1) * z i := by
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  unfold reinvest
  field_simp

section Groups

variable {C : Type*} [Fintype C] [DecidableEq C]

/-- The total share of a group of addresses. -/
noncomputable def groupShare (g : A → C) (a : A → ℝ) (c : C) : ℝ :=
  ∑ i with g i = c, a i

/-- What the group as a whole earns. -/
noncomputable def groupGain (g : A → C) (a w : A → ℝ) (c : C) : ℝ :=
  ∑ i with g i = c, a i * w i

/-- The bet of a group is its internal prediction. -/
noncomputable def groupBet (g : A → C) (a w : A → ℝ) (c : C) : ℝ :=
  groupGain g a w c / groupShare g a c

theorem groupGain_sum (g : A → C) (a w : A → ℝ) :
    ∑ c, groupGain g a w c = predicted a w := by
  unfold groupGain predicted
  exact Finset.sum_fiberwise Finset.univ g _

/-- Groups predict as addresses do. -/
theorem coarse_predicted (g : A → C) (a w : A → ℝ) (hs : ∀ c, groupShare g a c ≠ 0) :
    predicted (groupShare g a) (groupBet g a w) = predicted a w := by
  rw [← groupGain_sum g a w]
  unfold predicted groupBet
  exact Finset.sum_congr rfl fun c _ => by field_simp [hs c]

/-- **The same law at the next scale**: reinvestment of the group total is reinvestment of the group
as one address. -/
theorem coarse_reinvest (g : A → C) (a w : A → ℝ) (hs : ∀ c, groupShare g a c ≠ 0) (c : C) :
    reinvest (groupShare g a) (groupBet g a w) c = ∑ i with g i = c, reinvest a w i := by
  have hl : ∑ i with g i = c, reinvest a w i = groupGain g a w c / predicted a w := by
    unfold reinvest groupGain
    rw [Finset.sum_div]
  rw [hl]
  unfold reinvest
  rw [coarse_predicted g a w hs]
  unfold groupBet
  congr 1
  field_simp [hs c]

/-- Reinvestment normalised inside each group: exchange reaches only that far. -/
noncomputable def localReinvest (g : A → C) (a w : A → ℝ) (i : A) : ℝ :=
  a i * w i / groupGain g a w (g i) * groupShare g a (g i)

omit [Fintype C] in
/-- **A compartment keeps its total**: the sum of a group after a local exchange is its sum
before, whatever the bets are. -/
theorem local_keeps_group_total (g : A → C) (a w : A → ℝ) (c : C)
    (hc : groupGain g a w c ≠ 0) :
    ∑ i with g i = c, localReinvest g a w i = groupShare g a c := by
  have hrw : ∀ i ∈ Finset.univ.filter (fun i => g i = c),
      localReinvest g a w i = a i * w i * (groupShare g a c / groupGain g a w c) := fun i hi => by
    have hi' : g i = c := (Finset.mem_filter.mp hi).2
    unfold localReinvest
    rw [hi']
    ring
  rw [Finset.sum_congr rfl hrw, ← Finset.sum_mul]
  change groupGain g a w c * _ = _
  field_simp

omit [Fintype C] in
/-- **A compartment reads only its own bets**: changing bets outside the group changes nothing
inside it. -/
theorem local_ignores_outside (g : A → C) (a w w' : A → ℝ) (i : A)
    (hsame : ∀ j, g j = g i → w j = w' j) :
    localReinvest g a w i = localReinvest g a w' i := by
  have hgain : groupGain g a w (g i) = groupGain g a w' (g i) := by
    unfold groupGain
    refine Finset.sum_congr rfl fun j hj => ?_
    rw [hsame j (Finset.mem_filter.mp hj).2]
  unfold localReinvest
  rw [hgain, hsame i rfl]

end Groups

section Tags

/-- One forward step of unnormalised mass under the arrived byte's bets and a transport `T`. -/
noncomputable def step (T : A → A → ℝ) (w u : A → ℝ) (j : A) : ℝ :=
  ∑ i, u i * w i * T i j

/-- Likelihood of the later stream for any mass: a bookkeeping number. -/
noncomputable def lik (T : A → A → ℝ) (w : B → A → ℝ) : (A → ℝ) → List B → ℝ
  | u, [] => ∑ i, u i
  | u, y :: ys => lik T w (step T (w y) u) ys

theorem step_smul (T : A → A → ℝ) (w u : A → ℝ) (c : ℝ) (j : A) :
    step T w (fun i => c * u i) j = c * step T w u j := by
  unfold step
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => by ring

theorem lik_smul (T : A → A → ℝ) (w : B → A → ℝ) (ys : List B) (c : ℝ) (u : A → ℝ) :
    lik T w (fun i => c * u i) ys = c * lik T w u ys := by
  induction ys generalizing u with
  | nil => simp [lik, Finset.mul_sum]
  | cons y ys ih =>
    simp only [lik]
    rw [← ih]
    congr 1
    funext j
    exact step_smul T (w y) u c j

/-- Normalised forward step, shared by the mass and by its tagged part. -/
noncomputable def fwd (T : A → A → ℝ) (w a u : A → ℝ) (j : A) : ℝ :=
  (1 / predicted a w) * step T w u j

/-- The tagged fraction of the surviving mass, carried forward byte by byte. Nothing runs back. -/
noncomputable def tagFraction (T : A → A → ℝ) (w : B → A → ℝ) : (A → ℝ) → (A → ℝ) → List B → ℝ
  | a, b, [] => (∑ i, b i) / ∑ i, a i
  | a, b, y :: ys => tagFraction T w (fwd T (w y) a a) (fwd T (w y) a b) ys

theorem fwd_nonneg [Nonempty A] (T : A → A → ℝ) (hT : ∀ i j, 0 ≤ T i j) (w a : A → ℝ)
    (hw : ∀ i, 0 < w i) (ha : ∀ i, 0 ≤ a i) (hp : 0 < predicted a w) (j : A) :
    0 ≤ fwd T w a a j := by
  unfold fwd step
  refine mul_nonneg (one_div_pos.mpr hp).le (Finset.sum_nonneg fun i _ => ?_)
  exact mul_nonneg (mul_nonneg (ha i) (hw i).le) (hT i j)

theorem fwd_total (T : A → A → ℝ) (hR : ∀ i, ∑ j, T i j = 1) (w a : A → ℝ)
    (hp : predicted a w ≠ 0) : ∑ j, fwd T w a a j = 1 := by
  unfold fwd step
  rw [← Finset.mul_sum, Finset.sum_comm]
  have : ∑ i, ∑ j, a i * w i * T i j = predicted a w := by
    unfold predicted
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.mul_sum, hR i, mul_one]
  rw [this]
  field_simp

theorem predicted_pos_of_total [Nonempty A] (a w : A → ℝ) (hw : ∀ i, 0 < w i) (ha : ∀ i, 0 ≤ a i)
    (hs : ∑ i, a i = 1) : 0 < predicted a w := by
  have hex : ∃ i, 0 < a i := by
    by_contra hno
    simp only [not_exists, not_lt] at hno
    have : ∑ i, a i ≤ 0 := Finset.sum_nonpos fun i _ => hno i
    linarith
  obtain ⟨i, hi⟩ := hex
  unfold predicted
  exact Finset.sum_pos' (fun j _ => mul_nonneg (ha j) (hw j).le)
    ⟨i, Finset.mem_univ i, mul_pos hi (hw i)⟩

/-- **Delayed credit without a backward pass**: the tagged fraction of the mass that survives a
stream equals the likelihood of that stream through the tag over its likelihood overall, which is
the posterior of the tag given every later byte. The mass itself is the history carrier. -/
theorem tag_fraction_eq_posterior [Nonempty A] (T : A → A → ℝ) (hT : ∀ i j, 0 ≤ T i j)
    (hR : ∀ i, ∑ j, T i j = 1) (w : B → A → ℝ) (hw : ∀ y i, 0 < w y i) (ys : List B)
    (a b : A → ℝ) (ha : ∀ i, 0 ≤ a i) (hs : ∑ i, a i = 1) :
    tagFraction T w a b ys = lik T w b ys / lik T w a ys := by
  induction ys generalizing a b with
  | nil => simp [tagFraction, lik]
  | cons y ys ih =>
    have hp := predicted_pos_of_total a (w y) (hw y) ha hs
    have hc : (1 / predicted a (w y)) ≠ 0 := (one_div_pos.mpr hp).ne'
    simp only [tagFraction, lik]
    rw [ih _ _ (fwd_nonneg T hT (w y) a (hw y) ha hp) (fwd_total T hR (w y) a hp.ne')]
    have hfa : fwd T (w y) a a = fun j => (1 / predicted a (w y)) * step T (w y) a j := rfl
    have hfb : fwd T (w y) a b = fun j => (1 / predicted a (w y)) * step T (w y) b j := rfl
    rw [hfa, hfb, lik_smul, lik_smul, mul_div_mul_left _ _ hc]

theorem step_add (T : A → A → ℝ) (w u v : A → ℝ) (j : A) :
    step T w (fun i => u i + v i) j = step T w u j + step T w v j := by
  unfold step
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun i _ => by ring

theorem lik_add (T : A → A → ℝ) (w : B → A → ℝ) (ys : List B) (u v : A → ℝ) :
    lik T w (fun i => u i + v i) ys = lik T w u ys + lik T w v ys := by
  induction ys generalizing u v with
  | nil => simp [lik, Finset.sum_add_distrib]
  | cons y ys ih =>
    simp only [lik]
    rw [← ih]
    congr 1
    funext j
    exact step_add T (w y) u v j

theorem lik_pos [Nonempty A] (T : A → A → ℝ) (hT : ∀ i j, 0 ≤ T i j) (hR : ∀ i, ∑ j, T i j = 1)
    (w : B → A → ℝ) (hw : ∀ y i, 0 < w y i) (ys : List B) (a : A → ℝ) (ha : ∀ i, 0 ≤ a i)
    (hs : ∑ i, a i = 1) : 0 < lik T w a ys := by
  induction ys generalizing a with
  | nil => simp [lik, hs]
  | cons y ys ih =>
    have hp := predicted_pos_of_total a (w y) (hw y) ha hs
    have hind := ih _ (fwd_nonneg T hT (w y) a (hw y) ha hp) (fwd_total T hR (w y) a hp.ne')
    have hf : fwd T (w y) a a = fun j => (1 / predicted a (w y)) * step T (w y) a j := rfl
    rw [hf, lik_smul] at hind
    simp only [lik]
    exact (mul_pos_iff_of_pos_left (one_div_pos.mpr hp)).mp hind

/-- **The tag is the exact credit**: the derivative of the log likelihood of every later byte with
respect to mass added at an earlier event equals the tag's fraction of the surviving mass. This is
the derivative of the future with respect to a past contact, for fixed bets. The price is one more
field per tag: this is forward sensitivity, and it costs what forward sensitivity costs. -/
theorem tag_is_exact_credit [Nonempty A] (T : A → A → ℝ) (hT : ∀ i j, 0 ≤ T i j)
    (hR : ∀ i, ∑ j, T i j = 1) (w : B → A → ℝ) (hw : ∀ y i, 0 < w y i) (ys : List B)
    (a b : A → ℝ) (ha : ∀ i, 0 ≤ a i) (hs : ∑ i, a i = 1) :
    HasDerivAt (fun θ => Real.log (lik T w (fun i => a i + θ * b i) ys))
      (tagFraction T w a b ys) 0 := by
  have hl := lik_pos T hT hR w hw ys a ha hs
  have hf : HasDerivAt (fun θ : ℝ => lik T w a ys + θ * lik T w b ys) (lik T w b ys) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const (lik T w b ys)).const_add (lik T w a ys)
  have hlog := hf.log (by simpa using hl.ne')
  simp only [zero_mul, add_zero] at hlog
  rw [tag_fraction_eq_posterior T hT hR w hw ys a b ha hs]
  refine hlog.congr_of_eventuallyEq (Filter.Eventually.of_forall fun θ => ?_)
  simp only [lik_add, lik_smul]

end Tags

section Horizon

/-- The tag lies between `lo` and `hi` of the mass at every address. -/
def Bracketed (a b : A → ℝ) (lo hi : ℝ) : Prop := ∀ i, lo * a i ≤ b i ∧ b i ≤ hi * a i

/-- The reinvested tag. -/
noncomputable def tagReinvest (a w b : A → ℝ) (i : A) : ℝ := b i * w i / predicted a w

/-- The transported tag, moved by the same uniform transport as the mass. -/
noncomputable def tagMix (ε : ℝ) (q : A → ℝ) (i : A) : ℝ :=
  (1 - ε) * q i + ε * (∑ j, q j) / (Fintype.card A : ℝ)

theorem tagReinvest_total_bracket (a w b : A → ℝ) (lo hi : ℝ) (hb : Bracketed a b lo hi)
    (hw : ∀ i, 0 < w i) (hp : 0 < predicted a w) :
    lo ≤ ∑ i, tagReinvest a w b i ∧ ∑ i, tagReinvest a w b i ≤ hi := by
  have hsum : ∑ i, tagReinvest a w b i = (∑ i, b i * w i) / predicted a w := by
    unfold tagReinvest
    rw [Finset.sum_div]
  rw [hsum]
  constructor
  · rw [le_div_iff₀ hp]
    unfold predicted
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have := mul_le_mul_of_nonneg_right (hb i).1 (hw i).le
    linarith
  · rw [div_le_iff₀ hp]
    unfold predicted
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    have := mul_le_mul_of_nonneg_right (hb i).2 (hw i).le
    linarith

/-- **One step of forgetting**: uniform transport pulls the bracket of a tag toward its mean, by
`ε / n` of the gap at every byte. -/
theorem bracket_step [Nonempty A] (a w b : A → ℝ) (ε lo hi : ℝ) (h0 : 0 < ε) (h1 : ε ≤ 1)
    (hb : Bracketed a b lo hi) (ha : ∀ i, 0 ≤ a i) (hs : ∑ i, a i = 1) (hw : ∀ i, 0 < w i) :
    ∃ lo' hi', Bracketed (mix ε (reinvest a w)) (tagMix ε (tagReinvest a w b)) lo' hi' ∧
      hi' - lo' = (1 - ε / (Fintype.card A : ℝ)) * (hi - lo) := by
  have hp := predicted_pos_of_total a w hw ha hs
  obtain ⟨hQlo, hQhi⟩ := tagReinvest_total_bracket a w b lo hi hb hw hp
  set Q := ∑ i, tagReinvest a w b i with hQ
  have hn : (1 : ℝ) ≤ Fintype.card A := by exact_mod_cast Fintype.card_pos
  have hn0 : (0 : ℝ) < Fintype.card A := by linarith
  set l := ε / (Fintype.card A : ℝ) with hl
  have hl0 : 0 < l := div_pos h0 hn0
  have hle : l ≤ ε := div_le_self h0.le hn
  refine ⟨(1 - l) * lo + l * Q, (1 - l) * hi + l * Q, ?_, by ring⟩
  intro i
  have hri : 0 ≤ reinvest a w i :=
    div_nonneg (mul_nonneg (ha i) (hw i).le) hp.le
  have hri1 : reinvest a w i ≤ 1 := by
    have hsum := reinvest_total a w hp.ne'
    rw [← hsum]
    exact Finset.single_le_sum (f := fun j => reinvest a w j)
      (fun j _ => div_nonneg (mul_nonneg (ha j) (hw j).le) hp.le) (Finset.mem_univ i)
  have hqlo : lo * reinvest a w i ≤ tagReinvest a w b i := by
    unfold reinvest tagReinvest
    rw [← mul_div_assoc]
    apply div_le_div_of_nonneg_right _ hp.le
    have := mul_le_mul_of_nonneg_right (hb i).1 (hw i).le
    linarith
  have hqhi : tagReinvest a w b i ≤ hi * reinvest a w i := by
    unfold reinvest tagReinvest
    rw [← mul_div_assoc]
    apply div_le_div_of_nonneg_right _ hp.le
    have := mul_le_mul_of_nonneg_right (hb i).2 (hw i).le
    linarith
  have hmix : mix ε (reinvest a w) i = (1 - ε) * reinvest a w i + l := rfl
  have htag : tagMix ε (tagReinvest a w b) i = (1 - ε) * tagReinvest a w b i + l * Q := by
    unfold tagMix
    rw [hl, hQ]
    ring
  rw [hmix, htag]
  set p := reinvest a w i with hpdef
  set q := tagReinvest a w b i with hqdef
  have he : 0 ≤ 1 - ε := sub_nonneg.mpr h1
  have hfac : 0 ≤ (1 - l) - (1 - ε) * p := by nlinarith
  constructor
  · have h2 := mul_le_mul_of_nonneg_left hqlo he
    have h3 : 0 ≤ (Q - lo) * l * ((1 - l) - (1 - ε) * p) :=
      mul_nonneg (mul_nonneg (sub_nonneg.mpr hQlo) hl0.le) hfac
    nlinarith
  · have h2 := mul_le_mul_of_nonneg_left hqhi he
    have h3 : 0 ≤ (hi - Q) * l * ((1 - l) - (1 - ε) * p) :=
      mul_nonneg (mul_nonneg (sub_nonneg.mpr hQhi) hl0.le) hfac
    nlinarith

/-- The mass and the tag after a stream of reinvestment with uniform transport. -/
noncomputable def tagAfter (ε : ℝ) (w : B → A → ℝ) : (A → ℝ) × (A → ℝ) → List B → (A → ℝ) × (A → ℝ)
  | s, [] => s
  | s, y :: ys =>
    tagAfter ε w (mix ε (reinvest s.1 (w y)), tagMix ε (tagReinvest s.1 (w y) s.2)) ys

/-- **The credit horizon**: after `L` bytes the tag is the same fraction of the mass at every
address, up to `(1 - ε / n) ^ L` of the original gap. Later bytes can only read a difference that
is still there, so the transport that gives recovery also bounds how far back credit reaches. -/
theorem bracket_after [Nonempty A] (ε : ℝ) (h0 : 0 < ε) (h1 : ε ≤ 1) (w : B → A → ℝ)
    (hw : ∀ y i, 0 < w y i) (ys : List B) (a b : A → ℝ) (lo hi : ℝ) (hb : Bracketed a b lo hi)
    (ha : ∀ i, 0 ≤ a i) (hs : ∑ i, a i = 1) :
    ∃ lo' hi', Bracketed (tagAfter ε w (a, b) ys).1 (tagAfter ε w (a, b) ys).2 lo' hi' ∧
      hi' - lo' = (1 - ε / (Fintype.card A : ℝ)) ^ ys.length * (hi - lo) := by
  induction ys generalizing a b lo hi with
  | nil => exact ⟨lo, hi, hb, by simp⟩
  | cons y ys ih =>
    obtain ⟨lo₁, hi₁, hb₁, hgap⟩ := bracket_step a (w y) b ε lo hi h0 h1 hb ha hs (hw y)
    have hp := predicted_pos_of_total a (w y) (hw y) ha hs
    have hra : ∀ i, 0 ≤ reinvest a (w y) i := fun i =>
      div_nonneg (mul_nonneg (ha i) (hw y i).le) hp.le
    obtain ⟨lo₂, hi₂, hb₂, hgap₂⟩ := ih _ _ lo₁ hi₁ hb₁
      (fun i => (mix_pos ε h0 h1 _ hra i).le) (mix_total ε _ (reinvest_total a (w y) hp.ne'))
    refine ⟨lo₂, hi₂, hb₂, ?_⟩
    rw [hgap₂, hgap, List.length_cons, pow_succ]
    ring

end Horizon

end CGB.CountedPopulation
