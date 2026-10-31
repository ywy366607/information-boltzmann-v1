import Mathlib.Basic.Real.Basic
import Mathlib.LinearAlgebra.Dimension.Finite
import Mathlib.LinearAlgebra.FiniteDimensional.Basic

/-!
# Exact and mixed capacity of an associative linear state

A GDN fast-weight state is a superposition of value-weighted keys.  This file
records only the dimension-level facts needed before defining an error-tolerant
mixed capacity: independent keys cannot outnumber the key dimension, while an
excess family has two distinct coefficient assignments with the same
superposition.  No extra reserve variable is introduced here.
-/

namespace CGB.Generation

/-- A linear superposition of a finite family of keys. -/
def keySuperposition {d n : ℕ} (k : Fin n → (Fin d → ℝ))
    (a : Fin n → ℝ) : Fin d → ℝ :=
  ∑ i, a i • k i

/-- The associative matrix state `V Kᵀ`, written without a dense matrix type. -/
def associationState {dv dk n : ℕ}
    (k : Fin n → (Fin dk → ℝ)) (v : Fin n → (Fin dv → ℝ)) :
    Fin dv → Fin dk → ℝ :=
  fun o j => ∑ i, v i o * k i j

/-- Each output row of `V Kᵀ` is one scalar key superposition. -/
theorem associationState_row {dv dk n : ℕ}
    (k : Fin n → (Fin dk → ℝ)) (v : Fin n → (Fin dv → ℝ)) (o : Fin dv) :
    associationState k v o = keySuperposition k (fun i => v i o) := by
  funext j
  simp [associationState, keySuperposition]

/-- Slotwise natural-state retention transports each stored key by the same
slot factors. It applies to arbitrarily many prior associations at once;
the list of keys is a mathematical decomposition, not required runtime state. -/
theorem association_slot_retention {dv dk n : ℕ}
    (k : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ)
    (retention : Fin dk → ℝ) :
    (fun o j => retention j * associationState k v o j) =
      associationState (fun i j => retention j * k i j) v := by
  funext o j
  simp only [associationState, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Query a value-by-key state with a soft key. -/
def associationRead {dv dk : ℕ}
    (state : Fin dv → Fin dk → ℝ) (q : Fin dk → ℝ) : Fin dv → ℝ :=
  fun o => ∑ j, state o j * q j

/-- Reading `V Kᵀ` with a query is the corresponding weighted combination of
values; this is the algebraic reference for GDN probe readout. -/
theorem association_state_readout {dv dk n : ℕ}
    (k : Fin n → (Fin dk → ℝ)) (v : Fin n → (Fin dv → ℝ))
    (q : Fin dk → ℝ) :
    associationRead (associationState k v) q =
      fun o => ∑ i, v i o * (∑ j, k i j * q j) := by
  funext o
  simp only [associationRead, associationState, Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i hi
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j hj
  ring

/-- A single interference matrix bounds the readout error uniformly over
all bounded values. Keys and queries are the actual vectors presented to the
associative state; the bound is a mathematical certificate, not stored history. -/
theorem association_readout_interference_bound {dv dk n : ℕ}
    (k q : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ)
    (hv : ∀ i o, |v i o| ≤ 1) (i : Fin n) (o : Fin dv) :
    |associationRead (associationState k v) (q i) o - v i o| ≤
      ∑ j, |(∑ t, k j t * q i t) - (if j = i then 1 else 0)| := by
  rw [association_state_readout]
  have heq : (∑ j, v j o * (∑ t, k j t * q i t)) - v i o =
      ∑ j, v j o * ((∑ t, k j t * q i t) - (if j = i then 1 else 0)) := by
    simp only [mul_sub, Finset.sum_sub_distrib]
    simp
  rw [heq]
  calc
    |∑ j, v j o * ((∑ t, k j t * q i t) - (if j = i then 1 else 0))| ≤
        ∑ j, |v j o * ((∑ t, k j t * q i t) - (if j = i then 1 else 0))| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j, |(∑ t, k j t * q i t) - (if j = i then 1 else 0)| := by
      apply Finset.sum_le_sum
      intro j _
      rw [abs_mul]
      nlinarith [abs_nonneg ((∑ t, k j t * q i t) - (if j = i then 1 else 0)), hv j o]

/-- Accuracy of one declared value/key probe.  The tolerance is a measurement
contract, not a new state variable or a universal constant. -/
def probeAccurate {dv dk : ℕ}
    (state : Fin dv → Fin dk → ℝ) (key : Fin dk → ℝ)
    (target : Fin dv → ℝ) (ε : ℝ) : Prop :=
  0 ≤ ε ∧ ∀ o, |associationRead state key o - target o| ≤ ε

/-- Accuracy of a finite probe family, the Lean counterpart of the runtime
`association_probe_error` diagnostic. -/
def probeFamilyAccurate {dv dk m : ℕ}
    (state : Fin dv → Fin dk → ℝ) (keys : Fin m → Fin dk → ℝ)
    (targets : Fin m → Fin dv → ℝ) (ε : ℝ) : Prop :=
  0 ≤ ε ∧ ∀ i, ∀ o,
    |associationRead state (keys i) o - targets i o| ≤ ε

/-- Accuracy is monotone in the declared tolerance. -/
theorem probeFamilyAccurate_mono {dv dk m : ℕ}
    (state : Fin dv → Fin dk → ℝ) (keys : Fin m → Fin dk → ℝ)
    (targets : Fin m → Fin dv → ℝ) {ε ε' : ℝ}
    (h : probeFamilyAccurate state keys targets ε) (hle : ε ≤ ε') :
    probeFamilyAccurate state keys targets ε' := by
  rcases h with ⟨hε, hprobe⟩
  refine ⟨le_trans hε hle, ?_⟩
  intro i o
  exact (hprobe i o).trans hle

/-- A decoder that exactly recovers every coefficient assignment. -/
def ExactDecoder {d n : ℕ} (k : Fin n → (Fin d → ℝ)) : Prop :=
  ∃ D : (Fin d → ℝ) →ₗ[ℝ] (Fin n → ℝ),
    ∀ a, D (keySuperposition k a) = a

/-- Exact decoding itself forces the same key-dimension bound. -/
theorem exact_decoder_capacity (d n : ℕ) (k : Fin n → (Fin d → ℝ))
    (hD : ExactDecoder k) : n ≤ d := by
  rcases hD with ⟨D, hD⟩
  have hinj : Function.Injective (keySuperposition k) := by
    intro a b hab
    have hd := congrArg D hab
    simpa [hD] using hd
  have hk : LinearIndependent ℝ k := by
    rw [Fintype.linearIndependent_iffₛ]
    intro f g hfg
    have hab : keySuperposition k f = keySuperposition k g := by
      simpa [keySuperposition] using hfg
    exact fun i => congrFun (hinj hab) i
  simpa using hk.fintype_card_le_finrank

/-- An error-tolerant decoder is defined only after fixing a bounded probe set
(the coordinatewise unit box here) and a tolerance.  This is the entry point for mixed
capacity; it is deliberately a definition, not an unconditional capacity
claim. -/
def BoundedEpsilonDecoder {d n : ℕ} (k : Fin n → (Fin d → ℝ))
    (ε : ℝ) : Prop :=
  0 ≤ ε ∧ ∃ D : (Fin d → ℝ) →ₗ[ℝ] (Fin n → ℝ),
    ∀ a : Fin n → ℝ, (∀ i, |a i| ≤ 1) →
      ∀ i, |D (keySuperposition k a) i - a i| ≤ ε

/- The decoder contract is useful only when it is applied to the actual
   associative state.  The following two lemmas make that connection
   explicit: a row of `V Kᵀ` is the key superposition of the corresponding
   value coordinates, so the same bounded decoder gives a coefficient-level
   readout bound. -/

theorem exact_decoder_association_readout {dv dk n : ℕ}
    (k : Fin n → (Fin dk → ℝ)) (v : Fin n → (Fin dv → ℝ))
    (D : (Fin dk → ℝ) →ₗ[ℝ] (Fin n → ℝ))
    (hD : ∀ a, D (keySuperposition k a) = a) :
    ∀ o, D (associationState k v o) = (fun i => v i o) := by
  intro o
  rw [associationState_row]
  exact hD (fun i => v i o)

theorem bounded_decoder_association_readout {dv dk n : ℕ}
    (k : Fin n → (Fin dk → ℝ)) (v : Fin n → (Fin dv → ℝ))
    {ε : ℝ} (hε : BoundedEpsilonDecoder k ε)
    (hv : ∀ i o, |v i o| ≤ 1) :
    ∃ D : (Fin dk → ℝ) →ₗ[ℝ] (Fin n → ℝ),
      ∀ o i, |D (associationState k v o) i - v i o| ≤ ε := by
  rcases hε with ⟨hε_nonneg, D, hD⟩
  refine ⟨D, ?_⟩
  intro o i
  rw [associationState_row]
  exact hD (fun j => v j o) (fun j => hv j o) i

/-- A weighted natural-parameter write followed by slotwise normalization is
an associative superposition with effective keys `wᵢ kᵢⱼ / pⱼ`, plus the
normalized old state. If the existing bounded decoder handles those keys and
its decoded old-state contribution is bounded by `ρ`, the full decoded
error is at most `ε + ρ`. The event family is a proof-side decomposition,
not an additional persistent history. -/
theorem normalized_assimilation_decoder_bound {dv dk n : ℕ}
    (old : Fin dv → Fin dk → ℝ)
    (k : Fin n → Fin dk → ℝ) (v : Fin n → Fin dv → ℝ)
    (w : Fin n → ℝ) (p : Fin dk → ℝ) (ε ρ : ℝ)
    (hv : ∀ i o, |v i o| ≤ 1)
    (hdecoder : ∃ D : (Fin dk → ℝ) →ₗ[ℝ] (Fin n → ℝ),
      (∀ a : Fin n → ℝ, (∀ i, |a i| ≤ 1) → ∀ i,
        |D (keySuperposition (fun i j => w i * k i j / p j) a) i - a i| ≤ ε) ∧
      (∀ o i, |D (fun j => old o j / p j) i| ≤ ρ)) :
    ∃ D : (Fin dk → ℝ) →ₗ[ℝ] (Fin n → ℝ), ∀ o i,
      |D (fun j => (old o j + ∑ t, w t * v t o * k t j) / p j) i - v i o|
        ≤ ε + ρ := by
  obtain ⟨D, hD, hold⟩ := hdecoder
  refine ⟨D, ?_⟩
  intro o i
  have heq : (fun j => (old o j + ∑ t, w t * v t o * k t j) / p j) =
      (fun j => old o j / p j) +
        keySuperposition (fun t j => w t * k t j / p j) (fun t => v t o) := by
    funext j
    simp only [Pi.add_apply, keySuperposition, Finset.sum_apply, Pi.smul_apply,
      smul_eq_mul, add_div]
    congr 1
    rw [div_eq_mul_inv, Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro t _
    ring
  rw [heq, map_add]
  have htri := abs_add_le (D (fun j => old o j / p j) i)
    (D (keySuperposition (fun t j => w t * k t j / p j) (fun t => v t o)) i - v i o)
  have hnew := hD (fun t => v t o) (fun t => hv t o) i
  have hprior := hold o i
  simp only [Pi.add_apply] at *
  have he : D (fun j => old o j / p j) i +
      (D (keySuperposition (fun t j => w t * k t j / p j) (fun t => v t o)) i - v i o) =
      D (fun j => old o j / p j) i +
        D (keySuperposition (fun t j => w t * k t j / p j) (fun t => v t o)) i - v i o := by ring
  rw [he] at htri
  linarith

theorem exact_decoder_is_bounded_decoder {d n : ℕ}
    (k : Fin n → (Fin d → ℝ))
    (hD : ExactDecoder k) :
    BoundedEpsilonDecoder k 0 := by
  rcases hD with ⟨D, hD⟩
  refine ⟨le_rfl, D, ?_⟩
  intro a ha i
  rw [hD]
  simp

/-- A natural-parameter assimilation is an innovation update, with gain
`w / (p + w)` for the individual slot. This is a slotwise identity; it does
not assert equivalence with a key-directed delta-rule update. -/
theorem precision_assimilation_is_innovation (p w m y : ℝ)
    (hp : 0 < p) (hw : 0 ≤ w) :
    (p * m + w * y) / (p + w) = m + (w / (p + w)) * (y - m) := by
  have h : p + w ≠ 0 := ne_of_gt (by linarith)
  field_simp
  ring

/-- The precision-derived slot gain is bounded when prior precision is
positive and incoming precision is nonnegative. -/
theorem precision_assimilation_gain_bounds (p w : ℝ)
    (hp : 0 < p) (hw : 0 ≤ w) :
    0 ≤ w / (p + w) ∧ w / (p + w) < 1 := by
  have hd : 0 < p + w := by linarith
  constructor
  · exact div_nonneg hw (le_of_lt hd)
  · apply (div_lt_one hd).2
    linarith

/-- With strictly positive process variance and a lower bound on incoming
precision, one predict-then-assimilate step has a positive gain floor.  The
bound is conditional on applying the process step before the observation. -/
theorem process_noise_gain_floor (p q w wmin : ℝ)
    (hp : 0 < p) (hq : 0 < q) (hwmin : 0 < wmin) (hw : wmin ≤ w) :
    wmin / (1 / q + wmin) ≤ w / (p / (1 + q * p) + w) := by
  have hd : 0 < 1 + q * p := by positivity
  have hwpos : 0 < w := lt_of_lt_of_le hwmin hw
  have hprior : p / (1 + q * p) ≤ 1 / q := by
    apply (div_le_div_iff₀ hd hq).2
    nlinarith
  have hden1 : 0 < 1 / q + wmin := by positivity
  have hden2 : 0 < p / (1 + q * p) + w := by positivity
  apply (div_le_div_iff₀ hden1 hden2).2
  nlinarith [mul_nonneg (le_of_lt hwmin) (sub_nonneg.mpr hprior),
    mul_nonneg (sub_nonneg.mpr hw) (le_of_lt (one_div_pos.mpr hq))]
theorem exact_key_capacity (d n : ℕ) (k : Fin n → (Fin d → ℝ))
    (hk : LinearIndependent ℝ k) : n ≤ d := by
  simpa using hk.fintype_card_le_finrank

/-- Beyond key dimension, linear superposition is not injective: two distinct
coefficient assignments produce the same stored state.  This is the formal
collision statement behind mixed storage in an associative GDN matrix. -/
theorem collision_of_excess_keys (d n : ℕ) (h : d < n)
    (k : Fin n → (Fin d → ℝ)) :
    ∃ f g : Fin n → ℝ,
      keySuperposition k f = keySuperposition k g ∧ ∃ i, f i ≠ g i := by
  apply (Fintype.not_linearIndependent_iffₛ (R := ℝ) (v := k)).mp
  intro hk
  have hcap := hk.fintype_card_le_finrank
  have hcap' : n ≤ d := by simpa using hcap
  omega

end CGB.Generation
