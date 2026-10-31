import CGBAxioms.GenerationLinearMemory
import CGBAxioms.GenerationPersistence
import CGBAxioms.GenerationPredictiveHarvest
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Column memory for the language organism (generation chain, 2026-10-28)

The language organism's synaptic page is a set of columns `S_m` written by the gated delta rule
(`GenerationLinearMemory`), with the activity page relaxing on them. Design:
`docs/information_boltzmann/CGB_LANGUAGE_ORGANISM_THEOREM_USE_20261028.md`. Four edges:

* **Silence has one rest** (`silent_rest_unique`): with no write, the specific
  reference step `α * deltaStep` contracts (`gated_delta_contracts`), so its
  only fixed point is zero. Other GDN implementations need their own update
  and assumptions checked before this conclusion can be transferred.
* **Read then write with one key is error descent** (`tied_write_is_error_descent`,
  `tied_write_error_shrinks`, `exact_prediction_no_write`): reading `⟨k, x⟩` before the arrival
  and writing `x + β (v - ⟨k, x⟩) k` after it is a gradient step on `½ (⟨k, x⟩ - v)²`, the
  arrived-minus-predicted difference of G14 times the presynaptic key; with `‖k‖ = 1` the
  prediction error shrinks by `1 - β`; an exact prediction writes nothing
  (cf. `CausalLearning.agreement_no_write`).
* **Decay by use keeps rare topics** (`occurrence_decay_retains`, `unused_column_kept`): a column
  decaying by `α ^ w` when routed with weight `w ∈ [0, 1]` keeps at least what a clock decaying
  every tick keeps, and an unused column keeps everything (W3 `occurrence_clock_beats_time_clock`,
  G17b `retention_needs_selectivity`).
* **A latch needs gain above one and saturation** (`no_latch_below_unit_gain`,
  `latch_above_unit_gain`): activity relaxing as `x ↦ σ (g x)` with a 1-Lipschitz `σ`, `σ 0 = 0`,
  has only the rest `0` when `g < 1` (`contraction_unique_rest`); with the saturation of a finite
  occupancy (hard bound at `± 1`, cf. G25 `occupancy_le_one`) and `g ≥ 1` it has the two further
  rests `± 1`. The gain along a stored pattern is what the tied write drives toward its target
  (`tied_write_error_shrinks`), so a column latches a pattern written with target gain at least
  one, and releases it when adaptation (B) brings the gain below one.
-/

namespace CGB.Generation

open Finset
open CGB.PredictiveHarvest

variable {n : ℕ}

theorem dot_self_nonneg (x : Fin n → ℝ) : 0 ≤ dot x x :=
  sum_nonneg fun i _ => mul_self_nonneg (x i)

theorem eq_zero_of_dot_self (x : Fin n → ℝ) (h : dot x x = 0) : x = 0 := by
  funext i
  unfold dot at h
  have hs := (sum_eq_zero_iff_of_nonneg (fun j _ => mul_self_nonneg (x j))).mp h i (mem_univ i)
  simpa using mul_self_eq_zero.mp hs

/-- **Silence has one rest**: a gated delta step without a write (`α < 1`) fixes only zero. -/
theorem silent_rest_unique (k x : Fin n → ℝ) (hk : dot k k = 1) {α β : ℝ} (hα0 : 0 ≤ α)
    (hα1 : α < 1) (h0 : 0 ≤ β) (h2 : β ≤ 2) (hfix : (fun i => α * deltaStep k β x i) = x) :
    x = 0 := by
  have h := gated_delta_contracts k x hk (α := α) h0 h2
  rw [hfix] at h
  have hnn := dot_self_nonneg x
  have ha2 : α ^ 2 < 1 := by nlinarith
  have hz : dot x x = 0 := by nlinarith
  exact eq_zero_of_dot_self x hz

/-- The tied write on one row: read `⟨k, x⟩` before the arrival `v`, then write along `k`. -/
def tiedWrite (k : Fin n → ℝ) (β v : ℝ) (x : Fin n → ℝ) : Fin n → ℝ :=
  fun i => x i + β * (v - dot k x) * k i

/-- The tied write is the delta step plus the write of the arrival. -/
theorem tiedWrite_eq_delta_add (k x : Fin n → ℝ) (β v : ℝ) :
    tiedWrite k β v x = fun i => deltaStep k β x i + β * v * k i := by
  funext i
  simp only [tiedWrite, deltaStep]
  ring

/-- **Error descent**: the tied write is a gradient step of size `β` on `½ (⟨k, x⟩ - v)²`, whose
gradient in `x i` is `(⟨k, x⟩ - v) k i` (predicted minus arrived, times the presynaptic key). -/
theorem tied_write_is_error_descent (k x : Fin n → ℝ) (β v : ℝ) :
    tiedWrite k β v x = fun i => x i - β * ((dot k x - v) * k i) := by
  funext i
  simp only [tiedWrite]
  ring

/-- With a unit key the prediction error shrinks by `1 - β`. -/
theorem tied_write_error_shrinks (k x : Fin n → ℝ) (hk : dot k k = 1) (β v : ℝ) :
    dot k (tiedWrite k β v x) - v = (1 - β) * (dot k x - v) := by
  have e : dot k (tiedWrite k β v x) = dot k x + β * (v - dot k x) * dot k k := by
    have : ∀ c : ℝ, dot k (fun i => x i + c * k i) = dot k x + c * dot k k := by
      intro c
      simp only [dot, mul_add, sum_add_distrib, mul_sum]
      congr 1
      exact sum_congr rfl fun i _ => by ring
    exact this (β * (v - dot k x))
  rw [e, hk]
  ring

/-! A scalar binary column is the smallest information-economy instance of
the tied write.  Its update moves the probability assigned to the arrived bit
toward that bit, so the same local write that descends squared prediction
error also descends cross-entropy. -/

def binaryAllocation (p : ℝ) : Bool → ℝ := fun b => if b then p else 1 - p

def binaryPointLaw (y : Bool) : Bool → ℝ := fun b => if b = y then 1 else 0

theorem binary_delta_crossEntropy_decreases
    (p : ℝ) (y : Bool) (β : ℝ)
    (hp0 : 0 < p) (hp1 : p < 1)
    (hβ0 : 0 < β) (hβ1 : β ≤ 1) :
    let p' := p + β * ((if y then 1 else 0) - p)
    (0 ≤ p' ∧ p' ≤ 1 ∧ p' ≠ p) ∧
      crossEntropy (binaryPointLaw y) (binaryAllocation p') <
        crossEntropy (binaryPointLaw y) (binaryAllocation p) := by
  dsimp only
  by_cases hy : y
  · have hy1 : y = true := by simpa using hy
    simp only [hy1, ↓reduceIte]
    have hnonneg : 0 ≤ (1 - p) := by linarith
    have hprod : β * (1 - p) ≤ 1 * (1 - p) :=
      mul_le_mul_of_nonneg_right hβ1 hnonneg
    have hprod' : β * (1 - p) ≤ 1 - p := by simpa using hprod
    have hsum : p + β * (1 - p) ≤ p + (1 - p) := by
      have hsum0 := add_le_add_left hprod' p
      nlinarith [hsum0]
    have hpp : p < p + β * (1 - p) := by
      nlinarith
    have hp'0 : 0 < p + β * (1 - p) := by
      nlinarith
    have hlog : Real.log p < Real.log (p + β * (1 - p)) :=
      Real.strictMonoOn_log hp0 hp'0 hpp
    constructor
    · refine ⟨?_, ?_, ?_⟩
      · nlinarith
      · nlinarith [hsum]
      · exact ne_of_gt hpp
    · simp [PredictiveHarvest.crossEntropy, binaryPointLaw, binaryAllocation]
      linarith
  · have hy0 : y = false := by cases y <;> simp_all
    simp only [hy0, Bool.false_eq_true, ↓reduceIte]
    have hpp : p + β * (0 - p) < p := by
      nlinarith
    have hp'1 : p + β * (0 - p) < 1 := by
      nlinarith
    have hprod : 0 ≤ β * p := mul_nonneg hβ0.le hp0.le
    have hprod' : β * p ≤ p := by
      have h := mul_le_mul_of_nonneg_right hβ1 hp0.le
      simpa using h
    have hsum : 0 ≤ p + β * (0 - p) := by
      nlinarith [hprod']
    have hpold0 : 0 < 1 - p := by
      linarith
    have hpnew0 : 0 < 1 - (p + β * (0 - p)) := by
      nlinarith
    have hlog : Real.log (1 - p) <
        Real.log (1 - (p + β * (0 - p))) := by
      exact Real.strictMonoOn_log hpold0 hpnew0 (by nlinarith)
    have hlog' : Real.log (1 - p) <
        Real.log (1 - (p + -(β * p))) := by
      simpa [sub_eq_add_neg] using hlog
    constructor
    · refine ⟨?_, ?_, ?_⟩
      · nlinarith [hsum]
      · nlinarith [hp'1]
      · exact ne_of_lt hpp
    · simp [PredictiveHarvest.crossEntropy, binaryPointLaw, binaryAllocation]
      linarith

/-- The change in any other readout is exactly the prediction error times
the key overlap and write gain. Orthogonal readouts are therefore preserved. -/
theorem tied_write_cross_read (k q x : Fin n → ℝ) (β v : ℝ) :
    dot q (tiedWrite k β v x) - dot q x = β * (v - dot k x) * dot q k := by
  have he : dot q (tiedWrite k β v x) =
      dot q x + β * (v - dot k x) * dot q k := by
    simp only [dot, tiedWrite, mul_add, sum_add_distrib, mul_sum]
    congr 1
    exact sum_congr rfl fun i _ => by ring
  rw [he]
  ring

/-! ## Reciprocal local teaching paths

One local contact can expose a forward coefficient and a coefficient in the
reciprocal direction.  When both receive the *same local heterosynaptic
statistic* and have the same release rate, their mismatch is not manually
transposed: the shared statistic cancels from the mismatch update.  This is
the strictly proved splice-alignment portion of a self-assembling teaching
path.  It does not establish that the complete multi-hop teaching path has
small error; that remains a separately measurable admission condition. -/

structure ReciprocalPair (m n : ℕ) where
  forward : Fin m → Fin n → ℝ
  feedback : Fin n → Fin m → ℝ

def reciprocalStep {m n : ℕ} (release rate : ℝ)
    (shared : Fin m → Fin n → ℝ) (pair : ReciprocalPair m n) :
    ReciprocalPair m n where
  forward := fun i j => (1 - release) * pair.forward i j + rate * shared i j
  feedback := fun j i => (1 - release) * pair.feedback j i + rate * shared i j

def reciprocalMismatch {m n : ℕ} (pair : ReciprocalPair m n) (i : Fin m) (j : Fin n) : ℝ :=
  pair.forward i j - pair.feedback j i

/-- A matched reciprocal update contracts its splice mismatch by its release
factor.  No initial transpose relation is assumed. -/
theorem reciprocal_shared_update_mismatch {m n : ℕ} (release rate : ℝ)
    (shared : Fin m → Fin n → ℝ) (pair : ReciprocalPair m n)
    (i : Fin m) (j : Fin n) :
    reciprocalMismatch (reciprocalStep release rate shared pair) i j =
      (1 - release) * reciprocalMismatch pair i j := by
  simp only [reciprocalMismatch, reciprocalStep]
  ring

def reciprocalIter {m n : ℕ} (release rate : ℝ)
    (shared : ℕ → Fin m → Fin n → ℝ) (pair : ReciprocalPair m n) :
    ℕ → ReciprocalPair m n
  | 0 => pair
  | t + 1 => reciprocalStep release rate (shared t) (reciprocalIter release rate shared pair t)

/-- The mismatch is geometric even when the common local statistic changes at
each event.  Thus alignment does not require either contact to stop learning. -/
theorem reciprocal_iter_mismatch {m n : ℕ} (release rate : ℝ)
    (shared : ℕ → Fin m → Fin n → ℝ) (pair : ReciprocalPair m n)
    (t : ℕ) (i : Fin m) (j : Fin n) :
    reciprocalMismatch (reciprocalIter release rate shared pair t) i j =
      (1 - release) ^ t * reciprocalMismatch pair i j := by
  induction t with
  | zero => simp [reciprocalIter]
  | succ t ih =>
      rw [reciprocalIter, reciprocal_shared_update_mismatch, ih, pow_succ]
      ring

/-- Reuse one finite row by writing each arriving value at the same unit key.
The stream is an external input; only the current row is the memory state. -/
def reusedKeyState (k x : Fin n → ℝ) (v : ℕ → ℝ) : ℕ → (Fin n → ℝ)
  | 0 => x
  | t + 1 => tiedWrite k 1 (v t) (reusedKeyState k x v t)

/-- Every write recovers the latest value. Whenever consecutive arrivals
differ, the finite state must change again; reuse is not a monotonically
shrinking set of permanently excluded candidates. -/
theorem reused_key_tracks_and_rewrites (k x : Fin n → ℝ) (hk : dot k k = 1)
    (v : ℕ → ℝ) (t : ℕ) :
    dot k (reusedKeyState k x v (t + 1)) = v t ∧
    (v (t + 1) ≠ v t → reusedKeyState k x v (t + 2) ≠ reusedKeyState k x v (t + 1)) := by
  have hr : ∀ j, dot k (reusedKeyState k x v (j + 1)) = v j := by
    intro j
    have he := tied_write_error_shrinks k (reusedKeyState k x v j) hk 1 (v j)
    change dot k (reusedKeyState k x v (j + 1)) - v j = _ at he
    linarith
  refine ⟨hr t, ?_⟩
  intro hd heq
  apply hd
  have he := congrArg (dot k) heq
  rw [show t + 2 = (t + 1) + 1 by omega, hr (t + 1), hr t] at he
  exact he

/-- An exact prediction writes nothing. -/
theorem exact_prediction_no_write (k x : Fin n → ℝ) (β v : ℝ) (h : dot k x = v) :
    tiedWrite k β v x = x := by
  funext i
  simp [tiedWrite, h]

/-- **Decay by use keeps rare topics**: decaying by `α ^ w t` with `w t ∈ [0, 1]` keeps at least
what decaying by `α` every tick keeps. -/
theorem occurrence_decay_retains (α : ℝ) (hα0 : 0 < α) (hα1 : α ≤ 1) (w : ℕ → ℝ)
    (hw1 : ∀ t, w t ≤ 1) (T : ℕ) :
    α ^ T ≤ ∏ t ∈ range T, α ^ (w t) := by
  calc α ^ T = ∏ _t ∈ range T, α := by simp
    _ ≤ ∏ t ∈ range T, α ^ (w t) := by
      refine prod_le_prod₀ (fun _ _ => hα0.le) fun t _ => ?_
      have := Real.rpow_le_rpow_of_exponent_ge hα0 hα1 (hw1 t)
      rwa [Real.rpow_one] at this

/-- An unused column keeps everything. -/
theorem unused_column_kept (α : ℝ) (T : ℕ) : ∏ _t ∈ range T, α ^ (0 : ℝ) = 1 := by
  simp

/-- **No latch below unit gain**: `x ↦ σ (g x)` with a 1-Lipschitz `σ`, `σ 0 = 0` and
`0 ≤ g < 1` rests only at zero. -/
theorem no_latch_below_unit_gain (σ : ℝ → ℝ) (hσ : ∀ a b, |σ a - σ b| ≤ |a - b|) (hσ0 : σ 0 = 0)
    {g : ℝ} (hg0 : 0 ≤ g) (hg : g < 1) (x : ℝ) (hx : σ (g * x) = x) : x = 0 := by
  refine contraction_unique_rest (fun y => σ (g * y)) g hg (fun a b => ?_) x 0 hx (by simp [hσ0])
  calc |σ (g * a) - σ (g * b)| ≤ |g * a - g * b| := hσ _ _
    _ = g * |a - b| := by rw [← mul_sub, abs_mul, abs_of_nonneg hg0]

/-- Saturation of a finite occupancy, as a hard bound at `± 1`. -/
noncomputable def hardSat (y : ℝ) : ℝ := max (min y 1) (-1)

theorem hardSat_zero : hardSat 0 = 0 := by norm_num [hardSat]

theorem hardSat_lipschitz (a b : ℝ) : |hardSat a - hardSat b| ≤ |a - b| :=
  (abs_max_sub_max_le_abs _ _ _).trans <| (abs_min_sub_min_le_max a 1 b 1).trans (by simp)

/-- **A latch above unit gain**: with saturation and `g ≥ 1`, `± 1` rest besides `0`. -/
theorem latch_above_unit_gain {g : ℝ} (hg : 1 ≤ g) :
    hardSat (g * 1) = 1 ∧ hardSat (g * (-1)) = -1 ∧ hardSat (g * 0) = 0 := by
  refine ⟨?_, ?_, by simp [hardSat_zero]⟩
  · unfold hardSat
    rw [mul_one, min_eq_right hg, max_eq_left (by norm_num)]
  · unfold hardSat
    rw [min_eq_left (by linarith), max_eq_right (by linarith)]

/-! ## Self-listening and forward credit: a conditional feedback example

These identities describe one scalar, differentiable closed-loop example;
they do not assert that a mother-world response is differentiable or that
any stochastic gradient is unbiased.  A local derivative that freezes the
self-generated input can omit the feedback path. -/

def selfListeningStep (α β θ x : ℝ) : ℝ := α * x + β * θ * x + θ

/-- Exact two-trajectory increment, including the feedback contribution. -/
theorem self_listening_increment (α β θ θ' x x' : ℝ) :
    selfListeningStep α β θ' x' - selfListeningStep α β θ x =
      (α + β * θ) * (x' - x) + (1 + β * x') * (θ' - θ) := by
  unfold selfListeningStep
  ring

/-- Freezing the self-listening input loses the feedback part of state credit. -/
theorem self_listening_frozen_state_gap (α β θ sensitivity : ℝ) :
    ((α + β * θ) * sensitivity) - α * sensitivity =
      β * θ * sensitivity := by
  ring

/-- A concrete nonzero feedback path makes the frozen-input credit incorrect. -/
theorem self_listening_frozen_state_counterexample :
    ((0 : ℝ) + 1 * 1) * 1 ≠ 0 * 1 := by
  norm_num

end CGB.Generation
