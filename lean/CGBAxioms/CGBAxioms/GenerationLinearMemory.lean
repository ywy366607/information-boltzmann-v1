import CGBAxioms.GenerationIgnition

/-!
# Linear recurrent memory cannot latch (generation chain, corollary of G17/G24)

Gated DeltaNet (and every gated linear recurrence: Mamba, RWKV, linear attention with decay)
updates each row `x` of its state as `x ↦ α (x - β ⟪k, x⟫_ℝ k) + (write)`, with `‖k‖ = 1`,
`β ∈ [0, 2]`, `α < 1`. In the chain's terms `α` is unselective export (G17), the delta term is
selective export, the write is intake.

* `dot_delta`, `delta_nonexpansive`: the delta step never expands:
  `‖x - β⟨k,x⟩k‖² = ‖x‖² - β(2-β)⟨k,x⟩²`. `deltaStep_sub`: it is linear, so two histories'
  states differ by the step applied to their gap.
* `gated_delta_contracts`: with the gate the squared gap contracts by `α²`.
* `histories_merge`: two histories that differ before step 0 differ after `T` steps by at most
  `∏ α_t` times the initial gap. With a uniform bound `α_t ≤ ᾱ < 1` the memory horizon is finite.
* `linear_hold_keeps_error`: a linear map that holds a signal holds every perturbation along
  it just as well. Linear memory cannot correct errors without forgetting what it holds;
  error correction (an attractor, a latch) needs nonlinearity.
-/

namespace CGB.Generation

open Finset

variable {n : ℕ}

/-- Dot product of state rows. -/
def dot (x y : Fin n → ℝ) : ℝ := ∑ i, x i * y i

/-- The delta step on one state row: erase along the key `k` with strength `β`. -/
def deltaStep (k : Fin n → ℝ) (β : ℝ) (x : Fin n → ℝ) : Fin n → ℝ := fun i => x i - β * dot k x * k i

theorem dot_delta (k x : Fin n → ℝ) (hk : dot k k = 1) (β : ℝ) :
    dot (deltaStep k β x) (deltaStep k β x) = dot x x - β * (2 - β) * dot k x ^ 2 := by
  set p := dot k x with hp
  have e : dot (deltaStep k β x) (deltaStep k β x) =
      ∑ i, (x i * x i - 2 * (β * p) * (k i * x i) + (β * p) ^ 2 * (k i * k i)) := by
    unfold dot
    exact Finset.sum_congr rfl fun i _ => by simp only [deltaStep, ← hp]; ring
  rw [e, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  have h1 : ∑ i, k i * x i = p := rfl
  have h2 : ∑ i, k i * k i = 1 := hk
  have h3 : ∑ i, x i * x i = dot x x := rfl
  rw [h1, h2, h3]; ring

/-- The delta step never expands (squared norm), for `0 ≤ β ≤ 2`. -/
theorem delta_nonexpansive (k x : Fin n → ℝ) (hk : dot k k = 1) {β : ℝ} (h0 : 0 ≤ β)
    (h2 : β ≤ 2) : dot (deltaStep k β x) (deltaStep k β x) ≤ dot x x := by
  rw [dot_delta k x hk β]
  have : 0 ≤ β * (2 - β) * dot k x ^ 2 := by
    have : 0 ≤ 2 - β := by linarith
    positivity
  linarith

/-- One gated delta step on the difference of two states (the write cancels): the squared
gap shrinks by `α²`. Since `deltaStep` is linear in `x`, this is the step on the gap. -/
theorem gated_delta_contracts (k d : Fin n → ℝ) (hk : dot k k = 1) {α β : ℝ} (h0 : 0 ≤ β)
    (h2 : β ≤ 2) :
    dot (fun i => α * deltaStep k β d i) (fun i => α * deltaStep k β d i) ≤ α ^ 2 * dot d d := by
  have h := delta_nonexpansive k d hk h0 h2
  have e : dot (fun i => α * deltaStep k β d i) (fun i => α * deltaStep k β d i) =
      α ^ 2 * dot (deltaStep k β d) (deltaStep k β d) := by
    simp only [dot, Finset.mul_sum]; exact Finset.sum_congr rfl fun i _ => by ring
  rw [e]; exact mul_le_mul_of_nonneg_left h (sq_nonneg α)

theorem deltaStep_sub (k : Fin n → ℝ) (β : ℝ) (x y : Fin n → ℝ) :
    deltaStep k β x - deltaStep k β y = deltaStep k β (x - y) := by
  funext i
  simp only [deltaStep, dot, Pi.sub_apply, mul_sub, Finset.sum_sub_distrib]
  ring

/-- **Histories merge**: if every step shrinks the gap between two states by at most `α t`,
the gap after `T` steps is at most `∏ α t` times the initial gap. -/
theorem histories_merge (gap : ℕ → ℝ) (α : ℕ → ℝ) (hα : ∀ t, 0 ≤ α t)
    (hstep : ∀ t, gap (t + 1) ≤ α t * gap t) (T : ℕ) :
    gap T ≤ (∏ t ∈ Finset.range T, α t) * gap 0 := by
  induction T with
  | zero => simp
  | succ T ih =>
    rw [Finset.prod_range_succ]
    calc gap (T + 1) ≤ α T * gap T := hstep T
      _ ≤ α T * ((∏ t ∈ Finset.range T, α t) * gap 0) := mul_le_mul_of_nonneg_left ih (hα T)
      _ = (∏ t ∈ Finset.range T, α t) * α T * gap 0 := by ring

/-- With a uniform gate bound the gap decays at least geometrically. -/
theorem histories_merge_uniform (gap : ℕ → ℝ) (α : ℕ → ℝ) {ab : ℝ} (hα : ∀ t, 0 ≤ α t)
    (hb : ∀ t, α t ≤ ab) (h0 : 0 ≤ gap 0) (hstep : ∀ t, gap (t + 1) ≤ α t * gap t) (T : ℕ) :
    gap T ≤ ab ^ T * gap 0 := by
  refine (histories_merge gap α hα hstep T).trans ?_
  apply mul_le_mul_of_nonneg_right _ h0
  calc ∏ t ∈ Finset.range T, α t ≤ ∏ _t ∈ Finset.range T, ab :=
        Finset.prod_le_prod₀ (fun t _ => hα t) (fun t _ => hb t)
    _ = ab ^ T := by simp

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- **A linear hold cannot clean up**: if a linear map holds the signal `s` (`L s = s`), a
perturbation along `s` is held exactly as well, `L (s + t • s) - s = t • s`. Holding and
error correction are tied in linear memory, as writing and forgetting are tied in
equilibrium (G24); an attractor that maps `s + δ` back to `s` needs nonlinearity. -/
theorem linear_hold_keeps_error (L : E →ₗ[ℝ] E) {s : E} (hold : L s = s) (t : ℝ) :
    L (s + t • s) - s = t • s := by
  rw [map_add, map_smul, hold]; abel

end CGB.Generation
