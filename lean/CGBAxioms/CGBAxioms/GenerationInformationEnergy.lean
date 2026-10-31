import CGBAxioms.GenerationPredictiveHarvest
import CGBAxioms.GenerationGradientFilter

/-!
# Information and energy for a finite body in a stream (generation chain, 2026-10-28)

Selected mathematical identities and conditional bounds for the information
economics map. Their assumptions must be checked independently for any
language-world implementation.

* **N1, mismatch is free energy** (`freeEnergy_eq_kl`, `kl_nonneg`, `freeEnergy_at_law`): in the
  landscape `E = -log π` (temperature one), a body whose state law is `q` has free energy
  `Σ q E + Σ q log q = KL(q ‖ π) ≥ 0`, zero at `q = π`. What a body can be driven by is how far its
  state law differs from the chosen reference law. A physical interpretation
  requires the reference law and energy convention to be justified separately.
* **N2, one snapshot misses hidden memory** (`snapshot_misses_hidden`): if a measurement `M` reads
  only `X` (its channel `k x m` does not depend on the hidden part `Y`), the information it carries
  about `(X, Y)` equals the information about `X`. Hidden degrees of freedom (a topic, an intention,
  a world's slow part) are reached only through records across time: memory.
* **N3, an assumed work/erasure accounting bounds net harvest**
  (`finite_memory_bounds_net_work`): if the work drawn from acquired information `I` is at most
  `kT · I` (Sagawa–Ueda), and the part of `I` beyond the capacity must be erased (G16a
  `export_forced`) at a Landauer cost of at least `kT` per unit, the net is at most `kT · cap`,
  whatever `I` is. The theorem does not derive the two accounting bounds or
  establish an equilibrium-world model or a trajectory-level persistence result.
* **N4, passivity ends** (`passivity_ends`): with drift `q > 0` and no observation (W2 with gain
  zero), the variance of a hidden state exceeds any bound, so the information `½ log (1 + V / r)` of
  one observation exceeds any fixed numeric threshold. This alone does not
  imply that a particular policy acts, or that information and action cost have
  commensurate units or the assumed decision value.
* **N5, learning saturates** (`mixture_regret_le`): a learner mixing hypotheses `θ` with prior
  weights `w θ` codes any sequence at most `log (1 / w θ₀)` worse than the source's own law `θ₀`.
  This pathwise excess-code-length bound does not imply that a source stops
  providing useful information or that the learner's parameters stop changing.
-/

namespace CGB.InformationEnergy

open Finset

section FreeEnergy

variable {S : Type*} [Fintype S]

/-- Free energy (temperature one) of a state law `q` in the energy landscape `E = -log π`. -/
noncomputable def freeEnergy (π q : S → ℝ) : ℝ :=
  ∑ s, q s * (-Real.log (π s)) + ∑ s, q s * Real.log (q s)

/-- Relative entropy `KL(q ‖ π)`. -/
noncomputable def kl (q π : S → ℝ) : ℝ := ∑ s, q s * (Real.log (q s) - Real.log (π s))

/-- **N1**: the free energy in the landscape `-log π` is the relative entropy to `π`. -/
theorem freeEnergy_eq_kl (π q : S → ℝ) : freeEnergy π q = kl q π := by
  unfold freeEnergy kl
  rw [← sum_add_distrib]
  exact sum_congr rfl fun s _ => by ring

/-- The law itself has free energy zero. -/
theorem freeEnergy_at_law (π : S → ℝ) : freeEnergy π π = 0 := by
  rw [freeEnergy_eq_kl]
  unfold kl
  simp

/-- **N1**: free energy is never below that of the law (Gibbs' inequality). -/
theorem kl_nonneg (q π : S → ℝ) (hq0 : ∀ s, 0 ≤ q s) (hq : ∑ s, q s = 1) (hπ0 : ∀ s, 0 < π s)
    (hπ : ∑ s, π s = 1) : 0 ≤ kl q π := by
  have h := CGB.PredictiveHarvest.gibbs_inequality q π hq0 hq hπ0 hπ
  have e : kl q π = ∑ s, q s * Real.log (q s) - ∑ s, q s * Real.log (π s) := by
    unfold kl
    rw [← sum_sub_distrib]
    exact sum_congr rfl fun s _ => by ring
  linarith

end FreeEnergy

section Snapshot

variable {X Y M : Type*} [Fintype X] [Fintype Y] [Fintype M]

/-- Law of the measurement outcome when `(X, Y)` has law `p` and the measurement reads `X` through
the channel `k`. -/
noncomputable def outLaw (p : X → Y → ℝ) (k : X → M → ℝ) (m : M) : ℝ := ∑ x, ∑ y, p x y * k x m

/-- Mutual information between `(X, Y)` and the outcome. -/
noncomputable def infoJoint (p : X → Y → ℝ) (k : X → M → ℝ) : ℝ :=
  ∑ x, ∑ y, ∑ m, p x y * k x m * Real.log (p x y * k x m / (p x y * outLaw p k m))

/-- Mutual information between `X` alone and the outcome. -/
noncomputable def infoVisible (p : X → Y → ℝ) (k : X → M → ℝ) : ℝ :=
  ∑ x, ∑ m, (∑ y, p x y) * k x m * Real.log (k x m / outLaw p k m)

/-- **N2**: a measurement that reads only the visible part carries no information about the hidden
part beyond what it carries about the visible one. -/
theorem snapshot_misses_hidden (p : X → Y → ℝ) (k : X → M → ℝ) :
    infoJoint p k = infoVisible p k := by
  unfold infoJoint infoVisible
  refine sum_congr rfl fun x _ => ?_
  have h1 : ∀ y m, p x y * k x m * Real.log (p x y * k x m / (p x y * outLaw p k m)) =
      p x y * (k x m * Real.log (k x m / outLaw p k m)) := by
    intro y m
    by_cases hp : p x y = 0
    · simp [hp]
    · rw [mul_div_mul_left _ _ hp]
      ring
  simp_rw [h1]
  rw [sum_comm]
  refine sum_congr rfl fun m _ => ?_
  rw [mul_assoc, sum_mul]

end Snapshot

/-- **N3**: work drawn from acquired information `I` at most `kT · I`, erasure of everything beyond
the capacity at least `kT` per unit: the net never exceeds `kT · cap`. -/
theorem finite_memory_bounds_net_work (W D I cap kT : ℝ) (hk : 0 ≤ kT) (hwork : W ≤ kT * I)
    (herase : kT * (I - cap) ≤ D) : W - D ≤ kT * cap := by
  nlinarith

/-- **N4**: with drift `q > 0` and no observation (W2's recursion at gain zero), the information of
one observation, `½ log (1 + V / r)`, exceeds any cost `c > 0` after finitely many ticks. -/
theorem passivity_ends (V : ℕ → ℝ) (q r c : ℝ) (hq : 0 < q) (hr : 0 < r) (hc : 0 < c)
    (hV0 : 0 ≤ V 0) (hrec : ∀ t, V (t + 1) = V t + q) :
    ∃ t, c < Real.log (1 + V t / r) / 2 := by
  set Mb := r * (Real.exp (2 * c) - 1) with hMb
  have hexp : 1 < Real.exp (2 * c) := by
    have : 0 < 2 * c := by linarith
    have h2 := Real.add_one_lt_exp (ne_of_gt this)
    linarith
  have hM : 0 < Mb := mul_pos hr (by linarith)
  obtain ⟨t, ht⟩ := CGB.GradientFilter.vanishing_gain_loses_under_drift V (fun _ => 0) q r Mb hq hr.le hM
    hV0 (fun _ => le_refl 0) (fun _ => by positivity) (fun t => by rw [hrec t]; ring)
  refine ⟨t, ?_⟩
  have hdiv : Real.exp (2 * c) - 1 < V t / r := by
    rw [lt_div_iff₀ hr]
    linarith
  have hpos : 0 < 1 + V t / r := by linarith
  have hlog : 2 * c < Real.log (1 + V t / r) := by
    rw [Real.lt_log_iff_exp_lt hpos]
    linarith
  linarith

/-- **N5 (mixture regret)**: the Bayes mixture codes the sequence at most `log (1 / w θ₀)` worse
than the source's law `θ₀` (pathwise, for every sequence and length). `P θ` is the probability the
hypothesis `θ` gives to the whole sequence. -/
theorem mixture_regret_le {Θ : Type*} [Fintype Θ] (w P : Θ → ℝ) (hw : ∀ θ, 0 < w θ)
    (hP : ∀ θ, 0 ≤ P θ) (θ₀ : Θ) (h0 : 0 < P θ₀) :
    -Real.log (∑ θ, w θ * P θ) ≤ -Real.log (P θ₀) + Real.log (1 / w θ₀) := by
  have hle : w θ₀ * P θ₀ ≤ ∑ θ, w θ * P θ :=
    single_le_sum (f := fun θ => w θ * P θ) (fun θ _ => mul_nonneg (hw θ).le (hP θ)) (mem_univ θ₀)
  have hpos : 0 < w θ₀ * P θ₀ := mul_pos (hw θ₀) h0
  have hlog := Real.log_le_log hpos hle
  rw [Real.log_mul (hw θ₀).ne' h0.ne', one_div, Real.log_inv] at *
  linarith

end CGB.InformationEnergy
