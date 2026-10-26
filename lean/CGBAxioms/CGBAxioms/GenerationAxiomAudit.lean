import CGBAxioms.GenerationCRN

/-!
# Which axiom parts work (generation chain, audit)

Usage in the generation chain (G1–G22, M1–M2): P1b's locality parts (participants in
contact, writes and reads only participants), P2a, P2b and P3's boundary sensing enter
derivations; P2c and the symmetric birth state are what G21 turns into "differentiation
needs fluctuation, boundary, identity or history". Contact homogeneity (formerly the first part of
P1b) enters no derivation. This module proves what it cost; it was removed from P1b.
-/

namespace CGB.Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

/-- The number of units in contact with `u`. -/
noncomputable def degree [Fintype U] (u : U) : ℕ := by
  classical
  exact (Finset.univ.filter fun v => M.near u v).card

/-- **Homogeneity forces a regular contact graph**: every unit has the same number of
contacts. So with homogeneity in P1b (before 2026-10-26) a body whose contact has units of
different degree (a tree with leaves, a hub, a finite grid with a surface, a line graph of
compartments of different sizes) was not a model. -/
theorem homogeneous_regular [Fintype U] (h : M.HomogeneousContact) (u v : U) :
    M.degree u = M.degree v := by
  classical
  obtain ⟨σ, hσ, huv⟩ := h u v
  unfold degree
  refine Finset.card_bij (fun w _ => σ w) ?_ ?_ ?_
  · intro w hw
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
    rw [← huv]; exact (hσ u w).2 hw
  · intro a _ b _ hab; exact σ.injective hab
  · intro w hw
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hw
    refine ⟨σ.symm w, ?_, by simp⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    have := (hσ u (σ.symm w)).1
    rw [huv, Equiv.apply_symm_apply] at this
    exact this hw

/-- A unit with a different number of contacts than another rules out homogeneity. -/
theorem not_homogeneous_of_degrees [Fintype U] {u v : U} (hd : M.degree u ≠ M.degree v) :
    ¬ M.HomogeneousContact :=
  fun h => hd (M.homogeneous_regular h u v)

end CGB.Medium
