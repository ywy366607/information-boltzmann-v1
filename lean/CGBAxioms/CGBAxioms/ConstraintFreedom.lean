import CGBAxioms.ConstraintAudit

/-!
# Families admitted by the same clauses

These are logical model families, not proposed brain mechanisms or capability
experiments. They expose free functions instead of selecting a desired behaviour.
Any restrictions added later need their own general or carrier-specific basis.
-/

open Finset

namespace CGB.ConstraintFreedom

variable {Λ E : Type*} [Fintype Λ] [DecidableEq Λ] [Fintype E]

/-- P1a is qualitative: an arbitrarily large FINITE state alphabet may have
capacity 1. A quantitative information/content relation is not encoded in it. -/
theorem arbitrarily_many_states_at_unit_capacity (n : ℕ) :
    ∃ cost : Fin (n + 1) → ℕ, (∀ l, cost l ≤ 1) ∧
      (∃! l, cost l = 0) ∧ Fintype.card (Fin (n + 1)) = n + 1 := by
  refine ⟨fun l => if l = 0 then 0 else 1, ?_, ?_, Fintype.card_fin _⟩
  · intro l
    dsimp only
    split_ifs <;> norm_num
  · refine ⟨0, by simp, fun l hl => ?_⟩
    by_cases h : l = 0
    · exact h
    · simp [h] at hl

/-- An entirely boundary-exposed site. Under its calm input it is frozen; under
the other input the finite transition and rate tables are arbitrary. -/
noncomputable def openSite (cost : Λ → ℕ) (advance : E → Λ → Λ)
    (speed : E → Λ → ℝ) (hs : ∀ e l, 0 ≤ speed e l) (birth : Λ) :
    Medium Unit Λ E Bool where
  near _ _ := True
  ident _ := 0
  boundary := univ
  content := cost
  parts _ := univ
  act e s x := if s () then fun _ => advance e (x ()) else x
  rate e s x := if s () then speed e (x ()) else 0
  rate_nonneg e s x := by split <;> simp_all [hs]
  init _ := birth
  calm _ := false

/-- Thus the core alone does not determine a driven rule table. This does not
negate life existence, nor waive encoded budgets or separate world constraints. -/
theorem arbitrary_driven_tables_admitted (cost : Λ → ℕ) (hv : ∃! l, cost l = 0)
    (advance : E → Λ → Λ) (speed : E → Λ → ℝ) (hs : ∀ e l, 0 ≤ speed e l) (birth : Λ) :
    (openSite cost advance speed hs birth).Axioms := by
  let M := openSite cost advance speed hs birth
  have hr : M.Rigid := by
    intro σ _
    apply Equiv.ext
    intro u
    exact Subsingleton.elim _ _
  refine ⟨(M.p1a_iff_unique_vacuum).2 hv, ?_, ?_, ?_, M.p2c_of_rigid hr, ?_⟩
  · refine ⟨by intros; trivial, ?_, ?_⟩
    · intro e s x u hu
      simp [M, openSite] at hu
    · intro e s s' x y hh
      have hx : x = y := funext fun u => (hh u (by simp [M, openSite])).1
      have hst : s = s' := funext fun u => (hh u (by simp [M, openSite])).2
      subst y
      subst s'
      exact ⟨rfl, fun _ _ => rfl⟩
  · apply M.p2a_of_every_event_open
    intro e
    change ¬ Disjoint (univ : Finset Unit) univ
    decide
  · refine ⟨fun _ => 1, fun _ => one_pos, fun x y => ?_⟩
    simp [Medium.jump, M, openSite]
  · exact ⟨M.onlyBoundarySenses_of_all_boundary rfl, M.initial_invariant_of_rigid hr⟩

/-- With one event per target state, the exposed-site family reproduces the
off-diagonal rates of ANY finite Markov jump process under its driven input.
This is a statement about the permissiveness of the core representation, not
an encoded-budget or life witness. Self-jumps are omitted just as in Medium.jump. -/
theorem openSite_reproduces_process (cost : Λ → ℕ) (P : JumpProcess Λ) (birth x y : Λ) :
    ((openSite cost (fun e _ => e) (fun e l => P.rate l e)
      (fun e l => P.rate_nonneg l e) birth).jump (fun _ => true)).rate
      (fun _ => x) (fun _ => y) = if x = y then 0 else P.rate x y := by
  by_cases hxy : x = y <;> simp [Medium.jump, openSite, funext_iff, hxy]

/-- A constant-in-time stimulus already breaks detailed balance in the existing
seed witness. Time inhomogeneity and equilibrium reversal are different claims. -/
theorem constant_protocol_can_be_driven :
    (∀ n : ℕ, (fun _ : ℕ => Models.food) n = Models.food) ∧
      ¬ (Models.seed.jump Models.food).Reversible :=
  ⟨fun _ => rfl, Models.seed_driven⟩

end CGB.ConstraintFreedom
