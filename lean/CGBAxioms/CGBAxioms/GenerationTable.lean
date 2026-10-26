import CGBAxioms.GenerationCRN

/-!
# Rule-table media (generation chain, M3)

An organism is a finite table of reversible two-unit transitions: "a pair of units with
identity pages `(i, j)` in states `a` may go to states `b`", each with a prefactor `κ`.
The reverse event swaps `a` and `b`. Reversibility and symmetric prefactors then hold by
construction; the only obligation left to an organism is that rules between two interior
pages conserve content, which is decidable rule by rule.
-/

namespace CGB

/-- One reversible transition of the table. -/
structure RuleSpec (Λ : Type*) where
  i : ℕ
  j : ℕ
  a : Λ × Λ
  b : Λ × Λ
  κ : ℝ
  κ_nonneg : 0 ≤ κ

namespace RuleSpec

variable {Λ : Type*} [DecidableEq Λ]

/-- The effect of a rule in one direction (`false`: a ↦ b, `true`: b ↦ a). -/
def apply (s : RuleSpec Λ) (dir : Bool) (i j : ℕ) (p : Λ × Λ) : Λ × Λ :=
  let src := if dir then s.b else s.a
  let dst := if dir then s.a else s.b
  if i = s.i ∧ j = s.j ∧ p = src then dst else p

lemma apply_back (s : RuleSpec Λ) (dir : Bool) (i j : ℕ) (p : Λ × Λ)
    (h : s.apply dir i j p ≠ p) : s.apply (!dir) i j (s.apply dir i j p) = p := by
  unfold apply at h ⊢
  by_cases hm : i = s.i ∧ j = s.j ∧ p = (if dir then s.b else s.a)
  · rw [if_pos hm] at h ⊢
    obtain ⟨hi, hj, hp⟩ := hm
    cases dir <;> simp_all
  · rw [if_neg hm] at h
    exact absurd rfl h

end RuleSpec

/-- The pair reaction medium of a rule table. -/
def tableCRN {U Λ I : Type*} [DecidableEq Λ] {n : ℕ} (spec : Fin n → RuleSpec Λ)
    (ident : U → ℕ) (bid : Finset ℕ) (content : Λ → ℕ) (φ : ℕ → Λ → ℝ) (μ : I → ℝ)
    (birth : ℕ → Λ) (calm : I) : PairCRN U Λ I (Fin n × Bool) where
  ident := ident
  bid := bid
  content := content
  φ := φ
  μ := μ
  eff r i j p := (spec r.1).apply r.2 i j p
  κ r := (spec r.1).κ
  κ_nonneg r := (spec r.1).κ_nonneg
  rev r := (r.1, !r.2)
  birth := birth
  calm := calm

namespace tableCRN

variable {U Λ I : Type*} [Fintype U] [DecidableEq U] [DecidableEq Λ] {n : ℕ}
  (spec : Fin n → RuleSpec Λ) (ident : U → ℕ) (bid : Finset ℕ) (content : Λ → ℕ)
  (φ : ℕ → Λ → ℝ) (μ : I → ℝ) (birth : ℕ → Λ) (calm : I)

/-- **M3**: a rule-table medium is a model of the CGB axioms as soon as its content has a
unique free state over a finite local state type, the calm reservoir potential is zero,
and every rule between two interior
pages conserves content. -/
theorem axioms [Finite Λ] [DecidableEq (U → Λ)] (hcalm : μ calm = 0)
    (hvac : ∃! l, content l = 0)
    (hcons : ∀ k, (spec k).i ∉ bid → (spec k).j ∉ bid →
      content (spec k).a.1 + content (spec k).a.2 = content (spec k).b.1 + content (spec k).b.2) :
    (tableCRN spec ident bid content φ μ birth calm).medium.Axioms := by
  classical
  have hfin : Fintype Λ := Fintype.ofFinite Λ
  refine PairCRN.axioms _ hcalm ⟨fun _ => Set.toFinite _, ⟨Finset.univ.sup content, fun l =>
    Finset.le_sup (f := content) (Finset.mem_univ l)⟩, hvac⟩ ?_ ?_ ?_ ?_
  · rintro ⟨k, dir⟩ i j p hi hj
    change content ((spec k).apply dir i j p).1 + content ((spec k).apply dir i j p).2 =
      content p.1 + content p.2
    unfold RuleSpec.apply
    by_cases hm : i = (spec k).i ∧ j = (spec k).j ∧ p = (if dir then (spec k).b else (spec k).a)
    · rw [if_pos hm]
      obtain ⟨rfl, rfl, rfl⟩ := hm
      have := hcons k hi hj
      cases dir <;> simp <;> omega
    · rw [if_neg hm]
  · intro r; simp [tableCRN]
  · intro r; rfl
  · rintro ⟨k, dir⟩ i j p h
    exact (spec k).apply_back dir i j p h

end tableCRN

end CGB
