import CGBAxioms.Foundation

/-!
# Joint audit of irredundancy and generative freedom

No new physical clause is introduced. Atomized is a name-preserving factoring
of the old conjunctions, not a claim that these are globally minimal atoms.
The SAME event-permutation witness in P2c is retained: splitting its existential
into unrelated rate/effect witnesses would silently weaken the theory.
-/

open Finset

namespace CGB.Medium

variable {U Λ A I : Type*} (M : Medium U Λ A I)

structure Atomized [Fintype U] [Fintype A] [DecidableEq (U → Λ)] : Prop where
  finiteSublevels : ∀ n, {l : Λ | M.content l ≤ n}.Finite
  boundedContent : ∃ K, ∀ l, M.content l ≤ K
  uniqueVacuum : ∃! l, M.content l = 0
  participantsContact : ∀ a, ∀ u ∈ M.parts a, ∀ v ∈ M.parts a, M.near u v
  writesParticipants : ∀ a s x u, u ∉ M.parts a → M.act a s x u = x u
  rateReadsParticipants : ∀ a s s' x y,
    (∀ u ∈ M.parts a, x u = y u ∧ s u = s' u) → M.rate a s x = M.rate a s' y
  effectReadsParticipants : ∀ a s s' x y,
    (∀ u ∈ M.parts a, x u = y u ∧ s u = s' u) →
    ∀ u ∈ M.parts a, M.act a s x u = M.act a s' y u
  closedConservation : M.P2a
  calmBalance : M.P2b
  coherentEquivariance : M.P2c
  boundaryRate : ∀ a s s' x,
    (∀ u ∈ M.boundary, s u = s' u) → M.rate a s x = M.rate a s' x
  boundaryEffect : ∀ a s s' x,
    (∀ u ∈ M.boundary, s u = s' u) → M.act a s x = M.act a s' x
  initialInvariant : ∀ σ, M.IsAut σ → M.init ∘ σ.symm = M.init

theorem axioms_iff_atomized [Fintype U] [Fintype A] [DecidableEq (U → Λ)] :
    M.Axioms ↔ M.Atomized := by
  constructor
  · intro h
    exact ⟨h.p1a.1, h.p1a.2.1, h.p1a.2.2, h.p1b.1,
      h.p1b.2.1, fun a s s' x y hh => (h.p1b.2.2 a s s' x y hh).1,
      fun a s s' x y hh => (h.p1b.2.2 a s s' x y hh).2,
      h.p2a, h.p2b, h.p2c, fun a s s' x hh => (h.p3.1 a s s' x hh).1,
      fun a s s' x hh => (h.p3.1 a s s' x hh).2, h.p3.2⟩
  · intro h
    exact ⟨⟨h.finiteSublevels, h.boundedContent, h.uniqueVacuum⟩,
      ⟨h.participantsContact, h.writesParticipants,
        fun a s s' x y hh => ⟨h.rateReadsParticipants a s s' x y hh,
          h.effectReadsParticipants a s s' x y hh⟩⟩,
      h.closedConservation, h.calmBalance, h.coherentEquivariance,
      ⟨fun a s s' x hh => ⟨h.boundaryRate a s s' x hh,
        h.boundaryEffect a s s' x hh⟩, h.initialInvariant⟩⟩

/-- The first two cost clauses jointly say finite local state space in this
language. Once it is finite, the numerical cost function remains undetermined. -/
theorem p1a_iff_finite_and_unique :
    M.P1a ↔ Finite Λ ∧ ∃! l, M.content l = 0 := by
  constructor
  · intro h
    exact ⟨M.finite_local_states_of_p1a h, h.2.2⟩
  · rintro ⟨hf, hv⟩
    let _ : Finite Λ := hf
    exact (M.p1a_iff_unique_vacuum).2 hv

/-- Rigid structural data removes all nontrivial comparisons from equivariance.
This is a parameter regime, not global redundancy of P2c. -/
def Rigid : Prop := ∀ σ, M.IsAut σ → σ = Equiv.refl U

theorem rigid_of_injective_identity (hi : Function.Injective M.ident) : M.Rigid := by
  intro σ hσ
  apply Equiv.ext
  intro u
  exact hi (hσ.2.1 u)

theorem p2c_of_rigid (hr : M.Rigid) : M.P2c := by
  intro σ hσ
  obtain rfl := hr σ hσ
  exact ⟨Equiv.refl A, fun _ _ _ => ⟨rfl, rfl⟩⟩

theorem initial_invariant_of_rigid (hr : M.Rigid) :
    ∀ σ, M.IsAut σ → M.init ∘ σ.symm = M.init := by
  intro σ hσ
  obtain rfl := hr σ hσ
  rfl

theorem onlyBoundarySenses_of_all_boundary [Fintype U]
    (hb : M.boundary = univ) : M.OnlyBoundarySenses := by
  intro a s s' x hh
  have hs : s = s' := funext fun u => hh u (by simp [hb])
  subst s'
  exact ⟨rfl, rfl⟩

theorem p2a_of_every_event_open [Fintype U] (ho : ∀ a, ¬ M.Closed a) : M.P2a :=
  fun a _ _ hc _ => False.elim (ho a hc)

theorem empty_participants_no_effect (hl : ∀ a s x u,
    u ∉ M.parts a → M.act a s x u = x u) (a : A) (ha : M.parts a = ∅)
    (s : U → I) (x : U → Λ) : M.act a s x = x := by
  funext u
  exact hl a s x u (by simp [ha])

/-- A genuine joint consequence: event equivariance descends to the aggregate
jump kernel, including the single shared event-permutation witness. -/
theorem jump_equivariant [Fintype A] [DecidableEq (U → Λ)] (he : M.P2c)
    (σ : Equiv.Perm U) (hσ : M.IsAut σ) (s : U → I) (x y : U → Λ) :
    (M.jump (s ∘ σ.symm)).rate (x ∘ σ.symm) (y ∘ σ.symm) = (M.jump s).rate x y := by
  obtain ⟨τ, hτ⟩ := he σ hσ
  have hinj : Function.Injective (fun f : U → Λ => f ∘ σ.symm) := by
    intro f g hh
    funext u
    have hu := congrFun hh (σ u)
    simpa only [Function.comp_apply, Equiv.symm_apply_apply] using hu
  simp only [jump]
  rw [← Equiv.sum_comp τ]
  apply Finset.sum_congr rfl
  intro a _
  rw [(hτ a s x).1, (hτ a s x).2]
  simp only [ne_eq, hinj.eq_iff]

/-- P3 with no boundary removes ALL stimulus dependence. Combined with P2b,
every stimulus then gives the same reversible aggregate kernel. -/
theorem empty_boundary_every_stimulus_reversible [Fintype U] [DecidableEq U]
    [Fintype Λ] [DecidableEq Λ] [Fintype A]
    (hb : M.boundary = ∅) (hs : M.OnlyBoundarySenses) (hd : M.P2b) (s : U → I) :
    (M.jump s).Reversible := by
  have hh : ∀ u ∈ M.boundary, s u = M.calm u := by
    intro u hu
    simp [hb] at hu
  obtain ⟨π, hp, hdb⟩ := hd
  refine ⟨π, hp, fun x y => ?_⟩
  rw [M.jump_blanket hs hh x y, M.jump_blanket hs hh y x]
  exact hdb x y

/-- The six clauses admit a continuum of clock scales; physical clock admission
is an extra obligation when comparing implementations in a fixed world. -/
noncomputable def rescale (c : ℝ) (hc : 0 ≤ c) : Medium U Λ A I :=
  { M with rate := fun a s x => c * M.rate a s x
           rate_nonneg := fun a s x => mul_nonneg hc (M.rate_nonneg a s x) }

theorem rescale_jump [Fintype A] [DecidableEq (U → Λ)] (c : ℝ) (hc : 0 ≤ c)
    (s : U → I) (x y : U → Λ) :
    ((M.rescale c hc).jump s).rate x y = c * (M.jump s).rate x y := by
  simp only [jump, rescale, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a _
  by_cases ha : M.act a s x = y <;> by_cases hxy : x = y <;> simp [ha, hxy]

theorem rescale_axioms [Fintype U] [Fintype A] [DecidableEq (U → Λ)]
    (c : ℝ) (hc : 0 ≤ c) (h : M.Axioms) : (M.rescale c hc).Axioms := by
  refine ⟨h.p1a, ⟨h.p1b.1, h.p1b.2.1, ?_⟩, ?_, ?_, ?_, ?_⟩
  · intro a s s' x y hh
    obtain ⟨hr, ha⟩ := h.p1b.2.2 a s s' x y hh
    exact ⟨congrArg (c * ·) hr, ha⟩
  · intro a s x hclosed hpos
    apply h.p2a a s x hclosed
    by_contra hn
    have hz : M.rate a s x = 0 := le_antisymm (le_of_not_gt hn) (M.rate_nonneg a s x)
    simp [rescale, hz] at hpos
  · obtain ⟨π, hp, hd⟩ := h.p2b
    refine ⟨π, hp, fun x y => ?_⟩
    change π x * ((M.rescale c hc).jump M.calm).rate x y =
      π y * ((M.rescale c hc).jump M.calm).rate y x
    rw [rescale_jump, rescale_jump]
    linear_combination c * hd x y
  · intro σ hσ
    obtain ⟨τ, hτ⟩ := h.p2c σ hσ
    exact ⟨τ, fun a s x => ⟨congrArg (c * ·) (hτ a s x).1, (hτ a s x).2⟩⟩
  · exact ⟨fun a s s' x hh =>
      ⟨congrArg (c * ·) (h.p3.1 a s s' x hh).1, (h.p3.1 a s s' x hh).2⟩, h.p3.2⟩

theorem rescale_changes_nonzero_jump [Fintype A] [DecidableEq (U → Λ)]
    (s : U → I) (x y : U → Λ) (hp : 0 < (M.jump s).rate x y) :
    ((M.rescale 2 (by norm_num)).jump s).rate x y ≠ (M.jump s).rate x y := by
  rw [rescale_jump]
  linarith

end CGB.Medium

namespace CGB.JumpProcess

variable {S : Type*}

/-- Full free data under detailed balance: positive state weights plus arbitrary
nonnegative symmetric conductances. No transition support or time scale is fixed. -/
noncomputable def fromConductance (π : S → ℝ) (hp : ∀ x, 0 < π x)
    (C : S → S → ℝ) (hc : ∀ x y, 0 ≤ C x y) : JumpProcess S where
  rate x y := C x y / π x
  rate_nonneg x y := div_nonneg (hc x y) (hp x).le

theorem fromConductance_reversible (π : S → ℝ) (hp : ∀ x, 0 < π x)
    (C : S → S → ℝ) (hc : ∀ x y, 0 ≤ C x y) (hs : ∀ x y, C x y = C y x) :
    (fromConductance π hp C hc).Reversible := by
  refine ⟨π, hp, fun x y => ?_⟩
  change π x * (C x y / π x) = π y * (C y x / π y)
  field_simp [ne_of_gt (hp x), ne_of_gt (hp y)]
  exact hs x y

theorem reversible_conductance_representation (P : JumpProcess S) (h : P.Reversible) :
    ∃ π : S → ℝ, (∀ x, 0 < π x) ∧ ∃ C : S → S → ℝ,
      (∀ x y, 0 ≤ C x y) ∧ (∀ x y, C x y = C y x) ∧
        ∀ x y, P.rate x y = C x y / π x := by
  obtain ⟨π, hp, hd⟩ := h
  refine ⟨π, hp, fun x y => π x * P.rate x y,
    fun x y => mul_nonneg (hp x).le (P.rate_nonneg x y), hd, fun x y => ?_⟩
  field_simp [ne_of_gt (hp x)]

end CGB.JumpProcess
