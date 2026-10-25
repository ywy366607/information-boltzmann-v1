import CGBAxioms.SubconditionGraph

/-!
# Full remaining-thirteen certificates

Atomic irredundancy is relative to this fixed general Medium language, which does
not assume finite local states. Logical countermodels are not capability studies
or evidence that a condition is physically necessary for life.
-/

open Finset

namespace CGB.Medium

inductive AtomicCondition where
  | a01 | a02 | a03 | a04 | a05 | a06 | a07
  | a08 | a09 | a10 | a11 | a12 | a13 | a14
  deriving DecidableEq

variable {U Λ A I : Type*} (M : Medium U Λ A I)

def Holds [Fintype U] [Fintype A] [DecidableEq (U → Λ)] : AtomicCondition → Prop
  | .a01 => ∀ n, {l : Λ | M.content l ≤ n}.Finite
  | .a02 => ∃ K, ∀ l, M.content l ≤ K
  | .a03 => ∃! l, M.content l = 0
  | .a04 => M.HomogeneousContact
  | .a05 => M.ParticipantsContact
  | .a06 => M.WritesParticipants
  | .a07 => M.RateReadsParticipants
  | .a08 => M.EffectReadsParticipants
  | .a09 => M.P2a
  | .a10 => M.P2b
  | .a11 => M.P2c
  | .a12 => M.BoundaryRate
  | .a13 => M.BoundaryEffect
  | .a14 => M.InitialInvariant

/-- Every other condition holds, and exactly the designated condition fails. -/
def Isolates [Fintype U] [Fintype A] [DecidableEq (U → Λ)]
    (target : AtomicCondition) : Prop :=
  (∀ c, c ≠ target → M.Holds c) ∧ ¬ M.Holds target

theorem atomized_iff_all_conditions [Fintype U] [Fintype A]
    [DecidableEq (U → Λ)] : M.Atomized ↔ ∀ c, M.Holds c := by
  constructor
  · intro h c
    cases c
    all_goals first | exact h.finiteSublevels | exact h.boundedContent |
      exact h.uniqueVacuum | exact h.homogeneousContact | exact h.participantsContact |
      exact h.writesParticipants | exact h.rateReadsParticipants |
      exact h.effectReadsParticipants | exact h.closedConservation | exact h.calmBalance |
      exact h.coherentEquivariance | exact h.boundaryRate | exact h.boundaryEffect |
      exact h.initialInvariant
  · intro h
    exact ⟨h .a01, h .a02, h .a03, h .a04, h .a05, h .a06, h .a07,
      h .a08, h .a09, h .a10, h .a11, h .a12, h .a13, h .a14⟩

open Models

theorem a01_independent : twinTape.Isolates .a01 := by
  obtain ⟨hn, h1b, h9, h10, h11, hp3, _, h2, h3⟩ := twinTape_witness
  obtain ⟨h4, h5, h6, h7, h8⟩ := (p1b_iff_atoms twinTape).mp h1b
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms twinTape).mp hp3
  have hn1 : ¬ (∀ n, {l | twinTape.content l ≤ n}.Finite) := fun h => hn ⟨h, h2, h3⟩
  refine ⟨?_, hn1⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a02_independent : twinBag.Isolates .a02 := by
  obtain ⟨hn, h1b, h9, h10, h11, hp3, _, h1, h3⟩ := twinBag_witness
  obtain ⟨h4, h5, h6, h7, h8⟩ := (p1b_iff_atoms twinBag).mp h1b
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms twinBag).mp hp3
  have hn1 : ¬ (∃ K, ∀ l, twinBag.content l ≤ K) := fun h => hn ⟨h1, h, h3⟩
  refine ⟨?_, hn1⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a03_independent : twinBlank.Isolates .a03 := by
  obtain ⟨hn, h1b, h9, h10, h11, hp3, _, h1, h2⟩ := twinBlank_witness
  obtain ⟨h4, h5, h6, h7, h8⟩ := (p1b_iff_atoms twinBlank).mp h1b
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms twinBlank).mp hp3
  have hn1 : ¬ (∃! l, twinBlank.content l = 0) := fun h => hn ⟨h1, h2, h⟩
  refine ⟨?_, hn1⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a09_independent : twinToggle.Isolates .a09 := by
  obtain ⟨hn1, hp1a, h1b, h10, h11, hp3, _⟩ := twinToggle_witness
  obtain ⟨h1, h2, h3⟩ := hp1a
  obtain ⟨h4, h5, h6, h7, h8⟩ := (p1b_iff_atoms twinToggle).mp h1b
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms twinToggle).mp hp3
  refine ⟨?_, hn1⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a10_independent : twinCycle.Isolates .a10 := by
  obtain ⟨hn1, hp1a, h1b, h9, h11, hp3, _⟩ := twinCycle_witness
  obtain ⟨h1, h2, h3⟩ := hp1a
  obtain ⟨h4, h5, h6, h7, h8⟩ := (p1b_iff_atoms twinCycle).mp h1b
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms twinCycle).mp hp3
  refine ⟨?_, hn1⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a11_independent : triLookup.Isolates .a11 := by
  obtain ⟨hn1, hp1a, h1b, h9, h10, hp3⟩ := triLookup_witness
  obtain ⟨h1, h2, h3⟩ := hp1a
  obtain ⟨h4, h5, h6, h7, h8⟩ := (p1b_iff_atoms triLookup).mp h1b
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms triLookup).mp hp3
  refine ⟨?_, hn1⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a07_independent : farFlip.Isolates .a07 := by
  obtain ⟨_, hp1a, h9, h10, h11, hp3, _⟩ := farFlip_witness
  obtain ⟨h1, h2, h3⟩ := hp1a
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms farFlip).mp hp3
  obtain ⟨hn1, h8, h6, h5, h4⟩ := farFlip_isolates_a07
  refine ⟨?_, hn1⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a12_independent : innerClock.Isolates .a12 := by
  obtain ⟨_, hp1a, h1b, h9, h10, h11, _⟩ := innerClock_witness
  obtain ⟨h1, h2, h3⟩ := hp1a
  obtain ⟨h4, h5, h6, h7, h8⟩ := (p1b_iff_atoms innerClock).mp h1b
  obtain ⟨hn1, h13, h14⟩ := innerClock_isolates_a12
  refine ⟨?_, hn1⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a14_independent : pairPreset.Isolates .a14 := by
  obtain ⟨_, hp1a, h1b, h9, h10, h11, _⟩ := pairPreset_witness
  obtain ⟨h1, h2, h3⟩ := hp1a
  obtain ⟨h4, h5, h6, h7, h8⟩ := (p1b_iff_atoms pairPreset).mp h1b
  obtain ⟨hn1, h12, h13⟩ := pairPreset_isolates_a14
  refine ⟨?_, hn1⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a04_independent : path3Rigid.Isolates .a04 := by
  obtain ⟨h, hr, _⟩ := finite_rigid_reducedCore_not_atomized
  have h1 := a01_of_finite path3Rigid.content
  have h2 := a02_of_finite path3Rigid.content
  have h11 := p2c_of_rigid path3Rigid hr
  have h14 := initial_invariant_of_rigid path3Rigid hr
  obtain ⟨h3, h5, h6, h7, h8, h9, h10, h12, h13⟩ := h
  refine ⟨?_, path3_not_homogeneous⟩
  intro c hc
  cases c <;> simp_all [Holds, InitialInvariant]

/-- The same positive-rate involution with either all or no sites declared as
participants. It separates contact admissibility from the write footprint. -/
noncomputable def globalFlip (participants : Finset (Fin 2)) :
    Medium (Fin 2) (Fin 3) Unit Unit where
  near u v := u = v
  ident _ := 0
  boundary := ∅
  content := flipCost
  parts _ := participants
  act _ _ x u := flip12 (x u)
  rate _ _ _ := 1
  rate_nonneg _ _ _ := zero_le_one
  init _ := 1
  calm _ := ()

theorem globalFlip_other_clauses (participants : Finset (Fin 2)) :
    (globalFlip participants).P1a ∧ (globalFlip participants).P2a ∧
      (globalFlip participants).P2b ∧ (globalFlip participants).P2c ∧
      (globalFlip participants).P3 := by
  refine ⟨flipCost_p1a rfl, ?_, ?_, ?_, ?_⟩
  · intro a s x _ _
    exact Finset.sum_congr rfl (fun u _ => flipCost_flip12 (x u))
  · refine p2b_of_micro _ id (fun _ => rfl) ?_
    intro a x _
    exact ⟨funext (fun u => flip12_flip12 (x u)), rfl⟩
  · intro σ _
    exact ⟨Equiv.refl Unit, fun a s x => ⟨rfl, rfl⟩⟩
  · exact ⟨fun _ _ _ _ _ => ⟨rfl, rfl⟩, fun _ _ => rfl⟩

theorem a05_independent : (globalFlip univ).Isolates .a05 := by
  obtain ⟨⟨h1, h2, h3⟩, h9, h10, h11, hp3⟩ := globalFlip_other_clauses univ
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms (globalFlip univ)).mp hp3
  have h4 : (globalFlip univ).HomogeneousContact := eq_homogeneous
  obtain ⟨h6, h7, h8⟩ := universal_parts_satisfies_a06_a07_a08
    (globalFlip univ) (fun _ => rfl)
  have hn : ¬ (globalFlip univ).ParticipantsContact := by
    intro h
    have he : (0 : Fin 2) = 1 := h () 0 (by simp [globalFlip]) 1 (by simp [globalFlip])
    exact (by decide : (0 : Fin 2) ≠ 1) he
  refine ⟨?_, hn⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem a06_independent : (globalFlip ∅).Isolates .a06 := by
  obtain ⟨⟨h1, h2, h3⟩, h9, h10, h11, hp3⟩ := globalFlip_other_clauses ∅
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms (globalFlip ∅)).mp hp3
  have h4 : (globalFlip ∅).HomogeneousContact := eq_homogeneous
  have h5 : (globalFlip ∅).ParticipantsContact := by
    intro a u hu
    simp [globalFlip] at hu
  have h7 : (globalFlip ∅).RateReadsParticipants := fun _ _ _ _ _ _ => rfl
  have h8 : (globalFlip ∅).EffectReadsParticipants := by
    intro a s s' x y _ u hu
    simp [globalFlip] at hu
  have hn : ¬ (globalFlip ∅).WritesParticipants := by
    intro h
    have he := h () (fun _ => ()) (fun _ => 1) 0 (by simp [globalFlip])
    exact (by decide : flip12 (1 : Fin 3) ≠ 1) he
  refine ⟨?_, hn⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem farAct_other_clauses :
    farAct.P1a ∧ farAct.P2a ∧ farAct.P2b ∧ farAct.P2c ∧ farAct.P3 := by
  refine ⟨flipCost_p1a rfl, ?_, ?_, ?_, ?_⟩
  · intro a s x hclosed hpos
    clear hclosed hpos
    apply Finset.sum_congr rfl
    intro u hu
    clear hu
    change flipCost (farAct.act a s x u) = flipCost (x u)
    change flipCost (Function.update x a
      (if x (a + 1) = 2 then flip12 (x a) else x a) u) = flipCost (x u)
    revert a x u
    decide
  · refine p2b_of_micro _ id (fun _ => rfl) ?_
    intro a x _
    refine ⟨?_, rfl⟩
    funext u
    change farAct.act a farAct.calm (farAct.act a farAct.calm x) u = x u
    revert a x u
    decide
  · intro σ _
    refine ⟨σ, fun a s x => ⟨rfl, ?_⟩⟩
    have hs : σ.symm (σ a + 1) = a + 1 := by
      rw [← perm_fin2_succ, Equiv.symm_apply_apply]
    simp only [farAct, Function.comp_apply, Equiv.symm_apply_apply]
    rw [Function.update_comp_equiv]
    simp [hs]
  · exact ⟨fun _ _ _ _ _ => ⟨rfl, rfl⟩, fun _ _ => rfl⟩

theorem a08_independent : farAct.Isolates .a08 := by
  obtain ⟨⟨h1, h2, h3⟩, h9, h10, h11, hp3⟩ := farAct_other_clauses
  obtain ⟨h12, h13, h14⟩ := (p3_iff_atoms farAct).mp hp3
  obtain ⟨hn, h7, h6, h5, h4⟩ := farAct_isolates_a08
  refine ⟨?_, hn⟩
  intro c hc
  cases c <;> simp_all [Holds]

theorem innerActClock_other_clauses :
    innerActClock.P1a ∧ innerActClock.P1b ∧ innerActClock.P2a ∧
      innerActClock.P2b ∧ innerActClock.P2c := by
  refine ⟨flipCost_p1a rfl, ?_, ?_, ?_, ?_⟩
  · refine ⟨eq_homogeneous, ?_, ?_, ?_⟩
    · intro a u hu v hv
      simp only [innerActClock, Finset.mem_singleton] at hu hv
      exact hu.trans hv.symm
    · intro a s x u hu
      simp only [innerActClock, Finset.mem_singleton] at hu
      simp [innerActClock, Function.update_of_ne hu]
    · intro a s s' x y h
      refine ⟨rfl, ?_⟩
      intro u hu
      simp only [innerActClock, Finset.mem_singleton] at hu
      subst u
      obtain ⟨hx, hs⟩ := h a (by simp [innerActClock])
      simp [innerActClock, hx, hs]
  · intro a s x hclosed hpos
    clear hclosed hpos
    apply Finset.sum_congr rfl
    intro u hu
    clear hu
    change flipCost (innerActClock.act a s x u) = flipCost (x u)
    revert a s x u
    decide
  · refine p2b_of_micro _ id (fun _ => rfl) ?_
    intro a x _
    refine ⟨?_, rfl⟩
    simp [innerActClock]
  · intro σ _
    refine ⟨σ, fun a s x => ⟨rfl, ?_⟩⟩
    simp only [innerActClock, Function.comp_apply, Equiv.symm_apply_apply]
    rw [Function.update_comp_equiv]
    simp

theorem a13_independent : innerActClock.Isolates .a13 := by
  obtain ⟨⟨h1, h2, h3⟩, h1b, h9, h10, h11⟩ := innerActClock_other_clauses
  obtain ⟨h4, h5, h6, h7, h8⟩ := (p1b_iff_atoms innerActClock).mp h1b
  obtain ⟨hn, h12, h14⟩ := innerActClock_isolates_a13
  refine ⟨?_, hn⟩
  intro c hc
  cases c <;> simp_all [Holds]

/-- Consistency is witnessed separately from independence. -/
theorem twinFlip_all_conditions : ∀ c, twinFlip.Holds c :=
  (atomized_iff_all_conditions twinFlip).mp
    ((axioms_iff_atomized twinFlip).mp twinFlip_axioms)

/-- A packaged countermodel in the general finite-site, finite-event language.
The local-state type is allowed to be infinite, as in the original Medium. -/
structure AtomicCountermodel (target : AtomicCondition) where
  Carrier : Type
  Local : Type
  Event : Type
  Input : Type
  [carrierFinite : Fintype Carrier]
  [eventFinite : Fintype Event]
  [stateDecidable : DecidableEq (Carrier → Local)]
  medium : Medium Carrier Local Event Input
  certificate : medium.Isolates target

attribute [instance] AtomicCountermodel.carrierFinite AtomicCountermodel.eventFinite
  AtomicCountermodel.stateDecidable

theorem all_atoms_independent (c : AtomicCondition) : Nonempty (AtomicCountermodel c) := by
  cases c with
  | a01 =>
    exact ⟨{
      Carrier := Fin 2
      Local := ℕ
      Event := Fin 2 × Bool
      Input := Unit
      medium := twinTape
      certificate := a01_independent
    }⟩
  | a02 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Option (ℕ × Bool)
      Event := Fin 2 × Unit
      Input := Unit
      medium := twinBag
      certificate := a02_independent
    }⟩
  | a03 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Fin 3
      Event := Fin 2 × Unit
      Input := Unit
      medium := twinBlank
      certificate := a03_independent
    }⟩
  | a04 =>
    exact ⟨{
      Carrier := Fin 3
      Local := Fin 2
      Event := Empty
      Input := Unit
      medium := path3Rigid
      certificate := a04_independent
    }⟩
  | a05 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Fin 3
      Event := Unit
      Input := Unit
      medium := globalFlip univ
      certificate := a05_independent
    }⟩
  | a06 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Fin 3
      Event := Unit
      Input := Unit
      medium := globalFlip ∅
      certificate := a06_independent
    }⟩
  | a07 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Fin 3
      Event := Fin 2
      Input := Unit
      medium := farFlip
      certificate := a07_independent
    }⟩
  | a08 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Fin 3
      Event := Fin 2
      Input := Unit
      medium := farAct
      certificate := a08_independent
    }⟩
  | a09 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Bool
      Event := Fin 2 × Unit
      Input := Unit
      medium := twinToggle
      certificate := a09_independent
    }⟩
  | a10 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Fin 4
      Event := Fin 2 × Unit
      Input := Unit
      medium := twinCycle
      certificate := a10_independent
    }⟩
  | a11 =>
    exact ⟨{
      Carrier := Fin 3
      Local := Bool
      Event := Fin 3 × Fin 3
      Input := Unit
      medium := triLookup
      certificate := a11_independent
    }⟩
  | a12 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Fin 3
      Event := Fin 2
      Input := Bool
      medium := innerClock
      certificate := a12_independent
    }⟩
  | a13 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Fin 3
      Event := Fin 2
      Input := Bool
      medium := innerActClock
      certificate := a13_independent
    }⟩
  | a14 =>
    exact ⟨{
      Carrier := Fin 2
      Local := Bool
      Event := Fin 2 × Fin 2
      Input := Unit
      medium := pairPreset
      certificate := a14_independent
    }⟩

/-- The fixed nine conditions of ReducedCore, not a search over alternative languages. -/
def InReduced : AtomicCondition → Prop
  | .a03 | .a05 | .a06 | .a07 | .a08 | .a09 | .a10 | .a12 | .a13 => True
  | _ => False

theorem reducedCore_iff_selected_conditions [Fintype U] [Fintype A]
    [DecidableEq (U → Λ)] : M.ReducedCore ↔ ∀ c, InReduced c → M.Holds c := by
  constructor
  · rintro ⟨h3, h5, h6, h7, h8, h9, h10, h12, h13⟩ c hc
    cases c <;> simp_all [InReduced, Holds]
  · intro h
    exact ⟨h .a03 trivial, h .a05 trivial, h .a06 trivial, h .a07 trivial,
      h .a08 trivial, h .a09 trivial, h .a10 trivial, h .a12 trivial, h .a13 trivial⟩

/-- Assign distinct structural identities without changing contact or any dynamics. -/
noncomputable def rigidLabel [Fintype U] : Medium U Λ A I :=
  { M with ident := fun u => (Fintype.equivFin U u).val }

theorem rigidLabel_rigid [Fintype U] : M.rigidLabel.Rigid :=
  rigid_of_injective_identity _ (fun _ _ h => (Fintype.equivFin U).injective (Fin.ext h))

theorem holds_rigidLabel_iff [Fintype U] [Fintype A] [DecidableEq (U → Λ)]
    (c : AtomicCondition) (hc11 : c ≠ .a11) (hc14 : c ≠ .a14) :
    M.rigidLabel.Holds c ↔ M.Holds c := by
  cases c <;> first | rfl | contradiction

theorem isolates_rigidLabel [Fintype U] [Fintype A] [DecidableEq (U → Λ)]
    (c : AtomicCondition) (hc11 : c ≠ .a11) (hc14 : c ≠ .a14)
    (h : M.Isolates c) : M.rigidLabel.Isolates c := by
  refine ⟨?_, fun ht => h.2 ((M.holds_rigidLabel_iff c hc11 hc14).mp ht)⟩
  intro d hd
  by_cases hd11 : d = .a11
  · subst d
    exact p2c_of_rigid _ M.rigidLabel_rigid
  by_cases hd14 : d = .a14
  · subst d
    exact initial_invariant_of_rigid _ M.rigidLabel_rigid
  exact (M.holds_rigidLabel_iff d hd11 hd14).mpr (h.1 d hd)

/-- Every selected condition has a finite-local-state countermodel in the SAME
homogeneous-contact and rigid-structure regime used by the recovery bridge. -/
theorem every_reduced_condition_independent_in_regime (c : AtomicCondition)
    (hc : InReduced c) :
    ∃ w : AtomicCountermodel c, Finite w.Local ∧
      w.medium.HomogeneousContact ∧ w.medium.Rigid := by
  cases c with
  | a01 => exact False.elim hc
  | a02 => exact False.elim hc
  | a03 =>
    let w : AtomicCountermodel .a03 := {
      Carrier := Fin 2, Local := Fin 3, Event := Fin 2 × Unit, Input := Unit,
      medium := (twinBlank).rigidLabel,
      certificate := isolates_rigidLabel (twinBlank) .a03 (by decide) (by decide)
        a03_independent }
    refine ⟨w, inferInstance, ?_, rigidLabel_rigid (twinBlank)⟩
    exact (a03_independent.1 .a04 (by decide))
  | a04 => exact False.elim hc
  | a05 =>
    let w : AtomicCountermodel .a05 := {
      Carrier := Fin 2, Local := Fin 3, Event := Unit, Input := Unit,
      medium := (globalFlip univ).rigidLabel,
      certificate := isolates_rigidLabel (globalFlip univ) .a05 (by decide) (by decide)
        a05_independent }
    refine ⟨w, inferInstance, ?_, rigidLabel_rigid (globalFlip univ)⟩
    exact (a05_independent.1 .a04 (by decide))
  | a06 =>
    let w : AtomicCountermodel .a06 := {
      Carrier := Fin 2, Local := Fin 3, Event := Unit, Input := Unit,
      medium := (globalFlip ∅).rigidLabel,
      certificate := isolates_rigidLabel (globalFlip ∅) .a06 (by decide) (by decide)
        a06_independent }
    refine ⟨w, inferInstance, ?_, rigidLabel_rigid (globalFlip ∅)⟩
    exact (a06_independent.1 .a04 (by decide))
  | a07 =>
    let w : AtomicCountermodel .a07 := {
      Carrier := Fin 2, Local := Fin 3, Event := Fin 2, Input := Unit,
      medium := (farFlip).rigidLabel,
      certificate := isolates_rigidLabel (farFlip) .a07 (by decide) (by decide)
        a07_independent }
    refine ⟨w, inferInstance, ?_, rigidLabel_rigid (farFlip)⟩
    exact (a07_independent.1 .a04 (by decide))
  | a08 =>
    let w : AtomicCountermodel .a08 := {
      Carrier := Fin 2, Local := Fin 3, Event := Fin 2, Input := Unit,
      medium := (farAct).rigidLabel,
      certificate := isolates_rigidLabel (farAct) .a08 (by decide) (by decide)
        a08_independent }
    refine ⟨w, inferInstance, ?_, rigidLabel_rigid (farAct)⟩
    exact (a08_independent.1 .a04 (by decide))
  | a09 =>
    let w : AtomicCountermodel .a09 := {
      Carrier := Fin 2, Local := Bool, Event := Fin 2 × Unit, Input := Unit,
      medium := (twinToggle).rigidLabel,
      certificate := isolates_rigidLabel (twinToggle) .a09 (by decide) (by decide)
        a09_independent }
    refine ⟨w, inferInstance, ?_, rigidLabel_rigid (twinToggle)⟩
    exact (a09_independent.1 .a04 (by decide))
  | a10 =>
    let w : AtomicCountermodel .a10 := {
      Carrier := Fin 2, Local := Fin 4, Event := Fin 2 × Unit, Input := Unit,
      medium := (twinCycle).rigidLabel,
      certificate := isolates_rigidLabel (twinCycle) .a10 (by decide) (by decide)
        a10_independent }
    refine ⟨w, inferInstance, ?_, rigidLabel_rigid (twinCycle)⟩
    exact (a10_independent.1 .a04 (by decide))
  | a11 => exact False.elim hc
  | a12 =>
    let w : AtomicCountermodel .a12 := {
      Carrier := Fin 2, Local := Fin 3, Event := Fin 2, Input := Bool,
      medium := (innerClock).rigidLabel,
      certificate := isolates_rigidLabel (innerClock) .a12 (by decide) (by decide)
        a12_independent }
    refine ⟨w, inferInstance, ?_, rigidLabel_rigid (innerClock)⟩
    exact (a12_independent.1 .a04 (by decide))
  | a13 =>
    let w : AtomicCountermodel .a13 := {
      Carrier := Fin 2, Local := Fin 3, Event := Fin 2, Input := Bool,
      medium := (innerActClock).rigidLabel,
      certificate := isolates_rigidLabel (innerActClock) .a13 (by decide) (by decide)
        a13_independent }
    refine ⟨w, inferInstance, ?_, rigidLabel_rigid (innerActClock)⟩
    exact (a13_independent.1 .a04 (by decide))
  | a14 => exact False.elim hc

end CGB.Medium
