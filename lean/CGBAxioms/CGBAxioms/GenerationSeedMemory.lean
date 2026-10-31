import CGBAxioms.GenerationConservedMemory
import CGBAxioms.GenerationLifeGate

/-!
# An actual memory world for the unchanged physical seed

The first quiet tick moves the source particle and independently selects a
quiet or food world. The next input can add content at the source. Quiet
hopping preserves the resulting total-content distinction, not the exact
microstate. All prefix masses are those of `Medium.law`.

This is a nonempty physical `Remembers` component, not a GDN, a viability
proof, or a proof of `Lives`. The positive horizon and recovery fields only
complete the world definition; no survival claim is made for this world.
-/

namespace CGB.SeedMemory

open Finset ConservedMemory

/-- At birth the world branches fairly, independently of the body; each
selected environment then persists. -/
noncomputable def moves : Option Bool → (Fin 4 → Bool) → Option Bool → ℝ
  | none, _, none => 0
  | none, _, some _ => 1 / 2
  | some b, _, w => if w = some b then 1 else 0

/-- Only the food branch informs the original source and sink ports. -/
noncomputable abbrev world : World (Fin 4) Bool Bool where
  W := Option Bool
  sense
    | none => Models.seed.calm
    | some false => Models.seed.calm
    | some true => Models.food
  moves := moves
  moves_nonneg w x w' := by
    cases w with
    | none => cases w' <;> norm_num [moves]
    | some b => simp only [moves]; split_ifs <;> norm_num
  moves_sum w x := by
    cases w with
    | none => norm_num [Fintype.sum_option, Fintype.sum_bool, moves]
    | some b => simp [moves]
  start := none
  quiet := Models.seed.calm
  clock := 18
  clock_pos := by norm_num
  horizon := 1
  recovery := 1 / 2
  recovery_pos := by norm_num

/-- The existing structural event bound admits this world's exact clock. -/
theorem clocked : Models.seed.Clocked world.clock :=
  Models.seed.eventClocked_implies_clocked seed_event_clocked

/-- Collapse only the initial Dirac law, without enumerating body histories. -/
theorem law_one {U Λ A I : Type*}
    [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
    (M : Medium U Λ A I) (W : World U Λ I)
    (sched : ℕ → W.W → U → I) (y : U → Λ) (w' : W.W) :
    M.law W sched 1 y w' =
      W.moves W.start M.init w' * M.kernel W.clock (sched 0 W.start) M.init y := by
  classical
  simp only [Medium.law, ite_and, ite_mul, zero_mul, one_mul,
    sum_ite_irrel, sum_const_zero, sum_ite_eq', mem_univ, ↓reduceIte]

/-- Body reached by the actual first quiet hop, with source 0 now vacant. -/
def beforeWrite : Fin 4 → Bool := fun u => decide (u = 1)

/-- A source intake writes the extra unit into the same original body. -/
def afterWrite : Fin 4 → Bool := fun u => decide (u = 0 ∨ u = 1)

/-- The memory readout is a coarse total-content distinction. -/
noncomputable def observe (x : Fin 4 → Bool) : Bool := decide (2 ≤ Models.seed.total x)

theorem first_hop_act :
    Models.seed.act (.inl (0, 1)) Models.seed.calm Models.seed.init = beforeWrite := by
  decide

/-- A registered positive-rate hop yields a positive actual first transition. -/
theorem first_hop_pos :
    0 < Models.seed.kernel world.clock Models.seed.calm Models.seed.init beforeWrite := by
  rw [← first_hop_act]
  apply registered_event_kernel_pos Models.seed world.clock world.clock_pos
  · norm_num [Models.seed, Models.ringAdj]
  · rw [first_hop_act]
    decide

/-- Both input tags have positive probability at the very same reachable body. -/
theorem prefix_pos (b : Bool) :
    0 < Models.seed.law world (fun _ => world.sense) 1 beforeWrite (some b) := by
  rw [law_one]
  change 0 < (1 / 2 : ℝ) *
    Models.seed.kernel world.clock Models.seed.calm Models.seed.init beforeWrite
  exact mul_pos (by norm_num) first_hop_pos

theorem write_act :
    Models.seed.act (.inr true) Models.food beforeWrite = afterWrite := by
  decide

/-- Food really increases total from one to two through the existing source. -/
theorem write_totals : Models.seed.total beforeWrite = 1 ∧
    Models.seed.total afterWrite = 2 := by
  decide

theorem write_pos :
    0 < Models.seed.kernel world.clock (world.sense (some true)) beforeWrite afterWrite := by
  change 0 < Models.seed.kernel world.clock Models.food beforeWrite afterWrite
  rw [← write_act]
  apply registered_event_kernel_pos Models.seed world.clock world.clock_pos
  · norm_num [Models.seed, Models.food, beforeWrite]
  · rw [write_act]
    decide

/-- Positive quiet transitions preserve the readout even though hops continue. -/
theorem quiet_preserves (x y : Fin 4 → Bool)
    (hp : 0 < Models.seed.kernel world.clock world.quiet x y) : observe y = observe x := by
  have ht : Models.seed.total y = Models.seed.total x :=
    quiet_event_observable_kernel Models.seed world Models.seed.total
      (fun a z _ => seed_calm_event_conserves a z) x y hp
  simp only [observe, ht]

/-- The original full-body CMI memory clause, derived from actual positive
prefixes and source intake, with no assumed positive-information premise. -/
theorem remembers : Models.seed.Remembers world := by
  apply remembers_of_quiet_and_actual_write Models.seed world clocked observe
    quiet_preserves 1 beforeWrite afterWrite (some false) (some true)
  · rfl
  · exact prefix_pos false
  · exact prefix_pos true
  · exact write_pos
  · simp [observe, write_totals.1, write_totals.2]

/-- Quiet is not an identity dynamics: a positive-rate event changes the body. -/
theorem quiet_has_nontrivial_event :
    ∃ (a : (Fin 4 × Fin 4) ⊕ Bool) (x : Fin 4 → Bool),
      0 < Models.seed.rate a world.quiet x ∧ Models.seed.act a world.quiet x ≠ x :=
  seed_calm_has_nontrivial_event

/-- Finite Boolean graph fact only: every partially occupied ring has an
occupied-to-vacant contact. No real-rate decision procedure is involved. -/
theorem partial_has_hop (x : Fin 4 → Bool)
    (hlo : 0 < Models.seed.total x) (hhi : Models.seed.total x < 4) :
    ∃ u v : Fin 4, Models.ringAdj u v ∧ u ≠ v ∧ x u = true ∧ x v = false := by
  have graph : ∀ z : Fin 4 → Bool,
      0 < (∑ u, Bool.toNat (z u)) → (∑ u, Bool.toNat (z u)) < 4 →
      ∃ u v : Fin 4, Models.ringAdj u v ∧ u ≠ v ∧ z u = true ∧ z v = false := by
    decide
  exact graph x hlo hhi

/-- Every partial microstate has a state-changing rate-one hop under every
stimulus, including quiet. The rates and seed state space are unchanged. -/
theorem partial_rate_one (x : Fin 4 → Bool)
    (hlo : 0 < Models.seed.total x) (hhi : Models.seed.total x < 4)
    (s : Fin 4 → Bool) :
    ∃ a : (Fin 4 × Fin 4) ⊕ Bool,
      Models.seed.act a s x ≠ x ∧ Models.seed.rate a s x = 1 := by
  obtain ⟨u, v, hadj, hne, hu, hv⟩ := partial_has_hop x hlo hhi
  refine ⟨.inl (u, v), ?_, ?_⟩
  · change (if Models.ringAdj u v then Models.xhopAct u v x else x) ≠ x
    rw [ite_eq_left hadj, Models.xhopAct_of_ok ⟨hne, hu, hv⟩]
    intro h
    have hh := congrFun h u
    simp [hu, hv] at hh
  · simp [Models.seed, hadj, hne, hu, hv]

/-- The probability of never leaving a partial microstate has a geometric
bound, while the coarse memory remains. This is not a bound on return visits. -/
theorem partial_hold_bound (x : Fin 4 → Bool)
    (hlo : 0 < Models.seed.total x) (hhi : Models.seed.total x < 4)
    (sched : ℕ → Fin 4 → Bool) (n : ℕ) :
    ∏ t ∈ range n, Models.seed.kernel world.clock (sched t) x x ≤
      (1 - (1 : ℝ) / 18) ^ n := by
  apply Models.seed.hold_vanishes clocked (by norm_num : (0 : ℝ) < 1) x
  intro s
  obtain ⟨a, hne, hr⟩ := partial_rate_one x hlo hhi s
  exact ⟨a, hne, by rw [hr]⟩

/-- One unchanged admitted physical medium jointly has actual memory and
nontrivial quiet events in this world. No `Lives` conclusion is asserted. -/
theorem physical_memory_component : Models.seed.Axioms ∧
    Models.seed.Clocked world.clock ∧ Models.seed.Remembers world ∧
    (∃ (a : (Fin 4 × Fin 4) ⊕ Bool) (x : Fin 4 → Bool),
      0 < Models.seed.rate a world.quiet x ∧ Models.seed.act a world.quiet x ≠ x) :=
  ⟨Models.seed_axioms, clocked, remembers, quiet_has_nontrivial_event⟩

end CGB.SeedMemory
