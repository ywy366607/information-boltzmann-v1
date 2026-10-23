import CGBAxioms.InteractionBridge
import CGBAxioms.OperationalLife

/-!
# Semantic certificates and obstructions

These mathematical controls exercise exact representation, not brain capacity.
An independently resampled invisible bit can be removed by a proved morphism.
Equal immediate observations do not license merging states whose futures differ.
-/

open Finset

namespace CGB.InteractionControls

open Interaction

variable {S I O : Type*} [Fintype S] [Fintype O]

noncomputable def withIndependentBit (P : Machine S I O) : Machine (S × Bool) I O where
  step i s o s' := P.step i s.1 o s'.1 / 2
  nonneg _ _ _ _ := div_nonneg (P.nonneg _ _ _ _) (by norm_num)
  sum_one i s := by
    have hh : ∀ x : ℝ, x / 2 + x / 2 = x := by intro x; ring
    simpa only [Fintype.sum_prod_type, Fintype.sum_bool, hh] using P.sum_one i s.1

/-- Removing a bit is justified HERE by a universal transition equation. No
conclusion about ignoring an arbitrary biological cell/state follows from this. -/
theorem independent_bit_refines [DecidableEq S] (P : Machine S I O) :
    Refines (withIndependentBit P) P Prod.fst := by
  intro i s o t'
  simp [withIndependentBit, Fintype.sum_prod_type]
  ring

theorem independent_bit_sameBehavior (P : Machine S I O) (s : S) (b : Bool) :
    SameBehavior (withIndependentBit P) P (s, b) s := by
  classical
  exact Refines.sameBehavior (independent_bit_refines P) (s, b)

noncomputable def alternating : Machine Bool Unit Bool :=
  Machine.deterministic fun _ s => (s, !s)

noncomputable def constant : Machine Bool Unit Bool :=
  Machine.deterministic fun _ _ => (false, false)

theorem same_first_observation :
    alternating.trace (fun _ => ()) [] [false] false = 1 ∧
    constant.trace (fun _ => ()) [] [false] false = 1 := by
  norm_num [Machine.trace, alternating, constant, Machine.deterministic, Fintype.sum_bool]

theorem different_future :
    alternating.trace (fun _ => ()) [] [false, true] false = 1 ∧
    constant.trace (fun _ => ()) [] [false, true] false = 0 := by
  norm_num [Machine.trace, alternating, constant, Machine.deterministic, Fintype.sum_bool]
  decide

theorem no_bisimulation_from_first_observation : ¬ Bisimilar alternating constant false false := by
  intro h
  have he := bisimilar_sameBehavior h (fun _ => ()) [] [false, true]
  rw [different_future.1, different_future.2] at he
  norm_num at he

theorem physical_certificate_rejects_time_rescaling :
    ¬ TimedRefinement alternating alternating id 1 2 := by
  intro h
  have he := h.same_clock
  norm_num at he

/-- The tested representation is the identity on the actual continuous state;
the bridge must work for BOTH normal and cut futures with the same world clock. -/
theorem actual_recovery_bridge {U Λ A J : Type*} [Fintype U] [DecidableEq U]
    [Fintype Λ] [DecidableEq Λ] [Fintype A]
    (M : Medium U Λ A J) (Wd : World U Λ J) (hc : M.Clocked Wd.clock)
    (cut : Bool) (sched : ℕ → Wd.W → U → J) (history : List Bool) (k : ℕ)
    (z : (U → Λ) × Wd.W) :
    (M.intakeCoalgebra Wd hc cut sched).firstHit List.length id history k z =
      Operational.recovery M Wd hc cut sched history.length k z :=
  M.intakeCoalgebra_firstHit Wd hc cut sched history k z

end CGB.InteractionControls
