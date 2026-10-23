import CGBAxioms.Interaction

/-!
# The existing medium is a probabilistic interaction coalgebra

The observation wrapper preserves the full joint transition and its world tick.
It is not a new physical kernel, nor an assumption that chosen observations are
complete. A certificate can preserve only the observations named in its interface.
-/

open Finset

namespace CGB.Interaction.Machine

variable {S I O : Type*} [Fintype S] [Fintype O] [DecidableEq O]

/-- Label the transitions of an admitted kernel family. Input protocols are
external boundary conditions; the machine does not select them. -/
noncomputable def ofKernels (K : I → FirstHit.Kernel S) (observe : I → S → S → O) :
    Machine S I O where
  step i s o s' := if observe i s s' = o then (K i).step s s' else 0
  nonneg i s o s' := by split <;> simp_all [(K i).nonneg]
  sum_one i s := by
    rw [Finset.sum_comm]
    simp only [Finset.sum_ite_eq, Finset.mem_univ, ite_true, (K i).sum_one]

theorem ofKernels_step (K : I → FirstHit.Kernel S) (observe : I → S → S → O)
    (i : I) (s : S) (o : O) (s' : S) :
    (ofKernels K observe).step i s o s' =
      if observe i s s' = o then (K i).step s s' else 0 := rfl

/-- Forgetting the observer's label recovers EXACTLY the original transition. -/
theorem ofKernels_marginal (K : I → FirstHit.Kernel S) (observe : I → S → S → O)
    (i : I) (s s' : S) :
    ∑ o, (ofKernels K observe).step i s o s' = (K i).step s s' := by
  simp [ofKernels]

/-- Observe a time protocol in the existing external world clock. A list history
supplies the tick number to the test context, not extra internal body storage. -/
theorem ofKernels_firstHit (K : ℕ → FirstHit.Kernel S) (observe : S → S → O)
    (mark : O → Bool) (history : List O) (k : ℕ) (s : S) :
    (ofKernels K (fun _ => observe)).firstHit List.length mark history k s =
      FirstHit.during K (fun z z' => mark (observe z z')) history.length k s := by
  induction k generalizing history s with
  | zero => rfl
  | succ k ih =>
    simp only [firstHit, ofKernels_step, FirstHit.during]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro s' _
    simp only [ite_mul, zero_mul, Finset.sum_ite_eq, Finset.mem_univ, ite_true]
    split_ifs
    · rfl
    · simpa only [List.length_append, List.length_singleton] using
        congrArg ((K history.length).step s s' * ·) (ih (history ++ [observe s s']) s')

/-- Deterministic machines are a special case; used in semantic countermodels. -/
noncomputable def deterministic [DecidableEq S] (advance : I → S → O × S) : Machine S I O where
  step i s o s' := if (o, s') = advance i s then 1 else 0
  nonneg _ _ _ _ := by split <;> norm_num
  sum_one i s := by
    rcases he : advance i s with ⟨o, s'⟩
    simp only [Prod.mk.injEq]
    rw [Finset.sum_eq_single o]
    · simp
    · intro o' _ hne
      simp [hne]
    · simp

end CGB.Interaction.Machine

namespace CGB.Medium

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
  (M : Medium U Λ A I) (Wd : World U Λ I) (hc : M.Clocked Wd.clock)

/-- The sensory protocol is already explicit in law. This wrapper adds a
registered finite observation, keeping every physical body/world state. -/
noncomputable def interactionCoalgebra {J O : Type*} [Fintype O] [DecidableEq O]
    (protocol : J → Wd.W → U → I) (observe : J → ((U → Λ) × Wd.W) → ((U → Λ) × Wd.W) → O) :
    Interaction.Machine ((U → Λ) × Wd.W) J O :=
  Interaction.Machine.ofKernels (fun j => M.scheduledKernel Wd hc (fun _ => protocol j) 0) observe

/-- Direct bridge for the normal AND marginal-ghost-cut scheduled worlds. -/
noncomputable def intakeCoalgebra (cut : Bool) (sched : ℕ → Wd.W → U → I) :
    Interaction.Machine ((U → Λ) × Wd.W) ℕ Bool :=
  Interaction.Machine.ofKernels
    (if cut then M.scheduledCutKernel Wd hc sched else M.scheduledKernel Wd hc sched)
    (fun _ => M.materialIntake Wd)

theorem intakeCoalgebra_firstHit (cut : Bool) (sched : ℕ → Wd.W → U → I)
    (history : List Bool) (k : ℕ) (z : (U → Λ) × Wd.W) :
    (M.intakeCoalgebra Wd hc cut sched).firstHit List.length id history k z =
      FirstHit.during
        (if cut then M.scheduledCutKernel Wd hc sched else M.scheduledKernel Wd hc sched)
        (M.materialIntake Wd) history.length k z :=
  Interaction.Machine.ofKernels_firstHit _ _ _ _ _ _

/-- This normal-world observation yields the existing actualRecover, not a
different coalgebraic notion of success or a favourable-stimulus menu. -/
theorem intakeCoalgebra_actualRecover (history : List Bool) (k : ℕ)
    (x : U → Λ) (w : Wd.W) :
    (M.intakeCoalgebra Wd hc false (fun _ => Wd.sense)).firstHit List.length id history k (x, w) =
      M.actualRecover Wd hc k x w := by
  rw [M.intakeCoalgebra_firstHit]
  have he : M.scheduledKernel Wd hc (fun _ => Wd.sense) = fun _ => M.actualKernel Wd hc := rfl
  simp only [Bool.false_eq_true, ↓reduceIte, he, actualRecover]
  exact FirstHit.during_constant _ _ _ _ _

/-- Actual recovery AND elapsed time transfer along a certified implementation.
This is conditional on a universal refinement proof, not inferred from samples. -/
theorem refinement_preserves_recovery_and_time
    {V Θ B J : Type*} [Fintype V] [DecidableEq V] [Fintype Θ] [DecidableEq Θ] [Fintype B]
    (N : Medium V Θ B J) (Vd : World V Θ J) (hcN : N.Clocked Vd.clock)
    (f : ((U → Λ) × Wd.W) → ((V → Θ) × Vd.W))
    (h : Interaction.TimedRefinement
      (M.intakeCoalgebra Wd hc false (fun _ => Wd.sense))
      (N.intakeCoalgebra Vd hcN false (fun _ => Vd.sense)) f Wd.clock Vd.clock)
    (n k : ℕ) (z : (U → Λ) × Wd.W) :
    (M.actualRecover Wd hc k z.1 z.2 = N.actualRecover Vd hcN k (f z).1 (f z).2) ∧
      ((n : ℝ) / Wd.clock = (n : ℝ) / Vd.clock) := by
  constructor
  · rw [← M.intakeCoalgebra_actualRecover Wd hc [] k,
        ← N.intakeCoalgebra_actualRecover Vd hcN [] k]
    exact Interaction.Refines.firstHit_eq h.representation List.length id [] k z
  · exact Interaction.TimedRefinement.elapsed_eq h n

end CGB.Medium
