import CGBAxioms.Life
import CGBAxioms.Models

/-!
# Active inference, first rungs: the blanket and foresight

The life definition orders its emergent part as a ladder: memory → world model and
prediction → choosing actions by what is predicted. This file holds the rungs that are
settled; every notion is an information quantity, without assuming that the organism
carries an explicit generative model.

* **Markov blanket** as one-step sparse coupling (`Blanket`, `blanket_of_axioms`): the
  organism's step reads the world only through the boundary's stimuli (this follows from
  P3), and the world's step reads the organism only through the boundary (this is the
  assumption `ReadsBoundary` on the world). It is a statement about the generator, not
  conditional independence in the joint law, and it does not split the boundary into
  sensory and active states.
* **Foresight** (L3): `foresight` is `I(X_n; S_{n+1} | S_n)` with `S` the stimuli at the
  boundary (the only ones the organism can use). Measured in the loop-cut world
  (`worldForesight`, `Predicts`) the world does not react to the organism. Note that
  `cutMove` drives the cut world with a ghost drawn from the organism's marginal law,
  independently of the world state. Any memory gives some foresight in a world whose
  stimuli are correlated in time, so `Predicts` alone does not separate active from passive
  organisms.

Withdrawn after review (2026-10-23): a self-model as closed-minus-cut foresight (passed by
any noisy reflex in a world that echoes the body), and pragmatic and epistemic advantage
over the best reflex (unsatisfiable: a frozen copy of the body that already holds content
counts as alive forever under the THEN-current content-only test). The menu-intake
revision and then OperationalLife's actual-world recovery repaired that historical test;
LifeControls now excludes no-intake and fully absorbing states in the operational target.
The withdrawn reflex advantage and self-model are still not re-established. Quiet-only
freezing, useful memory, active recovery and action-conditioned prediction need separate
contracts and witnesses. See the axiom document, §4.4, and the Noether/life-controls report.
-/

open Finset

namespace CGB

namespace Medium

variable {U Λ A I : Type*} [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] [Fintype A]
  (M : Medium U Λ A I)

/-! ## The Markov blanket -/

omit [Fintype U] [DecidableEq U] [Fintype Λ] [DecidableEq Λ] in
/-- Under P3, the full dynamics reads the stimulus only at the boundary. -/
lemma jump_blanket [DecidableEq (U → Λ)] (h : M.OnlyBoundarySenses) {s s' : U → I}
    (hs : ∀ u ∈ M.boundary, s u = s' u) (x y : U → Λ) :
    (M.jump s).rate x y = (M.jump s').rate x y := by
  simp only [jump]
  exact Finset.sum_congr rfl fun a _ => by rw [(h a s s' x hs).1, (h a s s' x hs).2]

lemma kernel_blanket (h : M.OnlyBoundarySenses) (R : ℝ) {s s' : U → I}
    (hs : ∀ u ∈ M.boundary, s u = s' u) : M.kernel R s = M.kernel R s' := by
  funext x y
  simp only [kernel, M.jump_blanket h hs]

variable (Wd : World U Λ I)

/-- **Markov blanket** (one-step sparse coupling): the organism's step reads the world only
through the stimuli at the boundary, and the world's step reads the organism only through
the boundary. -/
def Blanket : Prop :=
  (∀ s s' : U → I, (∀ u ∈ M.boundary, s u = s' u) →
    M.kernel Wd.clock s = M.kernel Wd.clock s') ∧
    M.ReadsBoundary Wd

/-- P3 gives the organism's half of the blanket; a world that touches only the body gives
the other half. -/
theorem blanket_of_axioms (h : M.Axioms) (hw : M.ReadsBoundary Wd) : M.Blanket Wd :=
  ⟨fun _ _ hs => M.kernel_blanket h.p3.1 _ hs, hw⟩

/-! ## Foresight -/

section Foresight

variable [DecidableEq I] [Fintype I]

/-- The stimuli at the boundary. -/
def boundaryStim (s : U → I) : M.boundary → I := fun u => s u

/-- Joint law of (boundary stimulus at `n`, organism state at `n`, boundary stimulus at
`n + 1`) in the closed loop. -/
noncomputable def nextJoint (n : ℕ) (s : M.boundary → I) (x : U → Λ) (s' : M.boundary → I) :
    ℝ :=
  ∑ w, ∑ w', if M.boundaryStim (Wd.sense w) = s ∧ M.boundaryStim (Wd.sense w') = s' then
    M.law Wd (fun _ => Wd.sense) n x w * Wd.moves w x w' else 0

/-- The same joint law in the loop-cut world. -/
noncomputable def nextJointCut (n : ℕ) (s : M.boundary → I) (x : U → Λ)
    (s' : M.boundary → I) : ℝ :=
  ∑ w, ∑ w', if M.boundaryStim (Wd.sense w) = s ∧ M.boundaryStim (Wd.sense w') = s' then
    M.lawCut Wd (fun _ => Wd.sense) n x w *
      cutMove Wd (M.lawCut Wd (fun _ => Wd.sense) n) x w w' else 0

/-- **Foresight** in the closed loop: `I(X_n; S_{n+1} | S_n)`, boundary stimuli only. -/
noncomputable def foresight (n : ℕ) : ℝ := condMutualInfo (M.nextJoint Wd n)

/-- **World foresight** (L3): foresight in the loop-cut world, where the world does not
react to the organism. -/
noncomputable def worldForesight (n : ℕ) : ℝ := condMutualInfo (M.nextJointCut Wd n)

/-- The organism predicts its world: at some time it has at least `ε > 0` nats of world
foresight. -/
def Predicts : Prop := ∃ n ε, 0 < ε ∧ ε ≤ M.worldForesight Wd n

end Foresight

end Medium

namespace Models

open Medium

/-- The rock satisfies all six axioms but can never take anything in: it was never alive. -/
theorem rock_cannot_eat (x : Fin 2 → Fin 3) : ¬ rock.CanTakeIn x := by
  rintro ⟨a, s, hr, -⟩
  simp [rock, solo] at hr

/-- So the rock persists in no world. -/
theorem rock_not_persists (Wd : World (Fin 2) (Fin 3) Unit) (δ : ℝ) (T : ℕ) :
    ¬ rock.Persists Wd δ T :=
  rock.not_persists_of_no_intake Wd rock_cannot_eat δ T

end Models

/-- Any positive finite reweighting improves an observed event's normalized
probability exactly when its conditional mean multiplier exceeds the global
mean. This is a decision criterion for a specified change, not a learning law,
sampler guarantee, or claim that a shared potential is forced by the axioms. -/
theorem finite_reweighting_improves_iff {S : Type*} [Fintype S]
    (event : Finset S) (w r : S → ℝ) (hw : ∀ s, 0 < w s)
    (hr : ∀ s, 0 < r s) (hne : event.Nonempty) :
    (∑ s ∈ event, w s) / (∑ s, w s) <
        (∑ s ∈ event, w s * r s) / (∑ s, w s * r s) ↔
      (∑ s, w s * r s) / (∑ s, w s) <
        (∑ s ∈ event, w s * r s) / (∑ s ∈ event, w s) := by
  have : Nonempty S := ⟨hne.choose⟩
  have htotal : 0 < ∑ s, w s :=
    Finset.sum_pos (fun s _ => hw s) Finset.univ_nonempty
  have hchanged : 0 < ∑ s, w s * r s :=
    Finset.sum_pos (fun s _ => mul_pos (hw s) (hr s)) Finset.univ_nonempty
  have hevent : 0 < ∑ s ∈ event, w s :=
    Finset.sum_pos (fun s _ => hw s) hne
  rw [div_lt_div_iff₀ htotal hchanged, div_lt_div_iff₀ htotal hevent]
  rw [mul_comm (∑ s ∈ event, w s) (∑ s, w s * r s)]

end CGB
