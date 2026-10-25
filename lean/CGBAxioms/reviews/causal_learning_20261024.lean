import CGBAxioms.CausalLearning

/- Finite truth-table and payment checks, not a learning-capability benchmark.
   No training, fitted parameters or performance estimates are used.
   Run: lake env lean reviews/causal_learning_20261024.lean -/

open CGB CGB.CausalLearning

namespace CausalLearningReview20261024

abbrev H := Fin 2
abbrev A := Fin 3
abbrev O := Fin 4

def outcome (h : H) (a : A) : O :=
  if a = 0 then (if h = 0 then 0 else 3)
  else if a = 1 then (if h = 0 then 1 else 2) else 1

def possible (h : H) (_ : Unit) (a : A) (o : O) : Bool := decide (o = outcome h a)
def viable (_ : Unit) (_ : A) (o : O) : Bool := decide (o ≠ 0)
def before : Memory H 1 := initialMemory H 1
def after : Memory H 1 := advance possible before () 1 2
def policy (m : Memory H 1) (c : Unit) : A := certifiedOr possible viable 2 m c

-- The same policy uses feedback, not a true-model identifier, to change action.
example : policy before () = 1 ∧ after.candidates = {1} ∧ after.reserve.val = 0 ∧
    fee possible before () 1 2 = 1 ∧ policy after () = 0 := by decide

example : (advance possible before () 1 1).candidates = {0} ∧
    policy (advance possible before () 1 1) () = 1 := by decide

-- Shared predictions carry no exclusion evidence and no effective-write fee.
example : advance possible before () 2 1 = before ∧ fee possible before () 2 1 = 0 := by decide

def unfunded : Memory H 1 := ⟨Finset.univ, 0⟩
example : advance possible unfunded () 1 2 = unfunded ∧
    fee possible unfunded () 1 2 = 0 := by decide

noncomputable def world (h : H) : Interaction.Machine Unit (ℕ × A) O :=
  Interaction.Machine.deterministic (fun ta _ => (outcome h ta.2, ()))

example (h : H) : Faithful (world h) (fun _ => ()) possible h := by
  intro t s a o s' hp
  change 0 < (if (o, s') = (outcome h a, ()) then (1 : ℝ) else 0) at hp
  split_ifs at hp with he
  · have ho : o = outcome h a := congrArg Prod.fst he
    simp [possible, ho]
  · norm_num at hp

-- The physical world keeps producing outcomes even with zero write reserve.
example : (∑ m' : Memory H 1,
    (closed (world 1) (fun _ => ()) possible policy).step 0 ((), unfunded) (1, 2) ((), m')) = 1 := by
  rw [physical_marginal]
  have ha : policy unfunded () = 1 := by decide
  simp [ha, world, Interaction.Machine.deterministic, outcome]

-- Unsupported feedback is not silently treated as identifying a true model.
example : (advance possible before () 2 0).candidates = ∅ := by decide

example : certifiedActions possible viable (∅ : Finset H) () = ∅ := by decide

end CausalLearningReview20261024
