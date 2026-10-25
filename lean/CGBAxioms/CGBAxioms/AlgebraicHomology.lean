import CGBAxioms.AlgebraicCore

/-!
# Minimal algebraic homology core

Finite boundary operators are represented by matrices over a commutative
semiring. The only homological law registered here is that the composite of
two consecutive boundaries is zero; quotient homology and analytic structure
remain separate layers.
-/

open Finset

namespace CGB.AlgebraicHomology

variable {A B C R : Type*} [Fintype A] [Fintype B] [Fintype C] [CommSemiring R]

def matrixAction (d : A → B → R) (v : A → R) (b : B) : R :=
  ∑ a, v a * d a b

def matrixCompose (d₁ : A → B → R) (d₂ : B → C → R) (a : A) (c : C) : R :=
  ∑ b, d₁ a b * d₂ b c

def boundaryLaw (d₁ : A → B → R) (d₂ : B → C → R) : Prop :=
  ∀ a c, matrixCompose d₁ d₂ a c = 0

theorem matrixAction_compose (d₁ : A → B → R) (d₂ : B → C → R)
    (v : A → R) (c : C) :
    matrixAction d₂ (matrixAction d₁ v) c =
      ∑ a, v a * matrixCompose d₁ d₂ a c := by
  classical
  simp only [matrixAction, matrixCompose, Finset.sum_mul, Finset.mul_sum, mul_assoc]
  exact Finset.sum_comm

theorem boundary_boundary_zero (d₁ : A → B → R) (d₂ : B → C → R)
    (h : boundaryLaw d₁ d₂) (v : A → R) (c : C) :
    matrixAction d₂ (matrixAction d₁ v) c = 0 := by
  rw [matrixAction_compose]
  exact Finset.sum_eq_zero (fun a _ => by rw [h a c, mul_zero])

end CGB.AlgebraicHomology
