import CGBAxioms.AlgebraicHomology

open CGB.AlgebraicHomology

namespace AlgebraicHomologyReview20261024

example {A B C R : Type*} [Fintype A] [Fintype B] [Fintype C] [CommSemiring R]
    (d₁ : A → B → R) (d₂ : B → C → R)
    (h : boundaryLaw d₁ d₂) (v : A → R) (c : C) :
    matrixAction d₂ (matrixAction d₁ v) c = 0 :=
  boundary_boundary_zero d₁ d₂ h v c

example {A B C R : Type*} [Fintype A] [Fintype B] [Fintype C] [CommSemiring R]
    (d₁ : A → B → R) (d₂ : B → C → R) (v : A → R) (c : C) :
    matrixAction d₂ (matrixAction d₁ v) c =
      ∑ a, v a * matrixCompose d₁ d₂ a c :=
  matrixAction_compose d₁ d₂ v c

end AlgebraicHomologyReview20261024
