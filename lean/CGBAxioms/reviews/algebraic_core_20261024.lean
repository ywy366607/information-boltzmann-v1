import CGBAxioms.AlgebraicCore

open Finset
open CGB.AlgebraicCore

namespace AlgebraicCoreReview20261024

example {S E : Type*} [Fintype S] [Fintype E]
    (K : S → E → S → ℕ) (ρ : S → ℕ)
    (hrow : ∀ x, rowMass K x = 1) :
    ∑ y, pushForward K ρ y = ∑ x, ρ x :=
  pushForward_total K ρ hrow

example {R : Type*} [Semiring R] (s : CGB.SensoryMaintenance.Stim)
    (x : CGB.SensoryMaintenance.State) :
    (CGB.SensoryMaintenance.cost (CGB.SensoryMaintenance.next s x) : R) +
        CGB.SensoryMaintenance.outgoing s x =
      (CGB.SensoryMaintenance.cost x : R) + CGB.SensoryMaintenance.incoming s x :=
  cast_local_budget s x

example {R : Type*} [Semiring R] (stim : ℕ → CGB.SensoryMaintenance.Stim)
    (p : ℕ → CGB.SensoryMaintenance.State)
    (hp : ∀ t, p (t + 1) = CGB.SensoryMaintenance.next (stim t) (p t)) (n : ℕ) :
    (CGB.SensoryMaintenance.cost (p n) : R) +
        ∑ t ∈ range n, (CGB.SensoryMaintenance.outgoing (stim t) (p t) : R) =
      (CGB.SensoryMaintenance.cost (p 0) : R) +
        ∑ t ∈ range n, (CGB.SensoryMaintenance.incoming (stim t) (p t) : R) :=
  cast_path_budget stim p hp n

example {S E R : Type*} [Fintype S] [Fintype E] [CommSemiring R]
    (K : S → E → S → R) (p : ℕ → S) (e : ℕ → E) :
    pathWeight K p e 0 = 1 :=
  pathWeight_zero K p e

example {R : Type*} [Mul R] [Add R] (u v : R × R) :
    (jetMul u v).2 = u.1 * v.2 + u.2 * v.1 :=
  jetMul_product_rule u v

example {R : Type*} [AddMonoid R] [Mul R] (coeff : List R) (x dx : R) :
    (jetHorner coeff (x, dx)).1 = scalarHorner coeff x :=
  jetHorner_value coeff x dx

end AlgebraicCoreReview20261024
