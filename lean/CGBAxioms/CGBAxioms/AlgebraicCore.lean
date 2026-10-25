import CGBAxioms.SensoryMaintenanceBoundaries

/-!
# Algebraic finite-state core

Finite-state transport and exact budget identities can be expressed over a
commutative semiring. The construction deliberately stops before probability,
order, limits, or completeness: those remain separate real-valued interfaces.
-/

open Finset

namespace CGB.AlgebraicCore

variable {S E R : Type*} [Fintype S] [Fintype E] [CommSemiring R]

def pushForward (K : S → E → S → R) (ρ : S → R) (y : S) : R :=
  ∑ x, ∑ e, ρ x * K x e y

def rowMass (K : S → E → S → R) (x : S) : R :=
  ∑ e, ∑ y, K x e y

theorem pushForward_total (K : S → E → S → R) (ρ : S → R)
    (hrow : ∀ x, rowMass K x = 1) :
    ∑ y, pushForward K ρ y = ∑ x, ρ x := by
  classical
  calc
    ∑ y, pushForward K ρ y = ∑ x, ρ x * rowMass K x := by
      simp only [pushForward, rowMass, Finset.mul_sum]
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl (fun x _ => Finset.sum_comm)
    _ = ∑ x, ρ x := by simp [hrow]

def pathWeight (K : S → E → S → R) (p : ℕ → S) (e : ℕ → E) (n : ℕ) : R :=
  ∏ t ∈ range n, K (p t) (e t) (p (t + 1))

theorem pathWeight_zero (K : S → E → S → R) (p : ℕ → S) (e : ℕ → E) :
    pathWeight K p e 0 = 1 := by
  simp [pathWeight]

def jetAdd {R : Type*} [Add R] (u v : R × R) : R × R :=
  (u.1 + v.1, u.2 + v.2)

def jetMul {R : Type*} [Mul R] [Add R] (u v : R × R) : R × R :=
  (u.1 * v.1, u.1 * v.2 + u.2 * v.1)

theorem jetMul_product_rule {R : Type*} [Mul R] [Add R] (u v : R × R) :
    (jetMul u v).2 = u.1 * v.2 + u.2 * v.1 := by
  rfl

def scalarHorner {R : Type*} [AddMonoid R] [Mul R] : List R → R → R
  | [], _ => 0
  | a :: as, x => a + x * scalarHorner as x

def jetHorner {R : Type*} [AddMonoid R] [Mul R] : List R → (R × R) → R × R
  | [], _ => (0, 0)
  | a :: as, u => jetAdd (a, 0) (jetMul u (jetHorner as u))

theorem jetHorner_value {R : Type*} [AddMonoid R] [Mul R]
    (coeff : List R) (x dx : R) :
    (jetHorner coeff (x, dx)).1 = scalarHorner coeff x := by
  induction coeff with
  | nil => rfl
  | cons a as ih =>
    simp [jetHorner, jetAdd, jetMul, scalarHorner, ih]

theorem cast_local_budget {R : Type*} [Semiring R] (s : SensoryMaintenance.Stim)
    (x : SensoryMaintenance.State) :
    (SensoryMaintenance.cost (SensoryMaintenance.next s x) : R) +
        SensoryMaintenance.outgoing s x =
      (SensoryMaintenance.cost x : R) + SensoryMaintenance.incoming s x := by
  have h := congrArg (fun n : ℕ => (n : R)) (SensoryMaintenance.local_budgets s x).1
  simpa only [Nat.cast_add] using h

theorem cast_path_budget {R : Type*} [Semiring R] (stim : ℕ → SensoryMaintenance.Stim)
    (p : ℕ → SensoryMaintenance.State)
    (hp : ∀ t, p (t + 1) = SensoryMaintenance.next (stim t) (p t)) (n : ℕ) :
    (SensoryMaintenance.cost (p n) : R) +
        ∑ t ∈ range n, (SensoryMaintenance.outgoing (stim t) (p t) : R) =
      (SensoryMaintenance.cost (p 0) : R) +
        ∑ t ∈ range n, (SensoryMaintenance.incoming (stim t) (p t) : R) := by
  have h := congrArg (fun z : ℕ => (z : R))
    (SensoryMaintenance.path_budget stim p hp n)
  simpa only [Nat.cast_add, Nat.cast_sum] using h

end CGB.AlgebraicCore
