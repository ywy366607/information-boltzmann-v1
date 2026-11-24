import Mathlib.Tactic
import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Analysis.Matrix.Normed

/-!
Mathematical core of the current 32-layer, 8-head body, read from
bfss_gdn_capture.observed_momentum and adaptive_hamiltonian.potential/quiet.
This is a real-arithmetic abstraction, not verification of CUDA rounding.
The surrounding learned language, archive, actor, optimizer and environment
are accounted for by the execution abstraction; they are not frozen here.
-/
namespace CGB.NativeDynamics

noncomputable section
open scoped Matrix.Norms.Frobenius

abbrev Layer := Fin 32
abbrev Head := Fin 8
abbrev Axis := Fin 32
abbrev Mat := Matrix Axis Axis ℝ
abbrev Field := Layer → Head → Mat

def normSq (x : Mat) : ℝ := ∑ i, ∑ j, (x i j) ^ 2
def inner (x y : Mat) : ℝ := ∑ i, ∑ j, x i j * y i j
def softplus (x : ℝ) : ℝ := Real.log (1 + Real.exp x)
def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

structure Coefficients where
  intrinsic : Layer → Head → ℝ
  contentFrequency : Layer → ℝ
  rawStrength : Layer → ℝ
  correlationWeight : Layer → ℝ
  magnitudeWeight : Layer → ℝ
  context : Layer → Head → ℝ
  frequencyScale : Layer → Head → ℝ
  impedance : Layer → Head → ℝ

def omega (c : Coefficients) (l : Layer) (h : Head) : ℝ :=
  softplus (c.intrinsic l h + c.contentFrequency l * c.context l h) *
    c.frequencyScale l h

def pairWeight (c : Coefficients) (s : Field) (l : Layer) (h k : Head) : ℝ :=
  let x := s l h
  let y := s l k
  let correlation := inner x y / Real.sqrt ((1 + normSq x) * (1 + normSq y))
  2 * sigmoid (c.correlationWeight l * correlation + c.magnitudeWeight l *
    (Real.log (1 + normSq x) + Real.log (1 + normSq y)))

/-- Includes the S dependence of the content weights, as production grad does. -/
def potential (c : Coefficients) (s : Field) : ℝ :=
  ∑ l, softplus (c.rawStrength l) *
    (∑ h, ∑ k, pairWeight c s l h k *
      (normSq (s l h * (s l k).transpose - s l k * (s l h).transpose) +
       normSq ((s l h).transpose * s l k - (s l k).transpose * s l h))) / 16

def basis (l : Layer) (h : Head) (i j : Axis) : Field :=
  fun a b m n => if a = l ∧ b = h ∧ m = i ∧ n = j then 1 else 0

def force (c : Coefficients) (s : Field) : Field :=
  fun l h i j => fderiv ℝ (potential c) s (basis l h i j)

structure MatrixState where
  S : Field
  P : Field

/-- GDN write including actual admission correction; k is the normalized key. -/
def observedWrite (x : MatrixState) (alpha beta : Layer → Head → ℝ)
    (key value : Layer → Head → Axis → ℝ) (correction : Field) : MatrixState :=
  let innovation : Field := fun l h i j => beta l h * key l h i *
    (value l h j - ∑ m, (alpha l h * x.S l h m j) * key l h m)
  ⟨fun l h i j => alpha l h * x.S l h i j + innovation l h i j + correction l h i j,
   fun l h i j => alpha l h * x.P l h i j + innovation l h i j⟩

/-- One actual impedance-aware kick/rotation/kick substep, not an Euler proxy. -/
noncomputable def quietSubstep (c : Coefficients) (dt : ℝ) (x : MatrixState) : MatrixState :=
  let half : Field := fun l h i j => x.P l h i j - dt / 2 * force c x.S l h i j
  let nextS : Field := fun l h i j =>
    Real.cos (omega c l h * dt) * x.S l h i j +
    Real.sin (omega c l h * dt) / c.impedance l h * half l h i j
  let nextP : Field := fun l h i j =>
    Real.cos (omega c l h * dt) * half l h i j -
    c.impedance l h * Real.sin (omega c l h * dt) * x.S l h i j
  ⟨nextS, fun l h i j => nextP l h i j - dt / 2 * force c nextS l h i j⟩

/-- Exact algebra of the production write, used to check reduced observables. -/
theorem write_difference (x : MatrixState) (alpha beta : Layer → Head → ℝ)
    (key value : Layer → Head → Axis → ℝ) (correction : Field)
    (l : Layer) (h : Head) (i j : Axis) :
    let y := observedWrite x alpha beta key value correction
    y.S l h i j - y.P l h i j =
      alpha l h * (x.S l h i j - x.P l h i j) + correction l h i j := by
  dsimp [observedWrite]
  ring

end
end CGB.NativeDynamics
