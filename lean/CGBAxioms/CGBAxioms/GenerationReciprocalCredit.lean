import CGBAxioms.GenerationContactEligibility
import CGBAxioms.GenerationUnifiedOptimizer

/-!
# Reciprocal local credit

This is the common algebraic core for every slow map in a generated local
medium.  A contact does not need a separately designed optimizer: its forward
effect is a pre-synaptic value times its contact strength, while the reciprocal
read supplies the post-synaptic coefficient.  Their product is the local
forward response to changing that contact.

The language/world result is a third factor.  The existing bounded conserved
phase exchange decides whether this response occupies, releases, or holds a
finite local S coordinate.  Thus this file supplies one mechanism family, not
a rule per matrix name.
-/

namespace CGB.Generation

open Finset

/-- A reciprocal local contact field read from source values into target
coefficients.  The same contact array may be used in the forward field and in
the return field; no separately stored transpose is required. -/
noncomputable def reciprocalContactRead {Target Source : Type*}
    [Fintype Target] [Fintype Source]
    (post : Target → ℝ) (contact : Target → Source → ℝ) (pre : Source → ℝ) : ℝ :=
  ∑ i, ∑ j, post i * contact i j * pre j

/-- A local perturbation of one existing contact. -/
noncomputable def reciprocalContactPerturb {Target Source : Type*}
    [DecidableEq Target] [DecidableEq Source]
    (contact : Target → Source → ℝ) (target : Target) (source : Source) (θ : ℝ) :
    Target → Source → ℝ :=
  fun i j => contact i j + if i = target then (if j = source then θ else 0) else 0

/-- The exact local forward response of one reciprocal contact.  It is the
pre/post eligibility product, derived from the actual contact read rather
than stipulated as a biological analogy. -/
theorem reciprocal_contact_response_deriv {Target Source : Type*}
    [Fintype Target] [Fintype Source] [DecidableEq Target] [DecidableEq Source]
    (post : Target → ℝ) (contact : Target → Source → ℝ) (pre : Source → ℝ)
    (target : Target) (source : Source) :
    HasDerivAt
      (fun θ => reciprocalContactRead post
        (reciprocalContactPerturb contact target source θ) pre)
      (post target * pre source) 0 := by
  have hform :
      (fun θ => reciprocalContactRead post
        (reciprocalContactPerturb contact target source θ) pre) =
      fun θ => reciprocalContactRead post contact pre + θ * (post target * pre source) := by
    funext θ
    unfold reciprocalContactRead reciprocalContactPerturb
    calc
      (∑ i, ∑ j, post i * (contact i j +
        if i = target then (if j = source then θ else 0) else 0) * pre j) =
          ∑ i, ∑ j, (post i * contact i j * pre j +
            post i * (if i = target then (if j = source then θ else 0) else 0) * pre j) := by
        apply sum_congr rfl
        intro i _
        apply sum_congr rfl
        intro j _
        ring
      _ = (∑ i, ∑ j, post i * contact i j * pre j) +
          ∑ i, ∑ j, post i *
            (if i = target then (if j = source then θ else 0) else 0) * pre j := by
        calc
          (∑ i, ∑ j, (post i * contact i j * pre j +
              post i * (if i = target then (if j = source then θ else 0) else 0) * pre j)) =
              ∑ i, ((∑ j, post i * contact i j * pre j) +
                ∑ j, post i *
                  (if i = target then (if j = source then θ else 0) else 0) * pre j) := by
            apply sum_congr rfl
            intro i _
            rw [sum_add_distrib]
          _ = _ := by rw [sum_add_distrib]
    have hbase :
        (∑ i, ∑ j, post i *
          (if i = target then (if j = source then θ else 0) else 0) * pre j) =
          θ * (post target * pre source) := by
      classical
      simp [mul_ite]
      ring
    rw [hbase]
  rw [hform]
  change HasDerivAt (fun θ => reciprocalContactRead post contact pre +
    θ * (post target * pre source)) (post target * pre source) 0
  convert (hasDerivAt_const (0 : ℝ) (reciprocalContactRead post contact pre)).add
    ((hasDerivAt_id (0 : ℝ)).mul_const (post target * pre source)) using 1
  · ext θ
    rfl
  · ring

/-- Scaling a local response by a fresh boundary teaching signal preserves its
identity as the exact directional derivative.  This is the abstract
three-factor form: pre × post × outcome. -/
theorem reciprocal_contact_teaching_deriv {Target Source : Type*}
    [Fintype Target] [Fintype Source] [DecidableEq Target] [DecidableEq Source]
    (post : Target → ℝ) (contact : Target → Source → ℝ) (pre : Source → ℝ)
    (target : Target) (source : Source) (teaching : ℝ) :
    HasDerivAt
      (fun η => reciprocalContactRead post
        (reciprocalContactPerturb contact target source (η * teaching)) pre)
      (teaching * post target * pre source) 0 := by
  have h := reciprocal_contact_response_deriv post contact pre target source
  have hscale : HasDerivAt (fun η : ℝ => η * teaching) teaching 0 :=
    by simpa only [id, one_mul] using (hasDerivAt_id (0 : ℝ)).mul_const teaching
  simpa only [Function.comp_def, id, one_mul, mul_assoc, mul_comm, mul_left_comm] using
    h.comp_of_eq 0 hscale (by simp)

end CGB.Generation
