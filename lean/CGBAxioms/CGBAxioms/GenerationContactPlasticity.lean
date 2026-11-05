import CGBAxioms.GenerationResponseGatedCarrier
import Mathlib.Analysis.SpecialFunctions.Sigmoid

/-!
# Compartment tags modulate a current-event spatial pullback

The coefficient `c = lambda * eligibility` is fixed during this event's
parameter differentiation. `lambda` is the local return through the preceding
S-to-H/flux-to-current-score read; `eligibility = S * demand` is participation
history, not a historical parameter derivative. The potential below contracts
this stopped coefficient with the actual vector shared-pool update.

Only a current-event spatial VJP is described: old S, pre, gate, route and
source are fixed. No cross-event pullback, BPTT or RTRL is asserted. This local
surrogate is neither the total long-horizon score gradient nor a guarantee of
current/future CE improvement. Its finite algebra requires no runtime history.
-/

namespace CGB.Generation.ContactPlasticity

open Finset

/-- Transposing a finite linear contact read returns exactly the same scalar
contraction; the return is not an independently learned feedback matrix. -/
theorem linear_contact_return {I J : Type*} [Fintype I] [Fintype J]
    (post : I → ℝ) (contact : I → J → ℝ) (pre : J → ℝ) :
    (∑ i, post i * (∑ j, contact i j * pre j)) =
      ∑ j, pre j * (∑ i, contact i j * post i) := by
  simp_rw [mul_sum]
  rw [sum_comm]
  apply sum_congr rfl
  intro j _
  apply sum_congr rfl
  intro i _
  ring

/-- Incoming-minus-outgoing periodic flux has the next-minus-local adjoint.
`next.symm` is the predecessor, so this covers any finite periodic geometry. -/
theorem periodic_flux_return {I : Type*} [Fintype I]
    (next : Equiv.Perm I) (v flow : I → ℝ) :
    (∑ i, v i * (flow (next.symm i) - flow i)) =
      ∑ i, (v (next i) - v i) * flow i := by
  have hshift : (∑ i, v i * flow (next.symm i)) = ∑ i, v (next i) * flow i := by
    calc
      _ = ∑ i, v (next i) * flow (next.symm (next i)) :=
        (Equiv.sum_comp next (fun i => v i * flow (next.symm i))).symm
      _ = _ := by simp
  simp_rw [mul_sub, sub_mul]
  rw [sum_sub_distrib, sum_sub_distrib, hshift]

/-- Two transport directions add their local adjoints without retaining a
transport trajectory or its past pullbacks. -/
theorem two_direction_flux_return {I : Type*} [Fintype I]
    (p q : Equiv.Perm I) (v f g : I → ℝ) :
    (∑ i, v i * ((f (p.symm i) - f i) + (g (q.symm i) - g i))) =
      ∑ i, ((v (p i) - v i) * f i + (v (q i) - v i) * g i) := by
  simp_rw [mul_add]
  rw [sum_add_distrib, sum_add_distrib,
    periodic_flux_return p v f, periodic_flux_return q v g]

/-- The actual bounded transport law has the exact local factor used by the
runtime: an infinitesimal change of a pre-flow changes `tanh flow` by
`1 - tanh(flow)^2`. Combined with `periodic_flux_return`, this is a purely
current-event pullback through the conservative flow, not a stored temporal
gradient. -/
theorem tanh_offset_deriv (raw : ℝ) :
    HasDerivAt (fun θ : ℝ => Real.tanh (raw + θ))
      (1 - Real.tanh raw ^ 2) 0 := by
  have hplus : HasDerivAt (fun θ : ℝ => Real.exp (raw + θ)) (Real.exp raw) 0 := by
    simpa using ((hasDerivAt_const (0 : ℝ) raw).add (hasDerivAt_id (0 : ℝ))).exp
  have hinner : HasDerivAt (fun θ : ℝ => -(raw + θ)) (-1) 0 := by
    convert ((hasDerivAt_const (0 : ℝ) (-raw)).sub (hasDerivAt_id (0 : ℝ))) using 1
    · ext θ
      simp
      ring
    · norm_num
  have hminus : HasDerivAt (fun θ : ℝ => Real.exp (-(raw + θ))) (-Real.exp (-raw)) 0 := by
    convert hinner.exp using 1 <;> simp
  have hnum := hplus.sub hminus
  have hden := hplus.add hminus
  have hden0 : Real.exp raw + Real.exp (-raw) ≠ 0 := by positivity
  have hquot := hnum.div hden (by simpa using hden0)
  dsimp at hquot
  rw [show (fun θ : ℝ => Real.tanh (raw + θ)) =
      (fun θ : ℝ => (Real.exp (raw + θ) - Real.exp (-(raw + θ))) /
        (Real.exp (raw + θ) + Real.exp (-(raw + θ)))) by
        funext θ; rw [Real.tanh_eq]]
  convert hquot using 1
  rw [Real.tanh_eq]
  field_simp [hden0]
  ring

/-- Positive total drive cancels between admission and normalized target.
No assumption on the capacity divisor is needed for this algebraic identity. -/
theorem allocation_normalization_cancels (active totalDrive capacity drive : ℝ)
    (htotal : totalDrive ≠ 0) :
    (active * (totalDrive / capacity)) * (drive / totalDrive) =
      active * drive / capacity := by
  calc
    _ = (active * drive / capacity) * (totalDrive / totalDrive) := by ring
    _ = _ := by rw [div_self htotal, mul_one]

/-- The cancellation holds inside the actual shared reservoir, with its old
occupation and coordinatewise release unchanged. -/
theorem normalized_actual_pool {n : ℕ} (occupied drive release : Fin n → ℝ)
    (active capacity gate : ℝ) (htotal : (∑ j, drive j) ≠ 0) (i : Fin n) :
    sharedContactPoolVectorStep occupied (fun j => drive j / ∑ k, drive k) release
        (active * ((∑ j, drive j) / capacity)) gate i =
      occupied i + gate * (active * drive i / capacity) * (1 - ∑ j, occupied j) -
        gate * release i * occupied i := by
  unfold sharedContactPoolVectorStep
  have h := allocation_normalization_cancels active (∑ j, drive j) capacity (drive i) htotal
  calc
    _ = occupied i + gate * ((active * ((∑ j, drive j) / capacity)) *
        (drive i / ∑ j, drive j)) * (1 - ∑ j, occupied j) -
        gate * release i * occupied i := by ring
    _ = _ := by rw [h]

/-- The executable endpoint allocation uses the sigmoid of the mean logit. -/
noncomputable def endpointAllocation (left right : ℝ) : ℝ :=
  Real.sigmoid ((left + right) / 2)

/-- The executable endpoint retention averages the two endpoint sigmoids. -/
noncomputable def endpointRetention (left right : ℝ) : ℝ :=
  (Real.sigmoid left + Real.sigmoid right) / 2

/-- Mathlib's sigmoid is exactly the runtime logistic expression. -/
theorem sigmoid_runtime_formula (x : ℝ) :
    Real.sigmoid x = 1 / (1 + Real.exp (-x)) := by
  simp [Real.sigmoid, one_div]

/-- Exact allocation response to one endpoint's local pre-weight perturbation;
`pre` is fixed and the sigmoid supplies the current post-response factor. -/
theorem endpoint_allocation_deriv (left right pre : ℝ) :
    HasDerivAt (fun θ => endpointAllocation (left + θ * pre) right)
      (Real.sigmoid ((left + right) / 2) *
        (1 - Real.sigmoid ((left + right) / 2)) * (pre / 2)) 0 := by
  have hinner : HasDerivAt (fun θ : ℝ => (left + θ * pre + right) / 2) (pre / 2) 0 := by
    convert (((hasDerivAt_const (0 : ℝ) left).add
      ((hasDerivAt_id (0 : ℝ)).mul_const pre)).add_const right).div_const 2 using 1 <;> simp <;> rfl
  exact (Real.hasDerivAt_sigmoid ((left + right) / 2)).comp_of_eq 0 hinner (by simp)

/-- Exact mean-retention response at one perturbed endpoint. Unlike allocation,
the sigmoid is applied before the two endpoints are averaged. -/
theorem endpoint_retention_deriv (left right pre : ℝ) :
    HasDerivAt (fun θ => endpointRetention (left + θ * pre) right)
      (Real.sigmoid left * (1 - Real.sigmoid left) * pre / 2) 0 := by
  have hinner : HasDerivAt (fun θ : ℝ => left + θ * pre) pre 0 := by
    convert (hasDerivAt_const (0 : ℝ) left).add
      ((hasDerivAt_id (0 : ℝ)).mul_const pre) using 1 <;> simp <;> rfl
  exact (((Real.hasDerivAt_sigmoid left).comp_of_eq 0 hinner (by simp)).add_const
    (Real.sigmoid right)).div_const 2

/-- The current-event proxy contracts stopped `lambda * eligibility` with the
actual shared-pool transition, not an unrefined affine toy update. -/
noncomputable def localPotential {n : ℕ} (occupied target release : Fin n → ℝ)
    (inflow gate : ℝ) (lambda eligibility : Fin n → ℝ) : ℝ :=
  ∑ i, (lambda i * eligibility i) *
    sharedContactPoolVectorStep occupied target release inflow gate i

/-- Vanishing participation tags make the entire parameter-dependent proxy
identically zero, not merely its value at a chosen parameter. -/
theorem local_potential_zero_eligibility {n : ℕ} (occupied target release : Fin n → ℝ)
    (inflow gate : ℝ) (lambda eligibility : Fin n → ℝ) (hzero : ∀ i, eligibility i = 0) :
    localPotential occupied target release inflow gate lambda eligibility = 0 := by
  simp [localPotential, hzero]

/-- A zero current local return also makes the stopped-coefficient proxy zero. -/
theorem local_potential_zero_return {n : ℕ} (occupied target release : Fin n → ℝ)
    (inflow gate : ℝ) (lambda eligibility : Fin n → ℝ) (hzero : ∀ i, lambda i = 0) :
    localPotential occupied target release inflow gate lambda eligibility = 0 := by
  simp [localPotential, hzero]

/-- Exact current-event derivative through the actual inflow and coordinatewise
release. Old S, target, gate, lambda and eligibility are constant here; this is
not the derivative of a state trajectory or of the current language CE. -/
theorem local_potential_deriv {n : ℕ} (occupied target lambda eligibility : Fin n → ℝ)
    (inflow : ℝ → ℝ) (release : ℝ → Fin n → ℝ) (gate x dinflow : ℝ)
    (drelease : Fin n → ℝ) (ha : HasDerivAt inflow dinflow x)
    (hr : ∀ i, HasDerivAt (fun θ => release θ i) (drelease i) x) :
    HasDerivAt (fun θ => localPotential occupied target (release θ) (inflow θ)
        gate lambda eligibility)
      (∑ i, (lambda i * eligibility i) *
        (gate * target i * (1 - ∑ j, occupied j) * dinflow -
          gate * occupied i * drelease i)) x := by
  have h := HasDerivAt.fun_sum (u := univ) (fun i _ =>
    (((hasDerivAt_const x (occupied i)).add
      (((ha.const_mul gate).mul_const (target i)).mul_const (1 - ∑ j, occupied j))).sub
      (((hr i).const_mul gate).mul_const (occupied i))).const_mul (lambda i * eligibility i))
  convert h using 1
  · rfl
  · apply sum_congr rfl
    intro i _
    ring

/-- Actual normalized allocation, allowing target and total admission to vary
together. `capacity` is rank K, not the number of source-expanded coordinates.
Only the selected source's drive is nonzero in the runtime instantiation. -/
theorem allocation_pool_pre_post_coefficient {n : ℕ}
    (occupied release : Fin n → ℝ) (drive : ℝ → Fin n → ℝ)
    (active capacity gate left right pre c : ℝ) (i : Fin n)
    (htotal : ∀ θ, (∑ j, drive θ j) ≠ 0)
    (hdrive : ∀ θ, drive θ i = endpointAllocation (left + θ * pre) right) :
    HasDerivAt (fun θ => c * sharedContactPoolVectorStep occupied
      (fun j => drive θ j / ∑ k, drive θ k) release
      (active * ((∑ j, drive θ j) / capacity)) gate i)
      (pre * (Real.sigmoid ((left + right) / 2) *
        (1 - Real.sigmoid ((left + right) / 2)) / 2) *
        (c * gate * active * (1 - ∑ j, occupied j) / capacity)) 0 := by
  have hd : HasDerivAt (fun θ => drive θ i)
      (Real.sigmoid ((left + right) / 2) *
        (1 - Real.sigmoid ((left + right) / 2)) * (pre / 2)) 0 := by
    rw [show (fun θ => drive θ i) =
      (fun θ => endpointAllocation (left + θ * pre) right) from funext hdrive]
    exact endpoint_allocation_deriv left right pre
  have hform : (fun θ => c * sharedContactPoolVectorStep occupied
      (fun j => drive θ j / ∑ k, drive θ k) release
      (active * ((∑ j, drive θ j) / capacity)) gate i) =
      (fun θ => c * (occupied i + gate * (active * drive θ i / capacity) *
        (1 - ∑ j, occupied j) - gate * release i * occupied i)) := by
    funext θ
    rw [normalized_actual_pool occupied (drive θ) release active capacity gate (htotal θ) i]
  rw [hform]
  have h := ((hasDerivAt_const (0 : ℝ) (occupied i)).add
    ((((hd.const_mul active).div_const capacity).const_mul gate).mul_const
      (1 - ∑ j, occupied j))).sub
        (hasDerivAt_const (0 : ℝ) (gate * release i * occupied i))
  convert h.const_mul c using 1 <;> ring

/-- The actual release is A*(1-retention), so a retention logit's response has
a positive sign and is contracted with old S across all source coordinates.
Equal per-rank endpoint logits across sources specialize this sum directly. -/
theorem retention_pool_pre_post_coefficient {n : ℕ}
    (occupied target lambda eligibility left right pre : Fin n → ℝ)
    (inflow gate active : ℝ) :
    HasDerivAt (fun θ => localPotential occupied target
      (fun i => active * (1 - endpointRetention (left i + θ * pre i) (right i)))
      inflow gate lambda eligibility)
      (∑ i, pre i * (Real.sigmoid (left i) * (1 - Real.sigmoid (left i)) / 2) *
        ((lambda i * eligibility i) * gate * active * occupied i)) 0 := by
  have hr : ∀ i, HasDerivAt
      (fun θ => active * (1 - endpointRetention (left i + θ * pre i) (right i)))
      (-active * (Real.sigmoid (left i) * (1 - Real.sigmoid (left i)) * pre i / 2)) 0 := by
    intro i
    convert ((hasDerivAt_const (0 : ℝ) 1).sub
      (endpoint_retention_deriv (left i) (right i) (pre i))).const_mul active using 1 <;> ring
  have h := local_potential_deriv occupied target lambda eligibility (fun _ => inflow)
    (fun θ i => active * (1 - endpointRetention (left i + θ * pre i) (right i)))
    gate 0 0 _ (hasDerivAt_const (0 : ℝ) inflow) hr
  convert h using 1
  apply sum_congr rfl
  intro i _
  ring

/-- A parameter direction contracted with itself is nonnegative. This concerns
only the local proxy's own direction, not descent of a different CE objective. -/
theorem local_direction_self_nonneg {P : Type*} [Fintype P] (direction : P → ℝ) :
    0 ≤ ∑ p, direction p * direction p := by
  exact sum_nonneg (fun p _ => mul_self_nonneg (direction p))

end CGB.Generation.ContactPlasticity

