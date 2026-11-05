import CGBAxioms.GenerationContactEligibility

/-!
# A shared finite contact reservoir

`S` is already a bounded, typed local contact state.  A separate complement
for every coordinate prevents unbounded growth of each individual contact, but
does not make alternatives compete.  For one local contact bundle, use the
unoccupied part of its *total* `S` mass as the sole source of new occupation.

The update below is not another state page or a food account.  Its state is
only `occupied`; `1 - sum occupied` is the implicit unoccupied partner.  A
fixed causal route may select the bundle, while this rule redistributes the
finite capacity inside it.  Thus it cannot alter the address on which the GDN
value-credit theorem relies.
-/

namespace CGB.Generation

open Finset

/-- The bounded kinetic rate used when a nonnegative boundary receipt gates
an existing contact exchange.  It is a rate conversion, not another carrier
or a separate learning state. -/
noncomputable def receiptRate (x : ℝ) : ℝ := 1 - Real.exp (-x)

/-- A nonnegative receipt produces a legal exchange rate, even when its raw
magnitude is unbounded.  This is the precise range premise needed to apply the
shared-pool conservation and nonnegativity laws to the executable
``-expm1(-x)`` conversion. -/
theorem receipt_rate_in_unit_interval (x : ℝ) (hx : 0 ≤ x) :
    0 ≤ receiptRate x ∧ receiptRate x ≤ 1 := by
  unfold receiptRate
  have hupper : Real.exp (-x) ≤ 1 :=
    Real.exp_le_one_iff.mpr (neg_nonpos.mpr hx)
  have hlower : 0 < Real.exp (-x) := Real.exp_pos (-x)
  constructor <;> linarith

/-- One gate-controlled exchange between a local contact bundle and its
implicit shared free pool. `target` is the local distribution of an inflowted
increment; it is a transient readout, never stored as an event record. -/
noncomputable def sharedContactPoolStep {n : ℕ} (occupied target : Fin n → ℝ)
    (inflow release gate : ℝ) : Fin n → ℝ :=
  fun i => occupied i + gate *
    (inflow * target i * (1 - ∑ j, occupied j) - release * occupied i)

/-- The whole contact bundle follows one two-state exchange.  Coordinate
shares may differ, but their sum can draw only from the same implicit pool. -/
theorem shared_contact_pool_total {n : ℕ} (occupied target : Fin n → ℝ)
    (inflow release gate : ℝ) (htarget : ∑ i, target i = 1) :
    ∑ i, sharedContactPoolStep occupied target inflow release gate i =
      (∑ i, occupied i) + gate *
        (inflow * (1 - ∑ i, occupied i) - release * ∑ i, occupied i) := by
  unfold sharedContactPoolStep
  rw [sum_add_distrib]
  have hflow :
      (∑ i : Fin n, (inflow * target i * (1 - ∑ j, occupied j) -
        release * occupied i)) =
        inflow * (1 - ∑ j, occupied j) * (∑ i, target i) -
          release * (∑ i, occupied i) := by
    rw [sum_sub_distrib]
    congr 1
    · calc
        (∑ i, inflow * target i * (1 - ∑ j, occupied j)) =
            ∑ i, (inflow * (1 - ∑ j, occupied j)) * target i := by
              apply sum_congr rfl
              intro i _
              ring
        _ = inflow * (1 - ∑ j, occupied j) * (∑ i, target i) := by
              rw [← mul_sum]
    · rw [← mul_sum]
  rw [← mul_sum, hflow, htarget]
  ring

/-- Once a local bundle is full, no positive admission can add to any of its
contacts.  Only its release channel can change it. -/
theorem full_shared_pool_blocks_admission {n : ℕ} (occupied target : Fin n → ℝ)
    (inflow release gate : ℝ) (hfull : ∑ j, occupied j = 1) (i : Fin n) :
    sharedContactPoolStep occupied target inflow release gate i =
      occupied i - gate * release * occupied i := by
  unfold sharedContactPoolStep
  rw [hfull]
  ring

/-- For a positive local admission gate, an already more occupied bundle has
strictly less remaining marginal capacity.  This is the local anti-dominance
law required to prevent unlimited winner growth; it does not prescribe a
global balancing scalar. -/
theorem shared_pool_marginal_admission_is_anti_dominant
    (gate inflow target q₁ q₂ : ℝ)
    (hgate : 0 < gate) (hinflow : 0 < inflow) (htarget : 0 < target)
    (horder : q₁ < q₂) :
    gate * inflow * target * (1 - q₂) <
      gate * inflow * target * (1 - q₁) := by
  nlinarith [mul_pos (mul_pos hgate hinflow) htarget]

/-- When a local evidence gate is closed, the complete bundle holds exactly.
This lets delayed evidence regulate the existing S carrier without advancing a
separate trace or a global learning-rate state. -/
theorem shared_pool_closed_gate_holds {n : ℕ} (occupied target : Fin n → ℝ)
    (inflow release : ℝ) :
    sharedContactPoolStep occupied target inflow release 0 = occupied := by
  funext i
  simp [sharedContactPoolStep]

/-- The executable shared-pool transition: release is coordinatewise, whereas
admission draws from one common implicit free pool. -/
noncomputable def sharedContactPoolVectorStep {n : ℕ}
    (occupied target release : Fin n → ℝ) (inflow gate : ℝ) : Fin n → ℝ :=
  fun i => occupied i + gate * inflow * target i * (1 - ∑ j, occupied j) -
    gate * release i * occupied i

/-- Constant vector release recovers the original interface exactly. -/
theorem shared_contact_pool_vector_constant_release {n : ℕ}
    (occupied target : Fin n → ℝ) (inflow release gate : ℝ) :
    sharedContactPoolVectorStep occupied target (fun _ => release) inflow gate =
      sharedContactPoolStep occupied target inflow release gate := by
  funext i
  unfold sharedContactPoolVectorStep sharedContactPoolStep
  ring

theorem shared_contact_pool_vector_total {n : ℕ}
    (occupied target release : Fin n → ℝ) (inflow gate : ℝ)
    (htarget : ∑ i, target i = 1) :
    ∑ i, sharedContactPoolVectorStep occupied target release inflow gate i =
      (∑ i, occupied i) + gate * inflow * (1 - ∑ i, occupied i) -
        gate * ∑ i, release i * occupied i := by
  unfold sharedContactPoolVectorStep
  rw [sum_sub_distrib, sum_add_distrib]
  have hadmit : (∑ i, gate * inflow * target i * (1 - ∑ j, occupied j)) =
      gate * inflow * (1 - ∑ j, occupied j) := by
    calc
      _ = ∑ i, (gate * inflow * (1 - ∑ j, occupied j)) * target i := by
        apply sum_congr rfl
        intro i _
        ring
      _ = _ := by rw [← mul_sum, htarget, mul_one]
  rw [hadmit]
  congr 1
  rw [mul_sum]
  apply sum_congr rfl
  intro i _
  ring

/-- The implicit free partner loses admission and receives every coordinate's
release. It is an algebraic complement, not an additional state array. -/
noncomputable def sharedContactPoolFreeStep {n : ℕ}
    (occupied release : Fin n → ℝ) (inflow gate : ℝ) : ℝ :=
  (1 - gate * inflow) * (1 - ∑ i, occupied i) +
    gate * ∑ i, release i * occupied i

theorem shared_contact_pool_vector_conserves_total {n : ℕ}
    (occupied target release : Fin n → ℝ) (inflow gate : ℝ)
    (htarget : ∑ i, target i = 1) :
    (∑ i, sharedContactPoolVectorStep occupied target release inflow gate i) +
      sharedContactPoolFreeStep occupied release inflow gate = 1 := by
  rw [shared_contact_pool_vector_total occupied target release inflow gate htarget]
  unfold sharedContactPoolFreeStep
  ring

theorem shared_contact_pool_vector_nonneg {n : ℕ}
    (occupied target release : Fin n → ℝ) (inflow gate : ℝ)
    (hoccupied : ∀ i, 0 ≤ occupied i) (hsum : ∑ i, occupied i ≤ 1)
    (htarget : ∀ i, 0 ≤ target i)
    (hrelease : ∀ i, 0 ≤ release i ∧ release i ≤ 1)
    (hinflow : 0 ≤ inflow) (hgate : 0 ≤ gate ∧ gate ≤ 1) (i : Fin n) :
    0 ≤ sharedContactPoolVectorStep occupied target release inflow gate i := by
  have hgr : gate * release i ≤ 1 :=
    (mul_le_mul hgate.2 (hrelease i).2 (hrelease i).1 (by norm_num)).trans_eq
      (one_mul 1)
  have hretain : 0 ≤ (1 - gate * release i) * occupied i :=
    mul_nonneg (sub_nonneg.mpr hgr) (hoccupied i)
  have hadmit : 0 ≤ gate * inflow * target i * (1 - ∑ j, occupied j) :=
    mul_nonneg (mul_nonneg (mul_nonneg hgate.1 hinflow) (htarget i))
      (sub_nonneg.mpr hsum)
  unfold sharedContactPoolVectorStep
  nlinarith

theorem shared_contact_pool_free_nonneg {n : ℕ}
    (occupied release : Fin n → ℝ) (inflow gate : ℝ)
    (hoccupied : ∀ i, 0 ≤ occupied i) (hsum : ∑ i, occupied i ≤ 1)
    (hrelease : ∀ i, 0 ≤ release i)
    (hinflow : 0 ≤ inflow ∧ inflow ≤ 1) (hgate : 0 ≤ gate ∧ gate ≤ 1) :
    0 ≤ sharedContactPoolFreeStep occupied release inflow gate := by
  have hga : gate * inflow ≤ 1 :=
    (mul_le_mul hgate.2 hinflow.2 hinflow.1 (by norm_num)).trans_eq (one_mul 1)
  exact add_nonneg
    (mul_nonneg (sub_nonneg.mpr hga) (sub_nonneg.mpr hsum))
    (mul_nonneg hgate.1 (sum_nonneg fun i _ => mul_nonneg (hrelease i) (hoccupied i)))

theorem shared_contact_pool_vector_total_le_one {n : ℕ}
    (occupied target release : Fin n → ℝ) (inflow gate : ℝ)
    (hoccupied : ∀ i, 0 ≤ occupied i) (hsum : ∑ i, occupied i ≤ 1)
    (htarget : ∑ i, target i = 1) (hrelease : ∀ i, 0 ≤ release i)
    (hinflow : 0 ≤ inflow ∧ inflow ≤ 1) (hgate : 0 ≤ gate ∧ gate ≤ 1) :
    ∑ i, sharedContactPoolVectorStep occupied target release inflow gate i ≤ 1 := by
  have hfree := shared_contact_pool_free_nonneg occupied release inflow gate
    hoccupied hsum hrelease hinflow hgate
  have htotal := shared_contact_pool_vector_conserves_total
    occupied target release inflow gate htarget
  linarith

theorem full_shared_pool_vector_blocks_admission {n : ℕ}
    (occupied target release : Fin n → ℝ) (inflow gate : ℝ)
    (hfull : ∑ j, occupied j = 1) (i : Fin n) :
    sharedContactPoolVectorStep occupied target release inflow gate i =
      (1 - gate * release i) * occupied i := by
  unfold sharedContactPoolVectorStep
  rw [hfull]
  ring

theorem shared_pool_vector_closed_gate_holds {n : ℕ}
    (occupied target release : Fin n → ℝ) (inflow : ℝ) :
    sharedContactPoolVectorStep occupied target release inflow 0 = occupied := by
  funext i
  simp [sharedContactPoolVectorStep]

/-- Endpoints applying the same inputs to the same shared state cannot write
inconsistent copies of that edge's reservoir. -/
theorem shared_contact_pool_vector_endpoint_agreement {n : ℕ}
    (left right target release : Fin n → ℝ) (inflow gate : ℝ) (hstate : left = right) :
    sharedContactPoolVectorStep left target release inflow gate =
      sharedContactPoolVectorStep right target release inflow gate := by
  rw [hstate]

/-- Both endpoints use this one symmetric response gate. -/
noncomputable def sharedEdgeResponseGate (left right : ℝ) : ℝ := max left right

theorem shared_edge_response_gate_symmetric (left right : ℝ) :
    sharedEdgeResponseGate left right = sharedEdgeResponseGate right left :=
  max_comm left right

theorem shared_edge_response_gate_bounds (left right : ℝ)
    (hleft : 0 ≤ left ∧ left ≤ 1) (hright : 0 ≤ right ∧ right ≤ 1) :
    0 ≤ sharedEdgeResponseGate left right ∧ sharedEdgeResponseGate left right ≤ 1 :=
  ⟨hleft.1.trans (le_max_left left right), max_le hleft.2 hright.2⟩

theorem shared_edge_response_gate_closed : sharedEdgeResponseGate 0 0 = 0 := by
  simp [sharedEdgeResponseGate]

theorem shared_edge_response_gate_one (right : ℝ) (hright : right ≤ 1) :
    sharedEdgeResponseGate 1 right = 1 := max_eq_left hright

end CGB.Generation
