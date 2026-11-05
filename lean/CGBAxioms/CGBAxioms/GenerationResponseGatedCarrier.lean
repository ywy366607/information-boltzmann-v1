import CGBAxioms.GenerationContactCompetition

/-!
# Response-gated conserved contact carrier

`GenerationContactEligibility` establishes that a saturated local H contact has
zero forward response.  `GenerationContactCompetition` already supplies the
single finite S reservoir and its gate.  This file joins the two without
introducing a trace array: write the local response gate *into the existing S
exchange*.  A closed gate holds the complete typed contact bundle; if the open
exchange has nonzero net transfer, the two histories remain distinguishable in
S for a later reader.

This is a representation bridge, not a claim of exact arbitrary-horizon
parameter gradients.  The latter additionally needs a sufficient rank/readout
factorisation of the desired future credit.
-/

namespace CGB.Generation

open Finset

/-- A response gate closed at the source event leaves every coordinate of the
existing shared contact reservoir unchanged. -/
theorem response_gated_contact_closed_holds {n : ℕ}
    (occupied target : Fin n → ℝ) (inflow release : ℝ) :
    sharedContactPoolStep occupied target inflow release 0 = occupied :=
  shared_pool_closed_gate_holds occupied target inflow release

/-- If opening the existing shared exchange produces a nonzero local transfer,
the S coordinate distinguishes it from a saturated (closed-gate) event.  This
is exactly the missing condition for a delayed reader to retain the local
response distinction; no separate eligibility page is added. -/
theorem response_gated_contact_write_distinguishes {n : ℕ}
    (occupied target : Fin n → ℝ) (inflow release : ℝ) (i : Fin n)
    (htransfer : inflow * target i * (1 - ∑ j, occupied j) -
      release * occupied i ≠ 0) :
    sharedContactPoolStep occupied target inflow release 0 i ≠
      sharedContactPoolStep occupied target inflow release 1 i := by
  intro h
  unfold sharedContactPoolStep at h
  have : inflow * target i * (1 - ∑ j, occupied j) -
      release * occupied i = 0 := by
    linarith
  exact htransfer this

/-- Scalar retention of an existing coordinate, not a derivative or Jacobian. -/
noncomputable def contactRetention (gate release : ℝ) : ℝ := 1 - gate * release

/-- The actual shared-pool injection, including the current total occupation.
This is a transient expression, not a second state or a stored event history. -/
noncomputable def sharedContactInjection {n : ℕ}
    (occupied target : Fin n → ℝ) (inflow gate : ℝ) (i : Fin n) : ℝ :=
  gate * inflow * target i * (1 - ∑ j, occupied j)

/-- Exact connection of the retention/injection decomposition to the executable
coordinatewise-release shared reservoir. -/
theorem shared_contact_pool_vector_affine {n : ℕ}
    (occupied target release : Fin n → ℝ) (inflow gate : ℝ) (i : Fin n) :
    sharedContactPoolVectorStep occupied target release inflow gate i =
      contactRetention gate (release i) * occupied i +
        sharedContactInjection occupied target inflow gate i := by
  unfold sharedContactPoolVectorStep contactRetention sharedContactInjection
  ring

theorem contact_retention_bounds (gate release : ℝ)
    (hgate : 0 ≤ gate ∧ gate ≤ 1) (hrelease : 0 ≤ release ∧ release ≤ 1) :
    0 ≤ contactRetention gate release ∧ contactRetention gate release ≤ 1 := by
  have hprod : gate * release ≤ 1 :=
    (mul_le_mul hgate.2 hrelease.2 hrelease.1 (by norm_num)).trans_eq (one_mul 1)
  unfold contactRetention
  constructor
  · linarith
  · linarith [mul_nonneg hgate.1 hrelease.1]

/-- Algebraic forward expansion only. Public results below instantiate both
coefficients with the actual vector shared-pool transition. -/
private theorem affine_forward_history (state retention injection : ℕ → ℝ)
    (hstep : ∀ t, state (t + 1) = retention t * state t + injection t) (T : ℕ) :
    state T = state 0 * (∏ t ∈ range T, retention t) +
      ∑ k ∈ range T, injection k * (∏ t ∈ Ico (k + 1) T, retention t) := by
  induction T with
  | zero => simp
  | succ T ih =>
    rw [hstep T, ih, prod_range_succ, sum_range_succ]
    have htail :
        (∑ k ∈ range T, injection k * (∏ t ∈ Ico (k + 1) (T + 1), retention t)) =
          (∑ k ∈ range T, injection k * (∏ t ∈ Ico (k + 1) T, retention t)) *
            retention T := by
      rw [sum_mul]
      apply sum_congr rfl
      intro k hk
      rw [prod_Ico_succ_top (by have := mem_range.mp hk; omega)]
      ring
    rw [htail]
    simp only [Ico_self, prod_empty, mul_one]
    ring

/-- Every actual coordinate is its retained initial value plus its finitely
many injections, each weighted by subsequent retention. The formula does not
require the runtime to save or replay those injections. -/
theorem response_gated_vector_history_expansion {n : ℕ}
    (state target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ)
    (hstep : ∀ t, state (t + 1) =
      sharedContactPoolVectorStep (state t) (target t) (release t) (inflow t) (gate t))
    (T : ℕ) (i : Fin n) :
    state T i = state 0 i * (∏ t ∈ range T, contactRetention (gate t) (release t i)) +
      ∑ k ∈ range T, sharedContactInjection (state k) (target k) (inflow k) (gate k) i *
        (∏ t ∈ Ico (k + 1) T, contactRetention (gate t) (release t i)) := by
  apply affine_forward_history (fun t => state t i)
    (fun t => contactRetention (gate t) (release t i))
    (fun t => sharedContactInjection (state t) (target t) (inflow t) (gate t) i)
  intro t
  rw [hstep t, shared_contact_pool_vector_affine]

/-- With no new injection into this type during the interval, only the product
of its actual coordinatewise retention acts on its initial content. -/
theorem response_gated_vector_no_injection_retention {n : ℕ}
    (state target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ)
    (hstep : ∀ t, state (t + 1) =
      sharedContactPoolVectorStep (state t) (target t) (release t) (inflow t) (gate t))
    (T : ℕ) (i : Fin n)
    (hno : ∀ t < T, sharedContactInjection (state t) (target t) (inflow t) (gate t) i = 0) :
    state T i = state 0 i * (∏ t ∈ range T, contactRetention (gate t) (release t i)) := by
  rw [response_gated_vector_history_expansion state target release inflow gate hstep T i]
  have hsum : (∑ k ∈ range T,
      sharedContactInjection (state k) (target k) (inflow k) (gate k) i *
        (∏ t ∈ Ico (k + 1) T, contactRetention (gate t) (release t i))) = 0 := by
    apply sum_eq_zero
    intro k hk
    rw [hno k (mem_range.mp hk), zero_mul]
  rw [hsum, add_zero]

/-- A type absent at birth cannot appear without an injection of that type. -/
theorem response_gated_vector_zero_without_injection {n : ℕ}
    (state target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ)
    (hstep : ∀ t, state (t + 1) =
      sharedContactPoolVectorStep (state t) (target t) (release t) (inflow t) (gate t))
    (T : ℕ) (i : Fin n) (hzero : state 0 i = 0)
    (hno : ∀ t < T, sharedContactInjection (state t) (target t) (inflow t) (gate t) i = 0) :
    state T i = 0 := by
  rw [response_gated_vector_no_injection_retention
    state target release inflow gate hstep T i hno, hzero, zero_mul]

/-- Equal external inputs alone do not guarantee equal injections: the free
pool depends on each trajectory. This result explicitly requires equal actual
injections as well as the shared retention schedule. -/
theorem response_gated_vector_history_difference {n : ℕ}
    (left right target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ)
    (hleft : ∀ t, left (t + 1) =
      sharedContactPoolVectorStep (left t) (target t) (release t) (inflow t) (gate t))
    (hright : ∀ t, right (t + 1) =
      sharedContactPoolVectorStep (right t) (target t) (release t) (inflow t) (gate t))
    (T : ℕ) (i : Fin n)
    (hinjection : ∀ t < T,
      sharedContactInjection (left t) (target t) (inflow t) (gate t) i =
        sharedContactInjection (right t) (target t) (inflow t) (gate t) i) :
    left T i - right T i = (left 0 i - right 0 i) *
      (∏ t ∈ range T, contactRetention (gate t) (release t i)) := by
  induction T with
  | zero => simp
  | succ T ih =>
    have hprev := ih (fun t ht => hinjection t (by omega))
    rw [hleft T, hright T, shared_contact_pool_vector_affine,
      shared_contact_pool_vector_affine, prod_range_succ]
    rw [hinjection T (by omega)]
    calc
      _ = (left T i - right T i) * contactRetention (gate T) (release T i) := by ring
      _ = _ := by rw [hprev]; ring

/-- Nonzero retained product preserves an initial scalar distinction, not a
claim that the complete old pre×gate vector can be decoded. -/
theorem response_gated_vector_nonzero_retention_distinguishes {n : ℕ}
    (left right target release : ℕ → Fin n → ℝ) (inflow gate : ℕ → ℝ)
    (hleft : ∀ t, left (t + 1) =
      sharedContactPoolVectorStep (left t) (target t) (release t) (inflow t) (gate t))
    (hright : ∀ t, right (t + 1) =
      sharedContactPoolVectorStep (right t) (target t) (release t) (inflow t) (gate t))
    (T : ℕ) (i : Fin n)
    (hinjection : ∀ t < T,
      sharedContactInjection (left t) (target t) (inflow t) (gate t) i =
        sharedContactInjection (right t) (target t) (inflow t) (gate t) i)
    (hinitial : left 0 i ≠ right 0 i)
    (hproduct : (∏ t ∈ range T, contactRetention (gate t) (release t i)) ≠ 0) :
    left T i ≠ right T i := by
  apply sub_ne_zero.mp
  rw [response_gated_vector_history_difference
    left right target release inflow gate hleft hright T i hinjection]
  exact mul_ne_zero (sub_ne_zero.mpr hinitial) hproduct

/-- Equal rank-release rates give a direct source-contrast recurrence inside
one and the same reservoir; the common free pool is retained in the formula. -/
theorem response_gated_vector_source_difference {n : ℕ}
    (occupied target release : Fin n → ℝ) (inflow gate : ℝ) (i j : Fin n)
    (hrelease : release i = release j) :
    sharedContactPoolVectorStep occupied target release inflow gate i -
        sharedContactPoolVectorStep occupied target release inflow gate j =
      contactRetention gate (release i) * (occupied i - occupied j) +
        gate * inflow * (target i - target j) * (1 - ∑ k, occupied k) := by
  unfold sharedContactPoolVectorStep contactRetention
  rw [hrelease]
  ring

theorem positive_demand_preserves_contact_distinction (left right demand : ℝ)
    (hdemand : 0 < demand) (hdifference : left ≠ right) :
    demand * left ≠ demand * right := by
  intro h
  exact hdifference (mul_left_cancel₀ (ne_of_gt hdemand) h)

end CGB.Generation
