import CGBAxioms.GenerationColumnMemory

/-!
# Local contact eligibility is a forward response, not occupancy alone

For a column contact `w` with local participation `s`, the next activity is
`hardSat (base + w * s)`.  This small algebraic slice fixes the minimum
one-event eligibility factor before a language teaching signal is attached.

* In an unsaturated activity regime, a contact perturbation changes activity
  by exactly `δ * s`.
* When the activity is saturated, the same perturbation has zero local effect
  as long as it remains on that face.

Thus an implementation may not call `S * demand` alone the contact
eligibility: it must also read the score-side route and the local response
gate.  The theorem does not prescribe a new trace array or a feedback matrix.
-/

namespace CGB.Generation

/-- One scalar coordinate of the existing saturated S-to-H contact read. -/
noncomputable def contactResponse (base weight participation : ℝ) : ℝ :=
  hardSat (base + weight * participation)

theorem hardSat_eq_self {x : ℝ} (hlow : -1 ≤ x) (hhigh : x ≤ 1) :
    hardSat x = x := by
  unfold hardSat
  rw [min_eq_left hhigh, max_eq_left hlow]

theorem hardSat_eq_one {x : ℝ} (h : 1 ≤ x) : hardSat x = 1 := by
  unfold hardSat
  rw [min_eq_right h]
  norm_num

/-- On one unsaturated response segment, participation is the exact local
forward finite-difference factor of a contact-weight perturbation. -/
theorem contact_response_unsaturated_increment
    (base weight participation δ : ℝ)
    (hzero_low : -1 ≤ base + weight * participation)
    (hzero_high : base + weight * participation ≤ 1)
    (hdelta_low : -1 ≤ base + (weight + δ) * participation)
    (hdelta_high : base + (weight + δ) * participation ≤ 1) :
    contactResponse base (weight + δ) participation -
        contactResponse base weight participation = δ * participation := by
  unfold contactResponse
  rw [hardSat_eq_self hdelta_low hdelta_high,
    hardSat_eq_self hzero_low hzero_high]
  ring

/-- A bounded activity face carries no additional local contact evidence:
as long as both responses remain saturated at `+1`, their difference is zero. -/
theorem contact_response_saturated_increment_zero
    (base weight participation δ : ℝ)
    (hzero : 1 ≤ base + weight * participation)
    (hdelta : 1 ≤ base + (weight + δ) * participation) :
    contactResponse base (weight + δ) participation -
        contactResponse base weight participation = 0 := by
  unfold contactResponse
  rw [hardSat_eq_one hdelta, hardSat_eq_one hzero]
  ring

/-- A score whose local read coefficient is `q` inherits both cases.  This is
the one-contact part of the next-byte forward eligibility before its language
innovation is used as the third factor. -/
theorem contact_score_unsaturated_increment
    (q base weight participation δ : ℝ)
    (hzero_low : -1 ≤ base + weight * participation)
    (hzero_high : base + weight * participation ≤ 1)
    (hdelta_low : -1 ≤ base + (weight + δ) * participation)
    (hdelta_high : base + (weight + δ) * participation ≤ 1) :
    q * contactResponse base (weight + δ) participation -
        q * contactResponse base weight participation = q * δ * participation := by
  rw [show q * contactResponse base (weight + δ) participation -
      q * contactResponse base weight participation =
      q * (contactResponse base (weight + δ) participation -
        contactResponse base weight participation) by ring,
    contact_response_unsaturated_increment base weight participation δ
      hzero_low hzero_high hdelta_low hdelta_high]
  ring

end CGB.Generation
