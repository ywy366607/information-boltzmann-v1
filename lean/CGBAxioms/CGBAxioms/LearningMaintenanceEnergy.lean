import CGBAxioms.LearningMaintenance

/-!
# Declared fuel, actual consumption and paid learning

The same registered normal/cut event process pays maintenance fuel at active
steps and an additional write-fuel unit on the effective feedback update.
Food supplies one unit per enabled resource channel; scarcity supplies none.
Heat is this specified consumption, not a residual fitted to the first law.
The isothermal interpretation below declares T=1 and zero internal entropy;
it is an accounting-consistent carrier model, not a microscopic bath calibration.
-/

open Finset

namespace CGB.LearningMaintenanceEnergy

open LearningMaintenance Energetics MaterialSource

abbrev S := (Unit → State) × world.W

noncomputable def process (cut : Bool) : Interaction.Machine S ℕ (Option Unit) :=
  MaterialSource.machine (body.jointEventKernel world he cut (fun _ => world.sense))

noncomputable def initial : S := (body.init, world.start)

/-- The energy law is exactly the state law used in the viability comparison,
including the cut's own distribution-dependent future, not a second process. -/
theorem actual_law (cut : Bool) (n : ℕ) (z : S) :
    law (process cut) initial n z = if cut then
      body.lawCut world (fun _ => world.sense) n z.1 z.2 else
      body.law world (fun _ => world.sense) n z.1 z.2 := by
  have hi : (fun z : S => if z = initial then (1 : ℝ) else 0) =
      (fun z : S => if z.1 = body.init ∧ z.2 = world.start then 1 else 0) := by
    funext z
    simp [initial, Prod.ext_iff]
  change EventChannels.Kernel.stateLaw
    (body.jointEventKernel world he cut (fun _ => world.sense))
    (fun z : S => if z = initial then 1 else 0) n z = _
  rw [hi]
  exact body.jointEventKernel_stateLaw world he cut (fun _ => world.sense) n z

noncomputable def port (f : Stim → State → ℕ) (_t : ℕ) (z : S) (e : Option Unit) (_z' : S) : ℝ :=
  e.elim 0 (fun _ => (f (world.sense z.2 ()) (z.1 ()) : ℝ))

private theorem body_supported (cut : Bool) (t : ℕ) (z z' : S) (e : Option Unit)
    (hp : 0 < (process cut).step t z e z') :
    0 < body.eventStep world.clock (world.sense z.2) z.1 e z'.1 := by
  by_contra h
  have hw := body.eventWorldStep_nonneg world he cut (fun _ => world.sense) t z z'.2
  have hh := mul_nonpos_of_nonneg_of_nonpos hw (le_of_not_gt h)
  exact (not_lt_of_ge hh) hp

noncomputable def account (cut : Bool) : Account (process cut) where
  stored _ z := (cost (z.1 ()) : ℝ)
  drive _ _ _ _ := 0
  load _ _ _ _ := 0
  matter := port incoming
  heat t z e z' := -port outgoing t z e z'
  balance t z e z' hp := by
    have hb := body_supported cut t z z' e hp
    cases e with
    | none =>
      have hx : z.1 = z'.1 := by
        by_contra hn
        simp [Medium.eventStep, hn] at hb
      simp [port, ← hx]
    | some e =>
      have hx : body.act e (world.sense z.2) z.1 = z'.1 := by
        by_contra hn
        simp [Medium.eventStep, hn] at hb
      have hr : 0 < body.rate e (world.sense z.2) z.1 := by
        simpa only [Medium.eventStep, hx, ↓reduceIte, show world.clock = 1 from rfl,
          div_one] using hb
      have hbal := ledger.balance e (world.sense z.2) z.1 hr
      rw [hx] at hbal
      change cost (z'.1 ()) + outgoing (world.sense z.2 ()) (z.1 ()) =
        cost (z.1 ()) + incoming (world.sense z.2 ()) (z.1 ()) at hbal
      have hreal : (cost (z'.1 ()) : ℝ) + (outgoing (world.sense z.2 ()) (z.1 ()) : ℝ) =
          (cost (z.1 ()) : ℝ) + (incoming (world.sense z.2 ()) (z.1 ()) : ℝ) := by
        exact_mod_cast hbal
      simp only [port, Option.elim_some, add_zero]
      linarith

/-- Conditional mean second-law admission for the explicitly declared internal
entropy and bath units. Consumption, including the paid write, is fixed by outgoing material use. -/
theorem isothermal (cut : Bool) :
    Isothermal (process cut) (account cut) initial 1
      (fun _ _ => 0) (fun _ _ _ _ => 0) := by
  refine ⟨by norm_num, ?_⟩
  intro t
  have hq : flux (process cut) initial (account cut).heat t ≤ 0 := by
    apply Finset.sum_nonpos
    intro z _
    apply Finset.sum_nonpos
    intro e _
    apply Finset.sum_nonpos
    intro z' _
    apply mul_nonpos_of_nonneg_of_nonpos ((law_probability (process cut) initial t).1 z)
    apply mul_nonpos_of_nonneg_of_nonpos ((process cut).nonneg t z e z')
    change -port outgoing t z e z' ≤ 0
    cases e <;> simp [port]
  have hz : flux (process cut) initial (fun _ _ _ _ => 0) t = 0 := by simp [flux]
  simpa only [production, zero_sub, div_one, hz, sub_zero] using neg_nonneg.mpr hq

/-- Every finite prefix uses the SAME event law and tick. At T=1, zero declared
internal/carried entropy, imported energy pays actual consumption and reserve.
These imports are external energy supplies; no closed finite fuel source or
self-powered recycling is claimed for this learning witness. -/
theorem availability_budget (cut : Bool) (n : ℕ) :
    mean (law (process cut) initial n) ((account cut).stored n) +
        ∑ t ∈ range n, flux (process cut) initial (port outgoing) t =
      2 + ∑ t ∈ range n, flux (process cut) initial (port incoming) t := by
  have hb := (work_budget (process cut) (account cut) initial 1
    (fun _ _ => 0) (fun _ _ _ _ => 0) (isothermal cut) n).1
  have hp (t : ℕ) : production (process cut) (account cut) initial 1
      (fun _ _ => 0) (fun _ _ _ _ => 0) t =
        flux (process cut) initial (port outgoing) t := by
    have hn : flux (process cut) initial (account cut).heat t =
        -flux (process cut) initial (port outgoing) t := by
      simp only [flux, account, mul_neg, Finset.sum_neg_distrib]
    have hz : flux (process cut) initial (fun _ _ _ _ => 0) t = 0 := by simp [flux]
    simp only [production, zero_sub, div_one, hn, hz, sub_zero, neg_neg]
  have hs (t : ℕ) : suppliedAvailability (process cut) (account cut) initial 1
      (fun _ _ _ _ => 0) t = flux (process cut) initial (port incoming) t := by
    simp [suppliedAvailability, account, flux]
  have h0 : freeEnergy (process cut) (account cut) initial 1 (fun _ _ => 0) 0 = 2 := by
    simp only [freeEnergy, mul_zero, sub_zero]
    change (∑ z : S, (if z = initial then (1 : ℝ) else 0) * (cost (z.1 ()) : ℝ)) = 2
    rw [Finset.sum_eq_single initial]
    · norm_num [initial, body, plant, cost]
    · intro z _ hz
      simp [hz]
    · simp
  simp only [hp, hs, h0] at hb
  simpa [freeEnergy, account, flux] using hb

end CGB.LearningMaintenanceEnergy
