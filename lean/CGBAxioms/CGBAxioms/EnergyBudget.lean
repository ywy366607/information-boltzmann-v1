import CGBAxioms.MaterialSource

/-!
# Energy balance and conditional mean availability budget

Physical energy, work/heat ports, temperature and entropy require a carrier
interpretation. The six general medium predicates do not identify them. First
law is checked on supported events. The isothermal second-law condition is on
mean entropy production, not on each stochastic transition or trajectory.
-/

open Finset

namespace CGB.Energetics

variable {S O : Type*} [Fintype S] [DecidableEq S] [Fintype O]

/-- The existing finite-state law, using the same tick and process. -/
noncomputable def events (P : Interaction.Machine S ℕ O) (t : ℕ) :
    EventChannels.Kernel S O where
  step := P.step t
  nonneg := P.nonneg t
  sum_one := P.sum_one t

noncomputable abbrev law (P : Interaction.Machine S ℕ O) (initial : S) (t : ℕ) : S → ℝ :=
  EventChannels.Kernel.stateLaw (events P) (fun s => if s = initial then 1 else 0) t

theorem law_probability (P : Interaction.Machine S ℕ O) (initial : S) (t : ℕ) :
    (∀ s, 0 ≤ law P initial t s) ∧ (∑ s, law P initial t s) = 1 := by
  induction t with
  | zero =>
    constructor
    · intro s
      simp only [law, EventChannels.Kernel.stateLaw]
      split_ifs <;> norm_num
    · simp [law, EventChannels.Kernel.stateLaw]
  | succ t ih =>
    constructor
    · intro s'
      exact Finset.sum_nonneg fun s _ => mul_nonneg (ih.1 s)
        ((events P t).forget.nonneg s s')
    · simp only [law, EventChannels.Kernel.stateLaw]
      rw [Finset.sum_comm]
      simp only [← Finset.mul_sum, (events P t).forget.sum_one, mul_one]
      exact ih.2

noncomputable abbrev mean (ρ : S → ℝ) (f : S → ℝ) : ℝ := ∑ s, ρ s * f s

noncomputable def flux (P : Interaction.Machine S ℕ O) (initial : S)
    (j : ℕ → S → O → S → ℝ) (t : ℕ) : ℝ :=
  ∑ s, ∑ o, ∑ s', law P initial t s * (P.step t s o s' * j t s o s')

/-- Signed ports: drive is work into the system; load is work delivered out;
matter and heat are energy into the system. They may fluctuate in sign. -/
structure Account (P : Interaction.Machine S ℕ O) where
  stored : ℕ → S → ℝ
  drive : ℕ → S → O → S → ℝ
  load : ℕ → S → O → S → ℝ
  matter : ℕ → S → O → S → ℝ
  heat : ℕ → S → O → S → ℝ
  balance : ∀ t s o s', 0 < P.step t s o s' →
    stored (t + 1) s' + load t s o s' =
      stored t s + drive t s o s' + matter t s o s' + heat t s o s'

/-- Average the supported first law along the ORIGINAL process. -/
theorem expected_first_law (P : Interaction.Machine S ℕ O) (a : Account P)
    (initial : S) (t : ℕ) :
    mean (law P initial (t + 1)) (a.stored (t + 1)) + flux P initial a.load t =
      mean (law P initial t) (a.stored t) + flux P initial a.drive t +
        flux P initial a.matter t + flux P initial a.heat t := by
  have hn : mean (law P initial (t + 1)) (a.stored (t + 1)) =
      ∑ s, ∑ o, ∑ s', law P initial t s * (P.step t s o s' * a.stored (t + 1) s') := by
    simp only [law, EventChannels.Kernel.stateLaw, EventChannels.Kernel.forget, events, mean]
    simp only [Finset.sum_mul, mul_assoc, Finset.mul_sum]
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro s _
    rw [Finset.sum_comm]
  have ho : mean (law P initial t) (a.stored t) =
      ∑ s, ∑ o, ∑ s', law P initial t s * (P.step t s o s' * a.stored t s) := by
    simp only [← Finset.mul_sum, ← Finset.sum_mul, P.sum_one, one_mul]
  rw [hn, ho]
  simp only [flux, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro s _
  apply Finset.sum_congr rfl
  intro o _
  apply Finset.sum_congr rfl
  intro s' _
  rcases (P.nonneg t s o s').lt_or_eq with hp | hz
  · have hb := a.balance t s o s' hp
    linear_combination (law P initial t s * P.step t s o s') * hb
  · simp [hz.symm]

noncomputable def freeEnergy (P : Interaction.Machine S ℕ O) (a : Account P)
    (initial : S) (T : ℝ) (entropy : ℕ → (S → ℝ) → ℝ) (t : ℕ) : ℝ :=
  mean (law P initial t) (a.stored t) - T * entropy t (law P initial t)

/-- Entropy units and all non-thermal entropy transport must be calibrated.
This entropy is not predictive surprisal or a free choice of -log π. -/
noncomputable def production (P : Interaction.Machine S ℕ O) (a : Account P)
    (initial : S) (T : ℝ) (entropy : ℕ → (S → ℝ) → ℝ)
    (carriedEntropy : ℕ → S → O → S → ℝ) (t : ℕ) : ℝ :=
  entropy (t + 1) (law P initial (t + 1)) - entropy t (law P initial t) -
    flux P initial a.heat t / T - flux P initial carriedEntropy t

noncomputable def suppliedAvailability (P : Interaction.Machine S ℕ O) (a : Account P)
    (initial : S) (T : ℝ) (carriedEntropy : ℕ → S → O → S → ℝ) (t : ℕ) : ℝ :=
  flux P initial a.drive t + flux P initial a.matter t - T * flux P initial carriedEntropy t

/-- Conditional isothermal interpretation along this actual ensemble: one
fixed positive bath temperature, ideal work ports, complete boundary entropy,
and nonnegative MEAN production. Not derived from the general six axioms. -/
def Isothermal (P : Interaction.Machine S ℕ O) (a : Account P) (initial : S)
    (T : ℝ) (entropy : ℕ → (S → ℝ) → ℝ) (carriedEntropy : ℕ → S → O → S → ℝ) : Prop :=
  0 < T ∧ ∀ t, 0 ≤ production P a initial T entropy carriedEntropy t

theorem availability_balance (P : Interaction.Machine S ℕ O) (a : Account P)
    (initial : S) (T : ℝ) (entropy : ℕ → (S → ℝ) → ℝ)
    (carriedEntropy : ℕ → S → O → S → ℝ) (hT : 0 < T) (t : ℕ) :
    freeEnergy P a initial T entropy (t + 1) - freeEnergy P a initial T entropy t +
        flux P initial a.load t + T * production P a initial T entropy carriedEntropy t =
      suppliedAvailability P a initial T carriedEntropy t := by
  have hb := expected_first_law P a initial t
  unfold freeEnergy production suppliedAvailability
  have ht : T * (flux P initial a.heat t / T) = flux P initial a.heat t := by
    field_simp
  nlinarith

/-- One finite-prefix identity gives both the bounded-reserve work budget and
the cyclic bound. A lower free-energy reserve and mean second law are explicit
premises; no uniform per-tick loss or universal death condition is added. -/
theorem work_budget (P : Interaction.Machine S ℕ O) (a : Account P)
    (initial : S) (T : ℝ) (entropy : ℕ → (S → ℝ) → ℝ)
    (carriedEntropy : ℕ → S → O → S → ℝ)
    (h : Isothermal P a initial T entropy carriedEntropy) (n : ℕ) :
    freeEnergy P a initial T entropy n + (∑ t ∈ range n, flux P initial a.load t) +
        T * ∑ t ∈ range n, production P a initial T entropy carriedEntropy t =
      freeEnergy P a initial T entropy 0 +
        ∑ t ∈ range n, suppliedAvailability P a initial T carriedEntropy t ∧
    (∀ lower, lower ≤ freeEnergy P a initial T entropy n →
      (∑ t ∈ range n, flux P initial a.load t) ≤
        freeEnergy P a initial T entropy 0 - lower +
          ∑ t ∈ range n, suppliedAvailability P a initial T carriedEntropy t) ∧
    (freeEnergy P a initial T entropy n = freeEnergy P a initial T entropy 0 →
      (∑ t ∈ range n, flux P initial a.load t) ≤
        ∑ t ∈ range n, suppliedAvailability P a initial T carriedEntropy t) := by
  have he : ∀ k, freeEnergy P a initial T entropy k + (∑ t ∈ range k, flux P initial a.load t) +
      T * ∑ t ∈ range k, production P a initial T entropy carriedEntropy t =
        freeEnergy P a initial T entropy 0 +
          ∑ t ∈ range k, suppliedAvailability P a initial T carriedEntropy t := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      have hb := availability_balance P a initial T entropy carriedEntropy h.1 k
      simp only [sum_range_succ]
      linear_combination ih + hb
  have hd : 0 ≤ T * ∑ t ∈ range n, production P a initial T entropy carriedEntropy t :=
    mul_nonneg h.1.le (Finset.sum_nonneg fun t _ => h.2 t)
  refine ⟨he n, ?_, ?_⟩
  · intro lower hl; linarith [he n]
  · intro hc; linarith [he n]

/-- A positive mean availability-use requirement (delivered work plus T times
production) cannot persist on bounded total supply and reserve. This requirement
is an explicit property of
the mechanism, not a newly imposed maintenance cost for all living systems. -/
theorem no_sustained_use_on_finite_supply (P : Interaction.Machine S ℕ O) (a : Account P)
    (initial : S) (T : ℝ) (entropy : ℕ → (S → ℝ) → ℝ)
    (carriedEntropy : ℕ → S → O → S → ℝ)
    (h : Isothermal P a initial T entropy carriedEntropy)
    (lower supplyBound ε : ℝ) (hε : 0 < ε)
    (hlower : ∀ n, lower ≤ freeEnergy P a initial T entropy n)
    (hsupply : ∀ n, (∑ t ∈ range n,
      suppliedAvailability P a initial T carriedEntropy t) ≤ supplyBound) :
    ¬ ∀ n : ℕ, (n : ℝ) * ε ≤ ∑ t ∈ range n,
      (flux P initial a.load t + T * production P a initial T entropy carriedEntropy t) := by
  intro hout
  obtain ⟨n, hn⟩ := exists_nat_gt
    ((freeEnergy P a initial T entropy 0 - lower + supplyBound) / ε)
  have hn' := (div_lt_iff₀ hε).mp hn
  have hb := (work_budget P a initial T entropy carriedEntropy h n).1
  have ho := hout n
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum] at ho
  linarith [hlower n, hsupply n]

end CGB.Energetics
