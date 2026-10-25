import CGBAxioms.SensoryMaintenanceRecovery

/-! Finite blackout loss and actual-event energy accounting. -/
open Finset

namespace CGB.SensoryMaintenanceLimits
open RenewableLearning (Env Cell Signal buffer credit)
open SensoryMaintenance Energetics

private theorem support_closed {X E : Type*} [Fintype X] [DecidableEq X] [Fintype E]
    (K : ℕ → EventChannels.Kernel X E) (x₀ : X) (G : ℕ → X → Prop)
    (hi : G 0 x₀) (hc : ∀ t x e y, G t x → 0 < (K t).step x e y → G (t + 1) y)
    (n : ℕ) (x : X) (hx : ¬ G n x) :
    EventChannels.Kernel.stateLaw K (fun x => if x = x₀ then 1 else 0) n x = 0 := by
  classical
  induction n generalizing x with
  | zero =>
    have hn : x ≠ x₀ := by intro h; subst x; exact hx hi
    simp [EventChannels.Kernel.stateLaw, hn]
  | succ n ih =>
    change (∑ y, EventChannels.Kernel.stateLaw K (fun x => if x = x₀ then 1 else 0) n y *
      (∑ e, (K n).step y e x)) = 0
    apply Finset.sum_eq_zero
    intro y _
    by_cases hy : G n y
    · have hz (e : E) : (K n).step y e x = 0 := by
        by_cases hp : 0 < (K n).step y e x
        · exact False.elim (hx (hc n y e x hy hp))
        · exact le_antisymm (le_of_not_gt hp) ((K n).nonneg y e x)
      simp [hz]
    · rw [ih y hy, zero_mul]

private def lossRegion (n₀ t : ℕ) (z : SensoryMaintenance.S) : Prop :=
  t < n₀ ∨ ∃ x : State, z.1 () = burn x (t - n₀)

private theorem loss_closed (n₀ T : ℕ) (hT : 3 ≤ T) (t : ℕ)
    (z z' : SensoryMaintenance.S) (e : Option Unit) (hg : lossRegion n₀ t z)
    (hp : 0 < (process false (sensoryProtocol n₀ T)).step t z e z') :
    lossRegion n₀ (t + 1) z' := by
  have hx := actual_transition false (sensoryProtocol n₀ T) t z z' e hp
  by_cases ht : t < n₀
  · by_cases ht' : t + 1 < n₀
    · exact Or.inl ht'
    · right
      have he : t + 1 = n₀ := by omega
      exact ⟨z'.1 (), by simp [he, burn]⟩
  · obtain ⟨x,hx₀⟩ := hg.resolve_left ht
    right
    refine ⟨x, ?_⟩
    by_cases hk : t - n₀ < 3
    · have hs : sensoryProtocol n₀ T t z.2 () = none := by
        have h : n₀ ≤ t ∧ t < n₀ + T := by omega
        simp only [sensoryProtocol, if_pos h]
      rw [hs, hx₀] at hx
      rw [show t + 1 - n₀ = (t - n₀) + 1 by omega]
      simpa only [burn, Function.iterate_succ_apply'] using hx
    · rw [burn_all_later x _ (by omega)] at hx₀
      rw [hx₀, next_empty] at hx
      rw [hx, burn_all_later x _ (by omega)]

/-- After three consecutive absent-sensing ticks, every supported body is empty.
The world still advances at clock one throughout the interval. -/
theorem blackout_body_empty (n₀ T n : ℕ) (hT : 3 ≤ T) (hn : n₀ + 3 ≤ n)
    (z : SensoryMaintenance.S) (hp : 0 < body.law world (sensoryProtocol n₀ T) n z.1 z.2) :
    z.1 () = empty := by
  have hi : lossRegion n₀ 0 initial := by
    by_cases h : 0 < n₀
    · exact Or.inl h
    · right; refine ⟨body.init (), ?_⟩; simp [initial, burn, show n₀ = 0 by omega]
  have hg : lossRegion n₀ n z := by
    by_contra hh
    have hz := support_closed (body.jointEventKernel world ec false (sensoryProtocol n₀ T))
      initial (lossRegion n₀) hi (fun t z e z' => loss_closed n₀ T hT t z z' e) n z hh
    change Energetics.law (process false (sensoryProtocol n₀ T)) initial n z = 0 at hz
    rw [actual_law] at hz
    norm_num at hz
    linarith
  obtain ⟨x,hx⟩ := hg.resolve_left (by omega)
  rw [hx, burn_all_later x _ (by omega)]

theorem blackout_survival_zero (n₀ T n : ℕ) (hT : 3 ≤ T) (hn : n₀ + 3 ≤ n) :
    Exchange.survival body world ec material false (sensoryProtocol n₀ T) n = 0 := by
  change (∑ x, ∑ w, Exchange.alive body world ec material false (sensoryProtocol n₀ T)
    n (x,w) * body.law world (sensoryProtocol n₀ T) n x w) = 0
  apply Finset.sum_eq_zero; intro x _
  apply Finset.sum_eq_zero; intro w _
  by_cases hp : 0 < body.law world (sensoryProtocol n₀ T) n x w
  · have he := blackout_body_empty n₀ T n hT hn (x,w) hp
    have hx : x = fun _ => empty := by funext u; cases u; exact he
    rw [hx]
    simp [Exchange.alive, Viability.alive, body, Medium.total, SensoryMaintenance.cost,
      empty, RenewableLearning.cost, buffer, credit]
  · have hz : body.law world (sensoryProtocol n₀ T) n x w = 0 :=
      le_antisymm (le_of_not_gt hp)
        (body.law_nonneg_of_clocked world (body.eventClocked_implies_clocked ec) _ n x w)
    rw [hz, mul_zero]

theorem long_loss_not_tolerated (T : ℕ) (hT : 3 ≤ T) :
    ¬ SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 T := by
  intro h
  have hb := h.2 0 3
  change 1 - (0 : ℝ) ≤ Exchange.survival body world ec material false (sensoryProtocol 0 T) 3 at hb
  rw [blackout_survival_zero 0 T 3 hT (by omega)] at hb
  norm_num at hb

/-- A declared positive loss at every active tick yields a finite duration
bound from the actual first law. Clock advance alone is not the premise. -/
theorem no_input_active_duration (stim : ℕ → Stim) (p : ℕ → State)
    (hp : ∀ t, p (t + 1) = next (stim t) (p t)) (n : ℕ)
    (ha : ∀ t < n, stim t ≠ some .calm ∧ 0 < cost (p t))
    (hi : ∀ t < n, incoming (stim t) (p t) = 0) : n ≤ cost (p 0) := by
  have hb := path_budget stim p hp n
  have hz : (∑ t ∈ range n, incoming (stim t) (p t)) = 0 := by
    apply Finset.sum_eq_zero; intro t ht; exact hi t (Finset.mem_range.mp ht)
  have hl : n ≤ ∑ t ∈ range n, outgoing (stim t) (p t) := by
    calc
      n = ∑ _t ∈ range n, 1 := by simp
      _ ≤ _ := by
        apply Finset.sum_le_sum; intro t ht
        have h := ha t (Finset.mem_range.mp ht)
        exact active_loss (stim t) (p t) h.1 h.2
  omega

noncomputable def port (sched : ℕ → Env → Unit → Stim) (f : Stim → State → ℕ)
    (t : ℕ) (z : SensoryMaintenance.S) (e : Option Unit) (_z' : SensoryMaintenance.S) : ℝ :=
  e.elim 0 (fun _ => (f (sched t z.2 ()) (z.1 ()) : ℝ))

noncomputable def fuelAccount (cut : Bool) (sched : ℕ → Env → Unit → Stim) :
    Account (process cut sched) where
  stored _ z := (SensoryMaintenance.cost (z.1 ()) : ℝ)
  drive _ _ _ _ := 0
  load _ _ _ _ := 0
  matter := port sched SensoryMaintenance.incoming
  heat t z e z' := -port sched SensoryMaintenance.outgoing t z e z'
  balance t z e z' hp := by
    have hx := actual_transition cut sched t z z' e hp
    cases e with
    | none =>
      have hb : 0 < body.eventStep 1 (sched t z.2) z.1 none z'.1 := by
        by_contra hn
        exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos
          (body.eventWorldStep_nonneg world ec cut sched t z z'.2) (le_of_not_gt hn))) hp
      have he : z.1 = z'.1 := by by_contra hn; simp [Medium.eventStep, hn] at hb
      simp [port, ← he]
    | some u =>
      have h := (local_budgets (sched t z.2 ()) (z.1 ())).1
      rw [← hx] at h
      have hr : (SensoryMaintenance.cost (z'.1 ()) : ℝ) +
          (SensoryMaintenance.outgoing (sched t z.2 ()) (z.1 ()) : ℝ) =
        (SensoryMaintenance.cost (z.1 ()) : ℝ) +
          (SensoryMaintenance.incoming (sched t z.2 ()) (z.1 ()) : ℝ) := by exact_mod_cast h
      simp only [port, Option.elim_some, add_zero]
      linarith

theorem fuel_budget (cut : Bool) (sched : ℕ → Env → Unit → Stim) (n : ℕ) :
    mean (law (process cut sched) initial n) ((fuelAccount cut sched).stored n) +
      ∑ t ∈ range n, flux (process cut sched) initial (port sched SensoryMaintenance.outgoing) t =
    4 + ∑ t ∈ range n,
      flux (process cut sched) initial (port sched SensoryMaintenance.incoming) t := by
  induction n with
  | zero =>
    norm_num [mean, law, EventChannels.Kernel.stateLaw, fuelAccount, initial, body,
      SensoryMaintenance.cost, RenewableLearning.cost, buffer, credit]
  | succ n ih =>
    have hb := expected_first_law (process cut sched) (fuelAccount cut sched) initial n
    have hz : flux (process cut sched) initial (fun _ _ _ _ => 0) n = 0 := by simp [flux]
    have hh : flux (process cut sched) initial
        (fun t z e z' => -port sched SensoryMaintenance.outgoing t z e z') n =
      -flux (process cut sched) initial (port sched SensoryMaintenance.outgoing) n := by
      simp only [flux, mul_neg, Finset.sum_neg_distrib]
    change _ + flux (process cut sched) initial (fun _ _ _ _ => 0) n =
      _ + flux (process cut sched) initial (fun _ _ _ _ => 0) n +
        flux (process cut sched) initial (port sched SensoryMaintenance.incoming) n +
        flux (process cut sched) initial
          (fun t z e z' => -port sched SensoryMaintenance.outgoing t z e z') n at hb
    rw [hz, hh] at hb
    simp only [Finset.sum_range_succ]
    linarith

end CGB.SensoryMaintenanceLimits
