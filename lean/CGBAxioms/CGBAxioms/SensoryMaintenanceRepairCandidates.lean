import CGBAxioms.SensoryMaintenanceRepairAudit

/-!
# Minimal repair candidates after the two-tick refutation

The registered model is unchanged.  This module records two candidate repair
interfaces: an endogenous probe on the second lost tick, and a guarded
reachable-region contract.  Both keep the existing local budget theorem as a
required field; neither silently creates material.
-/

namespace CGB.SensoryMaintenanceRepairCandidates

open RenewableLearning

private theorem policy_table (q : Cell) : policy q = match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by cases q <;> decide


def protectedTwoTickProtocol (n₀ : ℕ) : ℕ → Env → Unit → SensoryMaintenance.Stim :=
  fun t w _ => if t = n₀ then none else
    if t = n₀ + 1 then some .probe else some (sense w ())

theorem protected_second_tick (n₀ : ℕ) (w : Env) (u : Unit) :
    protectedTwoTickProtocol n₀ (n₀ + 1) w u = some .probe := by
  simp [protectedTwoTickProtocol]

theorem protected_first_tick (n₀ : ℕ) (w : Env) (u : Unit) :
    protectedTwoTickProtocol n₀ n₀ w u = none := by
  simp [protectedTwoTickProtocol]

theorem protocol_repair_preserves_one_tick :
    SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1 :=
  SensoryMaintenanceRecovery.one_tick_tolerance

theorem probe_recharges_after_one_loss (x : SensoryMaintenance.State)
    (hq : x.1 ≠ .vacuum) (hr : 1 ≤ x.2.val) :
    0 < SensoryMaintenance.incoming (some .probe)
        (SensoryMaintenance.next none x) ∧
    (SensoryMaintenance.next (some .probe)
        (SensoryMaintenance.next none x)).1 ≠ .vacuum ∧
    (SensoryMaintenance.next (some .probe)
        (SensoryMaintenance.next none x)).2.val = 2 := by
  rcases x with ⟨q, r⟩
  fin_cases r <;> cases q <;>
    try norm_num [SensoryMaintenance.next, SensoryMaintenance.incoming,
      SensoryMaintenance.empty, RenewableLearning.next,
      RenewableLearning.incoming] at hq hr ⊢
  all_goals simp_all [SensoryMaintenance.next, SensoryMaintenance.empty]

theorem protected_probe_path_outgoing (x : SensoryMaintenance.State)
    (hq : x.1 ≠ .vacuum) (hr : 1 ≤ x.2.val) :
    SensoryMaintenance.outgoing none x +
        SensoryMaintenance.outgoing (some .probe)
          (SensoryMaintenance.next none x) = 2 := by
  rcases x with ⟨q, r⟩
  fin_cases r <;> cases q <;>
    try norm_num [SensoryMaintenance.next, SensoryMaintenance.outgoing,
      SensoryMaintenance.empty, RenewableLearning.outgoing,
      RenewableLearning.next] at hq hr ⊢
  all_goals simp_all [SensoryMaintenance.next, SensoryMaintenance.outgoing,
    SensoryMaintenance.empty, RenewableLearning.outgoing, RenewableLearning.next]

theorem protected_probe_path_incoming (x : SensoryMaintenance.State)
    (hq : x.1 ≠ .vacuum) (hr : 2 ≤ x.2.val) :
    SensoryMaintenance.incoming (some .probe)
        (SensoryMaintenance.next none x) = 2 := by
  rcases x with ⟨q, r⟩
  fin_cases r <;> cases q <;>
    try norm_num [SensoryMaintenance.next, SensoryMaintenance.incoming,
      SensoryMaintenance.empty, RenewableLearning.next,
      RenewableLearning.incoming] at hq hr ⊢
  all_goals simp_all [SensoryMaintenance.next, SensoryMaintenance.incoming,
    SensoryMaintenance.empty, RenewableLearning.next, RenewableLearning.incoming]

theorem protected_branch_excludes_local_blackout
    (x : SensoryMaintenance.State) (hq : x.1 ≠ .vacuum)
    (hr : 1 ≤ x.2.val) :
    SensoryMaintenance.next (some .probe)
        (SensoryMaintenance.next none x) ≠ SensoryMaintenance.empty := by
  have hp := probe_recharges_after_one_loss x hq hr
  intro he
  have hv := congrArg Prod.fst he
  exact hp.2.1 hv

/-- A guarded start region for two lost ticks.  It excludes spent and held
states, and therefore excludes the currently verified spent-miss branch. -/
def eligibleTwoTickStart (x : SensoryMaintenance.State) : Prop :=
  x.1 = .search ∨ x.1 = .aReady ∨ x.1 = .bReady

def guardedTwoTickStart (q : Cell) (w : Env) : Prop :=
  eligibleTwoTickStart (q, 2) ∧
    ∀ v₁ v₂,
      0 < GraceRecovery.moves w (fun _ => q) v₁ →
      0 < GraceRecovery.moves v₁ (fun _ => q) v₂ →
      0 < RenewableLearning.incoming true (sense v₂ ()) q ∧
      (SensoryMaintenance.next (some (sense v₂ ()))
        (SensoryMaintenance.burn (q, 2) 2)).1 ≠ .vacuum

theorem initial_guarded_two_tick_start :
    guardedTwoTickStart .search .probeA := by
  refine ⟨by simp [eligibleTwoTickStart], ?_⟩
  intro v₁ v₂ h₁ h₂
  cases v₁ <;> cases v₂ <;>
    try norm_num [GraceRecovery.moves, RenewableLearning.moves, policy_table,
      SensoryMaintenance.burn, SensoryMaintenance.next,
      SensoryMaintenance.incoming, RenewableLearning.incoming,
      RenewableLearning.next, RenewableLearning.sense] at h₁ h₂ ⊢
  all_goals simp_all [GraceRecovery.moves, RenewableLearning.moves, policy_table,
    SensoryMaintenance.burn, SensoryMaintenance.next,
    SensoryMaintenance.incoming, RenewableLearning.incoming,
    RenewableLearning.next, RenewableLearning.sense]

theorem guarded_third_tick_recharges (q : Cell) (w : Env)
    (hg : guardedTwoTickStart q w) (v₁ v₂ : Env)
    (h₁ : 0 < GraceRecovery.moves w (fun _ => q) v₁)
    (h₂ : 0 < GraceRecovery.moves v₁ (fun _ => q) v₂) :
    0 < SensoryMaintenance.incoming (some (sense v₂ ()))
        (SensoryMaintenance.burn (q, 2) 2) ∧
    (SensoryMaintenance.next (some (sense v₂ ()))
        (SensoryMaintenance.burn (q, 2) 2)).2.val = 2 := by
  have hh := hg.2 v₁ v₂ h₁ h₂
  have hq : q ≠ .vacuum := by
    rcases hg.1 with rfl | rfl | rfl <;> decide
  have hb : SensoryMaintenance.burn (q, 2) 2 = (q, 0) := by
    exact (SensoryMaintenance.funded_two_tick_bridge q hq).2.1
  rw [hb] at hh ⊢
  exact SensoryMaintenanceBoundaries.sensed_recharge (sense v₂ ())
    (q, 0) hh.1 hq hh.2

/-- Two lost ticks followed by the natural third tick.  This is the concrete
guarded schedule used by the first-hit instantiation below; it does not add a
port or stop the physical clock. -/
noncomputable def guardedLossProtocol (n₀ : ℕ) : ℕ → Env → Unit → SensoryMaintenance.Stim :=
  fun t w _ => if t = n₀ ∨ t = n₀ + 1 then none else some (sense w ())

noncomputable def guardedLayerOne (q : Cell) (w : Env)
    (z : SensoryMaintenance.S) : Prop :=
  ∃ v, z = (fun _ => (q, 1), v) ∧ 0 < GraceRecovery.moves w (fun _ => q) v

noncomputable def guardedLayerTwo (q : Cell) (w : Env)
    (z : SensoryMaintenance.S) : Prop :=
  ∃ v₁ v₂, z = (fun _ => (q, 0), v₂) ∧
    0 < GraceRecovery.moves w (fun _ => q) v₁ ∧
    0 < GraceRecovery.moves v₁ (fun _ => q) v₂

theorem guarded_repair_preserves_one_tick :
    SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1 :=
  SensoryMaintenanceRecovery.one_tick_tolerance

theorem firstHit_three_tick_layered
    {S E : Type*} [Fintype S] [Fintype E]
    (K : ℕ → EventChannels.Kernel S E)
    (mark : ℕ → S → E → S → Bool) (t : ℕ) (s₀ : S)
    (layer₁ layer₂ : S → Prop)
    (h₀ : ∀ e s₁, 0 < (K t).step s₀ e s₁ →
      layer₁ s₁ ∧ mark t s₀ e s₁ = false)
    (h₁ : ∀ s₁, layer₁ s₁ → ∀ e s₂,
      0 < (K (t + 1)).step s₁ e s₂ →
      layer₂ s₂ ∧ mark (t + 1) s₁ e s₂ = false)
    (h₂ : ∀ s₁, layer₁ s₁ → ∀ s₂, layer₂ s₂ → ∀ e s₃,
      0 < (K (t + 2)).step s₂ e s₃ →
      mark (t + 2) s₂ e s₃ = true) :
    EventChannels.Kernel.firstHit K mark t 3 s₀ = 1 := by
  have third (s₁ : S) (hs₁ : layer₁ s₁) (s₂ : S) (hs₂ : layer₂ s₂) :
      EventChannels.Kernel.firstHit K mark (t + 2) 1 s₂ = 1 := by
    change (∑ e, ∑ s₃, (K (t + 2)).step s₂ e s₃ *
      (if mark (t + 2) s₂ e s₃ then 1 else
        EventChannels.Kernel.firstHit K mark (t + 3) 0 s₃)) = 1
    calc
      _ = ∑ e, ∑ s₃, (K (t + 2)).step s₂ e s₃ := by
        apply Finset.sum_congr rfl
        intro e _
        apply Finset.sum_congr rfl
        intro s₃ _
        rcases ((K (t + 2)).nonneg s₂ e s₃).lt_or_eq with hp | hz
        · rw [h₂ s₁ hs₁ s₂ hs₂ e s₃ hp]
          simp
        · simp [← hz]
      _ = 1 := (K (t + 2)).sum_one s₂
  have second (s₁ : S) (hs₁ : layer₁ s₁) :
      EventChannels.Kernel.firstHit K mark (t + 1) 2 s₁ = 1 := by
    change (∑ e, ∑ s₂, (K (t + 1)).step s₁ e s₂ *
      (if mark (t + 1) s₁ e s₂ then 1 else
        EventChannels.Kernel.firstHit K mark (t + 2) 1 s₂)) = 1
    calc
      _ = ∑ e, ∑ s₂, (K (t + 1)).step s₁ e s₂ := by
        apply Finset.sum_congr rfl
        intro e _
        apply Finset.sum_congr rfl
        intro s₂ _
        rcases ((K (t + 1)).nonneg s₁ e s₂).lt_or_eq with hp | hz
        · obtain ⟨hs₂, hm⟩ := h₁ s₁ hs₁ e s₂ hp
          rw [hm, third s₁ hs₁ s₂ hs₂]
          simp
        · simp [← hz]
      _ = 1 := (K (t + 1)).sum_one s₁
  change (∑ e, ∑ s₁, (K t).step s₀ e s₁ *
    (if mark t s₀ e s₁ then 1 else
      EventChannels.Kernel.firstHit K mark (t + 1) 2 s₁)) = 1
  calc
    _ = ∑ e, ∑ s₁, (K t).step s₀ e s₁ := by
      apply Finset.sum_congr rfl
      intro e _
      apply Finset.sum_congr rfl
      intro s₁ _
      rcases ((K t).nonneg s₀ e s₁).lt_or_eq with hp | hz
      · obtain ⟨hs₁, hm⟩ := h₀ e s₁ hp
        rw [hm, second s₁ hs₁]
        simp
      · simp [← hz]
    _ = 1 := (K t).sum_one s₀

theorem guarded_firstHit (n₀ : ℕ) (q : Cell) (w : Env)
    (hg : guardedTwoTickStart q w) :
    let sched := guardedLossProtocol n₀
    let K := fun t => SensoryMaintenance.body.jointEventKernel
      SensoryMaintenance.world SensoryMaintenance.ec false sched t
    let mark := Exchange.intake SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.material sched
    EventChannels.Kernel.firstHit K mark n₀ 3
      ((fun _ => (q, 2)), w) = 1 := by
  let sched := guardedLossProtocol n₀
  let K := fun t => SensoryMaintenance.body.jointEventKernel
      SensoryMaintenance.world SensoryMaintenance.ec false sched t
  let mark := Exchange.intake SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.material sched
  have hq : q ≠ .vacuum := by
    rcases hg.1 with rfl | rfl | rfl <;> decide
  apply firstHit_three_tick_layered K mark n₀ ((fun _ => (q, 2)), w)
    (guardedLayerOne q w) (guardedLayerTwo q w)
  · intro e s₁ hp
    rcases s₁ with ⟨x, v⟩
    have hw : 0 < GraceRecovery.moves w (fun _ => q) v := by
      by_contra hn
      have he : 0 ≤ SensoryMaintenance.body.eventStep 1
          (sched n₀ w) (fun _ => (q, 2)) e x :=
        SensoryMaintenance.body.eventStep_nonneg SensoryMaintenance.ec _ _ _ _
      exact (not_lt_of_ge
        (mul_nonpos_of_nonpos_of_nonneg (le_of_not_gt hn) he)) hp
    have hx := SensoryMaintenance.actual_transition false sched n₀
      ((fun _ => (q, 2)), w) (x, v) e hp
    refine ⟨?_, ?_⟩
    · refine ⟨v, ?_, hw⟩
      have hx0 : x () = (q, 1) := by
        simpa [sched, guardedLossProtocol, SensoryMaintenance.next, hq] using hx
      apply Prod.ext
      · funext u
        cases u
        exact hx0
      · rfl
    · cases e <;> simp [mark, Exchange.intake, sched,
        SensoryMaintenance.material, SensoryMaintenance.incoming,
        guardedLossProtocol]
  · intro s₁ hs₁ e s₂ hp
    rcases hs₁ with ⟨v₁, hs₁, hw₁⟩
    rcases s₁ with ⟨x₁, w₁⟩
    rcases s₂ with ⟨x₂, v₂⟩
    have hs₁x : x₁ = (fun _ => (q, 1)) := congrArg Prod.fst hs₁
    have hs₁w : w₁ = v₁ := congrArg Prod.snd hs₁
    have hp' := hp
    rw [hs₁x, hs₁w] at hp'
    have hw₂ : 0 < GraceRecovery.moves v₁ (fun _ => q) v₂ := by
      by_contra hn
      have he : 0 ≤ SensoryMaintenance.body.eventStep 1
          (sched (n₀ + 1) v₁) (fun _ => (q, 1)) e x₂ :=
        SensoryMaintenance.body.eventStep_nonneg SensoryMaintenance.ec _ _ _ _
      exact (not_lt_of_ge
        (mul_nonpos_of_nonpos_of_nonneg (le_of_not_gt hn) he)) hp'
    have hx := SensoryMaintenance.actual_transition false sched (n₀ + 1)
      (x₁, w₁) (x₂, v₂) e hp
    have hx₂ : x₂ () = (q, 0) := by
      simpa [sched, guardedLossProtocol, SensoryMaintenance.next,
        hs₁x, hs₁w, hq] using hx
    refine ⟨?_, ?_⟩
    · refine ⟨v₁, v₂, ?_, hw₁, hw₂⟩
      apply Prod.ext
      · funext u
        exact hx₂
      · rfl
    · cases e <;> simp [mark, Exchange.intake, sched,
        SensoryMaintenance.material, SensoryMaintenance.incoming,
        guardedLossProtocol]
  · intro s₁ hs₁ s₂ hs₂ e s₃ hp
    rcases hs₂ with ⟨v₁, v₂, hs₂, hw₁, hw₂⟩
    rcases s₂ with ⟨x₂, w₂⟩
    rcases s₃ with ⟨x₃, v₃⟩
    have hs₂x : x₂ = (fun _ => (q, 0)) := congrArg Prod.fst hs₂
    have hs₂w : w₂ = v₂ := congrArg Prod.snd hs₂
    have hp' := hp
    rw [hs₂x, hs₂w] at hp'
    have hi : 0 < RenewableLearning.incoming true (sense v₂ ()) q :=
      hg.2 v₁ v₂ hw₁ hw₂ |>.1
    cases e with
    | none =>
        have hz := hp'
        cases v₂ <;>
          norm_num [K, SensoryMaintenance.body, Medium.jointEventKernel,
            Medium.eventWorldStep, Medium.eventStep, sched,
            guardedLossProtocol, SensoryMaintenance.next,
            SensoryMaintenance.empty, RenewableLearning.incoming,
            RenewableLearning.sense, hq] at hz
    | some a =>
        simp [mark, Exchange.intake, SensoryMaintenance.material,
          SensoryMaintenance.incoming, sched, guardedLossProtocol,
          hs₂x, hs₂w]
        by_cases hc : sense v₂ () = .calm
        · simp [hc, RenewableLearning.incoming] at hi
        · simp [hc, hi, hq]

theorem guarded_alive (n₀ : ℕ) (q : Cell) (w : Env)
    (hg : guardedTwoTickStart q w) :
    Exchange.alive SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (guardedLossProtocol n₀) n₀ ((fun _ => (q, 2)), w) = 1 := by
  unfold Exchange.alive Exchange.recovery
  rw [guarded_firstHit n₀ q w hg]
  have hq : q ≠ .vacuum := by
    rcases hg.1 with rfl | rfl | rfl <;> decide
  simp [SensoryMaintenance.body, Medium.total, SensoryMaintenance.cost,
    RenewableLearning.cost, RenewableLearning.buffer, RenewableLearning.credit,
    SensoryMaintenance.empty, SensoryMaintenance.world]
  norm_num

def GuardedTwoTickLocalTolerance (δ : ℝ) : Prop :=
  ∀ n₀ q w, guardedTwoTickStart q w →
    1 - δ ≤ Exchange.alive SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (guardedLossProtocol n₀) n₀ ((fun _ => (q, 2)), w)

theorem guarded_two_tick_local_tolerance :
    GuardedTwoTickLocalTolerance 0 := by
  intro n₀ q w hg
  rw [guarded_alive n₀ q w hg]
  norm_num

def GuardedTwoTickSupport (n₀ : ℕ) : Prop :=
  ∀ x w, 0 < SensoryMaintenance.body.law SensoryMaintenance.world
      (guardedLossProtocol n₀) n₀ x w →
    ∃ q, x = (fun _ => (q, 2)) ∧ guardedTwoTickStart q w

theorem guarded_survival_at (n₀ : ℕ)
    (hs : GuardedTwoTickSupport n₀) :
    Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (guardedLossProtocol n₀) n₀ = 1 := by
  let sched := guardedLossProtocol n₀
  have point (x : Unit → SensoryMaintenance.State) (w : Env) :
      Exchange.alive SensoryMaintenance.body SensoryMaintenance.world
          SensoryMaintenance.ec SensoryMaintenance.material false sched n₀ (x, w) *
        SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w =
        if 0 < SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w
        then SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w else 0 := by
    by_cases hL : 0 < SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w
    · obtain ⟨q, hx, hg⟩ := hs x w hL
      rw [hx, guarded_alive n₀ q w hg]
      have hL' : 0 < SensoryMaintenance.body.law SensoryMaintenance.world sched n₀
          (fun _ => (q, 2)) w := by simpa [hx] using hL
      simp [hL']
    · have hz : SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w = 0 := by
        exact le_antisymm (le_of_not_gt hL)
          (SensoryMaintenance.body.law_nonneg_of_clocked
            SensoryMaintenance.world
            (SensoryMaintenance.body.eventClocked_implies_clocked SensoryMaintenance.ec)
            sched n₀ x w)
      simp [hz]
  change (∑ x : Unit → SensoryMaintenance.State, ∑ w : Env,
      Exchange.alive SensoryMaintenance.body SensoryMaintenance.world
        SensoryMaintenance.ec SensoryMaintenance.material false sched n₀ (x, w) *
        SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w) = 1
  simp_rw [point]
  have hl := SensoryMaintenance.body.law_sum SensoryMaintenance.world sched n₀
  calc
    (∑ x : Unit → SensoryMaintenance.State, ∑ w : Env,
        if 0 < SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w
        then SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w else 0) =
        ∑ x : Unit → SensoryMaintenance.State, ∑ w : Env,
          SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w := by
      apply Finset.sum_congr rfl
      intro x _
      apply Finset.sum_congr rfl
      intro w _
      by_cases hL : 0 < SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w
      · simp [hL]
      · have hz : SensoryMaintenance.body.law SensoryMaintenance.world sched n₀ x w = 0 := by
          exact le_antisymm (le_of_not_gt hL)
            (SensoryMaintenance.body.law_nonneg_of_clocked
              SensoryMaintenance.world
              (SensoryMaintenance.body.eventClocked_implies_clocked SensoryMaintenance.ec)
              sched n₀ x w)
        simp [hz]
    _ = 1 := hl

theorem guarded_support_zero : GuardedTwoTickSupport 0 := by
  intro x w h
  have h0 : x = SensoryMaintenance.body.init ∧
      w = SensoryMaintenance.world.start := by
    by_cases heq : x = SensoryMaintenance.body.init ∧
        w = SensoryMaintenance.world.start
    · exact heq
    · simp [Medium.law, heq] at h
  rcases h0 with ⟨rfl, rfl⟩
  refine ⟨.search, ?_, ?_⟩
  · simp [SensoryMaintenance.body]
  · exact initial_guarded_two_tick_start

private noncomputable def guardedSupportPath : ℕ → SensoryMaintenance.S
  | 0 => SensoryMaintenance.initial
  | 1 => (fun _ => (.search, 2), .answerA)
  | 2 => (fun _ => (.aSpent, 2), .feedA)
  | _ => SensoryMaintenance.initial

private def guardedSupportEvent (_ : ℕ) : Option Unit := some ()

private noncomputable abbrev guardedSupportK := fun t =>
  SensoryMaintenance.body.jointEventKernel SensoryMaintenance.world
    SensoryMaintenance.ec false (guardedLossProtocol 2) t

private theorem guarded_support_path_step (t : ℕ) (ht : t < 2) :
    0 < (guardedSupportK t).step (guardedSupportPath t)
      (guardedSupportEvent t) (guardedSupportPath (t + 1)) := by
  have ht' : t = 0 ∨ t = 1 := by omega
  rcases ht' with rfl | rfl <;>
    norm_num [guardedSupportK, guardedSupportPath, guardedSupportEvent,
      guardedLossProtocol, Medium.jointEventKernel, Medium.eventWorldStep,
      Medium.eventStep, SensoryMaintenance.body, SensoryMaintenance.world,
      SensoryMaintenance.next, SensoryMaintenance.initial,
      SensoryMaintenance.empty, RenewableLearning.sense,
      SensoryMaintenance.incoming, SensoryMaintenance.outgoing,
      RenewableLearning.next, RenewableLearning.incoming,
      RenewableLearning.outgoing, GraceRecovery.moves, RenewableLearning.moves,
      policy_table]

theorem not_guarded_support_two : ¬ GuardedTwoTickSupport 2 := by
  intro hs
  have hp := SensoryMaintenanceTwoTickGlobal.stateLaw_pos_of_path
    guardedSupportK SensoryMaintenance.initial guardedSupportPath
    guardedSupportEvent (by rfl) 2 guarded_support_path_step
  have heq : (fun z : SensoryMaintenance.S =>
      if z = SensoryMaintenance.initial then (1 : ℝ) else 0) =
      (fun z => if z.1 = SensoryMaintenance.body.init ∧
        z.2 = SensoryMaintenance.world.start then 1 else 0) := by
    funext z
    simp [SensoryMaintenance.initial, Prod.ext_iff]
  rw [heq] at hp
  rw [SensoryMaintenance.body.jointEventKernel_stateLaw
    SensoryMaintenance.world SensoryMaintenance.ec false
    (guardedLossProtocol 2) 2 (guardedSupportPath 2)] at hp
  have hL : 0 < SensoryMaintenance.body.law SensoryMaintenance.world
      (guardedLossProtocol 2) 2 (guardedSupportPath 2).1 (guardedSupportPath 2).2 := by
    simpa [guardedSupportPath] using hp
  obtain ⟨q, hx, hg⟩ := hs (guardedSupportPath 2).1 (guardedSupportPath 2).2 hL
  have hq : q = .aSpent := by
    have hx0 := congrArg (fun f => (f ()).1) hx
    simpa [guardedSupportPath] using hx0.symm
  rcases hg.1 with h | h | h <;> simp [hq] at h

theorem guarded_initial_survival :
    Exchange.survival SensoryMaintenance.body SensoryMaintenance.world
      SensoryMaintenance.ec SensoryMaintenance.material false
      (guardedLossProtocol 0) 0 = 1 :=
  guarded_survival_at 0 guarded_support_zero

theorem eligible_excludes_spent (x : SensoryMaintenance.State)
    (hx : eligibleTwoTickStart x) :
    x.1 ≠ .aSpent ∧ x.1 ≠ .bSpent := by
  rcases x with ⟨q, r⟩
  cases q <;> simp [eligibleTwoTickStart] at hx ⊢

theorem eligible_excludes_verified_dangerous_branch
    (x : SensoryMaintenance.State) (w : Env)
    (hx : eligibleTwoTickStart x) :
    ¬(x.1 = .aSpent ∧ w = .feedA) := by
  intro h
  exact (eligible_excludes_spent x hx).1 h.1

theorem eligible_two_loss_preserves_cell
    (x : SensoryMaintenance.State) (hx : eligibleTwoTickStart x)
    (hr : 2 ≤ x.2.val) :
    (SensoryMaintenance.next none (SensoryMaintenance.next none x)).1 = x.1 ∧
    (SensoryMaintenance.next none (SensoryMaintenance.next none x)).2.val = 0 := by
  rcases x with ⟨q, r⟩
  fin_cases r <;> cases q <;>
    try norm_num [eligibleTwoTickStart, SensoryMaintenance.next,
      SensoryMaintenance.empty] at hx hr ⊢
  all_goals simp_all [eligibleTwoTickStart, SensoryMaintenance.next,
    SensoryMaintenance.empty]

theorem eligible_keeps_existing_budget_contract (s : SensoryMaintenance.Stim)
    (x : SensoryMaintenance.State) :
    SensoryMaintenance.cost (SensoryMaintenance.next s x) +
        SensoryMaintenance.outgoing s x =
      SensoryMaintenance.cost x + SensoryMaintenance.incoming s x :=
  SensoryMaintenance.local_budgets s x |>.1

theorem region_repair_preserves_one_tick :
    SensoryMaintenanceRecovery.ToleratesSensoryLoss 0 1 :=
  SensoryMaintenanceRecovery.one_tick_tolerance

end CGB.SensoryMaintenanceRepairCandidates
