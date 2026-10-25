import CGBAxioms.ScheduledRecovery
import CGBAxioms.RenewableRecoveryCertificate
import CGBAxioms.RenewableLearningControls

/-!
# A recoverable diagnostic opportunity after missed sensing

Keep the original body, learning payments, food amounts and clock. A wrong
probe action now returns conflict instead of immediate entrapment. The world
declares a three-tick recovery horizon, explicitly distinct from the old two.
-/

open Finset

namespace CGB.GraceRecovery
open RenewableLearning

noncomputable abbrev body := plant true true
noncomputable abbrev material := ledger true true

noncomputable def moves (w : Env) (x : Unit → Cell) (v : Env) : ℝ :=
  if x () ≠ .vacuum ∧ w = .probeA ∧ policy (x ()) ≠ 2 then
    (if v = .missA then 1 else 0)
  else if x () ≠ .vacuum ∧ w = .probeB ∧ policy (x ()) ≠ 2 then
    (if v = .missB then 1 else 0)
  else RenewableLearning.moves w x v

noncomputable abbrev worldWith (horizon : ℕ) : World Unit Cell Signal := {
  RenewableLearning.world with
  moves := moves
  moves_nonneg w x v := by
    unfold moves
    split_ifs
    all_goals try norm_num
    all_goals exact RenewableLearning.world.moves_nonneg _ _ _
  moves_sum w x := by
    unfold moves
    split_ifs
    all_goals try simp
    all_goals exact RenewableLearning.world.moves_sum _ _
  horizon := horizon }

noncomputable abbrev world := worldWith 3
abbrev ec : body.EventClocked world.clock := he true true
noncomputable abbrev natural : ℕ → Env → Unit → Signal := fun _ => sense
noncomputable def interrupted (n₀ : ℕ) := body.interrupt world n₀ 1
noncomputable def process (cut : Bool) (sched : ℕ → Env → Unit → Signal) :=
  MaterialSource.machine (body.jointEventKernel world ec cut sched)

private theorem policy_table (q : Cell) : policy q = match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2 := by cases q <;> decide

abbrev Good := RenewableLearningLife.Good

def Safe (q : Cell) : Env → Prop
  | .feedA | .feedB => q = .search ∨ q = .aSpent ∨ q = .bSpent ∨ q = .aReady ∨ q = .bReady
  | .missA | .missB => q = .aReady ∨ q = .bReady
  | .probeA | .probeB => q = .search ∨ q = .aReady ∨ q = .bReady
  | .answerA | .answerB => q = .search
  | .trap => False

def missed (w : Env) : Bool := decide (w = .missA ∨ w = .missB)
def lateFeed (q : Cell) (w : Env) : Bool :=
  decide ((w = .feedA ∧ q = .bReady) ∨ (w = .feedB ∧ q = .aReady))

theorem moves_unchanged_on_good (q : Cell) (w v : Env) (hg : Good q w) :
    moves w (fun _ => q) v = RenewableLearning.moves w (fun _ => q) v := by
  cases w <;> cases q <;> simp [Good, RenewableLearningLife.Good] at hg
  all_goals simp [moves, policy_table]

private theorem good_safe (q : Cell) (w : Env) (hg : Good q w) : Safe q w := by
  cases w <;> cases q <;> simp_all [Good, RenewableLearningLife.Good, Safe]

private theorem safe_nonvacuum (q : Cell) (w : Env) (hg : Safe q w) : q ≠ .vacuum := by
  cases w <;> cases q <;> simp_all [Safe]

private theorem normal_good (q : Cell) (w v : Env) (hg : Good q w)
    (hp : 0 < moves w (fun _ => q) v) : Good (next true true (sense w ()) q) v := by
  rw [moves_unchanged_on_good q w v hg] at hp
  cases w <;> cases q <;> simp [Good, RenewableLearningLife.Good] at hg
  all_goals cases v <;> norm_num [RenewableLearning.moves, policy_table] at hp
  all_goals norm_num [next, sense, Good, RenewableLearningLife.Good]

private theorem normal_safe (q : Cell) (w v : Env) (hg : Safe q w)
    (hp : 0 < moves w (fun _ => q) v) : Safe (next true true (sense w ()) q) v := by
  cases w <;> cases q <;> simp [Safe] at hg
  all_goals cases v <;> norm_num [moves, RenewableLearning.moves, policy_table] at hp
  all_goals norm_num [next, sense, Safe]

private theorem idle_safe (q : Cell) (w v : Env) (hg : Good q w)
    (hp : 0 < moves w (fun _ => q) v) : Safe q v := by
  rw [moves_unchanged_on_good q w v hg] at hp
  cases w <;> cases q <;> simp [Good, RenewableLearningLife.Good] at hg
  all_goals cases v <;> norm_num [RenewableLearning.moves, policy_table] at hp
  all_goals norm_num [Safe]

private theorem normal_zero (q : Cell) (w v : Env) (hg : Safe q w)
    (hp : 0 < moves w (fun _ => q) v) (hz : incoming true (sense w ()) q = 0) :
    missed w = true ∧ (v = .probeA ∨ v = .probeB) := by
  cases w <;> cases q <;> simp [Safe] at hg
  all_goals norm_num [sense, incoming] at hz
  all_goals cases v <;> norm_num [moves, RenewableLearning.moves, policy_table] at hp
  all_goals norm_num [missed]

private theorem idle_progress (q : Cell) (w v : Env) (hg : Good q w)
    (hp : 0 < moves w (fun _ => q) v) :
    (if missed v then (1 : ℕ) else 0) < (if lateFeed q w then 2 else 1) := by
  rw [moves_unchanged_on_good q w v hg] at hp
  cases w <;> cases q <;> simp [Good, RenewableLearningLife.Good] at hg
  all_goals cases v <;> norm_num [RenewableLearning.moves, policy_table] at hp
  all_goals norm_num [missed, lateFeed]

/-- Supported body events execute exactly the old local update; an idle slot
has zero incoming material. This lemma does not assume a favourable world. -/
theorem supported (cut : Bool) (sched : ℕ → Env → Unit → Signal)
    (t : ℕ) (z z' : S) (e : Option Unit)
    (hp : 0 < (process cut sched).step t z e z') :
    0 < body.eventWorldStep world cut sched t z z'.2 ∧
    z'.1 () = next true true (sched t z.2 ()) (z.1 ()) ∧
    Exchange.intake body world material sched t z e z' =
      decide (0 < incoming true (sched t z.2 ()) (z.1 ())) := by
  have hbody : 0 < body.eventStep 1 (sched t z.2) z.1 e z'.1 := by
    by_contra hn
    have hw := body.eventWorldStep_nonneg world ec cut sched t z z'.2
    exact (not_lt_of_ge (mul_nonpos_of_nonneg_of_nonpos hw (le_of_not_gt hn))) hp
  have hw : 0 < body.eventWorldStep world cut sched t z z'.2 := by
    by_contra hn
    exact (not_lt_of_ge (mul_nonpos_of_nonpos_of_nonneg (le_of_not_gt hn)
      (body.eventStep_nonneg ec _ _ _ _))) hp
  refine ⟨hw, ?_⟩
  cases e with
  | none =>
    have hx : z.1 = z'.1 := by by_contra hn; simp [Medium.eventStep, hn] at hbody
    have hc : sched t z.2 () = .calm ∨ z.1 () = .vacuum := by
      by_contra hn
      simp [Medium.eventStep, plant, body, hn] at hbody
    rw [← hx]
    rcases hc with hc | hc
    · simp [next, hc, Exchange.intake, incoming]
    · simp [next, hc, Exchange.intake, incoming]
  | some u =>
    have hx : body.act u (sched t z.2) z.1 = z'.1 := by
      by_contra hn
      simp [Medium.eventStep, hn] at hbody
    exact ⟨(congrFun hx ()).symm, rfl⟩

def region (n₀ t : ℕ) (z : S) : Prop :=
  if t ≤ n₀ then Good (z.1 ()) z.2 else Safe (z.1 ()) z.2

def potential (n₀ t : ℕ) (q : Cell) (w : Env) : ℕ :=
  if t < n₀ then (if missed w then 2 else 0)
  else if t = n₀ then (if lateFeed q w then 2 else 1)
  else if missed w then 1 else 0

private theorem probe_potential (n₀ t : ℕ) (q : Cell) (v : Env)
    (hv : v = .probeA ∨ v = .probeB) : potential n₀ t q v ≤ 1 := by
  rcases hv with rfl | rfl <;> simp [potential, missed, lateFeed]
  all_goals split_ifs <;> omega

private theorem stimulus_before (n₀ t : ℕ) (w : Env) (ht : t < n₀) :
    interrupted n₀ t w () = sense w () := by
  simp [interrupted, Medium.interrupt, world, worldWith, show ¬ n₀ ≤ t by omega]

private theorem stimulus_idle (n₀ : ℕ) (w : Env) : interrupted n₀ n₀ w () = .calm := by
  simp [interrupted, Medium.interrupt, body, plant]

private theorem stimulus_after (n₀ t : ℕ) (w : Env) (ht : n₀ < t) :
    interrupted n₀ t w () = sense w () := by
  simp [interrupted, Medium.interrupt, world, worldWith, show ¬ t < n₀ + 1 by omega]

private theorem scheduled_closed (n₀ t : ℕ) (z z' : S) (e : Option Unit)
    (hg : region n₀ t z) (hp : 0 < (process false (interrupted n₀)).step t z e z') :
    region n₀ (t + 1) z' := by
  obtain ⟨hw, hx, _⟩ := supported false (interrupted n₀) t z z' e hp
  change 0 < moves z.2 z.1 z'.2 at hw
  have he : z.1 = fun _ => z.1 () := by funext u; cases u; rfl
  rw [he] at hw
  rcases lt_trichotomy t n₀ with ht | heq | ht
  · have hG : Good (z.1 ()) z.2 := by simpa [region, le_of_lt ht] using hg
    rw [stimulus_before n₀ t z.2 ht] at hx
    have hn := normal_good (z.1 ()) z.2 z'.2 hG hw
    rw [← hx] at hn
    simpa [region, show t + 1 ≤ n₀ by omega] using hn
  · subst t
    have hG : Good (z.1 ()) z.2 := by simpa [region] using hg
    rw [stimulus_idle] at hx
    have hn := idle_safe (z.1 ()) z.2 z'.2 hG hw
    have hc (q : Cell) : next true true .calm q = q := by cases q <;> rfl
    rw [hc] at hx
    rw [← hx] at hn
    simpa [region, show ¬ n₀ + 1 ≤ n₀ by omega] using hn
  · have hG : Safe (z.1 ()) z.2 := by simpa [region, show ¬ t ≤ n₀ by omega] using hg
    rw [stimulus_after n₀ t z.2 ht] at hx
    have hn := normal_safe (z.1 ()) z.2 z'.2 hG hw
    rw [← hx] at hn
    simpa [region, show ¬ t + 1 ≤ n₀ by omega] using hn

private theorem scheduled_progress (n₀ t : ℕ) (z z' : S) (e : Option Unit)
    (hg : region n₀ t z) (hp : 0 < (process false (interrupted n₀)).step t z e z')
    (hm : Exchange.intake body world material (interrupted n₀) t z e z' = false) :
    potential n₀ (t + 1) (z'.1 ()) z'.2 < potential n₀ t (z.1 ()) z.2 := by
  obtain ⟨hw, _, hi⟩ := supported false (interrupted n₀) t z z' e hp
  have hz : incoming true (interrupted n₀ t z.2 ()) (z.1 ()) = 0 := by
    by_contra hn
    simp [hm, Nat.pos_of_ne_zero hn] at hi
  change 0 < moves z.2 z.1 z'.2 at hw
  have he : z.1 = fun _ => z.1 () := by funext u; cases u; rfl
  rw [he] at hw
  rcases lt_trichotomy t n₀ with ht | heq | ht
  · have hG : Good (z.1 ()) z.2 := by simpa [region, le_of_lt ht] using hg
    rw [stimulus_before n₀ t z.2 ht] at hz
    obtain ⟨hmw, hv⟩ := normal_zero _ _ _ (good_safe _ _ hG) hw hz
    have hnext := probe_potential n₀ (t + 1) (z'.1 ()) z'.2 hv
    have hc : potential n₀ t (z.1 ()) z.2 = 2 := by simp [potential, ht, hmw]
    omega
  · subst t
    have hG : Good (z.1 ()) z.2 := by simpa [region] using hg
    have hn := idle_progress (z.1 ()) z.2 z'.2 hG hw
    simpa [potential, show ¬ n₀ + 1 < n₀ by omega, show n₀ + 1 ≠ n₀ by omega] using hn
  · have hG : Safe (z.1 ()) z.2 := by simpa [region, show ¬ t ≤ n₀ by omega] using hg
    rw [stimulus_after n₀ t z.2 ht] at hz
    obtain ⟨hmw, hv⟩ := normal_zero _ _ _ hG hw hz
    simp only [potential, show ¬ t < n₀ by omega, show t ≠ n₀ by omega,
      show ¬ t + 1 < n₀ by omega, show t + 1 ≠ n₀ by omega, ↓reduceIte, hmw]
    rcases hv with hv | hv <;> simp [hv, missed]

/-- Protocol-time analysis only: no countdown, interruption start or rank is
stored in the organism or read by its policy. -/
noncomputable def certificate (n₀ : ℕ) : ScheduledRecovery.Ranked
    (body.jointEventKernel world ec false (interrupted n₀))
    (Exchange.intake body world material (interrupted n₀)) world.horizon where
  region := region n₀
  rank t z := potential n₀ t (z.1 ()) z.2
  bounded t z _ := by unfold potential; split_ifs <;> norm_num [world, worldWith]
  closed t z e z' := scheduled_closed n₀ t z z' e
  progresses t z e z' := scheduled_progress n₀ t z z' e

private theorem positive (n₀ t : ℕ) (z : S) (hg : region n₀ t z) : 0 < body.total z.1 := by
  have hS : Safe (z.1 ()) z.2 := by
    unfold region at hg
    split_ifs at hg with ht
    · exact good_safe _ _ hg
    · exact hg
  have hn := safe_nonvacuum _ _ hS
  change 0 < cost (z.1 ())
  generalize z.1 () = q at *
  cases q <;> norm_num [cost, buffer, credit] at *

theorem one_tick_survival (n₀ n : ℕ) :
    Exchange.survival body world ec material false (interrupted n₀) n = 1 :=
  ScheduledRecovery.survival_of_ranked body world ec material (interrupted n₀)
    (by norm_num [world, worldWith]) (certificate n₀)
    (by simp [certificate, region, body, plant, world, worldWith,
      Good, RenewableLearningLife.Good]) (positive n₀) n

theorem one_tick_tolerance : Exchange.ToleratesSensoryInterruption body world ec material 0 1 := by
  refine ⟨by norm_num, fun n₀ n => ?_⟩
  change 1 - (0 : ℝ) ≤ Exchange.survival body world ec material false (interrupted n₀) n
  rw [one_tick_survival]
  norm_num

noncomputable def naturalCertificate : RecoveryCriteria.Ranked
    (body.jointEventKernel world ec false natural)
    (Exchange.intake body world material natural) world.horizon where
  region z := Good (z.1 ()) z.2
  rank z := if missed z.2 then 1 else 0
  bounded z _ := by split_ifs <;> norm_num [world, worldWith]
  closed t z e z' hg hp := by
    obtain ⟨hw, hx, _⟩ := supported false natural t z z' e hp
    change 0 < moves z.2 z.1 z'.2 at hw
    have he : z.1 = fun _ => z.1 () := by funext u; cases u; rfl
    rw [he] at hw
    have hn := normal_good _ _ _ hg hw
    change z'.1 () = next true true (sense z.2 ()) (z.1 ()) at hx
    rw [← hx] at hn
    exact hn
  progresses t z e z' hg hp hm := by
    obtain ⟨hw, _, hi⟩ := supported false natural t z z' e hp
    have hz : incoming true (sense z.2 ()) (z.1 ()) = 0 := by
      by_contra hn
      simp [natural, hm, Nat.pos_of_ne_zero hn] at hi
    change 0 < moves z.2 z.1 z'.2 at hw
    have he : z.1 = fun _ => z.1 () := by funext u; cases u; rfl
    rw [he] at hw
    obtain ⟨hmw, hv⟩ := normal_zero _ _ _ (good_safe _ _ hg) hw hz
    rw [hmw]
    rcases hv with hv | hv <;> simp [hv, missed]

theorem natural_survival (n : ℕ) : Exchange.survival body world ec material false natural n = 1 :=
  RecoveryCriteria.survival_of_ranked body world ec material (by norm_num [world, worldWith])
    naturalCertificate (by rfl)
    (fun z hg => positive 0 0 z (by change Good (z.1 ()) z.2; exact hg)) n

end CGB.GraceRecovery
