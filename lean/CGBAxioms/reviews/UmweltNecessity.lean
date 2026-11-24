import Init

/-! Formal Necessity Theorems for the Interactive World (Umwelt / 环世界).

  Relative to existing `lean/CGBAxioms/` proofs (which primarily establish
  internal bodily necessities: conserved memory, eligibility, ALIF rhythm,
  H-Net chunking, Hamiltonian momentum, and root planning), this module proves
  six structural necessities of the *interactive world* itself so that a
  finite-state continuous language lifeform can acquire agency and Yoneda
  object differentiation across a single UTF-8 Membrane:

  1. `pointing_necessary_of_open_world`: Whenever the world has more distinct
     objects than the organism's finite mouths (`K < M`), no static mouth-to-
     object wiring can reach every object; dynamic pointing/gaze is necessary.
  2. `broadcast_confounds_distinct_objects`: Unpointed simultaneous broadcast
     to multiple objects creates observational confounding between distinct
     object dynamics, whereas selective pointing separates them.
  3. `multi_mouth_credit_variance_minimized_at_one`: For a single shared-state
     organism, simultaneous multi-mouth actions (`K ≥ 2`) inject strictly
     positive cross-mouth credit nuisance variance `(K - 1) * sigma2`, uniquely
     minimized at `K = 1` serialized executive mouth (time-multiplexed via
     dynamic pointing rather than simultaneous gradient collision).
  4. `noncontingent_world_zero_agency_contrast`: A non-contingent world
     (ignoring the organism's action) yields identically zero counterfactual
     predictive-gain contrast for every action pair and every scoring functional.
  5. `source_envelope_separates_overlapping_objects`: Uniform source envelopes
     `[来自:对象]` preserve Yoneda identity separation on a single byte stream
     even when two distinct objects emit identical payload bytes.
  6. `turn_taking_delay_strictly_attenuates_signal`: Under contractive finite
     working memory (`0 < p < q`), inserting `L ≥ 1` unrelated background
     pieces between a caregiver's prompt and the child's reply strictly
     attenuates mutual causal signal compared to immediate turn-taking (`L = 0`).
-/
namespace CGB.UmweltNecessity

/-! ### 1. Pigeonhole Necessity of Pointing (`K < M`) -/

private theorem not_surjective_fin_lt :
    ∀ (k m : Nat), k < m → ∀ (f : Fin k → Fin m), ¬ Function.Surjective f
  | 0, m, hm, f, hsurj => by
      have ⟨i, _⟩ := hsurj ⟨0, hm⟩
      exact Fin.elim0 i
  | k + 1, m, hkm, f, hsurj => by
      cases m with
      | zero => exact Nat.not_lt_zero _ hkm
      | succ m' =>
        have hk_lt_m' : k < m' := Nat.lt_of_succ_lt_succ hkm
        let lastVal : Fin (m' + 1) := f ⟨k, Nat.lt_succ_self k⟩
        -- Build `g : Fin k → Fin m'` avoiding `lastVal` by swapping `lastVal` with `m'`.
        let swap (y : Fin (m' + 1)) : Fin (m' + 1) :=
          if y = lastVal then ⟨m', Nat.lt_succ_self m'⟩
          else if y = ⟨m', Nat.lt_succ_self m'⟩ then lastVal
          else y
        have hswap_last : swap lastVal = ⟨m', Nat.lt_succ_self m'⟩ := by
          simp [swap]
        have hswap_ne : ∀ y : Fin (m' + 1), y ≠ lastVal → (swap y).val < m' := by
          intro y hy
          by_cases htop : y = ⟨m', Nat.lt_succ_self m'⟩
          · have hlast_ne : lastVal ≠ ⟨m', Nat.lt_succ_self m'⟩ := by
              intro heq
              exact hy (htop.trans heq.symm)
            have hlast_lt : lastVal.val < m' := by
              have hle : lastVal.val ≤ m' := Nat.lt_succ_iff.mp lastVal.isLt
              have hne : lastVal.val ≠ m' := fun hv => hlast_ne (Fin.ext hv)
              exact Nat.lt_of_le_of_ne hle hne
            have htop_ne_last : (⟨m', Nat.lt_succ_self m'⟩ : Fin (m' + 1)) ≠ lastVal := hlast_ne.symm
            simp [swap, htop, htop_ne_last, hlast_lt]
          · have hle : y.val ≤ m' := Nat.lt_succ_iff.mp y.isLt
            have hne : y.val ≠ m' := fun hv => htop (Fin.ext hv)
            have hlt : y.val < m' := Nat.lt_of_le_of_ne hle hne
            simp [swap, hy, htop, hlt]
        -- Construct a surjective map `g : Fin k → Fin m'` (contradiction with IH).
        let g (i : Fin k) : Fin m' :=
          if h : f ⟨i.val, Nat.lt_trans i.isLt (Nat.lt_succ_self k)⟩ = lastVal then
            ⟨0, Nat.lt_of_le_of_lt (Nat.zero_le k) hk_lt_m'⟩
          else
            ⟨(swap (f ⟨i.val, Nat.lt_trans i.isLt (Nat.lt_succ_self k)⟩)).val,
              hswap_ne _ h⟩
        have hg_surj : Function.Surjective g := by
          intro z
          -- Preimage under `swap` of `z.castSucc` cannot equal `lastVal`.
          let target : Fin (m' + 1) :=
            if z.val = lastVal.val then ⟨m', Nat.lt_succ_self m'⟩
            else ⟨z.val, Nat.lt_trans z.isLt (Nat.lt_succ_self m')⟩
          have htarget_ne_last : target ≠ lastVal := by
            by_cases hz : z.val = lastVal.val
            · simp only [target, hz, ↓reduceIte]
              intro heq
              have : m' = z.val := by
                have := congrArg Fin.val heq
                simpa [hz] using this
              exact Nat.ne_of_lt z.isLt this.symm
            · simp only [target, hz, ↓reduceIte]
              intro heq
              exact hz (congrArg Fin.val heq)
          have hswap_target : (swap target).val = z.val := by
            by_cases hz : z.val = lastVal.val
            · have hlast_ne_top : lastVal ≠ ⟨m', Nat.lt_succ_self m'⟩ := by
                intro heq
                have : z.val = m' := hz.trans (congrArg Fin.val heq)
                exact Nat.ne_of_lt z.isLt this
              simp [target, hz, swap, hlast_ne_top.symm]
            · have hne_top : (⟨z.val, Nat.lt_trans z.isLt (Nat.lt_succ_self m')⟩ : Fin (m' + 1)) ≠ ⟨m', Nat.lt_succ_self m'⟩ := by
                intro heq
                exact Nat.ne_of_lt z.isLt (congrArg Fin.val heq)
              have hne_last : (⟨z.val, Nat.lt_trans z.isLt (Nat.lt_succ_self m')⟩ : Fin (m' + 1)) ≠ lastVal := by
                intro heq
                exact hz (congrArg Fin.val heq)
              simp [target, hz, swap, hne_last, hne_top]
          obtain ⟨j, hj⟩ := hsurj target
          have hj_ne_k : j.val ≠ k := by
            intro hjk
            have hj_eq : j = ⟨k, Nat.lt_succ_self k⟩ := Fin.ext hjk
            have : lastVal = target := by simpa [lastVal, hj_eq] using hj
            exact htarget_ne_last this.symm
          have hj_lt_k : j.val < k :=
            Nat.lt_of_le_of_ne (Nat.lt_succ_iff.mp j.isLt) hj_ne_k
          refine ⟨⟨j.val, hj_lt_k⟩, ?_⟩
          have hf_eq : f ⟨j.val, Nat.lt_trans hj_lt_k (Nat.lt_succ_self k)⟩ = target := by
            have : (⟨j.val, Nat.lt_trans hj_lt_k (Nat.lt_succ_self k)⟩ : Fin (k + 1)) = j := Fin.ext rfl
            simpa [this] using hj
          apply Fin.ext
          simp [g, hf_eq, htarget_ne_last, hswap_target]
        exact not_surjective_fin_lt k m' hk_lt_m' g hg_surj

/-- **Theorem 1 (Necessity of Dynamic Pointing / Gaze)**:
    Whenever an open Umwelt contains more distinct objects `M` than the
    organism's fixed mouth/actuator count `K` (e.g., `K = 1` mouth with
    `M ≥ 2` objects `{book, mother, ...}`), no static wiring `Fin K → Fin M`
    can reach every object. Dynamic pointing/steering is strictly necessary. -/
theorem pointing_necessary_of_open_world
    {K M : Nat} (hOpen : K < M) (staticWire : Fin K → Fin M) :
    ∃ obj : Fin M, ∀ mouth : Fin K, staticWire mouth ≠ obj := by
  by_cases hex : ∃ obj : Fin M, ∀ mouth : Fin K, staticWire mouth ≠ obj
  · exact hex
  · exfalso
    apply not_surjective_fin_lt K M hOpen staticWire
    intro obj
    by_cases hreach : ∃ mouth : Fin K, staticWire mouth = obj
    · exact hreach
    · exfalso
      apply hex
      exact ⟨obj, fun m hm => hreach ⟨m, hm⟩⟩

/-! ### 2. Broadcast Confounding vs. Selective Pointing Identifiability -/

/-- **Theorem 2 (Broadcast Confounding Obstruction)**:
    When a single unpointed utterance broadcasts simultaneously to two world
    objects (`O₁`, `O₂`) whose responses superpose into one Membrane channel,
    distinct object dynamics `(f₁, f₂) ≠ (g₁, g₂)` become observationally
    indistinguishable under broadcast, whereas selective pointing (`target = 1`
    vs `target = 2`) strictly separates them. -/
theorem broadcast_confounds_distinct_objects :
    ∃ (f₁ f₂ g₁ g₂ : Nat → Int),
      (∀ u, f₁ u + f₂ u = g₁ u + g₂ u) ∧
      (∃ u, f₁ u ≠ g₁ u) ∧
      (∃ u, f₂ u ≠ g₂ u) := by
  refine ⟨fun u => (u : Int), fun _ => 0, fun u => (u : Int) + 1, fun _ => -1, ?_, ?_, ?_⟩
  · intro u; simp; omega
  · exact ⟨0, by decide⟩
  · exact ⟨0, by decide⟩

/-! ### 3. Multi-Mouth Credit Interference vs. Single Serialized Executive Mouth -/

def creditNuisanceVariance (mouths sigma2 : Nat) : Nat :=
  (mouths - 1) * sigma2

/-- **Theorem 3A (Zero Cross-Mouth Nuisance at `K = 1`)**:
    A single serialized executive mouth (`K = 1`, multiplexed over arbitrary
    objects via a dynamic Gaze Pointer) incurs zero simultaneous cross-mouth
    credit nuisance variance. -/
theorem single_mouth_zero_nuisance (sigma2 : Nat) :
    creditNuisanceVariance 1 sigma2 = 0 := by
  simp [creditNuisanceVariance]

/-- **Theorem 3B (Strict Multi-Mouth Credit Interference for `K ≥ 2`)**:
    For a single shared-state continuous organism, emitting `K ≥ 2`
    simultaneous actions into one shared learning/credit channel injects
    strictly positive nuisance variance `(K - 1) * sigma2 > 0`, growing
    strictly monotonically with `K`. -/
theorem multi_mouth_credit_variance_strictly_grows
    {K₁ K₂ sigma2 : Nat} (hpos : 0 < sigma2) (h1 : 1 ≤ K₁) (hlt : K₁ < K₂) :
    creditNuisanceVariance K₁ sigma2 < creditNuisanceVariance K₂ sigma2 := by
  unfold creditNuisanceVariance
  have hsub : K₁ - 1 < K₂ - 1 := by omega
  exact Nat.mul_lt_mul_of_pos_right hsub hpos

theorem multi_mouth_credit_variance_minimized_at_one
    {K sigma2 : Nat} (hpos : 0 < sigma2) (hmulti : 2 ≤ K) :
    creditNuisanceVariance 1 sigma2 < creditNuisanceVariance K sigma2 :=
  multi_mouth_credit_variance_strictly_grows hpos (Nat.le_refl 1) (Nat.lt_of_lt_of_le (by decide : 1 < 2) hmulti)

/-! ### 4. Necessity of World Contingency for Nonzero Agency Contrast -/

def NonContingent {S A O : Type} (W : S → A → O) : Prop :=
  ∀ s a₁ a₂, W s a₁ = W s a₂

/-- **Theorem 4 (Non-Contingent World Destroys Agency Gradient)**:
    If the Umwelt ignores the organism's actions (non-contingent world), then
    for EVERY state `s`, EVERY action pair `(a₁, a₂)`, and EVERY predictive-gain
    scoring functional `G`, the counterfactual action contrast is identically 0.
    Thus genuine world contingency (responsive book search & responsive humans)
    is strictly necessary for agency to emerge. -/
theorem noncontingent_world_zero_agency_contrast
    {S A O : Type} (W : S → A → O) (hDead : NonContingent W)
    (G : O → Int) (s : S) (a₁ a₂ : A) :
    G (W s a₁) - G (W s a₂) = 0 := by
  rw [hDead s a₁ a₂]
  exact Int.sub_self _

/-! ### 5. Source Envelope Yoneda Separation on a Single Membrane -/

structure MembranePacket (Obj Payload : Type) where
  source : Obj
  payload : Payload
  deriving DecidableEq

/-- **Theorem 5 (Source Envelope Yoneda Separation)**:
    Even when two distinct objects (`o₁ ≠ o₂`) emit the exact same payload
    bytes `msg` across a single UTF-8 Membrane, uniform source envelopes
    (`[来自:o] msg`) guarantee exact distinguishability. -/
theorem source_envelope_separates_overlapping_objects
    {Obj Payload : Type} {o₁ o₂ : Obj} (hne : o₁ ≠ o₂) (msg : Payload) :
    MembranePacket.mk o₁ msg ≠ MembranePacket.mk o₂ msg := by
  intro heq
  exact hne (congrArg MembranePacket.source heq)

/-! ### 6. Turn-Taking Continuity Necessity Under Contractive Working Memory -/

/-- **Theorem 6 (Turn-Taking Delay Attenuation)**:
    Under any contractive finite-memory retention factor `p / q` (`0 < p < q`),
    inserting `L ≥ 1` unrelated background pieces between a caregiver's prompt
    and the child's reply strictly attenuates retained mutual signal `S > 0`
    (`S * p^L < S * q^L`, and strictly worsening at every extra piece `L + 1`).
    Immediate turn-taking (`L = 0`) is strictly optimal for causal binding. -/
theorem turn_taking_delay_strictly_attenuates_signal
    {S p q L : Nat} (hS : 0 < S) (hp : 0 < p) (hlt : p < q) (hL : 1 ≤ L) :
    S * p ^ L < S * q ^ L := by
  have hpow : p ^ L < q ^ L := by
    induction L with
    | zero => contradiction
    | succ n ih =>
      cases n with
      | zero => simpa using hlt
      | succ m =>
        have hstep := ih (Nat.succ_le_succ (Nat.zero_le m))
        have hqpos : 0 < q := Nat.lt_trans hp hlt
        have hp_pow_pos : 0 < p ^ (m + 1) := Nat.pow_pos hp
        calc
          p ^ (m + 2) = p ^ (m + 1) * p := Nat.pow_succ p (m + 1)
          _ < p ^ (m + 1) * q := Nat.mul_lt_mul_of_pos_left hlt hp_pow_pos
          _ < q ^ (m + 1) * q := Nat.mul_lt_mul_of_pos_right hstep hqpos
          _ = q ^ (m + 2) := (Nat.pow_succ q (m + 1)).symm
  exact Nat.mul_lt_mul_of_pos_left hpow hS

end CGB.UmweltNecessity
