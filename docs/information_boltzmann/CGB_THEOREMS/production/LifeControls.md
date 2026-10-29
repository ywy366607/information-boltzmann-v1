# LifeControls

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean) · [返回导航](../README.md)

模块源码说明：Definition controls prompted by passive absorption and frozen-copy objections The old reflex/self-model advantages remain withdrawn.

## CGB.FirstHit.during_zero_on_closed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L20)

解释：若时变核的正概率转移在集合内封闭且均不标记目标事件，则从该集合出发任意有限期限的首次命中概率为零。

源码说明：Protocol-dependent extension of zero_on_closed, also valid for the future marginal-ghost cut. An observer cannot create a marked transition.

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (K : ℕ → CGB.FirstHit.Kernel S) (mark : S → S → Bool) (C : S → Prop),
  (∀ (t : ℕ) (z z' : S), C z → 0 < (K t).step z z' → C z') →
    (∀ (t : ℕ) (z z' : S), C z → 0 < (K t).step z z' → mark z z' = false) →
      ∀ (t k : ℕ) (z : S), C z → CGB.FirstHit.during K mark t k z = 0
```

## CGB.Operational.recovery_zero_without_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L44)

解释：若所有身体状态都不可能摄入，则任意指定协议、联合状态和有限期限的操作化摄入恢复概率为零。

源码说明：No possible positive-rate intake excludes recovery on BOTH registered future protocols, for every current joint state and every horizon.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock),
  (∀ (x : U → Λ), ¬M.CanTakeIn x) →
    ∀ (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ) (z : (U → Λ) × Wd.W),
      CGB.Operational.recovery M Wd hc cut sched t k z = 0
```

## CGB.Operational.alive_zero_without_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L62)

解释：若所有身体状态都不可能摄入，则任意协议、时刻和联合状态的操作化生命指标为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock),
  (∀ (x : U → Λ), ¬M.CanTakeIn x) →
    ∀ (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ) (z : (U → Λ) × Wd.W),
      CGB.Operational.alive M Wd hc cut sched t z = 0
```

## CGB.Operational.survival_zero_without_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L69)

解释：若所有身体状态都不可能摄入，则任意指定协议和时刻的操作化生存评分为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock),
  (∀ (x : U → Λ), ¬M.CanTakeIn x) →
    ∀ (cut : Bool) (sched : ℕ → Wd.W → U → I) (n : ℕ), CGB.Operational.survival M Wd hc cut sched n = 0
```

## CGB.Operational.not_persists_without_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L74)

解释：若所有身体状态都不可能摄入，则任意实数 δ 下操作化持续生存条件不成立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock), (∀ (x : U → Λ), ¬M.CanTakeIn x) → ∀ (δ : ℝ), ¬CGB.Operational.Persists M Wd hc δ
```

## CGB.Operational.not_selfMaintains_without_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L81)

解释：若所有身体状态都不可能摄入，则指定操作化闭环相对断环的自维持条件不成立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock), (∀ (x : U → Λ), ¬M.CanTakeIn x) → ¬CGB.Operational.SelfMaintains M Wd hc
```

## CGB.Operational.not_persists_of_frozen

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L92)

解释：若所有状态与刺激下的正速率事件均保持身体状态，则任意实数 δ 下操作化持续生存条件不成立。

源码说明：This migrates the fully-frozen-copy exclusion to the actual-world target. Freezing only under quiet, or only freezing a subset of state, is a different intervention and is not excluded by this sufficient hypothesis.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock),
  (∀ (a : A) (s : U → I) (x : U → Λ), 0 < M.rate a s x → M.act a s x = x) →
    ∀ (δ : ℝ), ¬CGB.Operational.Persists M Wd hc δ
```

## CGB.Operational.not_persists_of_closed_moments_and_empty_boundary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L104)

解释：若所有刺激下闭事件的一阶及二阶内容漂移为零且边界为空，则操作化持续生存条件不成立。

源码说明：Direct downstream use of the new Noether bridge: if closed moment drifts vanish and there are no boundary participants, actual intake is impossible. With an open boundary the flux term must instead be accounted for.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock),
  (∀ (s : U → I),
      (∀ (x : U → Λ), (M.closedJump s).gen M.contentObservable x = 0) ∧
        ∀ (x : U → Λ), (M.closedJump s).gen (fun y => M.contentObservable y ^ 2) x = 0) →
    M.boundary = ∅ → ∀ (δ : ℝ), ¬CGB.Operational.Persists M Wd hc δ
```

## CGB.Medium.kernel_from_frozen_state

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L126)

解释：若某状态在任何刺激下的所有正速率事件都保持该状态，则从该状态出发的一步核仅在原状态取一。

源码说明：The full initial state need only be absorbing for all possible stimuli. Other states of the same medium may still possess intake events.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (x : U → Λ),
  (∀ (a : A) (s : U → I), 0 < M.rate a s x → M.act a s x = x) →
    ∀ (R : ℝ) (s : U → I) (y : U → Λ), M.kernel R s x y = if x = y then 1 else 0
```

## CGB.Operational.recovery_zero_from_frozen_state

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L149)

解释：若某身体状态在所有刺激下的正速率事件均保持该状态，则该状态在任意指定协议与有限期限下的操作化摄入恢复概率为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock) (x : U → Λ),
  (∀ (a : A) (s : U → I), 0 < M.rate a s x → M.act a s x = x) →
    ∀ (cut : Bool) (sched : ℕ → Wd.W → U → I) (t k : ℕ) (w : Wd.W),
      CGB.Operational.recovery M Wd hc cut sched t k (x, w) = 0
```

## CGB.Operational.alive_zero_from_frozen_state

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L172)

解释：若某身体状态在所有刺激下的正速率事件均保持该状态，则该状态在任意时刻和世界状态的操作化生命指标为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock) (x : U → Λ),
  (∀ (a : A) (s : U → I), 0 < M.rate a s x → M.act a s x = x) →
    ∀ (cut : Bool) (sched : ℕ → Wd.W → U → I) (t : ℕ) (w : Wd.W), CGB.Operational.alive M Wd hc cut sched t (x, w) = 0
```

## CGB.Operational.not_persists_of_frozen_initial

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L181)

解释：若初始身体状态在所有刺激下被正速率事件保持，则任意实数 δ 下操作化持续生存条件不成立。

源码说明：Even holding material at birth cannot rescue an absorbing initial body.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (hc : M.Clocked Wd.clock),
  (∀ (a : A) (s : U → I), 0 < M.rate a s M.init → M.act a s M.init = M.init) →
    ∀ (δ : ℝ), ¬CGB.Operational.Persists M Wd hc δ
```

## CGB.RecoveryControls.passive_body_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L199)

解释：构造的被动吸收身体满足当前 Axioms 结构。

源码说明：The existing passive-absorption control also satisfies all six physical clauses; its failure of the life target is not a violation of those clauses.

形式陈述：

```lean
CGB.RecoveryControls.body.Axioms
```

## CGB.RecoveryControls.passive_body_can_take_in

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L232)

解释：指定被动吸收身体在空状态存在可能的内容摄入事件。

源码说明：Passive absorption really occurs, so this is stronger than a frozen body with no intake transition anywhere. It still does not certify ongoing life.

形式陈述：

```lean
CGB.RecoveryControls.body.CanTakeIn fun x => false
```

## CGB.RecoveryControls.passive_full_frozen

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L238)

解释：指定被动身体为满状态时，任意刺激下的正速率事件都保持该状态。

源码说明：Once full, the passive body's absorption stops in every environment.

形式陈述：

```lean
∀ (a : Unit) (s : Unit → Bool),
  (0 < CGB.RecoveryControls.body.rate a s fun x => true) →
    (CGB.RecoveryControls.body.act a s fun x => true) = fun x => true
```

## CGB.RecoveryControls.passive_full_not_alive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L245)

解释：对任意满足时钟条件的世界和指定协议，满状态被动身体的操作化生命指标为零。

源码说明：The current actual-world alive test excludes the full passive state for both normal and cut futures, at every horizon and in every world.

形式陈述：

```lean
∀ (Wd : CGB.World Unit Bool Bool) (hc : CGB.RecoveryControls.body.Clocked Wd.clock) (cut : Bool)
  (sched : ℕ → Wd.W → Unit → Bool) (t : ℕ) (w : Wd.W),
  CGB.Operational.alive CGB.RecoveryControls.body Wd hc cut sched t (fun x => true, w) = 0
```

## CGB.RecoveryControls.filledPassive_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L256)

解释：构造的满载被动身体满足当前 Axioms 结构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.RecoveryControls.filledPassive.Axioms
```

## CGB.RecoveryControls.filledPassive_clock

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L260)

解释：构造的满载被动身体满足速率一的旧版离散状态时钟条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.RecoveryControls.filledPassive.Clocked 1
```

## CGB.RecoveryControls.filledPassive_not_persists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L262)

解释：对任意世界、满足时钟条件的指定满载被动身体以及任意实数 δ，操作化持续生存条件不成立。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (Wd : CGB.World Unit Bool Bool) (hc : CGB.RecoveryControls.filledPassive.Clocked Wd.clock) (δ : ℝ),
  ¬CGB.Operational.Persists CGB.RecoveryControls.filledPassive Wd hc δ
```

## CGB.RecoveryControls.passive_intake_not_persistence

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L269)

解释：该满载被动身体满足公理、时钟、初始正内容及某状态可摄入，但对任意实数 δ 都不满足指定世界中的操作化持续生存。

源码说明：Explicit definition control: physical admissibility, initial material, and an intake event somewhere do not imply actual persistence in this world.

形式陈述：

```lean
∀ (δ : ℝ),
  CGB.RecoveryControls.filledPassive.Axioms ∧
    CGB.RecoveryControls.filledPassive.Clocked 1 ∧
      0 < CGB.RecoveryControls.filledPassive.total CGB.RecoveryControls.filledPassive.init ∧
        (∃ x, CGB.RecoveryControls.filledPassive.CanTakeIn x) ∧
          ¬CGB.Operational.Persists CGB.RecoveryControls.filledPassive (CGB.RecoveryControls.world true)
              CGB.RecoveryControls.filledPassive_clock δ
```

## CGB.NoetherControls.seed_closed_symmetry

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L284)

解释：对任意刺激，seed 的闭事件跳跃过程与总内容乘法算子满足指定可观测对称性。

源码说明：Existing admissible seed: its closed-event content symmetry is certified.

形式陈述：

```lean
∀ (s : Fin 4 → Bool), (CGB.Models.seed.closedJump s).ObservableSymmetry CGB.Models.seed.contentObservable
```

## CGB.NoetherControls.toggle_fails_closed_moments

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L289)

解释：twinToggle 不满足“每种刺激下闭事件的一阶和二阶内容漂移均为零”的联合条件。

源码说明：Existing nonconserving closed toggle is rejected by the new moment bridge.

形式陈述：

```lean
¬∀ (s : Fin 2 → Unit),
    (∀ (x : Fin 2 → Bool), (CGB.Models.twinToggle.closedJump s).gen CGB.Models.twinToggle.contentObservable x = 0) ∧
      ∀ (x : Fin 2 → Bool),
        (CGB.Models.twinToggle.closedJump s).gen (fun y => CGB.Models.twinToggle.contentObservable y ^ 2) x = 0
```

## CGB.NoetherControls.meanOnly_first_drift

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L303)

解释：指定三状态跳跃过程对 signedValue 在每个状态的一阶生成元漂移均为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (x : Fin 3), CGB.NoetherControls.meanOnly.gen CGB.NoetherControls.signedValue x = 0
```

## CGB.NoetherControls.meanOnly_second_drift

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L307)

解释：指定三状态见证中，signedValue 的平方在状态零处的生成元漂移等于二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.NoetherControls.meanOnly.gen (fun x => CGB.NoetherControls.signedValue x ^ 2) 0 = 2
```

## CGB.NoetherControls.meanOnly_not_conserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LifeControls.lean#L310)

解释：指定三状态见证的 signedValue 不是逐跳守恒量，说明一阶漂移为零本身不足以推出守恒。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.NoetherControls.meanOnly.Conserved CGB.NoetherControls.signedValue
```
