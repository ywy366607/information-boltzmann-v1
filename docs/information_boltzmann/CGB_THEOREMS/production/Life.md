# Life

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Life.lean) · [返回导航](../README.md)

模块源码说明：The infinite-stream life statement **Historical menu semantics (versioned 2026-10-23).** The `recover`, `aliveInd`, `Lives` and `LifeConjecture` below retain the old favourable-stimulus menu for compatibility.

## CGB.Medium.kernel_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L104)

解释：介质总转移速率不超过正时钟速率时，其离散一步转移核的每项非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) {R : ℝ},
  M.Clocked R → ∀ (s : U → I) (x y : U → Λ), 0 ≤ M.kernel R s x y
```

## CGB.Medium.kernel_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L111)

解释：按介质离散核定义，对下一状态求和得到一；非负性另需时钟条件。

源码说明：The kernel is stochastic.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (R : ℝ) (s : U → I)
  (x : U → Λ), ∑ y, M.kernel R s x y = 1
```

## CGB.Medium.sum4

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L131)

解释：有限身体状态与世界状态的四重求和可以交换前后两组变量的顺序。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {α : Type u_5} {β : Type u_6} [inst : Fintype α] [inst_1 : Fintype β] (F : α → β → α → β → ℝ),
  ∑ y, ∑ w', ∑ x, ∑ w, F x w y w' = ∑ x, ∑ w, ∑ y, ∑ w', F x w y w'
```

## CGB.Medium.law_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L138)

解释：按定义演化的身体—世界联合权重在每个时刻总和为一。

源码说明：The law is a probability distribution at every time.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (sched : ℕ → Wd.W → U → I) (n : ℕ), ∑ x, ∑ w, M.law Wd sched n x w = 1
```

## CGB.Medium.irrecoverable_absorbing

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L175)

解释：如果身体状态沿自身事件无法再到达可摄入状态，则任何一步自身事件的后继也不可恢复。

源码说明：**Losing recoverability is irreversible**: no event leads from an irrecoverable state back to a recoverable one.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] (M : CGB.Medium U Λ A I) {x y : U → Λ},
  ¬M.Recoverable x → M.OrgStep x y → ¬M.Recoverable y
```

## CGB.Medium.kernel_up_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L196)

解释：若所有状态都不能摄入内容，则任何内容严格增加的身体转移在离散核中的权重为零。

源码说明：Without any possible intake, no step raises content.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I),
  (∀ (x : U → Λ), ¬M.CanTakeIn x) → ∀ (R : ℝ) (s : U → I) {x y : U → Λ}, M.total x < M.total y → M.kernel R s x y = 0
```

## CGB.Medium.recover_eq_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L211)

解释：在旧版有利刺激菜单的恢复定义下，若任何身体状态都不能摄入内容，则任意期限的恢复评分为零。

源码说明：Without any possible intake, recovery is impossible.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I),
  (∀ (x : U → Λ), ¬M.CanTakeIn x) → ∀ (k : ℕ) (x : U → Λ), M.recover Wd k x = 0
```

## CGB.Medium.aliveInd_eq_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L229)

解释：在旧版生命指标下，若没有任何可能摄入且世界恢复阈值为正，则每个状态的生命指标均为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I),
  (∀ (x : U → Λ), ¬M.CanTakeIn x) → ∀ (x : U → Λ), M.aliveInd Wd x = 0
```

## CGB.Medium.stepSum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L242)

解释：当世界核归一且身体核保持测试函数时，一步联合更新保持该测试函数的加权和。

源码说明：One joint step, weighted by a test function `a` that the organism kernel preserves.

形式陈述：

```lean
∀ {α : Type u_5} {β : Type u_6} [inst : Fintype α] [inst_1 : Fintype β] (a : α → ℝ) (L : α → β → ℝ) (μ : α → β → β → ℝ)
  (K : β → α → α → ℝ),
  (∀ (x : α) (w : β), ∑ w', μ x w w' = 1) →
    (∀ (w : β) (x : α), ∑ y, a y * K w x y = a x) →
      ∑ y, ∑ w', a y * ∑ x, ∑ w, L x w * μ x w w' * K w x y = ∑ x, ∑ w, a x * L x w
```

## CGB.Medium.margStep

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L261)

解释：若身体转移核不依赖世界状态且世界核归一，联合更新后的身体边缘分布可由原身体边缘单独递推。

源码说明：One joint step of the organism's marginal, when its kernel ignores the world.

形式陈述：

```lean
∀ {α : Type u_5} {β : Type u_6} [inst : Fintype α] [inst_1 : Fintype β] (L : α → β → ℝ) (μ : α → β → β → ℝ)
  (K : α → α → ℝ),
  (∀ (x : α) (w : β), ∑ w', μ x w w' = 1) →
    ∀ (y : α), ∑ w', ∑ x, ∑ w, L x w * μ x w w' * K x y = ∑ x, (∑ w, L x w) * K x y
```

## CGB.Medium.cutMove_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L276)

解释：用于断环对照的独立身体混合世界转移，在输入联合权重总和为一时仍具有单位行和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U] [inst_2 : Fintype Λ]
  (Wd : CGB.World U Λ I) {L : (U → Λ) → Wd.W → ℝ},
  ∑ x, ∑ w, L x w = 1 → ∀ (x : U → Λ) (w : Wd.W), ∑ w', CGB.Medium.cutMove Wd L x w w' = 1
```

## CGB.Medium.lawCut_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L291)

解释：按断环对照递推得到的身体—世界联合权重在每个时刻总和为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  (sched : ℕ → Wd.W → U → I) (n : ℕ), ∑ x, ∑ w, M.lawCut Wd sched n x w = 1
```

## CGB.Medium.never_alive_without_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L311)

解释：按旧版生命指标，如果没有任何可能摄入，则任何感官安排下的生命评分在所有时刻为零。

源码说明：An organism that can never take anything in is never alive, whatever it senses.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I),
  (∀ (x : U → Λ), ¬M.CanTakeIn x) → ∀ (sched : ℕ → Wd.W → U → I) (n : ℕ), M.aliveProb Wd sched n = 0
```

## CGB.Medium.not_persists_of_no_intake

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L332)

解释：按旧版持续生存定义，若所有状态均不可能摄入内容，则无法满足持续生存条件。

源码说明：Without any possible intake there is no persistence.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I),
  (∀ (x : U → Λ), ¬M.CanTakeIn x) → ∀ (δ : ℝ) (T : ℕ), ¬M.Persists Wd δ T
```

## CGB.Medium.not_selfMaintains_of_unexposed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L367)

解释：按旧版闭环自维持定义，若介质的事件速率和作用完全不受刺激影响，则闭环无法相对断环获得要求的持续收益。

源码说明：An organism that ignores its stimuli cannot self-maintain: its own law does not depend on the world, so cutting the loop changes nothing.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I),
  ¬M.Exposed → ¬M.SelfMaintains Wd
```

## CGB.Medium.not_selfMaintains_of_closedMatter

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L397)

解释：按旧版基于摄入的自维持定义，若所有正速率事件都不改变总内容，则不能满足闭环自维持。

源码说明：An organism that exchanges no content with the world cannot self-maintain: whether it holds content never changes, in the loop or out of it.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I),
  (∀ (a : A) (s : U → I) (x : U → Λ), 0 < M.rate a s x → M.total (M.act a s x) = M.total x) → ¬M.SelfMaintains Wd
```

## CGB.Medium.Lives.exposed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L431)

解释：旧版 Lives 条件包含闭环自维持，因此满足它的介质必定有事件速率或作用依赖感官刺激。

源码说明：A living organism necessarily senses its world.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  {ρ δ : ℝ} {κ T : ℕ}, M.Lives Wd ρ δ κ T → M.Exposed
```

## CGB.Medium.Lives.metabolizes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Life.lean#L435)

解释：旧版 Lives 条件要求至少存在一个正速率事件改变主体总内容，即满足所定义的内容交换条件。

源码说明：A living organism necessarily exchanges content with its world (metabolism).

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A I) (Wd : CGB.World U Λ I)
  {ρ δ : ℝ} {κ T : ℕ}, M.Lives Wd ρ δ κ T → ∃ a s x, 0 < M.rate a s x ∧ M.total (M.act a s x) ≠ M.total x
```
