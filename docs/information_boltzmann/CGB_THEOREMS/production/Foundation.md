# Foundation

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean) · [返回导航](../README.md)

模块源码说明：Foundation audit: premises, specialization, and clocks These results audit the existing predicates; they add no physical axiom and do not change the organism dynamics or the life target.

## CGB.Medium.finite_local_states_of_p1a

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L18)

解释：公理P1a自身即内蕴局域状态类型的有限性，无需额外假设枚举。

源码说明：P1a itself supplies local-state finiteness, before a finite enumeration is chosen.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I), M.P1a → Finite Λ
```

## CGB.Medium.p1a_iff_unique_vacuum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L27)

解释：在局域状态有限前提下，公理P1a等价于系统中存在唯一的零能量真空基态。

源码说明：On an already finite local state type, P1a's first two conjuncts are automatic. The infinite tape/bag independence witnesses belong to the more general Medium layer.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [Finite Λ],
  M.P1a ↔ ∃! ℓ, M.content ℓ = 0
```

## CGB.Medium.exists_clock_of_finite_stimuli

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L42)

解释：若刺激字母表与机体均为有限集，则介质必存在统一的有界全局演化时钟速率。

源码说明：A globally bounded clock follows if the stimulus alphabet is also finite. Finiteness of the body alone does not provide this missing uniformity premise.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] [Finite I],
  ∃ R, M.Clocked R
```

## CGB.Medium.recover_ignores_world_moves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L65)

解释：现有恢复菜单仅依赖机体能力与环境状态空间，而独立于具体的环境转移矩阵设定。

源码说明：The current recovery menu does not consult actual world transitions. This audits the existing definition rather than calling its value a reachable-world probability.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (Wd : CGB.World U Λ I)
  (moves' : Wd.W → (U → Λ) → Wd.W → ℝ) (hnonneg : ∀ (w : Wd.W) (x : U → Λ) (w' : Wd.W), 0 ≤ moves' w x w')
  (hsum : ∀ (w : Wd.W) (x : U → Λ), ∑ w', moves' w x w' = 1) (k : ℕ) (x : U → Λ),
  M.recover
      { W := Wd.W, fin := Wd.fin, dec := Wd.dec, sense := Wd.sense, moves := moves', moves_nonneg := hnonneg,
        moves_sum := hsum, start := Wd.start, quiet := Wd.quiet, clock := Wd.clock, clock_pos := ⋯,
        horizon := Wd.horizon, recovery := Wd.recovery, recovery_pos := ⋯ }
      k x =
    M.recover Wd k x
```

## CGB.Medium.recover_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L77)

解释：在满足时钟准入假设下，机体在有限步内的恢复度量严格介于0和1之间。

源码说明：Recovery is a genuine probability only with the clock-admission hypothesis. Its favourable-stimulus supremum is not an actual-world reachability probability.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (Wd : CGB.World U Λ I),
  M.Clocked Wd.clock → ∀ (k : ℕ) (x : U → Λ), 0 ≤ M.recover Wd k x ∧ M.recover Wd k x ≤ 1
```

## CGB.Medium.law_nonneg_of_clocked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L112)

解释：当时钟准入条件满足时，联合系统的正常演化概率测度值恒非负。

源码说明：Normal and loop-cut laws must be nonnegative as well as normalized.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (Wd : CGB.World U Λ I),
  M.Clocked Wd.clock → ∀ (sched : ℕ → Wd.W → U → I) (n : ℕ) (x : U → Λ) (w : Wd.W), 0 ≤ M.law Wd sched n x w
```

## CGB.Medium.lawCut_nonneg_of_clocked

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L122)

解释：在时钟准入条件下，施加环路截断后的联合演化概率测度值恒非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (Wd : CGB.World U Λ I),
  M.Clocked Wd.clock → ∀ (sched : ℕ → Wd.W → U → I) (n : ℕ) (x : U → Λ) (w : Wd.W), 0 ≤ M.lawCut Wd sched n x w
```

## CGB.Medium.aliveProb_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L135)

解释：在时钟准入条件下，机体在任意调度与时间步下的存活概率度量严格介于0与1之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (Wd : CGB.World U Λ I),
  M.Clocked Wd.clock → ∀ (sched : ℕ → Wd.W → U → I) (n : ℕ), 0 ≤ M.aliveProb Wd sched n ∧ M.aliveProb Wd sched n ≤ 1
```

## CGB.Medium.aliveProb_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L152)

解释：系统在零时刻（第0步）的存活概率严格等于初始状态的存活指示函数值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (Wd : CGB.World U Λ I)
  (sched : ℕ → Wd.W → U → I), M.aliveProb Wd sched 0 = M.aliveInd Wd M.init
```

## CGB.Medium.persists_parameter_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L159)

解释：当机体满足持续生存条件时，其持久性阈值参数δ必然非负。

源码说明：Probability/horizon parameter bounds follow from a persistence witness and clock admission; they need not be introduced as additional physical axioms.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (Wd : CGB.World U Λ I),
  M.Clocked Wd.clock → ∀ {δ : ℝ} {T : ℕ}, M.Persists Wd δ T → 0 ≤ δ ∧ 0 < Wd.horizon ∧ Wd.recovery ≤ 1
```

## CGB.FoundationCounterexamples.unboundedStimulus_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L198)

解释：存在无界刺激反例系统完全满足微观介质的局部公理系统。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.FoundationCounterexamples.unboundedStimulus.Axioms
```

## CGB.FoundationCounterexamples.unboundedStimulus_no_clock

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L226)

解释：有限机体与状态空间并不自动蕴含全局有时钟速率上界（无界刺激反例构造）。

源码说明：Six clauses plus finite body/state/event types do not imply a global Clocked bound.

形式陈述：

```lean
∀ (R : ℝ), ¬CGB.FoundationCounterexamples.unboundedStimulus.Clocked R
```

## CGB.TheoryAudit.existence_survives_weakening

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L251)

解释：准入条件的放宽不会破坏已构造系统的存在性见证，确立弱化定理的单调传递性。

源码说明：Weakening admissibility cannot destroy an already constructed existence witness. Consequently a bad model is not a necessity proof for an existential life conjecture.

形式陈述：

```lean
∀ {S : Type u_1} (strong weak target : S → Prop),
  (∀ (x : S), strong x → weak x) → (∃ x, strong x ∧ target x) → ∃ x, weak x ∧ target x
```

## CGB.TheoryAudit.counterexample_blocks_guarantee

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Foundation.lean#L258)

解释：只要剩余前提下存在具体反例，即可彻底否定泛化存在性全称命题。

源码说明：A counterexample does refute a universal guarantee from the remaining premises.

形式陈述：

```lean
∀ {S : Type u_1} (remaining target : S → Prop), (∃ x, remaining x ∧ ¬target x) → ¬∀ (x : S), remaining x → target x
```
