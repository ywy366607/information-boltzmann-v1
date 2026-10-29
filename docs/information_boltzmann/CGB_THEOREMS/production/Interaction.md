# Interaction

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean) · [返回导航](../README.md)

模块源码说明：Finite probabilistic interaction coalgebras This is a semantic layer, not a physical axiom.

## CGB.Interaction.Machine.conditioned_next

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L33)

解释：当某次观测的总概率为正时，给定联合输出—后继状态核唯一确定归一化的条件后继分布。

源码说明：A positive-mass observation uniquely fixes the normalized next-state law. This conditions a given joint kernel; it does not identify its physical law.

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype O]
  (P : CGB.Interaction.Machine S I O) (i : I) (s : S) (o : O),
  0 < ∑ t, P.step i s o t →
    have mass := ∑ t, P.step i s o t;
    have post := fun t => P.step i s o t / mass;
    (∀ (t : S), 0 ≤ post t) ∧
      ∑ t, post t = 1 ∧ ∀ (candidate : S → ℝ), (∀ (t : S), mass * candidate t = P.step i s o t) → candidate = post
```

## CGB.Interaction.trace_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L54)

解释：对任意因果输入策略和有限输出序列，交互机器定义的轨迹权重非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype O]
  (P : CGB.Interaction.Machine S I O) (policy : CGB.Interaction.Policy I O) (history future : List O) (s : S),
  0 ≤ P.trace policy history future s
```

## CGB.Interaction.trace_prefix_consistent

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L64)

解释：对有限输出前缀的所有单符号扩展求和，恰好恢复该前缀的轨迹概率。

源码说明：Cylinder consistency: summing all one-symbol extensions recovers the prefix probability. Together with trace [] = 1 and nonnegativity this is a consistent family of finite observation laws, not just a collection of named numbers.

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype O]
  (P : CGB.Interaction.Machine S I O) (policy : CGB.Interaction.Policy I O) (history future : List O) (s : S),
  ∑ o, P.trace policy history (future ++ [o]) s = P.trace policy history future s
```

## CGB.Interaction.Coupling.expectation_at_output

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L98)

解释：若耦合只支持关系 R，且两个后继状态函数在 R 上相等，则同一输出的一步加权期望相等。

源码说明：One-step transfer of any observable continuation value. Nonnegativity is used to eliminate zero-mass pairs, not to assume the observable is positive.

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} {R : S → T → Prop}
  {s : S} {t : T} (c : CGB.Interaction.Coupling P Q R s t) (i : I) (o : O) (f : S → ℝ) (g : T → ℝ),
  (∀ (s' : S) (t' : T), R s' t' → f s' = g t') → ∑ s', P.step i s o s' * f s' = ∑ t', Q.step i t o t' * g t'
```

## CGB.Interaction.Bisimulation.trace_eq

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L115)

解释：互模拟关系关联的两个状态，在任意共同因果策略和有限输出轨迹下具有相同轨迹概率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} {R : S → T → Prop},
  CGB.Interaction.Bisimulation P Q R →
    ∀ (policy : CGB.Interaction.Policy I O) (history future : List O) {s : S} {t : T},
      R s t → P.trace policy history future s = Q.trace policy history future t
```

## CGB.Interaction.Bisimulation.sameBehavior

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L126)

解释：互模拟关系关联的状态满足所定义的所有有限输出柱事件行为相等。

源码说明：Coinductive soundness for observed infinite behaviour, represented by ALL finite cylinders. A finite sample of cylinders does not establish this premise.

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} {R : S → T → Prop},
  CGB.Interaction.Bisimulation P Q R → ∀ {s : S} {t : T}, R s t → CGB.Interaction.SameBehavior P Q s t
```

## CGB.Interaction.coinduction

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L130)

解释：若存在包含两状态的互模拟关系，则这两个状态按定义互模拟。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} {R : S → T → Prop},
  CGB.Interaction.Bisimulation P Q R → ∀ {s : S} {t : T}, R s t → CGB.Interaction.Bisimilar P Q s t
```

## CGB.Interaction.bisimilar_is_bisimulation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L133)

解释：所有互模拟状态对组成的关系本身也是互模拟关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O},
  CGB.Interaction.Bisimulation P Q (CGB.Interaction.Bisimilar P Q)
```

## CGB.Interaction.bisimilar_sameBehavior

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L138)

解释：两个状态互模拟时，在任意因果策略下的所有有限输出行为相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} {s : S} {t : T},
  CGB.Interaction.Bisimilar P Q s t → CGB.Interaction.SameBehavior P Q s t
```

## CGB.Interaction.distinguishing_trace

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L141)

解释：若某策略和有限轨迹给两个状态不同概率，则任何已给定互模拟关系都不能关联这两个状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} {R : S → T → Prop},
  CGB.Interaction.Bisimulation P Q R →
    ∀ (policy : CGB.Interaction.Policy I O) (history future : List O) {s : S} {t : T},
      P.trace policy history future s ≠ Q.trace policy history future t → ¬R s t
```

## CGB.Interaction.Refines.bisimulation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L152)

解释：若表示映射精确保留输出与后继状态的联合转移分布，则映射关系构成互模拟。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} [inst_3 : DecidableEq T]
  {f : S → T}, CGB.Interaction.Refines P Q f → CGB.Interaction.Bisimulation P Q fun s t => f s = t
```

## CGB.Interaction.Refines.sameBehavior

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L167)

解释：若一个机器通过给定映射精确细化另一个机器，则源状态与其映射像有相同输出行为。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} [inst_3 : DecidableEq T]
  {f : S → T}, CGB.Interaction.Refines P Q f → ∀ (s : S), CGB.Interaction.SameBehavior P Q s (f s)
```

## CGB.Interaction.Refines.refl

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L170)

解释：交互机器通过恒等状态映射精确细化自身。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype O]
  {P : CGB.Interaction.Machine S I O} [inst_2 : DecidableEq S], CGB.Interaction.Refines P P id
```

## CGB.Interaction.Refines.comp

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L175)

解释：两个精确细化证书可以沿状态映射复合，得到从原机器到最终机器的细化证书。

源码说明：Certificates compose, so a chain of representation changes does not need to recheck arbitrary long sample traces after each implementation pass.

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} {V : Type u_5}
  [inst_3 : Fintype V] [inst_4 : DecidableEq T] [inst_5 : DecidableEq V] {R : CGB.Interaction.Machine V I O} {f : S → T}
  {g : T → V}, CGB.Interaction.Refines P Q f → CGB.Interaction.Refines Q R g → CGB.Interaction.Refines P R (g ∘ f)
```

## CGB.Interaction.TimedRefinement.elapsed_eq

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L196)

解释：带时间的细化证书要求两时钟相等，因此相同步数所对应的物理时间相等。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} [inst_3 : DecidableEq T]
  {f : S → T} {a b : ℝ}, CGB.Interaction.TimedRefinement P Q f a b → ∀ (n : ℕ), ↑n / a = ↑n / b
```

## CGB.Interaction.Bisimulation.firstHit_eq

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L214)

解释：互模拟状态在相同策略、标记条件和有限期限下具有相同首次命中概率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} {R : S → T → Prop},
  CGB.Interaction.Bisimulation P Q R →
    ∀ (policy : CGB.Interaction.Policy I O) (mark : O → Bool) (history : List O) (k : ℕ) {s : S} {t : T},
      R s t → P.firstHit policy mark history k s = Q.firstHit policy mark history k t
```

## CGB.Interaction.Refines.firstHit_eq

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L230)

解释：精确细化映射保留任意共同策略、标记及期限的首次命中概率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} [inst_3 : DecidableEq T]
  {f : S → T},
  CGB.Interaction.Refines P Q f →
    ∀ (policy : CGB.Interaction.Policy I O) (mark : O → Bool) (history : List O) (k : ℕ) (s : S),
      P.firstHit policy mark history k s = Q.firstHit policy mark history k (f s)
```

## CGB.Interaction.Refines.recovery_test_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L237)

解释：精确细化同时保持状态内容时，源状态与目标状态的正内容及首次命中阈值测试等价。

源码说明：To preserve the existing nonempty-body + recovery-threshold test, content must ALSO be represented faithfully. Matching emitted words alone is insufficient.

形式陈述：

```lean
∀ {S : Type u_1} {T : Type u_2} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype T]
  [inst_2 : Fintype O] {P : CGB.Interaction.Machine S I O} {Q : CGB.Interaction.Machine T I O} [inst_3 : DecidableEq T]
  {f : S → T},
  CGB.Interaction.Refines P Q f →
    ∀ (contentS : S → ℕ) (contentT : T → ℕ),
      (∀ (s : S), contentS s = contentT (f s)) →
        ∀ (policy : CGB.Interaction.Policy I O) (mark : O → Bool) (history : List O) (k : ℕ) (p : ℝ) (s : S),
          0 < contentS s ∧ p ≤ P.firstHit policy mark history k s ↔
            0 < contentT (f s) ∧ p ≤ Q.firstHit policy mark history k (f s)
```

## CGB.Interaction.firstHit_bounds

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Interaction.lean#L246)

解释：交互机器按任意策略在有限期限内首次命中标记输出的概率处于零与一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_3} {O : Type u_4} [inst : Fintype S] [inst_1 : Fintype O]
  (P : CGB.Interaction.Machine S I O) (policy : CGB.Interaction.Policy I O) (mark : O → Bool) (history : List O) (k : ℕ)
  (s : S), 0 ≤ P.firstHit policy mark history k s ∧ P.firstHit policy mark history k s ≤ 1
```
