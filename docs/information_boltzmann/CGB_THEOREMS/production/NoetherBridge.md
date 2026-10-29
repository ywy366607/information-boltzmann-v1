# NoetherBridge

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean) · [返回导航](../README.md)

模块源码说明：Noether correspondence for the actual medium event language A closed-event restriction is an analysis object, not a change to the running medium.

## CGB.JumpProcess.observableSymmetry_iff_conserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L24)

解释：对于有限跳跃过程，可观测量 Q 的乘法算子与生成元交换，当且仅当每个正速率转移都保持 Q。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} [inst : Fintype S] (P : CGB.JumpProcess S) (Q : S → ℝ), P.ObservableSymmetry Q ↔ P.Conserved Q
```

## CGB.Medium.jump_gen_events

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L52)

解释：介质跳跃过程对任意观测函数的生成元等于逐事件的速率乘函数变化之和。

源码说明：Sum event contributions before aggregating transitions with the same output.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : Fintype (U → Λ)] [inst_2 : DecidableEq (U → Λ)] (s : U → I) (f : (U → Λ) → ℝ) (x : U → Λ),
  (M.jump s).gen f x = ∑ a, M.rate a s x * (f (M.act a s x) - f x)
```

## CGB.Medium.closedJump_conserved_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L80)

解释：固定刺激下，闭事件跳跃过程守恒 Q 当且仅当每个正速率闭事件都不改变 Q。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : DecidableEq U]
  [inst_1 : Fintype A] [inst_2 : DecidableEq (U → Λ)] (s : U → I) (Q : (U → Λ) → ℝ),
  (M.closedJump s).Conserved Q ↔ ∀ (a : A) (x : U → Λ), M.Closed a → 0 < M.rate a s x → Q (M.act a s x) = Q x
```

## CGB.Medium.p2a_iff_closed_conserved

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L120)

解释：介质的 P2a 闭事件内容守恒条件等价于任意刺激下闭事件跳跃过程守恒总内容。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : DecidableEq Λ] [inst_3 : Fintype A],
  M.P2a ↔ ∀ (s : U → I), (M.closedJump s).Conserved M.contentObservable
```

## CGB.Medium.p2a_iff_closed_moments

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L133)

解释：P2a 等价于各刺激下闭事件生成元对总内容及其平方均为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A],
  M.P2a ↔
    ∀ (s : U → I),
      (∀ (x : U → Λ), (M.closedJump s).gen M.contentObservable x = 0) ∧
        ∀ (x : U → Λ), (M.closedJump s).gen (fun y => M.contentObservable y ^ 2) x = 0
```

## CGB.Medium.p2a_iff_closed_expected_drifts

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L143)

解释：P2a 等价于各刺激、任意归一非负初始分布下，闭事件的总内容一阶与二阶矩瞬时漂移均为零。

源码说明：Every normalized initial distribution has zero first and second moment DRIFT. Constancy along a time evolution follows when that evolution solves the forward equation; this statement does not introduce a continuous semigroup.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A],
  M.P2a ↔
    ∀ (s : U → I),
      (∀ (p : (U → Λ) → ℝ),
          (∀ (x : U → Λ), 0 ≤ p x) → ∑ x, p x = 1 → ∑ x, p x * (M.closedJump s).gen M.contentObservable x = 0) ∧
        ∀ (p : (U → Λ) → ℝ),
          (∀ (x : U → Λ), 0 ≤ p x) →
            ∑ x, p x = 1 → ∑ x, p x * (M.closedJump s).gen (fun y => M.contentObservable y ^ 2) x = 0
```

## CGB.Medium.p2a_iff_closed_observable_symmetry

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L155)

解释：P2a 等价于各刺激下闭事件生成元与总内容乘法算子交换。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A],
  M.P2a ↔ ∀ (s : U → I), (M.closedJump s).ObservableSymmetry M.contentObservable
```

## CGB.Medium.content_drift_boundary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/NoetherBridge.lean#L164)

解释：在 P2a 成立时，开放介质中任何总内容函数的生成元漂移都可写为触及边界事件的贡献之和。

源码说明：In the original OPEN generator, all drift of any function of total content comes from boundary-touching events. This is an event balance, not a heat or negentropy identification.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U]
  [inst_1 : DecidableEq U] [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A],
  M.P2a →
    ∀ (s : U → I) (g : ℕ → ℝ) (x : U → Λ),
      (M.jump s).gen (fun y => g (M.total y)) x =
        ∑ a, if M.Closed a then 0 else M.rate a s x * (g (M.total (M.act a s x)) - g (M.total x))
```
