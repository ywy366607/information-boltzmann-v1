# InteractionControls

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean) · [返回导航](../README.md)

模块源码说明：Semantic certificates and obstructions These mathematical controls exercise exact representation, not brain capacity.

## CGB.InteractionControls.independent_bit_refines

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L29)

解释：当系统扩展位独立演化时，将其投影折叠构成对原机体的严格模拟细化（Refines）。

源码说明：Removing a bit is justified HERE by a universal transition equation. No conclusion about ignoring an arbitrary biological cell/state follows from this.

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_2} {O : Type u_3} [inst : Fintype S] [inst_1 : Fintype O] [inst_2 : DecidableEq S]
  (P : CGB.Interaction.Machine S I O), CGB.Interaction.Refines (CGB.InteractionControls.withIndependentBit P) P Prod.fst
```

## CGB.InteractionControls.independent_bit_sameBehavior

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L35)

解释：忽略独立的无关状态位不会改变机体在所有输入序列下的可观测行为等价性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ {S : Type u_1} {I : Type u_2} {O : Type u_3} [inst : Fintype S] [inst_1 : Fintype O]
  (P : CGB.Interaction.Machine S I O) (s : S) (b : Bool),
  CGB.Interaction.SameBehavior (CGB.InteractionControls.withIndependentBit P) P (s, b) s
```

## CGB.InteractionControls.same_first_observation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L46)

解释：交替机与常数机在空历史下的首次可观测输出概率分布完全相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.InteractionControls.alternating.trace (fun x => ()) [] [false] false = 1 ∧
  CGB.InteractionControls.constant.trace (fun x => ()) [] [false] false = 1
```

## CGB.InteractionControls.different_future

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L51)

解释：交替机与常数机在后续第二步输出概率上产生发散，展示仅凭单步观测无法区分内部动力学。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.InteractionControls.alternating.trace (fun x => ()) [] [false, true] false = 1 ∧
  CGB.InteractionControls.constant.trace (fun x => ()) [] [false, true] false = 0
```

## CGB.InteractionControls.no_bisimulation_from_first_observation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L57)

解释：基于首步单一观测所建立的等价关系不能提升为系统在未来交互流上的完全双互模拟。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Interaction.Bisimilar CGB.InteractionControls.alternating CGB.InteractionControls.constant false false
```

## CGB.InteractionControls.physical_certificate_rejects_time_rescaling

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L63)

解释：物理能量证书在时钟尺度变换下破缺，确立物理时钟不能被任意时间重参数化替代。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
¬CGB.Interaction.TimedRefinement CGB.InteractionControls.alternating CGB.InteractionControls.alternating id 1 2
```

## CGB.InteractionControls.actual_recovery_bridge

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InteractionControls.lean#L71)

解释：在受检的真实恢复子空间上该表征映射为恒等嵌入，构成了形式验证与真实行为的一致性桥梁。

源码说明：The tested representation is the identity on the actual continuous state; the bridge must work for BOTH normal and cut futures with the same world clock.

形式陈述：

```lean
∀ {U : Type u_4} {Λ : Type u_5} {A : Type u_6} {J : Type u_7} [inst : Fintype U] [inst_1 : DecidableEq U]
  [inst_2 : Fintype Λ] [inst_3 : DecidableEq Λ] [inst_4 : Fintype A] (M : CGB.Medium U Λ A J) (Wd : CGB.World U Λ J)
  (hc : M.Clocked Wd.clock) (cut : Bool) (sched : ℕ → Wd.W → U → J) (history : List Bool) (k : ℕ) (z : (U → Λ) × Wd.W),
  (M.intakeCoalgebra Wd hc cut sched).firstHit List.length id history k z =
    CGB.Operational.recovery M Wd hc cut sched history.length k z
```
