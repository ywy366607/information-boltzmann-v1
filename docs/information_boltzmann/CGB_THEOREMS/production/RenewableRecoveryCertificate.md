# RenewableRecoveryCertificate

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean) · [返回导航](../README.md)

模块源码说明：Instantiate the reusable local recovery criterion on the existing model.

## _private.CGBAxioms.RenewableRecoveryCertificate.0.CGB.RenewableRecoveryCertificate.policy_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L12)

解释：给出可再生恢复证书中各机体构型对应输出动作控制律的分段显式查找表。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell),
  CGB.RenewableLearning.policy q =
    match q with
    | CGB.RenewableLearning.Cell.aSpent => 0
    | CGB.RenewableLearning.Cell.aReady => 0
    | CGB.RenewableLearning.Cell.aHeld => 0
    | CGB.RenewableLearning.Cell.bSpent => 1
    | CGB.RenewableLearning.Cell.bReady => 1
    | CGB.RenewableLearning.Cell.bHeld => 1
    | x => 2
```

## _private.CGBAxioms.RenewableRecoveryCertificate.0.CGB.RenewableRecoveryCertificate.nonvacuum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L19)

解释：有效机体在支持集内的状态均具有非零正维持成本，从而排除平凡真空状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
  CGB.RenewableLearningLife.Good q w → q ≠ CGB.RenewableLearning.Cell.vacuum
```

## _private.CGBAxioms.RenewableRecoveryCertificate.0.CGB.RenewableRecoveryCertificate.active

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L22)

解释：在非空有效生存态下机体必定处于激活工作状态并产生可观测动作。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (w : CGB.RenewableLearning.Env), CGB.RenewableLearning.sense w () ≠ CGB.RenewableLearning.Signal.calm
```

## _private.CGBAxioms.RenewableRecoveryCertificate.0.CGB.RenewableRecoveryCertificate.next_good

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L26)

解释：良性状态在一步环境交互演化后仍以概率1保持在良性恢复集合中。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
  CGB.RenewableLearningLife.Good q w →
    0 < CGB.RenewableLearning.moves w (fun x => q) v →
      CGB.RenewableLearningLife.Good (CGB.RenewableLearning.next true true (CGB.RenewableLearning.sense w ()) q) v
```

## _private.CGBAxioms.RenewableRecoveryCertificate.0.CGB.RenewableRecoveryCertificate.supported_good

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L32)

解释：沿任意支持的交互历史演化，机体状态序列始终严格受限于良性状态集合内部。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t : ℕ) (z z' : CGB.RenewableLearning.S) (e : Option Unit),
  CGB.RenewableLearningLife.Good (z.1 ()) z.2 →
    0 < (CGB.RenewableLearning.process true true false).step t z e z' → CGB.RenewableLearningLife.Good (z'.1 ()) z'.2
```

## _private.CGBAxioms.RenewableRecoveryCertificate.0.CGB.RenewableRecoveryCertificate.unmarked_progress

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L55)

解释：未标记的微观状态进展保持持续的局部能量平衡与生存通量传递。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w v : CGB.RenewableLearning.Env),
  CGB.RenewableLearningLife.Good q w →
    0 < CGB.RenewableLearning.moves w (fun x => q) v →
      ¬0 < CGB.RenewableLearning.incoming true (CGB.RenewableLearning.sense w ()) q →
        (if v = CGB.RenewableLearning.Env.missA ∨ v = CGB.RenewableLearning.Env.missB then 1 else 0) <
          if w = CGB.RenewableLearning.Env.missA ∨ w = CGB.RenewableLearning.Env.missB then 1 else 0
```

## CGB.RenewableRecoveryCertificate.survival_by_criterion

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RenewableRecoveryCertificate.lean#L94)

解释：由局域可再生证书可严格导出机体全时期生存概率下界，完成了局域物理证书到全时生存性的逻辑闭环。

源码说明：The original all-time score now also follows from the shared local criterion; it is not inserted as a premise of that criterion.

形式陈述：

```lean
∀ (n : ℕ),
  CGB.Exchange.survival (CGB.RenewableLearning.plant true true) CGB.RenewableLearning.world ⋯
      (CGB.RenewableLearning.ledger true true) false (fun x => CGB.RenewableLearning.world.sense) n =
    1
```
