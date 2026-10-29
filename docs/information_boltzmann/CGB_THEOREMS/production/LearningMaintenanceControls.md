# LearningMaintenanceControls

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean) · [返回导航](../README.md)

模块源码说明：A paired write-disabled control: same initial fuel, sensing, world and clock.

## _private.CGBAxioms.LearningMaintenanceControls.0.CGB.LearningMaintenanceControls.sum_configs

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L27)

解释：对单位置、四种主体状态的任意实函数求和，等于其在四个常值构型上的值之和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : (Unit → CGB.LearningMaintenance.State) → ℝ),
  ∑ x, f x = (((f fun x => 0) + f fun x => 1) + f fun x => 2) + f fun x => 3
```

## _private.CGBAxioms.LearningMaintenanceControls.0.CGB.LearningMaintenanceControls.kernel_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L36)

解释：给出机体转移核在写禁用模式下的完整状态转移查找表。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s : Unit → CGB.LearningMaintenance.Stim) (x y : CGB.LearningMaintenance.State),
  (CGB.LearningMaintenance.frozen.kernel 1 s (fun x_1 => x) fun x => y) =
    if y = CGB.LearningMaintenance.frozenNext (s ()) x then 1 else 0
```

## _private.CGBAxioms.LearningMaintenanceControls.0.CGB.LearningMaintenanceControls.policy_value

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L53)

解释：给出冻结策略在各个机体构型下输出动作的确定性取值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.LearningMaintenance.State), CGB.LearningMaintenance.policy q = if q = 2 then 0 else if q = 3 then 1 else 2
```

## _private.CGBAxioms.LearningMaintenanceControls.0.CGB.LearningMaintenanceControls.frozen_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L59)

解释：给出在环境平稳输入驱动下冻结机体各步联合状态分布的显式解析解。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ) (a : CGB.LearningMaintenance.State) (w : CGB.LearningMaintenance.W),
  CGB.LearningMaintenance.frozen.law CGB.LearningMaintenance.world (fun x => CGB.LearningMaintenance.world.sense) n
      (fun x => a) w =
    CGB.LearningMaintenanceControls.density✝ n a w
```

## _private.CGBAxioms.LearningMaintenanceControls.0.CGB.LearningMaintenanceControls.starve_recovery

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L85)

解释：指定冻结对照在世界状态 5 下对任意主体状态和有限期限的交换恢复概率为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (t k : ℕ) (a : CGB.LearningMaintenance.State),
  CGB.Exchange.recovery CGB.LearningMaintenance.frozen CGB.LearningMaintenance.world CGB.LearningMaintenanceControls.hf
      CGB.LearningMaintenanceControls.ledger false (fun x => CGB.LearningMaintenance.world.sense) t k (fun x => a, 5) =
    0
```

## _private.CGBAxioms.LearningMaintenanceControls.0.CGB.LearningMaintenanceControls.starve_alive

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L98)

解释：指定冻结对照在世界状态 5 下的交换生命指标对任意时刻和主体状态均为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (n : ℕ) (a : CGB.LearningMaintenance.State),
  CGB.Exchange.alive CGB.LearningMaintenance.frozen CGB.LearningMaintenance.world CGB.LearningMaintenanceControls.hf
      CGB.LearningMaintenanceControls.ledger false (fun x => CGB.LearningMaintenance.world.sense) n (fun x => a, 5) =
    0
```

## CGB.LearningMaintenanceControls.frozen_survival

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L108)

解释：指定冻结学习对照在自然世界中从第 3 拍起交换生存评分为零。

源码说明：With writing disabled the initial two fuel units cannot preserve access to food. Viability is lost at tick three; the remaining body empties one tick later.

形式陈述：

```lean
∀ (n : ℕ),
  3 ≤ n →
    CGB.Exchange.survival CGB.LearningMaintenance.frozen CGB.LearningMaintenance.world
        CGB.LearningMaintenanceControls.hf CGB.LearningMaintenanceControls.ledger false
        (fun x => CGB.LearningMaintenance.world.sense) n =
      0
```

## CGB.LearningMaintenanceControls.frozen_not_persists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/LearningMaintenanceControls.lean#L126)

解释：指定冻结学习对照在其给定世界和账本下，对任意实数 δ 都不满足交换持续性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (δ : ℝ),
  ¬CGB.Exchange.Persists CGB.LearningMaintenance.frozen CGB.LearningMaintenance.world CGB.LearningMaintenanceControls.hf
      CGB.LearningMaintenanceControls.ledger δ
```
