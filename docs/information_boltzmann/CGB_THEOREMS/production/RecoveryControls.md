# RecoveryControls

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean) · [返回导航](../README.md)

模块源码说明：Recovery semantic controls (not organism capability experiments) Identical body and stimulus menus; only world accessibility differs.

## CGB.RecoveryControls.sum_functions

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L41)

解释：对单位置布尔状态的任意实函数求和等于函数在假和真两种状态的值之和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : (Unit → Bool) → ℝ), ∑ x, f x = (f fun x => false) + f fun x => true
```

## CGB.RecoveryControls.body_kernel

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L50)

解释：该布尔身体在速率一时的一步转移核由后继状态是否等于当前状态或刺激给出。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (s x y : Unit → Bool), CGB.RecoveryControls.body.kernel 1 s x y = if y () = (x () || s ()) then 1 else 0
```

## CGB.RecoveryControls.body_clock

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L60)

解释：指定恢复对照身体满足速率一的离散时钟条件。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.RecoveryControls.body.Clocked 1
```

## CGB.RecoveryControls.oracle_ignores_access

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L72)

解释：在该对照中，旧版有利刺激菜单的恢复评分对两个可达性不同的世界仍相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (k : ℕ) (x : Unit → Bool),
  CGB.RecoveryControls.body.recover (CGB.RecoveryControls.world false) k x =
    CGB.RecoveryControls.body.recover (CGB.RecoveryControls.world true) k x
```

## CGB.RecoveryControls.actual_recovery_distinguishes_world_moves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/RecoveryControls.lean#L78)

解释：指定身体与同一状态下，两个不同世界转移核在两拍内的真实恢复概率不同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.RecoveryControls.body.actualRecover (CGB.RecoveryControls.world false) CGB.RecoveryControls.body_clock 2
      (fun x => false) false =
    0 ∧
  CGB.RecoveryControls.body.actualRecover (CGB.RecoveryControls.world true) CGB.RecoveryControls.body_clock 2
      (fun x => false) false =
    1
```
