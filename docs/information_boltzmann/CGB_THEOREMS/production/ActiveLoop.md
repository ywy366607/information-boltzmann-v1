# ActiveLoop

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean) · [返回导航](../README.md)

模块源码说明：Proposed responsive loop and algebraic energy account The proved results are medium axioms, clock and boundary admission, world-row normalization, and an expected first-law identity.

## CGB.ActiveLoop.unique_vacuum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L48)

解释：主动环路状态中存在唯一的零成本真空基态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∃! l, CGB.ActiveLoop.cost l = 0
```

## CGB.ActiveLoop.body_clock

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L77)

解释：主动环路生命机体受速率为1的全局时钟严格节拍驱动。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.ActiveLoop.body.Clocked 1
```

## CGB.ActiveLoop.body_axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L91)

解释：主动环路生命机体完全满足局部非负性与概率守恒等公理体系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.ActiveLoop.body.Axioms
```

## CGB.ActiveLoop.fin6_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L122)

解释：六状态有限环境空间上任意实值函数的求和展开等价性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : CGB.ActiveLoop.WorldState → ℝ), ∑ w, f w = f 0 + f 1 + f 2 + f 3 + f 4 + f 5
```

## CGB.ActiveLoop.worldMoves_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L151)

解释：在机体状态作用下环境状态转移矩阵元素恒非负。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (w : CGB.ActiveLoop.WorldState) (x : Unit → CGB.ActiveLoop.State) (w' : CGB.ActiveLoop.WorldState),
  0 ≤ CGB.ActiveLoop.worldMoves w x w'
```

## CGB.ActiveLoop.worldMoves_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L155)

解释：环境在机体任意固定构型驱动下的转移概率归一性（行和为1）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (w : CGB.ActiveLoop.WorldState) (x : Unit → CGB.ActiveLoop.State), ∑ w', CGB.ActiveLoop.worldMoves w x w' = 1
```

## CGB.ActiveLoop.body_readsBoundary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L180)

解释：主动环路机体的内部演化仅读取环境边界接触信息而满足马尔可夫毯隔离。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.ActiveLoop.body.ReadsBoundary CGB.ActiveLoop.world
```

## CGB.ActiveLoop.active_expected_first_law

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/ActiveLoop.lean#L234)

解释：沿真实主动闭环耦合过程机体与环境的平均能量满足热力学第一定律（内能增量等于功、驱动力、物质流与热流之和）。

源码说明：Average first law holds along the actual active process.

形式陈述：

```lean
∀ (a w_act : ℝ) (cut : Bool) (initial : (Unit → CGB.ActiveLoop.State) × CGB.ActiveLoop.world.W) (t : ℕ),
  CGB.Energetics.mean (CGB.Energetics.law (CGB.ActiveLoop.jointMachine cut) initial (t + 1))
        ((CGB.ActiveLoop.activeAccount a w_act cut).stored (t + 1)) +
      CGB.Energetics.flux (CGB.ActiveLoop.jointMachine cut) initial (CGB.ActiveLoop.activeAccount a w_act cut).load t =
    CGB.Energetics.mean (CGB.Energetics.law (CGB.ActiveLoop.jointMachine cut) initial t)
            ((CGB.ActiveLoop.activeAccount a w_act cut).stored t) +
          CGB.Energetics.flux (CGB.ActiveLoop.jointMachine cut) initial (CGB.ActiveLoop.activeAccount a w_act cut).drive
            t +
        CGB.Energetics.flux (CGB.ActiveLoop.jointMachine cut) initial (CGB.ActiveLoop.activeAccount a w_act cut).matter
          t +
      CGB.Energetics.flux (CGB.ActiveLoop.jointMachine cut) initial (CGB.ActiveLoop.activeAccount a w_act cut).heat t
```
