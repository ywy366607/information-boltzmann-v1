# PortFlowHomology

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean) · [返回导航](../README.md)

模块源码说明：Finite flow chain for the separated port witness Each current state/input pair is a directed transition edge.

## CGB.PortFlowHomology.incidence_augmented_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean#L30)

解释：增广关联矩阵在同调链复合下满足闭合性，即端口边界链算子的二次复合作用为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.AlgebraicHomology.boundaryLaw CGB.PortFlowHomology.incidence CGB.PortFlowHomology.augmentation
```

## CGB.PortFlowHomology.weighted_incidence_augmented_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean#L43)

解释：加权增广关联矩阵与同调边界矩阵正交，确认了端口间流动的拓扑守恒性。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
CGB.AlgebraicHomology.boundaryLaw CGB.PortFlowHomology.weightedIncidence CGB.PortFlowHomology.augmentation
```

## CGB.PortFlowHomology.weighted_flow_boundary_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/PortFlowHomology.lean#L59)

解释：加权物质流在端口网络边界上的净代数发散恒等于零（局部电流守恒定律）。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (e : CGB.PortFlowHomology.Edge),
  CGB.AlgebraicHomology.matrixAction CGB.PortFlowHomology.augmentation
      (CGB.AlgebraicHomology.matrixAction CGB.PortFlowHomology.weightedIncidence fun e' => if e' = e then 1 else 0) () =
    0
```
