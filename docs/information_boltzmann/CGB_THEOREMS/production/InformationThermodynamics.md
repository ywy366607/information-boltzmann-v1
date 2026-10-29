# InformationThermodynamics

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean) · [返回导航](../README.md)

模块源码说明：Information acquisition as a conditional thermodynamic contract The general medium axioms do not assign physical energy to a recorded symbol.

## CGB.InformationThermodynamics.erased_information_floor

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean#L38)

解释：记录器合同直接要求耗散不低于温度、擦除成本和擦除位数的乘积。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (r : CGB.InformationThermodynamics.RecorderContract), r.bathTemperature * r.erasureCost * r.erasedBits ≤ r.dissipation
```

## CGB.InformationThermodynamics.positive_erasure_requires_positive_dissipation

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean#L42)

解释：在记录器合同的热力学下界下，正擦除位数和正单位擦除成本蕴含正耗散。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (r : CGB.InformationThermodynamics.RecorderContract), 0 < r.erasureCost → 0 < r.erasedBits → 0 < r.dissipation
```

## CGB.InformationThermodynamics.efficient_recorder_is_not_free

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/InformationThermodynamics.lean#L51)

解释：满足合同的帕累托高效记录器若擦除正位数且擦除成本为正，耗散仍严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (r : CGB.InformationThermodynamics.RecorderContract),
  0 < r.erasedBits → 0 < r.erasureCost → CGB.InformationThermodynamics.ParetoEfficient r → 0 < r.dissipation
```
