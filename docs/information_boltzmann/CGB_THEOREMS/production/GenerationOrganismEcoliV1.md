# GenerationOrganismEcoliV1

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean) · [返回导航](../README.md)

模块源码说明：Organism `ecoli_v1`: generated from its rule table Source: `information_boltzmann/organisms/ecoli_v1.json`, sha256 4ab0545552e85d06779b9d1da4a1c49b88bd976203fae895833f9a148ccb6a65.

## CGB.Organisms.EcoliV1.axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L52)

解释：大肠杆菌 V1 演化形态模型实例严格满足 CGB 物理介质六公理。

源码说明：**The organism is a model of the CGB axioms** (M3: unique free state, and every rule between two interior pages conserves content).

形式陈述：

```lean
CGB.Organisms.EcoliV1.medium.Axioms
```

## CGB.Organisms.EcoliV1.metabolizes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L58)

解释：大肠杆菌 V1 模型存在能够净改变机体分子数总量的代谢反应路径。

源码说明：**Life gate L-metabolism** (`Lives.metabolizes`): rule 0 on units 17 and 1 changes content.

形式陈述：

```lean
∃ a s x,
  0 < CGB.Organisms.EcoliV1.medium.rate a s x ∧
    CGB.Organisms.EcoliV1.medium.total (CGB.Organisms.EcoliV1.medium.act a s x) ≠ CGB.Organisms.EcoliV1.medium.total x
```

## CGB.Organisms.EcoliV1.exposed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliV1.lean#L65)

解释：大肠杆菌 V1 模型具备与外界环境进行能量与信息交换的暴露边界。

源码说明：**Life gate L-sense** (`Lives.exposed`): the same event exchanges content with a boundary reservoir, so its rate depends on the stimulus.

形式陈述：

```lean
CGB.Organisms.EcoliV1.medium.Exposed
```
