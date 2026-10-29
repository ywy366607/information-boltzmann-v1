# GenerationOrganismEcoliMsS0

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean) · [返回导航](../README.md)

模块源码说明：Organism `ecoli_ms_s0`: generated from its rule table Source: `information_boltzmann/organisms/ecoli_ms_s0.json`, sha256 38ff4daa364a8d382398d91208cc7908b61c82d1a1324e64111291ef4801fa1b.

## CGB.Organisms.EcoliMsS0.axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L52)

解释：大肠杆菌 MsS0 基础形态发生模型实例严格满足 CGB 物理介质六公理。

源码说明：**The organism is a model of the CGB axioms** (M3: unique free state, and every rule between two interior pages conserves content).

形式陈述：

```lean
CGB.Organisms.EcoliMsS0.medium.Axioms
```

## CGB.Organisms.EcoliMsS0.metabolizes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L58)

解释：大肠杆菌 MsS0 模型存在非零动作能够改变局部机体质量总量，具备真实新陈代谢能力。

源码说明：**Life gate L-metabolism** (`Lives.metabolizes`): rule 0 on units 1024 and 1735 changes content.

形式陈述：

```lean
∃ a s x,
  0 < CGB.Organisms.EcoliMsS0.medium.rate a s x ∧
    CGB.Organisms.EcoliMsS0.medium.total (CGB.Organisms.EcoliMsS0.medium.act a s x) ≠
      CGB.Organisms.EcoliMsS0.medium.total x
```

## CGB.Organisms.EcoliMsS0.exposed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismEcoliMsS0.lean#L65)

解释：大肠杆菌 MsS0 模型具有暴露于外部世界驱动的有效外边界通道。

源码说明：**Life gate L-sense** (`Lives.exposed`): the same event exchanges content with a boundary reservoir, so its rate depends on the stimulus.

形式陈述：

```lean
CGB.Organisms.EcoliMsS0.medium.Exposed
```
