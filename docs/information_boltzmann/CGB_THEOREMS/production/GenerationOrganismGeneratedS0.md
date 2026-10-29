# GenerationOrganismGeneratedS0

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean) · [返回导航](../README.md)

模块源码说明：Organism `generated_s0`: generated from its rule table Source: `information_boltzmann/organisms/generated_s0.json`, sha256 d57065ecec7500d38a2b40e1ef94f9e8412f45e47f7a7df63d20421010ccd6b0.

## CGB.Organisms.GeneratedS0.axioms

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L50)

解释：从第一性原理生成的生命形态 S0 实例严格满足 CGB 物理介质六公理。

源码说明：**The organism is a model of the CGB axioms** (M3: unique free state, and every rule between two interior pages conserves content).

形式陈述：

```lean
CGB.Organisms.GeneratedS0.medium.Axioms
```

## CGB.Organisms.GeneratedS0.metabolizes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L56)

解释：生成的 S0 生命形态存在正速率动作驱动质量流入流出，证实具有自维持代谢能力。

源码说明：**Life gate L-metabolism** (`Lives.metabolizes`): rule 0 on units 14 and 0 changes content.

形式陈述：

```lean
∃ a s x,
  0 < CGB.Organisms.GeneratedS0.medium.rate a s x ∧
    CGB.Organisms.GeneratedS0.medium.total (CGB.Organisms.GeneratedS0.medium.act a s x) ≠
      CGB.Organisms.GeneratedS0.medium.total x
```

## CGB.Organisms.GeneratedS0.exposed

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationOrganismGeneratedS0.lean#L63)

解释：生成的 S0 生命形态具有非空的外部感官与环境相互作用边界。

源码说明：**Life gate L-sense** (`Lives.exposed`): the same event exchanges content with a boundary reservoir, so its rate depends on the stimulus.

形式陈述：

```lean
CGB.Organisms.GeneratedS0.medium.Exposed
```
