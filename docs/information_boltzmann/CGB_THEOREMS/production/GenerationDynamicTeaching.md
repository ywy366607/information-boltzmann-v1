# GenerationDynamicTeaching

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationDynamicTeaching.lean) · [返回导航](../README.md)

模块源码说明：Dynamic boundary teaching has a necessary cross term `GenerationContactPlasticity` proves the current spatial pullback while its returned teaching coefficient is stopped.

## CGB.Generation.DynamicTeaching.teaching_response_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDynamicTeaching.lean#L24)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当边界教学信号与局部接触响应同时变动时，总导数等于教学变动乘响应加上教学值乘局部响应导数；仅保留后者仅构成固定系数的局部代理。

源码说明：A boundary teaching coefficient times a contact response. When both vary with the same parameter, its derivative contains both the local spatial term and the teaching-motion term.

形式陈述：

```lean
∀ (teaching response : ℝ → ℝ) (x teaching' response' : ℝ),
  HasDerivAt teaching teaching' x →
    HasDerivAt response response' x →
      HasDerivAt (fun θ => teaching θ * response θ) (teaching' * response x + teaching x * response') x
```

## CGB.Generation.DynamicTeaching.stopped_teaching_complete_iff

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDynamicTeaching.lean#L35)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：冻结教学信号的局部导数与全耦合复合导数严格相等，当且仅当被省略的教学信号变动量与当前接触响应的乘积精确为零。

源码说明：The stopped-teaching spatial term is complete exactly when its omitted teaching-motion product vanishes. This equality is the admission condition for reusing a frozen-coefficient local derivative as a coupled derivative.

形式陈述：

```lean
∀ (teachingAt responseAt teaching' response' : ℝ),
  teaching' * responseAt + teachingAt * response' = teachingAt * response' ↔ teaching' * responseAt = 0
```
