# GenerationDynamicTeaching

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationDynamicTeaching.lean) · [返回导航](../README.md)

模块源码说明：Dynamic boundary teaching has a necessary cross term `GenerationContactPlasticity` proves the current spatial pullback while its returned teaching coefficient is stopped.

## CGB.Generation.DynamicTeaching.teaching_response_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationDynamicTeaching.lean#L24)

解释：When boundary teaching and local contact response both vary, the derivative is teaching-motion times response plus teaching times local response; the latter alone is only a stopped-coefficient proxy.

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

解释：A frozen-teaching local derivative equals the coupled derivative exactly when the omitted teaching-motion times current response is zero.

源码说明：The stopped-teaching spatial term is complete exactly when its omitted teaching-motion product vanishes. This equality is the admission condition for reusing a frozen-coefficient local derivative as a coupled derivative.

形式陈述：

```lean
∀ (teachingAt responseAt teaching' response' : ℝ),
  teaching' * responseAt + teachingAt * response' = teachingAt * response' ↔ teaching' * responseAt = 0
```
