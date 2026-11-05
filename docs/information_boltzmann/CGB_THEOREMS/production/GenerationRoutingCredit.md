# GenerationRoutingCredit

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationRoutingCredit.lean) · [返回导航](../README.md)

模块源码说明：Finite sparse-route credit A fixed sparse support is a finite boundary alphabet.

## CGB.Generation.log_softmax_direction_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationRoutingCredit.lean#L28)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：任意有限方向向量下的对数 Softmax 一阶方向导数具有精确的中心化内积形式，支持选定路径对其固定支撑集上全部 logits 的协同更新。

源码说明：The log-softmax derivative for an arbitrary finite direction. G14b is the one-coordinate instance; a selected route may change all of its fixed support logits together.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (logits direction : B → ℝ) (target : B),
  HasDerivAt (fun θ => Real.log (CGB.Medium.softmax (fun b => logits b + θ * direction b) target))
    (direction target - ∑ b, CGB.Medium.softmax logits b * direction b) 0
```

## CGB.Generation.softmax_direction_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationRoutingCredit.lean#L72)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：任意选定路由的发生概率在方向微扰下具有解析的标准 Softmax 响应导数形式。

源码说明：Each selected route probability has the expected softmax response.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (logits direction : B → ℝ) (target : B),
  HasDerivAt (fun θ => CGB.Medium.softmax (fun b => logits b + θ * direction b) target)
    (CGB.Medium.softmax logits target * (direction target - ∑ b, CGB.Medium.softmax logits b * direction b)) 0
```

## CGB.Generation.softmax_read_direction_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationRoutingCredit.lean#L98)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：路由读出向量的方向导数等于其各局部中心化 Softmax 响应的总和，精确对应即时评分固定路由的回执导数。

源码说明：A route read changes by the sum of its local, centered softmax responses. This is the exact receipt used by an immediately scored fixed-support route.

形式陈述：

```lean
∀ {B : Type u_1} [inst : Fintype B] [Nonempty B] (logits direction value : B → ℝ),
  HasDerivAt (fun θ => CGB.Generation.softmaxRead (fun b => logits b + θ * direction b) value)
    (∑ b, CGB.Medium.softmax logits b * (direction b - ∑ c, CGB.Medium.softmax logits c * direction c) * value b) 0
```
