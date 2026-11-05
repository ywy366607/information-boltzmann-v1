# GenerationContactPlasticity

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean) · [返回导航](../README.md)

模块源码说明：Compartment tags modulate a current-event spatial pullback The coefficient `c = lambda * eligibility` is fixed during this event's parameter differentiation.

## CGB.Generation.ContactPlasticity.linear_contact_return

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L25)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：转置有限线性接触读出所得到的标量收缩算子与原前向读出伴随完全一致；反向返回系数直接对应同一物理接触权重，而非独立的外部反馈矩阵。

源码说明：Transposing a finite linear contact read returns exactly the same scalar contraction; the return is not an independently learned feedback matrix.

形式陈述：

```lean
∀ {I : Type u_1} {J : Type u_2} [inst : Fintype I] [inst_1 : Fintype J] (post : I → ℝ) (contact : I → J → ℝ)
  (pre : J → ℝ), ∑ i, post i * ∑ j, contact i j * pre j = ∑ j, pre j * ∑ i, contact i j * post i
```

## CGB.Generation.ContactPlasticity.periodic_flux_return

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L39)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：对于任意有限周期几何上的流入减流出通量，其伴随算子精确等于后继减去局部的前向差分；周期置换的前驱即其逆映射。

源码说明：Incoming-minus-outgoing periodic flux has the next-minus-local adjoint. `next.symm` is the predecessor, so this covers any finite periodic geometry.

形式陈述：

```lean
∀ {I : Type u_1} [inst : Fintype I] (next : Equiv.Perm I) (v flow : I → ℝ),
  ∑ i, v i * (flow ((Equiv.symm next) i) - flow i) = ∑ i, (v (next i) - v i) * flow i
```

## CGB.Generation.ContactPlasticity.two_direction_flux_return

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L53)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：两个传输方向上的局部通量伴随可直接线性相加，无需保存历史传输轨迹或执行过去的伴随反传。

源码说明：Two transport directions add their local adjoints without retaining a transport trajectory or its past pullbacks.

形式陈述：

```lean
∀ {I : Type u_1} [inst : Fintype I] (p q : Equiv.Perm I) (v f g : I → ℝ),
  ∑ i, v i * (f ((Equiv.symm p) i) - f i + (g ((Equiv.symm q) i) - g i)) =
    ∑ i, ((v (p i) - v i) * f i + (v (q i) - v i) * g i)
```

## CGB.Generation.ContactPlasticity.tanh_offset_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L66)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：双曲正切函数在任意实数基值偏移下的局部一阶导数精确等于一减去该点双曲正切值的平方。

源码说明：The actual bounded transport law has the exact local factor used by the runtime: an infinitesimal change of a pre-flow changes `tanh flow` by `1 - tanh(flow)^2`. Combined with `periodic_flux_return`, this is a purely current-event pullback through the conservative flow, not a stored temporal gradient.

形式陈述：

```lean
∀ (raw : ℝ), HasDerivAt (fun θ => Real.tanh (raw + θ)) (1 - Real.tanh raw ^ 2) 0
```

## CGB.Generation.ContactPlasticity.allocation_normalization_cancels

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L95)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当总驱动量非零时，准入有效驱动项与归一化目标项中的总驱动因子精确相消，证明两步归一化代数形式在任何正容量除数下均保持不变。

源码说明：Positive total drive cancels between admission and normalized target. No assumption on the capacity divisor is needed for this algebraic identity.

形式陈述：

```lean
∀ (active totalDrive capacity drive : ℝ),
  totalDrive ≠ 0 → active * (totalDrive / capacity) * (drive / totalDrive) = active * drive / capacity
```

## CGB.Generation.ContactPlasticity.normalized_actual_pool

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L105)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在实际共享池中，总驱动量非零时归一化因子的代数对消依然严格成立，旧占用分布与逐坐标释放率保持不变。

源码说明：The cancellation holds inside the actual shared reservoir, with its old occupation and coordinatewise release unchanged.

形式陈述：

```lean
∀ {n : ℕ} (occupied drive release : Fin n → ℝ) (active capacity gate : ℝ),
  ∑ j, drive j ≠ 0 →
    ∀ (i : Fin n),
      CGB.Generation.sharedContactPoolVectorStep occupied (fun j => drive j / ∑ k, drive k) release
          (active * ((∑ j, drive j) / capacity)) gate i =
        occupied i + gate * (active * drive i / capacity) * (1 - ∑ j, occupied j) - gate * release i * occupied i
```

## CGB.Generation.ContactPlasticity.sigmoid_runtime_formula

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L128)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：标准 Sigmoid 函数的值域和代数形式严格等于数值运行时的 logistic 表达式 1 / (1 + exp(-x))。

源码说明：Mathlib's sigmoid is exactly the runtime logistic expression.

形式陈述：

```lean
∀ (x : ℝ), x.sigmoid = 1 / (1 + Real.exp (-x))
```

## CGB.Generation.ContactPlasticity.endpoint_allocation_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L134)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在端点分配中，关于单侧参数扰动在原点处的一阶导数精确等于均值 Sigmoid 导数形式乘以 pre/2。

源码说明：Exact allocation response to one endpoint's local pre-weight perturbation; `pre` is fixed and the sigmoid supplies the current post-response factor.

形式陈述：

```lean
∀ (left right pre : ℝ),
  HasDerivAt (fun θ => CGB.Generation.ContactPlasticity.endpointAllocation (left + θ * pre) right)
    (((left + right) / 2).sigmoid * (1 - ((left + right) / 2).sigmoid) * (pre / 2)) 0
```

## CGB.Generation.ContactPlasticity.endpoint_retention_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L145)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在端点保持中，关于单侧参数扰动在原点处的一阶导数精确等于单侧 Sigmoid 导数形式乘以 pre/2。

源码说明：Exact mean-retention response at one perturbed endpoint. Unlike allocation, the sigmoid is applied before the two endpoints are averaged.

形式陈述：

```lean
∀ (left right pre : ℝ),
  HasDerivAt (fun θ => CGB.Generation.ContactPlasticity.endpointRetention (left + θ * pre) right)
    (left.sigmoid * (1 - left.sigmoid) * pre / 2) 0
```

## CGB.Generation.ContactPlasticity.local_potential_zero_eligibility

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L163)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当所有槽位的参与资格标记均为零时，整个依赖参数的局部更新代理量在全空间恒等于零，而非仅在选定参数点处为零。

源码说明：Vanishing participation tags make the entire parameter-dependent proxy identically zero, not merely its value at a chosen parameter.

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ) (lambda eligibility : Fin n → ℝ),
  (∀ (i : Fin n), eligibility i = 0) →
    CGB.Generation.ContactPlasticity.localPotential occupied target release inflow gate lambda eligibility = 0
```

## CGB.Generation.ContactPlasticity.local_potential_zero_return

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L169)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当当拍局部语言返回标量为零时，具有停止梯度的局部更新代理项全部分量恒等于零。

源码说明：A zero current local return also makes the stopped-coefficient proxy zero.

形式陈述：

```lean
∀ {n : ℕ} (occupied target release : Fin n → ℝ) (inflow gate : ℝ) (lambda eligibility : Fin n → ℝ),
  (∀ (i : Fin n), lambda i = 0) →
    CGB.Generation.ContactPlasticity.localPotential occupied target release inflow gate lambda eligibility = 0
```

## CGB.Generation.ContactPlasticity.local_potential_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L177)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：当前事件通过实际流入与逐坐标释放所产生的精确局部势能导数，在历史状态、门控与资格迹固定时构成严格的单拍因果 VJP，不跨越时间反传。

源码说明：Exact current-event derivative through the actual inflow and coordinatewise release. Old S, target, gate, lambda and eligibility are constant here; this is not the derivative of a state trajectory or of the current language CE.

形式陈述：

```lean
∀ {n : ℕ} (occupied target lambda eligibility : Fin n → ℝ) (inflow : ℝ → ℝ) (release : ℝ → Fin n → ℝ)
  (gate x dinflow : ℝ) (drelease : Fin n → ℝ),
  HasDerivAt inflow dinflow x →
    (∀ (i : Fin n), HasDerivAt (fun θ => release θ i) (drelease i) x) →
      HasDerivAt
        (fun θ =>
          CGB.Generation.ContactPlasticity.localPotential occupied target (release θ) (inflow θ) gate lambda
            eligibility)
        (∑ i,
          lambda i * eligibility i *
            (gate * target i * (1 - ∑ j, occupied j) * dinflow - gate * occupied i * drelease i))
        x
```

## CGB.Generation.ContactPlasticity.allocation_pool_pre_post_coefficient

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L199)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：在目标分布与总准入量同时变动的归一化分配更新下，任意槽位的一阶导数精确分解为选通驱动增量、总体准入驱动及池保持因子的代数线性组合。

源码说明：Actual normalized allocation, allowing target and total admission to vary together. `capacity` is rank K, not the number of source-expanded coordinates. Only the selected source's drive is nonzero in the runtime instantiation.

形式陈述：

```lean
∀ {n : ℕ} (occupied release : Fin n → ℝ) (drive : ℝ → Fin n → ℝ) (active capacity gate left right pre c : ℝ)
  (i : Fin n),
  (∀ (θ : ℝ), ∑ j, drive θ j ≠ 0) →
    (∀ (θ : ℝ), drive θ i = CGB.Generation.ContactPlasticity.endpointAllocation (left + θ * pre) right) →
      HasDerivAt
        (fun θ =>
          c *
            CGB.Generation.sharedContactPoolVectorStep occupied (fun j => drive θ j / ∑ k, drive θ k) release
              (active * ((∑ j, drive θ j) / capacity)) gate i)
        (pre * (((left + right) / 2).sigmoid * (1 - ((left + right) / 2).sigmoid) / 2) *
          (c * gate * active * (1 - ∑ j, occupied j) / capacity))
        0
```

## CGB.Generation.ContactPlasticity.retention_pool_pre_post_coefficient

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L233)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：实际释放形式为 A*(1-retention)，因此保持度 logits 的导数具有正号并与各源坐标上的旧 S 状态收缩，直接给出逐 rank 端点 logits 的梯度解析形式。

源码说明：The actual release is A*(1-retention), so a retention logit's response has a positive sign and is contracted with old S across all source coordinates. Equal per-rank endpoint logits across sources specialize this sum directly.

形式陈述：

```lean
∀ {n : ℕ} (occupied target lambda eligibility left right pre : Fin n → ℝ) (inflow gate active : ℝ),
  HasDerivAt
    (fun θ =>
      CGB.Generation.ContactPlasticity.localPotential occupied target
        (fun i => active * (1 - CGB.Generation.ContactPlasticity.endpointRetention (left i + θ * pre i) (right i)))
        inflow gate lambda eligibility)
    (∑ i,
      pre i * ((left i).sigmoid * (1 - (left i).sigmoid) / 2) * (lambda i * eligibility i * gate * active * occupied i))
    0
```

## CGB.Generation.ContactPlasticity.local_direction_self_nonneg

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationContactPlasticity.lean#L257)

分类：design；适用范围：candidate；审核：reviewed；候选：Column

分类说明：针对 Column 候选机制的具体算法、更新导数、结构收敛性或介质性质。

解释：任意有限维参数方向向量与自身的内积（平方和）严格非负；该性质保证局部代理梯度的自乘积始终构成非负上升度量。

源码说明：A parameter direction contracted with itself is nonnegative. This concerns only the local proxy's own direction, not descent of a different CE objective.

形式陈述：

```lean
∀ {P : Type u_1} [inst : Fintype P] (direction : P → ℝ), 0 ≤ ∑ p, direction p * direction p
```
