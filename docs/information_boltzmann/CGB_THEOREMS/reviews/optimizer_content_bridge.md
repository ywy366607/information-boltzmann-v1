# optimizer_content_bridge

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean) · [返回导航](../README.md)

模块源码说明：Content bridge for the EXISTING binary p/q/w reaction unit.

## CGB.OptimizerContentReview.finite_capacity_content_bridge

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L58)

解释：在给定容量和驱动力区间内，同一有限内容载体满足定理所列的能量屏障与单缺陷转移条件。

源码说明：Generative conditions at arbitrary capacity c>=1. One-quantum defects can be repositioned by existing transfers at equal energy, then repaired at the Q port. Their combined exit barrier is J*c^2. Gradient work shifts the same chemical drive. The theorem certifies TWO endpoint codes, not a separate stable memory for every intermediate readout level.

形式陈述：

```lean
theorem finite_capacity_content_bridge (a J mu c scale offset gradient : ℝ)
    (hc : 1 ≤ c) (hlo : a - J * c ^ 2 < mu - gradient * scale)
    (hhi : mu - gradient * scale < a) :
    let drive
```

## CGB.OptimizerContentReview.one_defect_regeneration_margin

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L97)

解释：正耦合和正整数容量下，该共同载体在一个量子缺陷之后仍具有定理给出的后续损失能垒。

源码说明：At equal endpoint energy, the existing cubic interaction retains a strict barrier against a SECOND loss after one quantum is removed iff the integer field capacity is at least two. Capacity one loses the entire cooperative term after the first loss. This is a local energy-margin result, not a kinetic recovery guarantee or a new carrier.

形式陈述：

```lean
theorem one_defect_regeneration_margin (J : ℝ) (c : ℕ)
    (hJ : 0 < J) (hc : 1 ≤ c) :
    let a
```

## CGB.OptimizerContentReview.cooperative_content_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L142)

解释：在指定驱动区间下，构造的两个编码状态均具有所列翻转能垒，并满足单缺陷转移能量关系。

源码说明：Same physical carrier: both code states have barriers in the declared drive interval; every one-charge defect has a nonincreasing-energy legal repair path, with a strict drop on the Q exchange. This is a SUPPORT/energy certificate, not a probability-one or bounded-time recovery claim.

形式陈述：

```lean
theorem cooperative_content_contract (a J mu : ℝ) (hlo : a - J < mu) (hhi : mu < a) :
    (0 < energy a J mu (flipQ off) - energy a J mu off) ∧
    (0 < energy a J mu (flipQ on) - energy a J mu on) ∧
    (energy a J mu (flipQ off) - energy a J mu off +
      (energy a J mu (flipQ on) - energy a J mu on) = J) ∧
    (∀ s : Cell, count s = 1 ∨ count s = 2 →
      let t
```

## CGB.OptimizerContentReview.drive_removes_content_barriers

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L169)

解释：在非负耦合下，边界势越过给定阈值时，该内容模型中列出的释放路径能量下降。

源码说明：Release thresholds follow from the SAME boundary energy differences. Above a every Q addition on the actual 000->010->100->110->101->111 path is downhill. Below a-J the reverse erasure path is downhill. Neutral transfers conserve charge. Rates, paid time and noise still need their own certificates.

形式陈述：

```lean
theorem drive_removes_content_barriers (a J mu : ℝ) (hJ : 0 ≤ J) :
    (a < mu →
      energy a J mu (false, true, false) < energy a J mu off ∧
      energy a J mu (true, true, false) < energy a J mu (true, false, false) ∧
      energy a J mu on < energy a J mu (true, false, true)) ∧
    (mu < a - J →
      energy a J mu (true, false, true) < energy a J mu on ∧
      energy a J mu (true, false, false) < energy a J mu (true, true, false) ∧
      energy a J mu off < energy a J mu (false, true, false))
```

## CGB.OptimizerContentReview.content_gradient_port

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L185)

解释：正尺度及有效驱动区间下，冻结的线性任务梯度可等价写成边界势的平移并保持所列内容能垒条件。

源码说明：The loss gradient has a conjugate physical port: frozen linear task work g*dtheta shifts mu to mu-g*scale. The SAME existing repair paths then restore decoded parameter content, under an explicit drive interval. This is not a utility gate, a proof of actual-loss descent, or probability-one repair.

形式陈述：

```lean
theorem content_gradient_port (a J mu scale offset gradient : ℝ)
    (hscale : 0 < scale) (hlo : a - J < mu - gradient * scale)
    (hhi : mu - gradient * scale < a) :
    (∀ s t : Cell,
      (energy a J mu t + gradient * contentParameter scale offset t) -
        (energy a J mu s + gradient * contentParameter scale offset s) =
      energy a J (mu - gradient * scale) t -
        energy a J (mu - gradient * scale) s) ∧
    (contentParameter scale offset on - contentParameter scale offset off = 3 * scale ∧
      contentParameter scale offset off < contentParameter scale offset on) ∧
    (∀ s : Cell, count s = 1 ∨ count s = 2 →
      let t
```

## CGB.OptimizerContentReview.transfer_gradient_port

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L228)

解释：单次材料转移引起的冻结线性任务功由两端梯度差决定，并满足给定端口误差条件。

源码说明：A material transfer changes TWO decoded weights. Frozen task work is the gradient difference, the incidence adjoint, not just the receiving gradient. Uniform offsets of this force cancel for closed transfers. Force quantization has a local two-end error bound without retaining a dense incidence matrix.

形式陈述：

```lean
theorem transfer_gradient_port (scale offsetA offsetB nA nB amount gA gB common fA fB
    epsA epsB : ℝ) (hA : |fA - scale * gA| ≤ epsA)
    (hB : |fB - scale * gB| ≤ epsB) :
    (gA * ((offsetA + scale * (nA - amount)) - (offsetA + scale * nA)) +
      gB * ((offsetB + scale * (nB + amount)) - (offsetB + scale * nB)) =
        scale * (gB - gA) * amount) ∧
    (scale * ((gB + common) - (gA + common)) * amount = scale * (gB - gA) * amount) ∧
    |(fB - fA) - scale * (gB - gA)| ≤ epsA + epsB
```

## CGB.OptimizerContentReview.event_port_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L247)

解释：对任意正反事件指数与任务功，所构造的共同事件时钟满足定理列出的正反端口关系。

源码说明：Exact paired event port. Source forward/reverse exponents need not be opposites. Shift them by -work/+work and use their smallest common ceiling above the old clock bound. The log2 probability ratio changes by -2*work. At zero work the old clock is unchanged if both original rates were legal.

形式陈述：

```lean
theorem event_port_contract (baseForward baseReverse taskWork baseBound : ℝ) :
    let f
```

## CGB.OptimizerContentReview.maximal_event_clock

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L270)

解释：对给定正反指数和功，构造的配对时钟满足定理所述的速率界与正反比关系。

源码说明：Removing the obsolete global rejection ceiling gives the smallest common pair clock. The rate odds are unchanged; each direction is at least as fast as under any larger legal ceiling. It is NOT a theorem that faster kinetics improves retention: damaging events also accelerate.

形式陈述：

```lean
theorem maximal_event_clock (forward reverse work : ℝ) :
    let f
```

## CGB.OptimizerContentReview.force_response_dissipative

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L300)

解释：给定事件速率关于外力单调不增，且外力乘受迫与未受迫速率差不大于零。

源码说明：The learning channel is dissipative relative to the unforced physical motion: work times its rate change is nonpositive. Under an unchanged ceiling the local response is multiplied by the ORIGINAL local rate, so a common force unit does not imply a common effective gain at every compartment. This is not an assertion that autonomous motion always lowers task loss.

形式陈述：

```lean
theorem force_response_dissipative (forward reverse bound : ℝ) :
    Antitone (fun work : ℝ => (2 : ℝ) ^ eventLogRate forward reverse bound work) ∧
    (∀ work : ℝ, work * ((2 : ℝ) ^ eventLogRate forward reverse bound work -
      (2 : ℝ) ^ eventLogRate forward reverse bound 0) ≤ 0) ∧
    (∀ work : ℝ, forward ≤ bound → reverse ≤ bound → forward - work ≤ bound →
      reverse + work ≤ bound →
      (2 : ℝ) ^ eventLogRate forward reverse bound work =
        (2 : ℝ) ^ eventLogRate forward reverse bound 0 * (2 : ℝ) ^ (-work))
```

## CGB.OptimizerContentReview.certified_root_refinement

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L359)

解释：若根及其平方区间证书满足所列不等式，则新边界形成所述平方根包络。

源码说明：A finite integer square certificate refines a square-root enclosure. The runtime produces these inequalities using isqrt, without floating powers.

形式陈述：

```lean
theorem certified_root_refinement (x root lo hi nextLo nextHi : ℝ)
    (hroot0 : 0 ≤ root) (hroot : root ^ 2 = x)
    (hxlo : lo ≤ x) (hxhi : x ≤ hi)
    (hh0 : 0 ≤ nextHi)
    (hlsq : nextLo ^ 2 ≤ lo) (hhsq : hi ≤ nextHi ^ 2) :
    nextLo ≤ root ∧ root ≤ nextHi
```

## CGB.OptimizerContentReview.fractional_probability_enclosure

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L372)

解释：在尾部权重与前缀幂满足给定界时，构造的上下界包住指定分数幂概率。

源码说明：Certified prefix-product enclosure of 2^(-r), including its unexpanded tail. A lazy uniform prefix may decide only after its entire interval lies on one side. Ambiguity is confined to an interval whose width is <=4*eps when the probability enclosure width is <=2*eps. This is a numerical sampling contract, not a new physical mechanism or a useful-retention assumption.

形式陈述：

```lean
theorem fractional_probability_enclosure (headPower tail lo hi : ℝ)
    (ht0 : 0 ≤ tail) (ht1 : tail ≤ 1)
    (hlo : lo ≤ (2 : ℝ) ^ (-headPower)) (hhi : (2 : ℝ) ^ (-headPower) ≤ hi) :
    let low
```

## CGB.OptimizerContentReview.kinetic_survival_envelope

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L432)

解释：若非负时变转移核和权重函数满足所列逐行界，则任意有限期限的生存量满足相应递推界。

源码说明：One finite weighted-row certificate controls EVERY funded horizon. No tick-by-tick expansion, stationarity, eigensolver or fitted retention rule is needed. The verifier supplies the same actual event kernel and checks the finite row inequalities, including controlled rate intervals when present.

形式陈述：

```lean
theorem kinetic_survival_envelope {X : Type*} [Fintype X]
    (K : ℕ → X → X → ℝ) (v : X → ℝ) (rho : ℝ)
    (hK : ∀ n x y, 0 ≤ K n x y) (hr : 0 ≤ rho) :
    ((∀ x, v x ≤ 1) → (∀ n x, rho * v x ≤ ∑ y, K n x y * v y) →
      ∀ T start x, rho ^ T * v x ≤ kernelSurvival K T start x) ∧
    ((∀ x, 1 ≤ v x) → (∀ n x, (∑ y, K n x y * v y) ≤ rho * v x) →
      ∀ T start x, kernelSurvival K T start x ≤ rho ^ T * v x)
```

## CGB.OptimizerContentReview.hidden_basin_calibration

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L494)

解释：在给定正参数与顺序条件下，构造的隐藏微观状态权重满足所列盆地分布关系。

源码说明：Hidden-state accounting is necessary: endpoint neutrality k*x^3=1 still biases the majority-readout distribution when 0<x<1. The exact target odds require a cubic calibration, rather than an endpoint-only linear tilt. The final identity proves strict monotonicity of the odds for x,y,k>0.

形式陈述：

```lean
theorem hidden_basin_calibration (k x y target : ℝ) (hk : 0 < k)
    (hx : 0 < x) (hy : x < y) :
    (microWeight k x off + microWeight k x (true, false, false) +
      microWeight k x (false, true, false) + microWeight k x (false, false, true) = offMass x ∧
      microWeight k x (true, true, false) + microWeight k x (true, false, true) +
        microWeight k x (false, true, true) + microWeight k x on = onMass k x) ∧
    (0 < offMass x ∧ 0 < onMass k x) ∧
    (onMass k x / offMass x = target ↔
      k * x ^ 3 + 3 * x ^ 2 - 3 * target * x - target = 0) ∧
    (k * x ^ 3 = 1 → x < 1 → onMass k x < offMass x) ∧
    onMass k x / offMass x < onMass k y / offMass y
```

## CGB.OptimizerContentReview.maximal_reversible_acceptance

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L532)

解释：正平衡比下，满足指定正反平衡及接受率不超过一的任意方案都不超过对应 Metropolis 接受率。

源码说明：Given local detailed-balance ratio r and acceptance probabilities <=1, Metropolis is the pointwise maximal reversible acceptance. It minimizes rejection in THIS fixed proposal class, not all optimization costs or memory lifetimes. No biology-specific counter, window or extra array is introduced.

形式陈述：

```lean
theorem maximal_reversible_acceptance (r f b : ℝ) (hr : 0 < r)
    (_hf0 : 0 ≤ f) (hf1 : f ≤ 1) (_hb0 : 0 ≤ b) (hb1 : b ≤ 1)
    (hbalance : f = r * b) :
    (min 1 r = r * min 1 (1 / r)) ∧ f ≤ min 1 r ∧ b ≤ min 1 (1 / r)
```

## CGB.ResourceEnergyReview.localBounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L580)

解释：指定离散局部场的势与编码满足非负剩余和三个内容计数下界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem localBounds (p q w : Field) :
    0 ≤ phi p q w + 12 * code p q w ∧
    6 * code p q w ≤ p.val + q.val + w.val ∧
    2 * code p q w ≤ p.val ∧ 2 * code p q w ≤ q.val
```

## CGB.ResourceEnergyReview.slackZero

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L586)

解释：指定局部势与编码的剩余为零，当且仅当三个场全部为零或全部为二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem slackZero (p q w : Field) :
    phi p q w + 12 * code p q w = 0 ↔
      (p.val = 0 ∧ q.val = 0 ∧ w.val = 0) ∨ (p.val = 2 ∧ q.val = 2 ∧ w.val = 2)
```

## CGB.ResourceEnergyReview.poolBound

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L605)

解释：若总内容至少为计数的两倍，则两种计数表达式都受总内容平方的给定上界约束。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem poolBound (count total : ℕ) (h : 2 * count ≤ total) :
    4 * (count : ℝ) ≤ (total : ℝ)^2 ∧ 4 * (count : ℝ)^2 ≤ (total : ℝ)^2
```

## CGB.ResourceEnergyReview.rowTwiceBound

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L619)

解释：每行的两倍编码数不超过该行 P 场总量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem rowTwiceBound {n : ℕ} (p q w : Fin n → Fin n → Field) (i : Fin n) :
    2 * rowCodes p q w i ≤ rowP p i
```

## CGB.ResourceEnergyReview.colTwiceBound

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L624)

解释：每列的两倍编码数不超过该列 Q 场总量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem colTwiceBound {n : ℕ} (p q w : Fin n → Fin n → Field) (j : Fin n) :
    2 * colCodes p q w j ≤ colQ q j
```

## CGB.ResourceEnergyReview.poolsUnique

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L629)

解释：在逐位置预算、编码总数和资源平方和达到给定条件时，每个位置的编码计数都为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem poolsUnique {n : ℕ} (a b : Fin n → ℕ)
    (h2 : ∀ i, 2 * a i ≤ b i) (hsum : ∑ i, a i = n)
    (hsq : ∑ i, (b i : ℝ)^2 = 4 * n) : ∀ i, a i = 1
```

## CGB.ResourceEnergyReview.resource_ground_bound

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L652)

解释：任意大小的该离散矩阵在给定总供给下具有指定能量下界，等号成立时满足所列结构条件。

源码说明：Same actual energy, arbitrary matrix size: limited material and local quadratic resource costs generate a global lower bound without assuming row/column allocation constraints. No state-space enumeration in n.

形式陈述：

```lean
theorem resource_ground_bound {n : ℕ} (p q w : Fin n → Fin n → Field)
    (hsupply : mass p q w ≤ 6 * n) :
    (-(4 : ℝ) * n ≤ energy p q w) ∧
    (energy p q w = -(4 : ℝ) * n →
      (∀ i j, ((p i j).val = 0 ∧ (q i j).val = 0 ∧ (w i j).val = 0) ∨
        ((p i j).val = 2 ∧ (q i j).val = 2 ∧ (w i j).val = 2)) ∧
      (∀ i, rowCodes p q w i = 1) ∧ (∀ j, colCodes p q w j = 1))
```

## CGB.ResourceEnergyReview.diagonalPhi

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L707)

解释：指定对角构造的局部势在对角线上为 −12、其他位置为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem diagonalPhi {n : ℕ} (i j : Fin n) :
    phi (diagonal i j) (diagonal i j) (diagonal i j) = if i = j then -12 else 0
```

## CGB.ResourceEnergyReview.diagonalRow

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L710)

解释：指定对角构造每一行的 P 资源总量为二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem diagonalRow {n : ℕ} (i : Fin n) : rowP diagonal i = 2
```

## CGB.ResourceEnergyReview.diagonalCol

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L712)

解释：指定对角构造每一列的 Q 资源总量为二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem diagonalCol {n : ℕ} (j : Fin n) : colQ diagonal j = 2
```

## CGB.ResourceEnergyReview.resource_ground_witness

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L718)

解释：任意大小的对角分配构造恰好使用 6n 内容并达到能量 −4n。

源码说明：A dispersed configuration attains the bound at every size; its material is exactly the stated supply. This is a ground-state witness, not mixing or future predictive utility.

形式陈述：

```lean
theorem resource_ground_witness (n : ℕ) :
    mass (@diagonal n) diagonal diagonal = 6 * n ∧
      energy (@diagonal n) diagonal diagonal = -(4 : ℝ) * n
```

## CGB.ResourceEnergyReview.resource_monopoly_gap

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L736)

解释：在指定 2×2 等材料构造中，单行垄断和复制同列的能量都比对角分配高八。

源码说明：Explicit same-material counter-configurations: both single-row ownership and copying one column pay a strict energy gap in the SAME source energy.

形式陈述：

```lean
theorem resource_monopoly_gap :
    mass monopoly monopoly monopoly = 12 ∧ mass copies copies copies = 12 ∧
    energy monopoly monopoly monopoly - energy (@diagonal 2) diagonal diagonal = 8 ∧
    energy copies copies copies - energy (@diagonal 2) diagonal diagonal = 8
```

## CGB.ResourceEnergyReview.scaled_pool_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L747)

解释：若尺度成本满足所列复制一致性，则系数按规模倒数变化并满足给定池缩放关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem scaled_pool_contract (A : ℝ → ℝ)
    (hcopy : ∀ D : ℝ, 0 < D → A D * D^2 = (D/2) * (A 2 * 2^2)) :
    (∀ D : ℝ, 0 < D → A D = 2 * A 2 / D) ∧
    (∀ D r total : ℝ, 0 < D → 0 < r →
      scaledPool (r*D) (r*total) = r * scaledPool D total) ∧
    (∀ D total : ℝ, 1 ≤ D → 0 ≤ total → total+1 ≤ 2*D →
      let delta
```

## CGB.ResourceEnergyReview.codeSum_row

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L790)

解释：逐行编码数之和等于整个矩阵的编码总数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem codeSum_row {n : ℕ} (p q w : Fin n → Fin n → Field) :
    (∑ i, (rowCodes p q w i : ℝ)) = codes p q w
```

## CGB.ResourceEnergyReview.codeSum_col

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L794)

解释：逐列编码数之和等于整个矩阵的编码总数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem codeSum_col {n : ℕ} (p q w : Fin n → Fin n → Field) :
    (∑ i, (colCodes p q w i : ℝ)) = codes p q w
```

## CGB.ResourceEnergyReview.variance_expansion

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L799)

解释：定义的平衡方差可展开成行列编码数平方和及指定总量项。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem variance_expansion {n : ℕ} (p q w : Fin n → Fin n → Field) :
    balanceVariance p q w =
      (∑ i, (rowCodes p q w i : ℝ)^2) + (∑ j, (colCodes p q w j : ℝ)^2) -
      (3*(n:ℝ)/2) * codes p q w + 9*(n:ℝ)^3/32
```

## CGB.ResourceEnergyReview.scaled_ground_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L817)

解释：正矩阵大小下，指定缩放能量具有给定下界，等号成立时满足所列局部结构。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem scaled_ground_bounds {n : ℕ} (hn : 0 < n)
    (p q w : Fin n → Fin n → Field) :
    -(9:ℝ)*(n:ℝ)^2/4 ≤ scaledEnergy p q w ∧
    (scaledEnergy p q w = -(9:ℝ)*(n:ℝ)^2/4 →
      (∀ i j, ((p i j).val = 0 ∧ (q i j).val = 0 ∧ (w i j).val = 0) ∨
        ((p i j).val = 2 ∧ (q i j).val = 2 ∧ (w i j).val = 2)) ∧
      (∀ i, (rowCodes p q w i : ℝ) = 3*n/8) ∧
      (∀ j, (colCodes p q w j : ℝ) = 3*n/8))
```

## CGB.ResourceEnergyReview.band8_row

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L891)

解释：指定 8×8 布尔带状构造的每行恰有三个真值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem band8_row (bit : Bool) (a : Fin 8) : ∑ b, (band8 bit a b).toNat = 3
```

## CGB.ResourceEnergyReview.band8_col

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L894)

解释：指定 8×8 布尔带状构造的每列恰有三个真值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem band8_col (bit : Bool) (b : Fin 8) : ∑ a, (band8 bit a b).toNat = 3
```

## CGB.ResourceEnergyReview.fieldCode_values

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L900)

解释：布尔码映射到三个相同场后，其编码、场值和局部势分别满足给定显式公式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem fieldCode_values (bit : Bool) :
    code (fieldCode bit) (fieldCode bit) (fieldCode bit) = bit.toNat ∧
    (fieldCode bit).val = 2*bit.toNat ∧
    phi (fieldCode bit) (fieldCode bit) (fieldCode bit) = -12*(bit.toNat:ℝ)
```

## CGB.ResourceEnergyReview.spread_code

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L910)

解释：按给定掩码展开的矩阵在任意位置的编码等于定理中对应的带状构造值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem spread_code (m : ℕ) (mask : Fin m → Fin m → Bool) (i j : Fin (8*m)) :
    code (spread m mask i j) (spread m mask i j) (spread m mask i j) =
      let e : Fin 8 × Fin m ≃ Fin (8*m)
```

## CGB.ResourceEnergyReview.spread_row

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L916)

解释：大小为 8m 的展开矩阵每一行的编码计数为 3m。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem spread_row (m : ℕ) (mask : Fin m → Fin m → Bool) (i : Fin (8*m)) :
    rowCodes (spread m mask) (spread m mask) (spread m mask) i = 3*m
```

## CGB.ResourceEnergyReview.spread_col

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L926)

解释：大小为 8m 的展开矩阵每一列的编码计数为 3m。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem spread_col (m : ℕ) (mask : Fin m → Fin m → Bool) (j : Fin (8*m)) :
    colCodes (spread m mask) (spread m mask) (spread m mask) j = 3*m
```

## CGB.ResourceEnergyReview.spread_values

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L936)

解释：展开矩阵各位置的场值与局部势由对应编码值按所列公式确定。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem spread_values (m : ℕ) (mask : Fin m → Fin m → Bool) (i j : Fin (8*m)) :
    (spread m mask i j).val = 2*code (spread m mask i j) (spread m mask i j) (spread m mask i j) ∧
    phi (spread m mask i j) (spread m mask i j) (spread m mask i j) =
      -12*(code (spread m mask i j) (spread m mask i j) (spread m mask i j) : ℝ)
```

## CGB.ResourceEnergyReview.spread_rowP

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L943)

解释：大小为 8m 的展开矩阵每行 P 场总量为 6m。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem spread_rowP (m : ℕ) (mask : Fin m → Fin m → Bool) (i : Fin (8*m)) :
    rowP (spread m mask) i = 6*m
```

## CGB.ResourceEnergyReview.spread_colQ

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L950)

解释：大小为 8m 的展开矩阵每列 Q 场总量为 6m。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem spread_colQ (m : ℕ) (mask : Fin m → Fin m → Bool) (j : Fin (8*m)) :
    colQ (spread m mask) j = 6*m
```

## CGB.ResourceEnergyReview.spread_attains

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L957)

解释：m>0 时，掩码展开构造的编码总数、材料量和缩放能量分别为 24m²、144m² 和 −144m²。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem spread_attains (m : ℕ) (hm : 0 < m) (mask : Fin m → Fin m → Bool) :
    codes (spread m mask) (spread m mask) (spread m mask) = 24*m^2 ∧
    mass (spread m mask) (spread m mask) (spread m mask) = 144*m^2 ∧
    scaledEnergy (spread m mask) (spread m mask) (spread m mask) = -144*(m:ℝ)^2
```

## CGB.ResourceEnergyReview.spread_injective

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L990)

解释：固定 m 时，不同布尔掩码经展开构造得到不同矩阵。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem spread_injective (m : ℕ) : Function.Injective (spread m)
```

## CGB.ResourceEnergyReview.scaled_resource_generation

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1003)

解释：指定缩放资源模型对任意正规模有能量下界，且 8m 规模存在由掩码生成的多种达到该界的构造。

源码说明：Generated lower bound/equality structure at every size. Sizes8m have at least2^(m^2) distinct attaining configurations via an injective Boolean mask. The witness is not an initialization policy or a learning theorem.

形式陈述：

```lean
theorem scaled_resource_generation :
    (∀ n : ℕ, 0 < n → ∀ p q w : Fin n → Fin n → Field,
      -(9:ℝ)*(n:ℝ)^2/4 ≤ scaledEnergy p q w ∧
      (scaledEnergy p q w = -(9:ℝ)*(n:ℝ)^2/4 →
        (∀ i j, ((p i j).val = 0 ∧ (q i j).val = 0 ∧ (w i j).val = 0) ∨
          ((p i j).val = 2 ∧ (q i j).val = 2 ∧ (w i j).val = 2)) ∧
        (∀ i, (rowCodes p q w i : ℝ) = 3*n/8) ∧
        (∀ j, (colCodes p q w j : ℝ) = 3*n/8))) ∧
    (∀ m : ℕ, 0 < m → ∃ G : (Fin m → Fin m → Bool) → (Fin (8*m) → Fin (8*m) → Field),
      Function.Injective G ∧ ∀ mask,
        codes (G mask) (G mask) (G mask) = 24*m^2 ∧
        mass (G mask) (G mask) (G mask) = 144*m^2 ∧
        scaledEnergy (G mask) (G mask) (G mask) = -144*(m:ℝ)^2)
```

## CGB.ResourceReleaseReview.resource_feedback_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1042)

解释：共享池负载增加会改变局部势垒；该构造中双稳外力区间宽度为 46，额外非负负载不增大指定转移速率。

源码说明：Same content interaction and shared pool: local Q-port energetic margins shift with other users of the pool. The 46-unit overlap is not a complete ignition, dynamic refractory-period or useful-memory theorem. No new state.

形式陈述：

```lean
theorem resource_feedback_contract (load force extra : ℝ) (he : 0 ≤ extra) :
    let E
```

## CGB.ResourceReleaseReview.rate_unit

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1072)

解释：指定事件速率始终位于零和一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem rate_unit (energy work : ℝ) :
    0 ≤ rate energy work ∧ rate energy work ≤ 1
```

## CGB.ResourceReleaseReview.downhill

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1078)

解释：当事件能量变化与外功之和非正时，指定接受速率为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem downhill (energy work : ℝ) (h : energy + work ≤ 0) :
    rate energy work = 1
```

## CGB.ResourceReleaseReview.primitive_release_port

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1089)

解释：在指定局部场范围、资源条件和至少 96 的外力下，给定释放事件满足所列速率关系。

源码说明：Source energy increments, plus the same conjugate task work. Force>=96 is a declared evidence domain, not an internal detector or learning rule. The intake bound retains 2^-126 for the requested billion-cell scale. Conversion statements apply when q=0; their guards are checked separately.

形式陈述：

```lean
theorem primitive_release_port (p w P Q force : ℝ)
    (hp : 0 ≤ p) (hp2 : p ≤ 2) (hw : 0 ≤ w) (hw2 : w ≤ 2)
    (hP : p ≤ P) (hQ : 0 ≤ Q) (hforce : 96 ≤ force) :
    rate (-13 + 12*p*w - 2*Q) (-force) = 1 ∧
    rate (15 - 12*p*w + 2*Q) force ≤ 1/85070591730234615865843651857942052864 ∧
    (1 ≤ p → rate (-12*w*(p-1) - 2*P + 2) 0 = 1) ∧
    (1 ≤ w → 1/4 ≤ rate (-12*p*(w-1) + 1) 0)
```

## CGB.ResourceReleaseReview.gated_unit

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1122)

解释：带使能条件的指定事件速率始终位于零和一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem gated_unit (enabled : Prop) [Decidable enabled] (energy work : ℝ) :
    0 ≤ gated enabled energy work ∧ gated enabled energy work ≤ 1
```

## CGB.ResourceReleaseReview.local_release

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1150)

解释：在指定局部场和至少 96 的外力条件下，局部释放漂移受定理列出的数值上界约束。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem local_release (p q w : Fin 3) (P Q force : ℝ) (stocked : Bool)
    (hP : (p.val : ℝ) ≤ P) (hQ : 0 ≤ Q) (hforce : 96 ≤ force) :
    drift (rates p q w P Q force stocked) ≤ 9/85070591730234615865843651857942052864 ∧
    ((0 < q.val ∨ (q.val = 0 ∧ Q = 0 ∧ 0 < p.val + w.val)) →
      drift (rates p q w P Q force stocked) ≤ -1 + 9/85070591730234615865843651857942052864)
```

## CGB.ResourceReleaseReview.resource_release_drift

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1217)

解释：对任意规模的非空主体，若每个局部外力至少为 96，则构造的整体释放观测量满足给定漂移界。

源码说明：All-size nonempty-body drift from the actual quadratic-pool local rate formula. V=13(Ptotal+Wtotal)+9Qtotal is a proof observer only. Its coefficients balance the outlet and enabling margins; they do not modify the update. Multiply the displayed sum by the common proposal probability 1/(2*F*n). Cross-cell p and q transfers preserve V, regardless of their acceptance. Runtime refinement and conversion to first-hit time are separate obligations.

形式陈述：

```lean
theorem resource_release_drift {n : ℕ} (p q w : Fin n → Fin 3)
    (row col : Fin n → ℕ) (force : Fin n → ℝ) (stocked : Bool)
    (hforce : ∀ i, 96 ≤ force i)
    (hnonempty : ∃ i, 0 < (p i).val + (q i).val + (w i).val) :
    (∑ i, drift (rates (p i) (q i) (w i) (pool p row i) (pool q col i) (force i) stocked))
      ≤ -1 + n * (9/85070591730234615865843651857942052864)
```

## CGB.ResourceReleaseReview.releaseValue_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1255)

解释：当四位置主体总量不超过十二且非空时，指定释放观测值位于定理给出的上下界内。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem releaseValue_bounds (p q w : Fin 4 → Fin 3)
    (hmass : (∑ i, ((p i).val + (q i).val + (w i).val)) ≤ 12)
    (hne : ∃ i, 0 < (p i).val + (q i).val + (w i).val) :
    9 ≤ releaseValue p q w ∧ releaseValue p q w ≤ 156
```

## CGB.ResourceReleaseReview.local_change_effect

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1322)

解释：对被使能的局部事件，局部内容量与观测值变化恰为事件定义的变化量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem local_change_effect (c : NativeCell) (m : Fin 6) (h : localEnabled c m) :
    ((nativeCellMass (localChange c m) : ℤ) - nativeCellMass c = localMassDelta m) ∧
    (nativeCellValue (localChange c m) - nativeCellValue c = localValueDelta m)
```

## CGB.ResourceReleaseReview.sum_update_nat

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1330)

解释：替换主体一个位置后，新总内容加旧局部内容等于旧总内容加新局部内容。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem sum_update_nat (b : NativeBody) (i : Fin 4) (c : NativeCell) :
    nativeMass (Function.update b i c) + nativeCellMass (b i) = nativeMass b + nativeCellMass c
```

## CGB.ResourceReleaseReview.sum_update_value

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1341)

解释：替换主体一个位置导致的总观测值变化，等于该位置局部观测值变化。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem sum_update_value (b : NativeBody) (i : Fin 4) (c : NativeCell) :
    nativeValue (Function.update b i c) - nativeValue b = nativeCellValue c - nativeCellValue (b i)
```

## CGB.ResourceReleaseReview.shift_up_effect

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1358)

解释：选定局部场尚未满时，上移一单位使局部内容增加一并产生给定观测值变化。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem shift_up_effect (c : NativeCell) (isQ : Bool)
    (h : (selectedField c isQ).val < 2) :
    nativeCellMass (shiftCell c isQ true) = nativeCellMass c + 1 ∧
    nativeCellValue (shiftCell c isQ true) - nativeCellValue c = fieldWeight isQ
```

## CGB.ResourceReleaseReview.shift_down_effect

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1369)

解释：选定局部场为正时，下移一单位使局部内容减少一并产生给定观测值变化。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem shift_down_effect (c : NativeCell) (isQ : Bool)
    (h : 0 < (selectedField c isQ).val) :
    nativeCellMass (shiftCell c isQ false) + 1 = nativeCellMass c ∧
    nativeCellValue (shiftCell c isQ false) - nativeCellValue c = -fieldWeight isQ
```

## CGB.ResourceReleaseReview.partner_ne

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1383)

解释：指定四位置配对函数选择的伙伴位置始终不同于原位置。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem partner_ne (isQ : Bool) (i : Fin 4) : partner isQ i ≠ i
```

## CGB.ResourceReleaseReview.transfer_change_effect

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1390)

解释：当来源有库存且伙伴未满时，指定内部转移满足所列总内容与观测值变化关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem transfer_change_effect (b : NativeBody) (i : Fin 4) (isQ : Bool)
    (hd : 0 < (selectedField (b i) isQ).val)
    (hu : (selectedField (b (partner isQ i)) isQ).val < 2) :
    nativeMass (transferChange b i isQ) = nativeMass b ∧
    nativeValue (transferChange b i isQ) = nativeValue b
```

## CGB.ResourceReleaseReview.native_candidate_effect

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1434)

解释：在主体容量和事件使能条件下，构造的后继候选仍在容量内并具有所列观测值变化。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_candidate_effect (b : NativeBody) (e : NativeEvent)
    (hmass : nativeMass b ≤ 12) (h : nativeEnabled b e) :
    nativeMass (nativeCandidate b e) ≤ 12 ∧
      nativeValue (nativeCandidate b e) - nativeValue b = choiceValueDelta e.2
```

## CGB.ResourceReleaseReview.native_value_zero

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1470)

解释：指定主体总内容为零时，其释放观测值也为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_value_zero (b : NativeBody) (h : nativeMass b = 0) : nativeValue b = 0
```

## CGB.ResourceReleaseReview.native_pack_value

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1484)

解释：当主体满足容量界时，打包表示保留原主体的观测值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_pack_value (b : NativeBody) (hcap : nativeMass b ≤ 12) :
    (nativePack b).elim 0 (fun x => nativeValue x.val) = nativeValue b
```

## CGB.ResourceReleaseReview.local_rate_unit

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1509)

解释：指定局部事件速率始终位于零和一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem local_rate_unit (p q w : Fin 3) (P Q f : ℝ) (stocked : Bool) (m : Fin 6) :
    0 ≤ rates p q w P Q f stocked m ∧ rates p q w P Q f stocked m ≤ 1
```

## CGB.ResourceReleaseReview.local_rate_disabled

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1512)

解释：当指定局部事件不满足使能及库存条件时，其速率为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem local_rate_disabled (p q w : Fin 3) (P Q f : ℝ) (stocked : Bool) (m : Fin 6)
    (h : ¬(localEnabled (p, q, w) m ∧ (m.val = 4 → stocked = true))) :
    rates p q w P Q f stocked m = 0
```

## CGB.ResourceReleaseReview.native_rate_unit

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1517)

解释：指定主体级事件速率始终位于零和一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_rate_unit (f : Fin 4 → ℝ) (b : NativeBody) (e : NativeEvent) :
    0 ≤ nativeRate f b e ∧ nativeRate f b e ≤ 1
```

## CGB.ResourceReleaseReview.native_rate_disabled

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1523)

解释：当指定主体级事件未使能时，其速率为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_rate_disabled (f : Fin 4 → ℝ) (b : NativeBody) (e : NativeEvent)
    (h : ¬nativeEnabled b e) : nativeRate f b e = 0
```

## CGB.ResourceReleaseReview.jumpKernel_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1542)

解释：当事件提议概率归一且接受率位于零到一之间时，指定跳跃核满足所列非负性与质量界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem jumpKernel_contract {X E : Type*} [Fintype X] [DecidableEq X] [Fintype E]
    (p : E → ℝ) (a : X → E → ℝ) (d : X → E → Option X)
    (hp : ∀ e, 0 ≤ p e) (hsum : ∑ e, p e = 1)
    (ha : ∀ x e, 0 ≤ a x e ∧ a x e ≤ 1) :
    (∀ x y, 0 ≤ jumpKernel p a d x y) ∧
    (∀ v x, (∑ y, jumpKernel p a d x y * v y) =
      v x + ∑ e, p e * a x e * ((d x e).elim 0 v - v x))
```

## CGB.ResourceReleaseReview.native_event_change

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1590)

解释：指定主体事件的速率乘打包后观测值变化，可改写为所定义的事件变化表达式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_event_change (f : Fin 4 → ℝ) (s : NativeState) (e : NativeEvent) :
    nativeRate f s.val e *
        ((nativePack (nativeCandidate s.val e)).elim 0 (fun x => nativeValue x.val) - nativeValue s.val) =
      nativeRate f s.val e * choiceValueDelta e.2
```

## CGB.ResourceReleaseReview.native_rate_collapse

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1600)

解释：对所有原生事件的加权观测变化求和，可化为各位置局部漂移之和的四十分之一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_rate_collapse (f : Fin 4 → ℝ) (b : NativeBody) :
    (∑ e, nativeProposal e * nativeRate f b e * choiceValueDelta e.2) =
      (1/40:ℝ) * ∑ i, drift (rates (b i).1 (b i).2.1 (b i).2.2
        (pool (fun j => (b j).1) nativeRow i) (pool (fun j => (b j).2.1) nativeCol i)
        (f i) (decide (nativeMass b < 12)))
```

## CGB.ResourceReleaseReview.native_kernel_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1612)

解释：构造的时变原生核非负、每行和不超过一，且满足定理所列观测值关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_kernel_contract (f : ℕ → NativeState → Fin 4 → ℝ) :
    (∀ t x y, 0 ≤ nativeKernel f t x y) ∧
    (∀ t x, (∑ y, nativeKernel f t x y) ≤ 1) ∧
    (∀ t x, (∑ y, nativeKernel f t x y * nativeValue y.val) = nativeValue x.val +
      (1/40:ℝ) * ∑ i, drift (rates (x.val i).1 (x.val i).2.1 (x.val i).2.2
        (pool (fun j => (x.val j).1) nativeRow i) (pool (fun j => (x.val j).2.1) nativeCol i)
        (f t x i) (decide (nativeMass x.val < 12))))
```

## CGB.ResourceReleaseReview.decOrSame_le

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1655)

解释：指定三态场减一或保持操作的值不超过原值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem decOrSame_le (x : Fin 3) : (decOrSame x).val ≤ x.val
```

## CGB.ResourceReleaseReview.damage_cell_effect

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1662)

解释：指定单元损伤操作不会增加该单元的内容量或观测值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem damage_cell_effect (c : NativeCell) (field : Fin 3) :
    nativeCellMass (damageCell c field) ≤ nativeCellMass c ∧
    nativeCellValue (damageCell c field) ≤ nativeCellValue c
```

## CGB.ResourceReleaseReview.native_damage_effect

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1676)

解释：指定主体损伤候选不会增加主体总内容或观测值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_damage_effect (b : NativeBody) (e : Fin 4 × Fin 3) :
    nativeMass (nativeDamageCandidate b e) ≤ nativeMass b ∧
    nativeValue (nativeDamageCandidate b e) ≤ nativeValue b
```

## CGB.ResourceReleaseReview.native_damage_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1691)

解释：指定损伤核非负，且损伤后的加权观测值不超过损伤前观测值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_damage_contract :
    (∀ x y, 0 ≤ nativeDamageKernel x y) ∧
    (∀ x, (∑ y, nativeDamageKernel x y * (nativeValue y.val / 9)) ≤ nativeValue x.val / 9)
```

## CGB.ResourceReleaseReview.native_nonempty

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1716)

解释：每个合法原生状态都至少有一个位置的三种局部内容之和为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_nonempty (x : NativeState) :
    ∃ i, 0 < (x.val i).1.val + (x.val i).2.1.val + (x.val i).2.2.val
```

## CGB.ResourceReleaseReview.native_value_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1725)

解释：每个合法原生状态的指定观测值位于九到一百五十六之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_value_bounds (x : NativeState) : 9 ≤ nativeValue x.val ∧ nativeValue x.val ≤ 156
```

## CGB.ResourceReleaseReview.native_weighted_release

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1730)

解释：若每个时刻每个位置的外力至少为 96，则原生核的加权观测值满足 6399/6400 收缩界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem native_weighted_release (f : ℕ → NativeState → Fin 4 → ℝ)
    (hf : ∀ t x i, 96 ≤ f t x i) (t : ℕ) (x : NativeState) :
    (∑ y, nativeKernel f t x y * (nativeValue y.val / 9)) ≤
      (6399/6400:ℝ) * (nativeValue x.val / 9)
```

## CGB.ResourceReleaseReview.resource_release_budget

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1751)

解释：在持续至少 96 的外力及指定损伤和资金协议下，有限期限生存量受定理给出的几何上界约束。

源码说明：Same external wear as holding: wear then funded physical opportunity, first empty hit observed at either substep. Only genuine paid opportunities are counted; the kernel includes all native proposals and every legal body. No row-bookkeeping, desired memory or desired release property is assumed.

形式陈述：

```lean
theorem resource_release_budget (f : ℕ → NativeState → Fin 4 → ℝ)
    (hf : ∀ t x i, 96 ≤ f t x i) :
    ∀ T start x, CGB.OptimizerContentReview.kernelSurvival (nativeOpenKernel f) T start x ≤
      (6399/6400:ℝ)^T * (nativeValue x.val / 9)
```

## CGB.RationalSamplerReview.coefficient_step

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1794)

解释：指定级数系数的下一项可由当前项乘给定有理因子递推。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem coefficient_step (x : ℝ) (n : ℕ) :
    coefficient x (n+1) = coefficient x n * ((n+1:ℝ)-x) / (n+2)
```

## CGB.RationalSamplerReview.coefficient_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1808)

解释：参数 x 位于零到一时，指定级数每项系数都位于零到一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem coefficient_bounds (x : ℝ) (hx : 0 ≤ x) (hx1 : x ≤ 1) (n : ℕ) :
    0 ≤ coefficient x n ∧ coefficient x n ≤ 1
```

## CGB.RationalSamplerReview.series_identity

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1823)

解释：指定系数乘二进制权重的无穷级数求和等于 1−2^(−x)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem series_identity (x : ℝ) :
    HasSum (fun n : ℕ => coefficient x n * (1/2:ℝ)^(n+1)) (1-(2:ℝ)^(-x))
```

## CGB.RationalSamplerReview.coefficientQ_refinement

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1850)

解释：有理数实现的级数系数转为实数后，等于实数定义的同一系数。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem coefficientQ_refinement (x : ℚ) (n : ℕ) :
    (coefficientQ x n : ℝ) = coefficient x n
```

## CGB.RationalSamplerReview.coinNextQ_refinement

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1877)

解释：有理数实现的一步硬币状态转为实数后，等于实数定义的一步状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem coinNextQ_refinement (p : ℚ) (bit : Bool) :
    (coinNextQ p bit : ℝ) = coinNext (p:ℝ) bit
```

## CGB.RationalSamplerReview.coinAcceptedQ_refinement

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1881)

解释：有理数实现的有限前缀接受量转为实数后，等于实数版本。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem coinAcceptedQ_refinement (n : ℕ) (p : ℚ) :
    (coinAcceptedQ n p : ℝ) = coinAccepted n (p:ℝ)
```

## CGB.RationalSamplerReview.coin_endpoints

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1901)

解释：目标概率为零或一时，有限前缀的接受量等于该端点且待定量为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem coin_endpoints (n : ℕ) :
    coinAccepted n 0 = 0 ∧ coinAccepted n 1 = 1 ∧
      coinPending n 0 = 0 ∧ coinPending n 1 = 0
```

## CGB.RationalSamplerReview.coin_next_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1906)

解释：概率位于零到一时，两个二进制分支的下一概率均在零到一之间，且两者之和为原概率的两倍。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem coin_next_bounds (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    (0 ≤ coinNext p false ∧ coinNext p false ≤ 1) ∧
    (0 ≤ coinNext p true ∧ coinNext p true ≤ 1) ∧
    coinNext p false + coinNext p true = 2*p
```

## CGB.RationalSamplerReview.coin_prefix

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1922)

解释：目标概率位于零到一时，有限前缀接受量与待定量形成概率包络，待定量不超过 2^(−n)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem coin_prefix (n : ℕ) (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    0 ≤ coinAccepted n p ∧ coinAccepted n p ≤ p ∧
    p ≤ coinAccepted n p + coinPending n p ∧
    0 ≤ coinPending n p ∧ coinPending n p ≤ (1/2:ℝ)^n
```

## CGB.RationalSamplerReview.rational_mixture_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L1966)

解释：有理参数位于零到一时，有理系数满足逐项界，并按指定二进制混合级数收敛到目标幂表达式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem rational_mixture_contract (x : ℚ) (hx : 0 ≤ x) (hx1 : x ≤ 1) :
    (∀ n, 0 ≤ coefficientQ x n ∧ coefficientQ x n ≤ 1) ∧
    HasSum (fun n : ℕ => (1/2:ℝ)^(n+1) * (coefficientQ x n : ℝ))
      (1-(2:ℝ)^(-(x:ℝ))) ∧
    HasSum (fun n : ℕ => (1/2:ℝ)^(n+1) * (1-(coefficientQ x n : ℝ)))
      ((2:ℝ)^(-(x:ℝ))) ∧
    HasSum (fun n : ℕ => (1/2:ℝ)^(n+1)) 1 ∧
    HasSum (fun n : ℕ => (n:ℝ) * (1/2:ℝ)^(n+1)) 1 ∧
    HasSum (fun n : ℕ => (n+1:ℝ) * (1/2:ℝ)^(n+1)) 2
```

## CGB.RationalSamplerReview.rational_binary_coin_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2000)

解释：对合法目标概率，有理实现细化实数前缀算法，且接受量与待定量满足给定包络。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem rational_binary_coin_contract (p : ℝ) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    (∀ q : ℚ, ∀ n, (coinAcceptedQ n q : ℝ) = coinAccepted n (q:ℝ)) ∧
    (∀ n, 0 ≤ coinAccepted n p ∧ coinAccepted n p ≤ p ∧
      p ≤ coinAccepted n p + coinPending n p ∧
      0 ≤ coinPending n p ∧ coinPending n p ≤ (1/2:ℝ)^n) ∧
    Filter.Tendsto (fun n => coinAccepted n p) Filter.atTop (nhds p) ∧
    Filter.Tendsto (fun n => coinPending n p) Filter.atTop (nhds 0) ∧
    (∀ n, (∑ k ∈ Finset.range n, coinPending k p) ≤ 2)
```

## CGB.FiberOptimizerReview.blockFloor_le

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2045)

解释：非空有限状态块的最小能量不超过块内任一状态的能量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem blockFloor_le {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) (x : S) (hx : x ∈ B) :
    blockFloor B hB E ≤ E x
```

## CGB.FiberOptimizerReview.blockFloor_attained

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2051)

解释：非空有限状态块中至少有一个状态达到该块最小能量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem blockFloor_attained {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) :
    ∃ x ∈ B, E x = blockFloor B hB E
```

## CGB.FiberOptimizerReview.block_rate_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2066)

解释：指定非空读出块的单状态速率位于零与块大小倒数之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem block_rate_bounds {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) (y : S) :
    0 ≤ blockRate B hB E y ∧ blockRate B hB E y ≤ 1/(B.card:ℝ)
```

## CGB.FiberOptimizerReview.block_activity

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2080)

解释：指定非空读出块的总活动率位于块大小倒数与一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem block_activity {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) :
    1/(B.card:ℝ) ≤ ∑ z, blockRate B hB E z ∧
      (∑ z, blockRate B hB E z) ≤ 1
```

## CGB.FiberOptimizerReview.readout_block_kernel_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2102)

解释：指定非空块的条件热浴核非负、行和为一，并满足定理所列的平衡与活动条件。

源码说明：Exact conditional heatbath with a lazy factor >=1/|B|. The floor used by the implementation is the maximal legal common envelope for uniform block proposals. Equilibrium weights, not a target-retention premise, determine it. The identity concerns the new kernel; old time certificates do not transfer.

形式陈述：

```lean
theorem readout_block_kernel_contract {S : Type*} [Fintype S] [DecidableEq S]
    (B : Finset S) (hB : B.Nonempty) (E : S → ℝ) :
    (∀ x y, 0 ≤ blockKernel B hB E x y) ∧
    (∀ x, (∑ y, blockKernel B hB E x y) = 1) ∧
    (∀ x y, (2:ℝ)^(-2*E x) * blockKernel B hB E x y =
      (2:ℝ)^(-2*E y) * blockKernel B hB E y x) ∧
    (1/(B.card:ℝ) ≤ ∑ z, blockRate B hB E z ∧
      (∑ z, blockRate B hB E z) ≤ 1) ∧
    (∀ x ∈ B, ∀ y,
      let a
```

## CGB.FiberOptimizerReview.readout_card_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2164)

解释：指定六个相邻读出块的大小都介于四和十三之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem readout_card_bounds (k : Fin 6) :
    4 ≤ (readoutBlock k).card ∧ (readoutBlock k).card ≤ 13
```

## CGB.FiberOptimizerReview.coord_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2174)

解释：三态场的自然数值作为实数时位于零到二之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem coord_bounds (p : Fin 3) : 0 ≤ (p.val:ℝ) ∧ (p.val:ℝ) ≤ 2
```

## CGB.FiberOptimizerReview.product_bound

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2179)

解释：任意两个三态场值的乘积不超过四。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem product_bound (p q : Fin 3) : (p.val:ℝ)*q.val ≤ 4
```

## CGB.FiberOptimizerReview.pool_decrement

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2185)

解释：在正规模、非负剩余资源及局部量至少一的条件下，指定二次池成本减少一单位的变化非正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem pool_decrement (D R p : ℝ) (hD : 0 < D) (hR : 0 ≤ R) (hp : 1 ≤ p) :
    2*(R+p-1)^2/D - 2*(R+p)^2/D ≤ 0
```

## CGB.FiberOptimizerReview.pool_increment

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2191)

解释：在规模、资源及容量满足给定条件时，指定二次池成本增加一单位的变化小于八。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem pool_increment (D R p : ℝ) (hD : 1 ≤ D) (hR : 0 ≤ R)
    (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hcap : R+2 ≤ 2*D) :
    2*(R+p+1)^2/D - 2*(R+p)^2/D < 8
```

## CGB.FiberOptimizerReview.remove_mass

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2209)

解释：内容非空的指定单元移除一单位后，新内容加一等于原内容。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem remove_mass (s : Cell) (hs : 0 < cellMass s) :
    cellMass (removeOne s)+1 = cellMass s
```

## CGB.FiberOptimizerReview.add_mass

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2214)

解释：内容少于六的指定单元增加一单位后，新内容恰等于原内容加一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem add_mass (s : Cell) (hs : cellMass s < 6) :
    cellMass (addOne s) = cellMass s+1
```

## CGB.FiberOptimizerReview.remove_energy

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2219)

解释：在指定正池规模和非负资源条件下，非空单元移除一单位的能量变化不超过三十四。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem remove_energy (a b R C : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (s : Cell) (hs : 0 < cellMass s) :
    cellEnergy a b R C (removeOne s) - cellEnergy a b R C s ≤ 34
```

## CGB.FiberOptimizerReview.add_energy

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2248)

解释：在指定池规模、资源及容量条件下，未满单元增加一单位的能量变化满足给定上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem add_energy (a b R C : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (s : Cell) (hs : cellMass s < 6) :
    cellEnergy a b R C (addOne s) - cellEnergy a b R C s ≤ 22
```

## CGB.FiberOptimizerReview.block_nonempty

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2278)

解释：六个指定相邻读出块均非空。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem block_nonempty (k : Fin 6) : (readoutBlock k).Nonempty
```

## CGB.FiberOptimizerReview.active_card

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2286)

解释：指定预算与单元状态下，活跃读出块的数量最多为二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem active_card (budget : Fin 7) (x : Cell) :
    (activeBlocks budget x).card ≤ 2
```

## CGB.FiberOptimizerReview.off_block_zero

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2301)

解释：若目标状态不在指定块内且不同于源状态，该块转移核的对应项为零。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem off_block_zero (B : Finset Cell) (hB : B.Nonempty) (E : Cell → ℝ)
    (x y : Cell) (hy : y ∉ B) (hxy : x ≠ y) : blockKernel B hB E x y = 0
```

## CGB.FiberOptimizerReview.readout_cell_kernel_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2310)

解释：读出核非负、归一且满足细致平衡；合法状态不会跳过预算上界，也不会一次跨越多个内容层。

源码说明：Uniform cell/direction selection gives this kernel. Each adjacent readout block is drawn with probability 1/2 when legal, otherwise the attempt stays. The physical cell has 27 states but at most13 energies are examined per attempt. Finite stock is the invariant budget N+stock, not a direction-dependent filter inside a block. All old holding/release time certificates must be rederived.

形式陈述：

```lean
theorem readout_cell_kernel_contract (budget : Fin 7) (E : Cell → ℝ) :
    (∀ k : Fin 6, 4 ≤ (readoutBlock k).card ∧ (readoutBlock k).card ≤ 13) ∧
    (∀ x y, 0 ≤ cellKernel budget E x y) ∧
    (∀ x, (∑ y, cellKernel budget E x y) = 1) ∧
    (∀ x y, (2:ℝ)^(-2*E x)*cellKernel budget E x y =
      (2:ℝ)^(-2*E y)*cellKernel budget E y x) ∧
    (∀ x y, cellMass x ≤ budget.val → budget.val < cellMass y →
      cellKernel budget E x y = 0) ∧
    (∀ x y, cellMass x+1 < cellMass y ∨ cellMass y+1 < cellMass x →
      cellKernel budget E x y = 0)
```

## CGB.FiberOptimizerReview.remove_tilt

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2395)

解释：在指定参数条件下，倾斜能量中移除一个量子的能量变化不超过 34−f。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem remove_tilt (a b R C f : ℝ) (ha : 0 < a) (hb : 0 < b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (s : Cell) (hs : 0 < cellMass s) :
    tiltedEnergy a b R C f (removeOne s) - tiltedEnergy a b R C f s ≤ 34-f
```

## CGB.FiberOptimizerReview.add_tilt

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2404)

解释：在指定参数和容量条件下，倾斜能量中增加一个量子的变化满足定理给出的上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem add_tilt (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (s : Cell) (hs : cellMass s < 6) :
    tiltedEnergy a b R C f (addOne s) - tiltedEnergy a b R C f s ≤ 22+f
```

## CGB.FiberOptimizerReview.floor_direction

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2414)

解释：当外力和池资源满足给定条件时，相邻读出块存在定理所述方向的极小能量状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem floor_direction (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (k : Fin 6) :
    (34 < f → ∃ y ∈ readoutBlock k, cellMass y = k.val ∧
      blockRate (readoutBlock k) (block_nonempty k) (tiltedEnergy a b R C f) y =
        1/((readoutBlock k).card:ℝ)) ∧
    (f < -22 → ∃ y ∈ readoutBlock k, cellMass y = k.val+1 ∧
      blockRate (readoutBlock k) (block_nonempty k) (tiltedEnergy a b R C f) y =
        1/((readoutBlock k).card:ℝ))
```

## CGB.FiberOptimizerReview.cell_block_lower

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2460)

解释：当指定块对当前单元活跃时，单元核至少包含该块速率的一半。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem cell_block_lower (budget : Fin 7) (E : Cell → ℝ)
    (x y : Cell) (k : Fin 6) (hk : k ∈ activeBlocks budget x) :
    blockRate (readoutBlock k) (block_nonempty k) E y/2 ≤ cellKernel budget E x y
```

## CGB.FiberOptimizerReview.readout_force_response_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2486)

解释：在指定池规模、容量和方向力条件下，读出核对相邻内容块的响应满足给定下界。

源码说明：In the SAME scaled-pool energy, a sufficiently strong signed force moves an adjacent-block minimizer in its descent direction. There is a concrete successor of probability >=1/26 per selected cell, independent of body size. This is a transition/service bound, not a guarantee that an estimated gradient is useful, nor a transfer of the old microscopic holding/release time bound.

形式陈述：

```lean
theorem readout_force_response_contract (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (budget : Fin 7) (k : Fin 6) (hk : k.val+1 ≤ budget.val) (x : Cell) :
    (34 < f → cellMass x = k.val+1 → ∃ y, cellMass y+1 = cellMass x ∧
      1/26 ≤ cellKernel budget (tiltedEnergy a b R C f) x y) ∧
    (f < -22 → cellMass x = k.val → ∃ y, cellMass y = cellMass x+1 ∧
      1/26 ≤ cellKernel budget (tiltedEnergy a b R C f) x y)
```

## CGB.FiberOptimizerReview.full_of_mass

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2522)

解释：三字段单元的总内容为六时，该单元等于全部字段均满的状态。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem full_of_mass (s : Cell) (hs : cellMass s = 6) : s = fullCell
```

## CGB.FiberOptimizerReview.full_repair_gap

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2533)

解释：在指定资源与力条件下，内容为五的单元修复到满状态的倾斜能量差满足所列界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem full_repair_gap (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (s : Cell) (hs : cellMass s = 5) :
    tiltedEnergy a b R C f fullCell - tiltedEnergy a b R C f s ≤ -26+f
```

## CGB.FiberOptimizerReview.readout_regeneration_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2553)

解释：在指定资源条件、力不大于一且单元缺少一个量子时，读出核恢复满状态的概率至少为八分之一。

源码说明：Existing content restores a missing physical quantum directly: no stored reference value or loss test selects this successor. With stock available, all three one-quantum defects have repair probability >=1/8 per selected cell under the SAME block clock and scaled-pool H, uniformly in body size. Other transitions and subsequent wear still run; this is not yet a lifetime bound.

形式陈述：

```lean
theorem readout_regeneration_contract (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (hf : f ≤ 1) (x : Cell) (hx : cellMass x = 5) :
    1/8 ≤ cellKernel (6:Fin 7) (tiltedEnergy a b R C f) x fullCell
```

## CGB.FiberOptimizerReview.mass_le_six

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2593)

解释：指定三字段单元的内容总量不超过六。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem mass_le_six (s : Cell) : cellMass s ≤ 6
```

## CGB.FiberOptimizerReview.upper_block_rate

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2600)

解释：当指定外力至少为 96 时，向上相邻读出状态的块速率满足给定上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem upper_block_rate (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hf : 96 ≤ f) (k : Fin 6)
    (y : Cell) (hy : cellMass y = k.val+1) :
    blockRate (readoutBlock k) (block_nonempty k) (tiltedEnergy a b R C f) y ≤
      (1:ℝ)/2^124
```

## CGB.FiberOptimizerReview.upward_cell_rate

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2630)

解释：当指定外力至少为 96 时，内容增加的单元转移速率满足给定上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem upward_cell_rate (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hf : 96 ≤ f) (budget : Fin 7)
    (x y : Cell) (hy : cellMass x < cellMass y) :
    cellKernel budget (tiltedEnergy a b R C f) x y ≤ (1:ℝ)/2^124
```

## CGB.FiberOptimizerReview.drift_up_bound

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2664)

解释：若非负核只允许相邻内容变化，则单个转移的内容漂移满足定理所列上下分解界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem drift_up_bound (K : Cell → Cell → ℝ)
    (hn : ∀ x y, 0 ≤ K x y)
    (hfar : ∀ x y, cellMass x+1 < cellMass y ∨ cellMass y+1 < cellMass x → K x y = 0)
    (x y : Cell) :
    K x y*((cellMass y:ℝ)-cellMass x) ≤ if cellMass x < cellMass y then K x y else 0
```

## CGB.FiberOptimizerReview.readout_release_drift

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2684)

解释：在池容量与持续强力等前提下，指定相邻读出核的内容漂移满足给定释放上界。

源码说明：Actual adjacent-readout drift under a sustained strong positive force. Increasing readouts are exponentially rare; a nonempty cell has an explicit lower-level successor. The state-dependent force may vary between funded opportunities. This bound is size-independent per selected physical cell.

形式陈述：

```lean
theorem readout_release_drift (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (hf : 96 ≤ f) (budget : Fin 7) (x : Cell) (hx : cellMass x ≤ budget.val) :
    (∑ y, cellKernel budget (tiltedEnergy a b R C f) x y *
      ((cellMass y:ℝ)-cellMass x)) ≤
      (1:ℝ)/2^118 - (if 0 < cellMass x then 1/26 else 0)
```

## CGB.FiberOptimizerReview.push_cell_value

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2735)

解释：将单元分布通过可选映射推到主体状态后求观测期望，等于直接对各单元的映射值加权求和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem push_cell_value (r : Cell → ℝ) (d : Cell → Option NativeState)
    (v : NativeState → ℝ) :
    (∑ y, pushCell r d y*v y) = ∑ c, r c*(d c).elim 0 v
```

## CGB.FiberOptimizerReview.budget_holds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2749)

解释：合法主体状态中，任意位置的单元内容不超过该位置由主体状态生成的预算。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem budget_holds (x : NativeState) (i : Fin 4) :
    cellMass (x.val i) ≤ (bodyBudget x i).val
```

## CGB.FiberOptimizerReview.pack_mass

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2764)

解释：主体总内容不超过容量时，打包后的总内容与原主体总内容相同。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem pack_mass (b : NativeBody) (hcap : nativeMass b ≤ 12) :
    (nativePack b).elim 0 (fun x => (nativeMass x.val:ℝ)) = nativeMass b
```

## CGB.FiberOptimizerReview.weighted_update_mass

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2772)

解释：单元更新核权重乘打包后主体总内容可按定理所列未打包表达式计算。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem weighted_update_mass (f : Fin 4 → ℝ) (x : NativeState) (i : Fin 4) (c : Cell) :
    bodyCellKernel f x i (x.val i) c *
      (nativePack (Function.update x.val i c)).elim 0 (fun y => (nativeMass y.val:ℝ)) =
    bodyCellKernel f x i (x.val i) c *
      ((nativeMass x.val:ℝ)+(cellMass c:ℝ)-cellMass (x.val i))
```

## CGB.FiberOptimizerReview.body_kernel_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2797)

解释：指定时变主体读出核非负、每行和不超过一，并满足所列内容观测关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem body_kernel_contract (f : ℕ → NativeState → Fin 4 → ℝ) :
    (∀ t x y, 0 ≤ readoutBodyKernel f t x y) ∧
    (∀ t x, (∑ y, readoutBodyKernel f t x y) ≤ 1) ∧
    (∀ t x, (∑ y, readoutBodyKernel f t x y*(nativeMass y.val:ℝ)) =
      (nativeMass x.val:ℝ)+(1/4:ℝ)*∑ i, ∑ c,
        bodyCellKernel (f t x) x i (x.val i) c*((cellMass c:ℝ)-cellMass (x.val i)))
```

## CGB.FiberOptimizerReview.body_release_row

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2860)

解释：若每个时刻每个位置的外力至少为 96，则指定主体核的期望总内容至多为原内容的 1279/1280。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem body_release_row (f : ℕ → NativeState → Fin 4 → ℝ)
    (hf : ∀ t x i, 96 ≤ f t x i) (t : ℕ) (x : NativeState) :
    (∑ y, readoutBodyKernel f t x y*(nativeMass y.val:ℝ)) ≤
      (1279/1280:ℝ)*(nativeMass x.val:ℝ)
```

## CGB.FiberOptimizerReview.damage_mass_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2887)

解释：指定损伤核非负且行和不超过一，并且不会增加主体的加权期望总内容。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem damage_mass_contract :
    (∀ x y, 0 ≤ nativeDamageKernel x y) ∧
    (∀ x, (∑ y, nativeDamageKernel x y) ≤ 1) ∧
    (∀ x, (∑ y, nativeDamageKernel x y*(nativeMass y.val:ℝ)) ≤ nativeMass x.val)
```

## CGB.FiberOptimizerReview.readout_release_budget

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2930)

解释：在持续强力和指定损伤、库存与时钟条件下，四单元主体的开放核满足有限期限首次清空预算界。

源码说明：First empty hit for the actual four-cell readout clock, including ALL legal nonempty bodies, finite stock and the SAME continuing external wear. Forces may vary by time/state/cell but remain>=96. No assumed row certificate, retention property or target-directed repair selector is used. Fee is13 base units per counted funded opportunity, so old microscopic budgets do not transfer.

形式陈述：

```lean
theorem readout_release_budget (f : ℕ → NativeState → Fin 4 → ℝ)
    (hf : ∀ t x i, 96 ≤ f t x i) :
    (∀ t x y, 0 ≤ readoutOpenKernel f t x y) ∧
    (∀ t x, (∑ y, readoutOpenKernel f t x y) ≤ 1) ∧
    (∀ T start x, CGB.OptimizerContentReview.kernelSurvival (readoutOpenKernel f) T start x ≤
      (1279/1280:ℝ)^T*(nativeMass x.val:ℝ))
```

## CGB.FiberOptimizerReview.lower_block_rate

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L2977)

解释：在指定资源及不大于 −96 的力条件下，较低内容相邻状态的块速率满足给定上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem lower_block_rate (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (hf : f ≤ -96) (k : Fin 6) (y : Cell) (hy : cellMass y = k.val) :
    blockRate (readoutBlock k) (block_nonempty k) (tiltedEnergy a b R C f) y ≤
      (1:ℝ)/2^124
```

## CGB.FiberOptimizerReview.downward_cell_rate

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3006)

解释：在指定资源及不大于 −96 的力条件下，内容减少的单元转移速率满足给定上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem downward_cell_rate (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (hf : f ≤ -96) (budget : Fin 7) (x y : Cell) (hy : cellMass y < cellMass x) :
    cellKernel budget (tiltedEnergy a b R C f) x y ≤ (1:ℝ)/2^124
```

## CGB.FiberOptimizerReview.move_lower

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3043)

解释：非负核中，某个相邻方向单一后继状态的速率不超过该方向的全部移动概率。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem move_lower (K : Cell → Cell → ℝ) (hn : ∀ x y, 0 ≤ K x y)
    (x z : Cell) (up : Bool)
    (hz : if up then cellMass z = cellMass x+1 else cellMass z+1 = cellMass x) :
    K x z ≤ moveProbability K x up
```

## CGB.FiberOptimizerReview.move_small

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3055)

解释：若指定方向所有相邻后继的单项核速率不超过给定小量，则该方向移动概率满足定理所列上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem move_small (K : Cell → Cell → ℝ) (x : Cell) (up : Bool)
    (hb : ∀ y, (if up then cellMass y = cellMass x+1 else cellMass y+1 = cellMass x) →
      K x y ≤ (1:ℝ)/2^124) : moveProbability K x up ≤ (1:ℝ)/2^118
```

## CGB.FiberOptimizerReview.finite_risk_moves

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3067)

解释：在给定观测误差、平滑风险模型及相邻移动概率界下，有限单元的一步任务风险满足所列界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem finite_risk_moves (K : Cell → Cell → ℝ) (x : Cell) (risk : Cell → ℝ)
    (q estimate error L actual : ℝ) (positive : Bool)
    (hq : 0 ≤ q) (herror : |actual-(if positive then estimate else -estimate)| ≤ error)
    (hn : ∀ x y, 0 ≤ K x y)
    (hfar : ∀ x y, cellMass x+1 < cellMass y ∨ cellMass y+1 < cellMass x → K x y = 0)
    (hmodel : ∀ y, risk y-risk x ≤ actual*q*((cellMass y:ℝ)-cellMass x) +
      L*q^2/2*((cellMass y:ℝ)-cellMass x)^2) :
    let D
```

## CGB.FiberOptimizerReview.readout_task_risk_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3133)

解释：在指定力、梯度误差和局部二阶风险上界成立时，同一随机读出核满足给定任务风险界。

源码说明：The actual task-risk bridge for this SAME stochastic kernel. A local Taylor upper model and a gradient-error envelope are explicit properties of the objective/observation, not an assumed descent or a runtime acceptance criterion. No loss oracle, curvature array, reference weight or gate is added. At a feasible strong signed signal the source probabilities close the gain inequality; the final clause allows50% gradient error with L*q<=estimate/2. The force is in the existing port's energy units, so a caller must also verify its positive unit conversion and stale/noisy observation envelope.

形式陈述：

```lean
theorem readout_task_risk_contract (a b R C f q estimate error L actual : ℝ)
    (ha : 1 ≤ a) (hb : 1 ≤ b) (hR : 0 ≤ R) (hC : 0 ≤ C)
    (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b) (hf : 96 ≤ f)
    (hq : 0 ≤ q) (he : 0 ≤ estimate) (herr : 0 ≤ error) (hL : 0 ≤ L)
    (hbudget : q*error+L*q^2/2 ≤ q*estimate)
    (budget : Fin 7) (x : Cell) (hx : cellMass x ≤ budget.val) (risk : Cell → ℝ)
    (hmodel : ∀ y, risk y-risk x ≤ actual*q*((cellMass y:ℝ)-cellMass x) +
      L*q^2/2*((cellMass y:ℝ)-cellMass x)^2) :
    let D
```

## CGB.FiberOptimizerReview.autonomous_task_noise_floor

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3213)

解释：指定有限速率自主核在构造的标量损失最小点仍有定理给出的非零随机风险下界。

源码说明：A quantified noise floor is unavoidable for this autonomous finite-rate kernel: at the minimum of the scalar loss theta^2 and supplied gradient0, there remains a strictly positive expected loss increment. This is an exact analytic counterexample, not a capability task or a new optimizer mechanism.

形式陈述：

```lean
theorem autonomous_task_noise_floor :
    let E
```

## CGB.FiberOptimizerReview.outside_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3285)

解释：正行列规模下，位置以外的同行和同列内容分别加二后不超过对应容量上限。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem outside_bounds {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (b : Population r c) (i : Site r c) :
    outsideRow b i+2 ≤ 2*c ∧ outsideCol b i+2 ≤ 2*r
```

## CGB.FiberOptimizerReview.population_budget_holds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3304)

解释：任意群体位置的单元内容不超过根据供给与其他位置生成的局部预算。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_budget_holds {r c : ℕ} (supply : ℕ) (b : Population r c) (i : Site r c) :
    cellMass (b i) ≤ (populationBudget supply b i).val
```

## CGB.FiberOptimizerReview.population_mass_update

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3310)

解释：替换群体中的一个单元后，新总内容加旧单元内容等于旧总内容加新单元内容。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_mass_update {r c : ℕ} (b : Population r c) (i : Site r c) (z : Cell) :
    populationMass (Function.update b i z)+cellMass (b i) = populationMass b+cellMass z
```

## CGB.FiberOptimizerReview.push_population_value

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3335)

解释：将有限事件分布映射到群体后求观测期望，等于直接对事件映射后的群体观测值加权求和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem push_population_value {r c : ℕ} {E : Type*} [Fintype E] (p : E → ℝ)
    (d : E → Population r c) (v : Population r c → ℝ) :
    (∑ y, pushPopulation p d y*v y) = ∑ z, p z*v (d z)
```

## CGB.FiberOptimizerReview.population_value

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3345)

解释：群体核作用于任意观测函数的期望可表示为逐位置、逐单元状态的平均。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_value {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b : Population r c) (v : Population r c → ℝ) :
    (∑ y, populationKernel supply f b y*v y) =
      (1:ℝ)/(r*c:ℕ)*∑ i, ∑ z, populationCellKernel supply f b i (b i) z*
        v (Function.update b i z)
```

## CGB.FiberOptimizerReview.population_cell_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3357)

解释：群体中指定位置的单元核逐项非负，且从当前单元状态出发的转移概率之和为一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_cell_contract {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b : Population r c) (i : Site r c) :
    (∀ x z, 0 ≤ populationCellKernel supply f b i x z) ∧
    (∑ z, populationCellKernel supply f b i (b i) z) = 1
```

## CGB.FiberOptimizerReview.population_positive_edit

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3365)

解释：群体核以正概率转移时，后继可表示为只替换一个位置且该单元满足生成预算。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_positive_edit {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b y : Population r c) (h : 0 < populationKernel supply f b y) :
    ∃ i z, y=Function.update b i z ∧ cellMass z ≤ (populationBudget supply b i).val ∧
      |(cellMass z:ℝ)-cellMass (b i)| ≤ 1
```

## CGB.FiberOptimizerReview.population_kernel_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3415)

解释：对任意正矩形规模，指定未断环群体核非负、行和为一，并满足所列预算与单位置更新条件。

源码说明：Full uncut population at EVERY positive rectangular size. Empty bodies are retained. The same conditional cells and one conserved supply generate a Markov kernel, preserve the legal supply bound and have an exact task-value pushforward. No finite population is enumerated and no row identity is assumed.

形式陈述：

```lean
theorem population_kernel_contract {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) :
    (∀ b y, 0 ≤ populationKernel supply f b y) ∧
    (∀ b, (∑ y, populationKernel supply f b y)=1) ∧
    (∀ b y, populationMass b ≤ supply → 0 < populationKernel supply f b y → populationMass y ≤ supply) ∧
    (∀ b (F : Population r c → ℝ),
      (∑ y, populationKernel supply f b y*(F y-F b)) =
      (1:ℝ)/(r*c:ℕ)*∑ i, ∑ z, populationCellKernel supply f b i (b i) z*
        (F (Function.update b i z)-F b))
```

## CGB.FiberOptimizerReview.population_risk_row

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3464)

解释：在正矩形规模、非负步幅及所列局部风险条件下，群体核一步任务风险满足给定行级界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_risk_row {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) (q : ℝ) (hq : 0 ≤ q)
    (actual observed error L : Site r c → ℝ) (b : Population r c) (F : Population r c → ℝ)
    (he : ∀ i, |actual i-observed i| ≤ error i)
    (hm : ∀ i z, F (Function.update b i z)-F b ≤
      actual i*q*((cellMass z:ℝ)-cellMass (b i))+
        L i*q^2/2*((cellMass z:ℝ)-cellMass (b i))^2) :
    (∑ y, populationKernel supply f b y*(F y-F b)) ≤
      populationAllowance supply f q observed error L b
```

## CGB.FiberOptimizerReview.population_damage_value

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3505)

解释：指定群体损伤核对观测函数的期望，等于不损伤分支与按事件平均的损伤分支的加权和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_damage_value {r c : ℕ} (eta : ℝ) (b : Population r c)
    (F : Population r c → ℝ) :
    (∑ y, populationDamageKernel eta b y*F y) =
      (1-eta)*F b+eta*((1:ℝ)/(3*(r*c):ℕ)*∑ e : Site r c × Fin 3, F (populationDamage b e))
```

## CGB.FiberOptimizerReview.population_damage_markov

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3519)

解释：正矩形规模且损伤概率位于零到一时，指定群体损伤核非负并归一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_damage_markov {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1) :
    (∀ b y : Population r c, 0 ≤ populationDamageKernel eta b y) ∧
    (∀ b : Population r c, (∑ y, populationDamageKernel eta b y)=1)
```

## CGB.FiberOptimizerReview.population_world_value

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3548)

解释：指定群体世界核的观测期望可表示为先取中间状态再对后继状态求和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_world_value {r c : ℕ} (supply : ℕ) (eta : ℝ)
    (f : Site r c → ℝ) (b : Population r c) (F : Population r c → ℝ) :
    (∑ y, populationWorldKernel supply eta f b y*F y) =
      ∑ u, populationDamageKernel eta b u*(∑ y, populationKernel supply f u y*F y)
```

## CGB.FiberOptimizerReview.population_world_markov

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3556)

解释：正矩形规模且损伤概率位于零到一时，指定物理与损伤组合世界核非负并归一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_world_markov {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1) (f : Site r c → ℝ) :
    (∀ b y, 0 ≤ populationWorldKernel supply eta f b y) ∧
    (∀ b, (∑ y, populationWorldKernel supply eta f b y)=1)
```

## CGB.FiberOptimizerReview.population_world_risk

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3580)

解释：在给定群体世界核、观测误差和风险模型条件下，一步世界风险满足所列上界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_world_risk {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1) (f : Site r c → ℝ)
    (q : ℝ) (hq : 0 ≤ q) (observed : Site r c → ℝ)
    (actual error : Population r c → Site r c → ℝ) (L : Site r c → ℝ) (F : Population r c → ℝ)
    (he : ∀ b i, |actual b i-observed i| ≤ error b i)
    (hm : ∀ b i z, F (Function.update b i z)-F b ≤
      actual b i*q*((cellMass z:ℝ)-cellMass (b i))+
        L i*q^2/2*((cellMass z:ℝ)-cellMass (b i))^2) (b : Population r c) :
    (∑ y, populationWorldKernel supply eta f b y*(F y-F b)) ≤
      populationStepAllowance supply eta f q observed error L F b
```

## CGB.FiberOptimizerReview.population_task_window_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3630)

解释：在所声明误差、平滑和世界核条件下，同一未断环群体的有限窗口任务风险满足给定累计包络。

源码说明：Exact whole-window task-risk envelope of the SAME uncut population, including all weak/blocked coordinates and external wear. Frozen-observation error at later states is DERIVED from the initial error and objective-gradient Lipschitz bound. No negative task increment, task oracle, shadow weights or fitted admission score enters the algorithm. The signed envelope can expose harmful windows; it is a theorem observer, not an acceptance mechanism.

形式陈述：

```lean
theorem population_task_window_contract {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1)
    (f : ℕ → Site r c → ℝ) (q : ℝ) (hq : 0 ≤ q)
    (observed initialError sensitivity : Site r c → ℝ)
    (actual : Population r c → Site r c → ℝ) (L : Site r c → ℝ)
    (F : Population r c → ℝ) (origin : Population r c)
    (hInitial : ∀ i, |actual origin i-observed i| ≤ initialError i)
    (hLip : ∀ b y i, |actual b i-actual y i| ≤ sensitivity i*q*populationDistance b y)
    (hm : ∀ b i z, F (Function.update b i z)-F b ≤
      actual b i*q*((cellMass z:ℝ)-cellMass (b i))+
        L i*q^2/2*((cellMass z:ℝ)-cellMass (b i))^2) :
    let error
```

## CGB.FiberOptimizerReview.population_distance_triangle

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3688)

解释：指定群体距离满足三角不等式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_distance_triangle {r c : ℕ} (b u y : Population r c) :
    populationDistance b y ≤ populationDistance b u+populationDistance u y
```

## CGB.FiberOptimizerReview.population_distance_update

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3694)

解释：替换单个群体位置时，前后群体距离等于该位置内容量变化的绝对值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_distance_update {r c : ℕ} (b : Population r c) (i : Site r c) (z : Cell) :
    populationDistance b (Function.update b i z)=|(cellMass (b i):ℝ)-cellMass z|
```

## CGB.FiberOptimizerReview.population_physical_distance

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3704)

解释：群体物理核正概率支持的任意一步，指定群体距离不超过一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_physical_distance {r c : ℕ} (supply : ℕ) (f : Site r c → ℝ)
    (b y : Population r c) (h : 0 < populationKernel supply f b y) : populationDistance b y ≤ 1
```

## CGB.FiberOptimizerReview.damage_cell_one

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3710)

解释：一次指定单字段损伤不增加单元内容，且内容改变量的绝对值不超过一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem damage_cell_one (s : Cell) (field : Fin 3) :
    cellMass (damageCell s field) ≤ cellMass s ∧
      |(cellMass (damageCell s field):ℝ)-cellMass s| ≤ 1
```

## CGB.FiberOptimizerReview.population_damage_event

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3717)

解释：一次指定群体损伤事件不增加总内容，且改变前后的群体距离不超过一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_damage_event {r c : ℕ} (b : Population r c) (e : Site r c × Fin 3) :
    populationMass (populationDamage b e) ≤ populationMass b ∧
      populationDistance b (populationDamage b e) ≤ 1
```

## CGB.FiberOptimizerReview.population_damage_support

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3728)

解释：损伤核非负概率支持的后继状态不增加总内容，并满足定理给出的距离约束。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_damage_support {r c : ℕ} (eta : ℝ) (he0 : 0 ≤ eta)
    (b y : Population r c) (h : 0 < populationDamageKernel eta b y) :
    populationMass y ≤ populationMass b ∧ populationDistance b y ≤ 1
```

## CGB.FiberOptimizerReview.population_world_support

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3752)

解释：指定物理和损伤组合世界核的正概率后继满足定理所列的一步支持范围。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_world_support {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1) (f : Site r c → ℝ)
    (b y : Population r c) (h : 0 < populationWorldKernel supply eta f b y) :
    (populationMass b ≤ supply → populationMass y ≤ supply) ∧ populationDistance b y ≤ 2
```

## CGB.FiberOptimizerReview.population_distance_capacity

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3772)

解释：两个群体总内容均不超过供给时，它们的指定距离不超过供给的两倍。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_distance_capacity {r c : ℕ} (supply : ℕ) (b y : Population r c)
    (hb : populationMass b ≤ supply) (hy : populationMass y ≤ supply) :
    populationDistance b y ≤ 2*(supply:ℝ)
```

## CGB.FiberOptimizerReview.population_frozen_gradient_budget

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3793)

解释：沿满足指定世界支持条件的群体轨迹，冻结观测的有限窗口变化由实际一步支持和容量界共同约束。

源码说明：Frozen observations need no shadow weights or per-parameter age. Source support bounds each actual funded opportunity (wear then physical event) by two readout quanta; supply also bounds the total reachable displacement. A true objective-gradient Lipschitz bound and the INITIAL observation error therefore derive the later error envelope, rather than assuming it each tick. The trace is in support of the ACTUAL uncut kernel, including no-change steps.

形式陈述：

```lean
theorem population_frozen_gradient_budget {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1)
    (f : ℕ → Site r c → ℝ) (trace : ℕ → Population r c)
    (hs : populationMass (trace 0) ≤ supply)
    (ht : ∀ t, 0 < populationWorldKernel supply eta (f t) (trace t) (trace (t+1)))
    (q : ℝ) (hq : 0 ≤ q) (sensitivity initialError observed : Site r c → ℝ)
    (hL : ∀ i, 0 ≤ sensitivity i) (actual : Population r c → Site r c → ℝ)
    (hLip : ∀ b y i, |actual b i-actual y i| ≤ sensitivity i*q*populationDistance b y)
    (hInitial : ∀ i, |actual (trace 0) i-observed i| ≤ initialError i) :
    (∀ T, populationMass (trace T) ≤ supply) ∧
    (∀ T, populationDistance (trace 0) (trace T) ≤ min (2*(T:ℝ)) (2*(supply:ℝ))) ∧
    (∀ T i, |actual (trace T) i-observed i| ≤ initialError i+
      sensitivity i*q*min (2*(T:ℝ)) (2*(supply:ℝ)))
```

## CGB.FiberOptimizerReview.reduced_geometry

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3858)

解释：每个简化读出块非空、包含于完整块、大小至多三，并满足给定代表元映射关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem reduced_geometry (k : Fin 6) :
    (reducedReadoutBlock k).Nonempty ∧
    reducedReadoutBlock k ⊆ readoutBlock k ∧
    (reducedReadoutBlock k).card ≤ 3 ∧
    (∀ s ∈ readoutBlock k, levelRepresentative s ∈ reducedReadoutBlock k ∧
      cellMass (levelRepresentative s) = cellMass s)
```

## CGB.FiberOptimizerReview.representative_min

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3869)

解释：在指定池增量单调且受界时，单元的代表状态满足定理所列池能量最小化关系。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem representative_min (P Q : Fin 3 → ℝ) (f : ℝ)
    (hp0 : 0 ≤ P 1-P 0) (hp1 : 0 ≤ P 2-P 1)
    (hq0 : 0 ≤ Q 1-Q 0) (hq1 : 0 ≤ Q 2-Q 1)
    (hp8 : P 1-P 0 ≤ 8) (hq8 : Q 1-Q 0 ≤ 8) (s : Cell) :
    poolForm P Q f (levelRepresentative s) ≤ poolForm P Q f s
```

## CGB.FiberOptimizerReview.positive_pool_increment

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3878)

解释：正规模及非负局部资源条件下，指定二次池成本增加一单位的变化严格为正。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem positive_pool_increment (D R p : ℝ) (hD : 0 < D) (hR : 0 ≤ R) (hp : 0 ≤ p) :
    0 < 2*(R+p+1)^2/D-2*(R+p)^2/D
```

## CGB.FiberOptimizerReview.readout_floor_reduction

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3887)

解释：在指定池条件下，完整读出块的最小能量可通过不超过三个代表能量精确计算。

源码说明：For the registered H the exact block floor uses at most three energies. All original microstates remain in the uniform proposal; only evaluation work is reduced. Arbitrary signed force is constant within each readout level.

形式陈述：

```lean
theorem readout_floor_reduction (a b R C f : ℝ) (ha : 1 ≤ a) (hb : 1 ≤ b)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hRp : R+2 ≤ 2*a) (hCq : C+2 ≤ 2*b)
    (k : Fin 6) (hfull : (readoutBlock k).Nonempty) :
    ∃ (B : Finset Cell) (hB : B.Nonempty), B ⊆ readoutBlock k ∧ B.card ≤ 3 ∧
      blockFloor B hB (tiltedEnergy a b R C f) =
        blockFloor (readoutBlock k) hfull (tiltedEnergy a b R C f)
```

## CGB.FiberOptimizerReview.population_up_choice

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3915)

解释：在指定强负力、单元未满和群体资源条件下，存在符合定理要求的单元增长选择。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_up_choice {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) (b : Population r c) (i : Site r c)
    (hf : f i < -22) (ht : cellMass (b i) < 6) (hs : populationMass b < supply) :
    ∃ z, cellMass z=cellMass (b i)+1 ∧
      (1:ℝ)/26 ≤ populationCellKernel supply f b i (b i) z
```

## CGB.FiberOptimizerReview.population_down_choice

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3930)

解释：在指定强正力且单元非空时，存在符合定理要求的单元减少选择。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_down_choice {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) (b : Population r c) (i : Site r c)
    (hf : 34 < f i) (ht : 0 < cellMass (b i)) :
    ∃ z, cellMass z+1=cellMass (b i) ∧
      (1:ℝ)/26 ≤ populationCellKernel supply f b i (b i) z
```

## CGB.FiberOptimizerReview.population_world_expect_lower

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3947)

解释：对非负观测函数及所列世界核条件，组合世界一步期望满足指定下界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_world_expect_lower {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he : 0 ≤ eta) (f : Site r c → ℝ)
    (b : Population r c) (F : Population r c → ℝ) (hF : ∀ y, 0 ≤ F y) :
    (1-eta)*(∑ y, populationKernel supply f b y*F y) ≤
      ∑ y, populationWorldKernel supply eta f b y*F y
```

## CGB.FiberOptimizerReview.population_up_expect

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3962)

解释：在指定强负力、容量及资源条件下，群体核的增长事件期望具有给定下界。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem population_up_expect {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (f : Site r c → ℝ) (b : Population r c) (i : Site r c)
    (hf : f i < -22) (ht : cellMass (b i) < 6) (hs : populationMass b < supply) :
    (1:ℝ)/(r*c:ℕ)/26 ≤ ∑ y, populationKernel supply f b y*
      (if cellMass (y i)=cellMass (b i)+1 then 1 else 0)
```

## CGB.FiberOptimizerReview.population_resource_service

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L3992)

解释：在所列群体资源、供需和力条件下，指定材料交换过程可以为目标位置的增长提供定理所述服务。

源码说明：The same conservative population can serve a new growth request even with empty stock: actual donor release, then actual target uptake. Donor set is a proof observer of signed evidence, never a runtime queue or score. The bound retains all other choices and external damage. It is a finite-window service probability, not an assumed task benefit or all-request fairness.

形式陈述：

```lean
theorem population_resource_service {r c : ℕ} (hr : 0 < r) (hc : 0 < c)
    (supply : ℕ) (eta : ℝ) (he0 : 0 ≤ eta) (he1 : eta ≤ 1)
    (f0 f1 : Site r c → ℝ) (b : Population r c) (target : Site r c)
    (hf : f1 target < -22) (ht : cellMass (b target) < 6) :
    (populationMass b < supply → (1-eta)/(r*c:ℕ)/26 ≤
      ∑ y, populationWorldKernel supply eta f1 b y*
        (if cellMass (y target)=cellMass (b target)+1 then 1 else 0)) ∧
    (populationMass b=supply → ∀ D : Finset (Site r c),
      (∀ i ∈ D, i≠target ∧ 0 < cellMass (b i) ∧ 34 < f0 i) →
      (D.card:ℝ)*((1-eta)/(r*c:ℕ)/26)^2 ≤
        ∑ u, populationWorldKernel supply eta f0 b u*
          (∑ y, populationWorldKernel supply eta f1 u y*
            (if cellMass (y target)=cellMass (b target)+1 then 1 else 0)))
```

## CGB.FiberOptimizerReview.matching_outside

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L4062)

解释：两个位置不同行且不同列时，修改其中一个不会改变另一个位置的同行及同列外部资源量。

源码说明：Two different rows AND columns have no shared pool dependence. This is exactly the matching restriction used by parallel preparation, not an assumed independence of arbitrary material sites.

形式陈述：

```lean
private theorem matching_outside {r c : ℕ} (b : Population r c) (i j : Site r c)
    (hr : i.1 ≠ j.1) (hc : i.2 ≠ j.2) (z : Cell) :
    outsideRow (Function.update b j z) i=outsideRow b i ∧
    outsideCol (Function.update b j z) i=outsideCol b i ∧
    Function.update b j z i=b i
```

## CGB.FiberOptimizerReview.matching_parallel_preparation

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L4084)

解释：不同行不同列位置的局部修改保持所列能量独立关系，并满足指定并行库存累计账本。

源码说明：Exact reflected-cumulative stock accounting. Prefix is cumulative PROPOSED material change, peak starts at initial stock. At a zero-stock upward step, reflection cancels the unfunded intake; every other step is conserved.

形式陈述：

```lean
theorem matching_parallel_preparation {r c : ℕ} (b : Population r c) (i j : Site r c)
    (hr : i.1≠j.1) (hc : i.2≠j.2) (z x : Cell) (f : ℝ) :
    tiltedEnergy c r (outsideRow (Function.update b j z) i)
      (outsideCol (Function.update b j z) i) f x =
      tiltedEnergy c r (outsideRow b i) (outsideCol b i) f x ∧
    Function.update (Function.update b i x) j z = Function.update (Function.update b j z) i x ∧
    (∀ cumulative peak delta : ℤ, cumulative ≤ peak → -1 ≤ delta → delta ≤ 1 →
      let stock
```

## CGB.FiberOptimizerReview.matching_coin_error

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L4111)

解释：当有限随机数级数为正且输入概率合法时，有限硬币误差受一格量化误差和所供概率误差共同约束。

源码说明：Finite GPU coin contributes at most one grid unit in addition to the explicit error of its supplied probability. This does not assume a GPU exp2 implementation is exact; its error remains a separate numerical obligation.

形式陈述：

```lean
private theorem matching_coin_error (p q : ℝ) (levels : ℕ) (hN : 0 < levels)
    (hq0 : 0 ≤ q) (hq1 : q ≤ 1) :
    let k
```

## CGB.FiberOptimizerReview.matching_coin_task_error

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_content_bridge.lean#L4143)

解释：逐事件概率近似误差有界时，其对有限事件加权任务增量的影响满足所列累计误差界。

源码说明：Propagate the actual finite coin error into task increments. This is an error budget, not an assumption that the ideal or implemented step descends. The physical stock guard is shared when comparing the two increments.

形式陈述：

```lean
private theorem matching_coin_task_error {ι : Type*} [Fintype ι]
    (p q d : ι → ℝ) (levels : ℕ) (hN : 0 < levels) (epsilon : ℝ)
    (hq0 : ∀ i, 0 ≤ q i) (hq1 : ∀ i, q i ≤ 1)
    (herr : ∀ i, |q i-p i| ≤ epsilon) :
    (∑ i, (Int.floor (q i*levels):ℝ)/levels*d i) ≤
      (∑ i, p i*d i)+(epsilon+1/(levels:ℝ))*(∑ i, |d i|)
```
