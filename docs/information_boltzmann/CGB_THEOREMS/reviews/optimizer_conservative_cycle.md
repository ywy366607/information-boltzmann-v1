# optimizer_conservative_cycle

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean) · [返回导航](../README.md)

模块源码说明：Conditional generative construction using double-budget geometry and a G20 pool.

## CGB.ConservativeCycleReview.clip_at_one_iff

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L39)

解释：指定截断函数输出一当且仅当输入不小于一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem clip_at_one_iff (x : ℝ) : clip x = 1 ↔ 1 ≤ x
```

## CGB.ConservativeCycleReview.clip_at_neg_one_iff

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L51)

解释：指定截断函数对 −x 输出负一当且仅当 x 不小于一。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem clip_at_neg_one_iff (x : ℝ) : clip (-x) = -1 ↔ 1 ≤ x
```

## CGB.ConservativeCycleReview.regeneration_threshold

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L68)

解释：在指定正补充率和非负使用率下，饱和模式成为固定点的条件等价于给定资源平衡与反馈阈值。

源码说明：Generative threshold of the DECLARED law, not an assumed retention score. At a saturated mode the resource balance fixes r=s/(s+u). Both opposite modes are fixed precisely when feedback through that resource compensates passive loss. The uncommitted mode remains fixed too. This classifies equilibria only; attraction, stochastic escape, useful content and optimality do not follow.

形式陈述：

```lean
theorem regeneration_threshold (q coupling cueScale refill useRate : ℝ)
    (hs : 0 < refill) (hu : 0 ≤ useRate) :
    let r
```

## CGB.ConservativeCycleReview.clip_bounds

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L97)

解释：指定截断函数的输出始终位于负一到一之间。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem clip_bounds (x : ℝ) : -1 ≤ clip x ∧ clip x ≤ 1
```

## CGB.ConservativeCycleReview.clip_lower

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L102)

解释：若 y 不大于 x，则截断后的 x 至少为 min(1,y)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem clip_lower {x y : ℝ} (h : y ≤ x) : min 1 y ≤ clip x
```

## CGB.ConservativeCycleReview.cycle_geometry

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L108)

解释：当相位 m 位于负一到一时，给定 2×2 分配非负、行列预算均为一且平方收益等于 1+m²。

源码说明：The local allocation always preserves BOTH budgets; its positive-feedback payoff is exactly 1+m^2. Differentiate this quadratic to obtain the linear regenerative force; no post-hoc Sinkhorn modification of a parameter step.

形式陈述：

```lean
theorem cycle_geometry (m : ℝ) (hm0 : -1 ≤ m) (hm1 : m ≤ 1) :
    (∀ i j, 0 ≤ allocation m i j) ∧
    (∀ i, allocation m i 0 + allocation m i 1 = 1) ∧
    (∀ j, allocation m 0 j + allocation m 1 j = 1) ∧
    (allocation m 0 0) ^ 2 + (allocation m 0 1) ^ 2 +
      (allocation m 1 0) ^ 2 + (allocation m 1 1) ^ 2 = 1 + m ^ 2
```

## CGB.ConservativeCycleReview.joint_step_invariant

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L140)

解释：在指定补给、使用和扰动界下，同一局部递推保持相位合法和资源下界，并满足所列资源收支。

源码说明：In the SAME recursion, the state stays legal and reserve stays above a strictly positive chosen floor whenever supply supports the declared worst use. The last identity is the actual replenishment/use ledger.

形式陈述：

```lean
theorem joint_step_invariant (q coupling cueScale refill useRate m reserve cue noise floor : ℝ)
    (_hs : 0 ≤ refill) (hu : 0 ≤ useRate) (hsu : refill + useRate ≤ 1)
    (hf0 : 0 ≤ floor) (hfr : floor ≤ reserve) (hr1 : reserve ≤ 1)
    (hfuel : (refill + useRate) * floor ≤ refill)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1) :
    let z
```

## CGB.ConservativeCycleReview.quiet_restoration

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L166)

解释：在正记忆盆地及足够反馈、资源和有界相反输入与噪声的条件下，该递推具有正的恢复余量。

源码说明：Within the positive memory basin, bounded contrary input and bounded noise still give a uniform restoring margin. This is active regeneration against q<1, not a zero-update gate. The reserve lower bound is maintained by joint_step_invariant, rather than assumed afresh at each time step.

形式陈述：

```lean
theorem quiet_restoration (q coupling cueScale refill useRate m reserve cue noise
    floor threshold cueBound noiseBound margin : ℝ)
    (hk : 0 ≤ coupling) (he : 0 ≤ cueScale)
    (hfloor : floor ≤ reserve) (hr0 : 0 ≤ reserve) (hr1 : reserve ≤ 1)
    (hth : 0 ≤ threshold) (hm : threshold ≤ m)
    (hg : 1 ≤ q + coupling * floor)
    (hcue0 : 0 ≤ cueBound) (hcue : |cue| ≤ cueBound)
    (hnoise : |noise| ≤ noiseBound)
    (hmargin : margin ≤ (q + coupling * floor - 1) * threshold -
      cueScale * cueBound - noiseBound) :
    min 1 (m + margin) ≤
      (step q coupling cueScale refill useRate m reserve cue noise).1
```

## CGB.ConservativeCycleReview.strong_cue_release

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L193)

解释：当相反线索足够强并满足所列噪声与资源条件时，同一递推的相位沿释放方向推进。

源码说明：A sufficiently contrary supplied cue overcomes the SAME recurrent force. No separate release bit or duration counter is added. Repeated such steps cross the old basin once the accumulated margin exceeds its width.

形式陈述：

```lean
theorem strong_cue_release (q coupling cueScale refill useRate m reserve cue noise
    floor cueSize noiseBound margin : ℝ)
    (hk : 0 ≤ coupling) (he : 0 ≤ cueScale)
    (hfloor : floor ≤ reserve) (hr0 : 0 ≤ reserve)
    (hm1 : m ≤ 1) (hg : 1 ≤ q + coupling * floor)
    (hsize : coupling ≤ cueScale * cueSize) (hcue : cue ≤ -cueSize)
    (hnoise : |noise| ≤ noiseBound)
    (hmargin : margin ≤ 1 - q + (cueScale * cueSize - coupling) * floor - noiseBound) :
    (step q coupling cueScale refill useRate m reserve cue noise).1 ≤
      max (-1) (m - margin)
```

## CGB.ConservativeCycleReview.clip_neg

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L218)

解释：指定截断函数是奇函数，即对负输入的输出等于原输出的负值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem clip_neg (x : ℝ) : clip (-x) = -clip x
```

## CGB.ConservativeCycleReview.normalized_joint_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L228)

解释：对合法相位和资源及有界扰动，指定归一化一步递推满足其联合状态和预算约束。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem normalized_joint_contract (m reserve cue noise : ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ)) :
    let z
```

## CGB.ConservativeCycleReview.restoration_by_budget

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L270)

解释：若每拍推进至少给定余量且状态不超过一，当累计余量覆盖初始缺口时，该拍状态达到一。

源码说明：Quantitative finite-time bridge: repeated restoring progress reaches the capacity endpoint by any N whose margin covers the initial deficit. This does not assume that a trajectory merely visits the endpoint eventually.

形式陈述：

```lean
theorem restoration_by_budget (x : ℕ → ℝ) (margin : ℝ) (hm : 0 ≤ margin)
    (hbound : ∀ t, x t ≤ 1)
    (hprogress : ∀ t, min 1 (x t + margin) ≤ x (t + 1))
    (N : ℕ) (hN : 1 ≤ x 0 + (N : ℝ) * margin) : x N = 1
```

## CGB.ConservativeCycleReview.normalized_trajectory_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L307)

解释：在所列合法初态和噪声界下，同一归一化轨迹满足资源约束及定理规定的有限步恢复与释放性质。

源码说明：Closed trajectory certificate for one actual recursion and all input sequences in the declared disturbance class. Quiet-basin perturbations are repaired within 12 steps; persistent strong contrary evidence switches even the most deeply held state within 8 steps. Budgets/resources remain legal throughout. These are local kernel bounds, not language quality evidence.

形式陈述：

```lean
theorem normalized_trajectory_contract (cue noise : ℕ → ℝ) (m₀ r₀ : ℝ)
    (hm0 : -1 ≤ m₀) (hm1 : m₀ ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ r₀) (hr1 : r₀ ≤ 1)
    (hnoise : ∀ t, |noise t| ≤ (1 / 32 : ℝ)) :
    let z
```

## CGB.ConservativeCycleReview.gradient_cycle_response

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L380)

解释：指定 2×2 梯度分配的工作量对相位修改的变化，等于修改量乘对角与交叉梯度平方差的一半。

源码说明：The supplied gradient determines the EXACT finite response of the allocated first-order descent work to a cycle edit. Thus the cue direction has an objective-derived source, rather than being an unrelated trigger. Its statistical reliability and normalization are not assumed proved here.

形式陈述：

```lean
theorem gradient_cycle_response (m change : ℝ) (g : Fin 2 → Fin 2 → ℝ) :
    gradientWork (m + change) g - gradientWork m g = change / 2 *
      ((g 0 0) ^ 2 + (g 1 1) ^ 2 - (g 0 1) ^ 2 - (g 1 0) ^ 2)
```

## CGB.ConservativeCycleReview.scalar_readout_safe

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L386)

解释：当分配比例在零到一、步长非负且 L·t≤1 时，指定标量二次下降估计至少保留给定的一半项。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem scalar_readout_safe (a t L g : ℝ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (ht : 0 ≤ t) (hL : 0 ≤ L) (hstep : L * t ≤ 1) :
    t / 2 * a * g ^ 2 ≤ g * (t * a * g) - L / 2 * (t * a * g) ^ 2
```

## CGB.ConservativeCycleReview.allocated_readout_safe

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L400)

解释：在合法 2×2 分配和 L·t≤1 的条件下，带分配的读出满足所列二次下降下界。

源码说明：The parameter readout is derived WITH its resource mobility, not renormalized afterward. For u_ij=t*A_ij*g_ij and an L-smooth objective's quadratic upper bound, t*L<=1 leaves at least half the first-order gain. Together with normalized_trajectory_contract this covers the actual step for every evolving legal allocation, rather than only the raw proposal.

形式陈述：

```lean
theorem allocated_readout_safe (m t L : ℝ) (g : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1) (ht : 0 ≤ t) (hL : 0 ≤ L)
    (hstep : L * t ≤ 1) :
    let u
```

## CGB.ConservativeCycleReview.relative_contrast_coverage

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L426)

解释：在相位、资源、相对对比度和噪声满足指定界时，再生更新后的分配覆盖给定比例的当前工作机会。

源码说明：Coupling and bounded disturbances do not allow a stale phase to suppress almost all of the CURRENT descent opportunity once the cue is the relative work contrast. No positive per-coordinate floor is imposed.

形式陈述：

```lean
theorem relative_contrast_coverage (m reserve contrast noise : ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hc0 : -1 ≤ contrast) (hc1 : contrast ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ)) :
    (1 / 16 : ℝ) ≤
      (1 + (normalizedStep m reserve (2 * contrast) noise).1 * contrast) / 2
```

## CGB.ConservativeCycleReview.encoded_gradient_cue_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L478)

解释：指定整数梯度线索位于 −64 到 64，且归一化后距实数相对线索不超过 1/64。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem encoded_gradient_cue_contract (g : Fin 2 → Fin 2 → ℝ) :
    (-64 ≤ encodedGradientCue g ∧ encodedGradientCue g ≤ 64) ∧
    |(encodedGradientCue g : ℝ) / 32 - relativeGradientCue g| ≤ (1 / 64 : ℝ)
```

## CGB.ConservativeCycleReview.relative_gradient_coverage

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L512)

解释：在给定资源、相位和扰动界下，精确梯度驱动的再生分配保留至少 1/16 的总平方梯度工作量。

源码说明：Exact-gradient interface for the same regenerative step: at least 1/16 of total squared-gradient work remains available, regardless of the old phase. This addresses useful current participation, not future retention or a proof that each coordinate must update on every step.

形式陈述：

```lean
theorem relative_gradient_coverage (m reserve noise : ℝ) (g : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ)) :
    let next
```

## CGB.ConservativeCycleReview.exact_optimizer_step

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L565)

解释：在指定状态、噪声及步长条件下，包含资源与再生分配的精确梯度步满足给定二次下降下界。

源码说明：A complete exact-gradient parameter step, including the OLD reserve, NEW regenerated allocation and actual final readout. Its quadratic decrease is bounded below uniformly over the old phase; the state update simultaneously preserves both resource budgets and the reserve invariant. This does not ascribe weight restoration to a multiplicative mobility.

形式陈述：

```lean
theorem exact_optimizer_step (m reserve noise η L : ℝ) (g : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1) :
    let z
```

## CGB.ConservativeCycleReview.perturbed_scalar_readout_safe

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L597)

解释：当分配和步长满足给定界时，标量估计梯度步的二次下降下界包含估计平方项及误差平方惩罚。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem perturbed_scalar_readout_safe (a t L estimate actual : ℝ)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1) (ht : 0 ≤ t) (hL : 0 ≤ L) (hstep : L * t ≤ 1) :
    t / 4 * a * estimate ^ 2 - t * (actual - estimate) ^ 2 ≤
      actual * (t * a * estimate) - L / 2 * (t * a * estimate) ^ 2
```

## CGB.ConservativeCycleReview.noisy_optimizer_step

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L613)

解释：若估计误差和完整目标的平滑上界满足定理条件，则含再生分配的估计梯度步满足对应下降保证。

源码说明：Conditional estimated-gradient guarantee for the FINAL step. The cue is formed from the supplied estimate, while actual may be a different gradient. The relative error and the smooth upper model are explicit obligations; no claim that a batch, variance estimate or trust controller satisfies them. The same state/resource recursion still applies.

形式陈述：

```lean
theorem noisy_optimizer_step (m reserve noise η L before after : ℝ)
    (estimate actual : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1)
    (herror : (actual 0 0 - estimate 0 0) ^ 2 + (actual 0 1 - estimate 0 1) ^ 2 +
      (actual 1 0 - estimate 1 0) ^ 2 + (actual 1 1 - estimate 1 1) ^ 2 ≤
      ((estimate 0 0) ^ 2 + (estimate 1 1) ^ 2 +
        (estimate 0 1) ^ 2 + (estimate 1 0) ^ 2) / 128)
    (hmodel :
      let z
```

## CGB.ConservativeCycleReview.integer_cycle_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L690)

解释：对合法整数相位、资源和线索，该局部整数一步递推保持给定取值与资源账本范围。

源码说明：Actual integer state, supply/use ledger, packing bound and arithmetic range in ONE step. Ceil supply and floor use keep R in [8,16]; no floor on individual parameter updates is added. The 585 possible state pairs fit in 10 information bits (two ordinary bytes); all intermediates fit INT16.

形式陈述：

```lean
theorem integer_cycle_contract (K R C : ℤ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16) (hc0 : -64 ≤ C) (hc1 : C ≤ 64) :
    let z
```

## CGB.ConservativeCycleReview.integer_clip_cast

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L722)

解释：整数截断至 −32 到 32 再除以 32，等于先缩放再应用实数截断。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem integer_clip_cast (q : ℤ) :
    ((max (-32) (min 32 q) : ℤ) : ℝ) / 32 = clip ((q : ℝ) / 32)
```

## CGB.ConservativeCycleReview.integer_phase_refines

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L747)

解释：给定整数资源及线索量化误差界时，整数相位一步可表示为实数归一化递推加至多 1/32 的扰动。

源码说明：The integer phase update REFINES the proved real phase update with a bounded disturbance. Half of the 1/32 budget pays for phase rounding and half pays for cue quantization; no claim of lossless quantization is made. The integer reserve law has its own exact ledger above, rather than being silently identified with the unrounded G20 recurrence.

形式陈述：

```lean
theorem integer_phase_refines (K R C : ℤ) (cue : ℝ)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16)
    (hcue : |(C : ℝ) / 32 - cue| ≤ (1 / 64 : ℝ)) :
    ∃ noise : ℝ, |noise| ≤ (1 / 32 : ℝ) ∧
      ((integerCycleStep K R C).1 : ℝ) / 32 =
        (normalizedStep ((K : ℝ) / 32) ((R : ℝ) / 16) cue noise).1
```

## CGB.ConservativeCycleReview.integer_code_roundtrip

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L788)

解释：在所列合法范围内，打包编码可精确恢复整数相位和资源两个字段。

源码说明：Representation proof only: both shared fields are recovered exactly. The compact code has 585 legal values; this is a sufficient encoding, not a claim that ten bits is necessary for every functional optimizer.

形式陈述：

```lean
theorem integer_code_roundtrip (K R : ℤ) (hk0 : -32 ≤ K) (hk1 : K ≤ 32)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16) :
    let code
```

## CGB.ConservativeCycleReview.integer_joint_optimizer_step

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L800)

解释：在给定整数范围、线索误差和步长条件下，同一局部递推联合满足状态合法、保持释放和参数步条件。

源码说明：Joint integer optimizer certificate: the SAME finite recursion provides legal state, restorative phase retention, contrary-evidence release and a safe final parameter step. No separate witness is substituted for one of these properties. Historical FUNCTION preservation and task evidence are not inferred from phase retention.

形式陈述：

```lean
theorem integer_joint_optimizer_step (K R C : ℤ) (η L : ℝ)
    (g : Fin 2 → Fin 2 → ℝ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16) (hc0 : -64 ≤ C) (hc1 : C ≤ 64)
    (hcue : |(C : ℝ) / 32 - relativeGradientCue g| ≤ (1 / 64 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1) :
    let z
```

## CGB.ConservativeCycleReview.finite_block_optimizer_step

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L888)

解释：对有限个不重叠参数块，若整体目标满足给定平滑上界且每块条件成立，可组合出整步下降证书。

源码说明：Composition over any finite number of DISJOINT coordinate blocks. The smooth upper model is for the whole objective, so off-block curvature is not silently set to zero. This proves an actual parameter update bound; the tensor layout must separately realize these coordinates without loss. Local state restoration/release remains supplied by the very same integer step, rather than by an independently chosen block witness.

形式陈述：

```lean
theorem finite_block_optimizer_step {n : ℕ} (K R C : Fin n → ℤ) (η L : ℝ)
    (F : Parameters n → ℝ) (θ g : Parameters n)
    (hlegal : ∀ b,
      (-32 ≤ K b ∧ K b ≤ 32) ∧ (8 ≤ R b ∧ R b ≤ 16) ∧
      (-64 ≤ C b ∧ C b ≤ 64) ∧
      |(C b : ℝ) / 32 - relativeGradientCue (g b)| ≤ (1 / 64 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1)
    (hmodel :
      F (θ - integerReadout K R C η g) ≤ F θ -
        (∑ b, dotFour (g b) (integerReadout K R C η g b)) +
        L / 2 * ∑ b, normFour (integerReadout K R C η g b)) :
    (∀ b, let z
```

## CGB.ConservativeCycleReview.unrestricted_parameter_initialization

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L925)

解释：任意实数参数向量都可不改动地放入具有合法初始相位和资源的优化器状态。

源码说明：Internal finite-state encoding imposes NO alphabet/range on parameters. Every supplied real vector is a legal initial parameter vector, unchanged. This is an interface contract, not a new dynamical mechanism.

形式陈述：

```lean
theorem unrestricted_parameter_initialization {n : ℕ} (θ : Parameters n) :
    ∃ s : OptimizerState n, s.parameter=θ ∧
      (∀ b, s.phase b=0 ∧ s.reserve b=16) ∧
      (∀ b, (-32 ≤ s.phase b ∧ s.phase b ≤ 32) ∧
        (8 ≤ s.reserve b ∧ s.reserve b ≤ 16))
```

## CGB.ConservativeCycleReview.functional_recovery_tracking

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L954)

解释：在声明的平滑、PL 型反馈与漂移等条件下，同一状态轨迹满足给定功能损失恢复和跟踪界。

源码说明：Actual same-trace FUNCTION recovery/tracking under declared smooth and PL-type evidence conditions. The loss feedback supplies parameter correction; the optimizer's own phase regeneration is a distinct property. No old weight copy or external target is stored in the state. With drift=0, functional error contracts geometrically. With drift, the bound exposes the tracking cost. This is a sufficient task class, not a guarantee for every neural loss or a claim that preserving phase alone preserves an unobserved past function.

形式陈述：

```lean
theorem functional_recovery_tracking {n : ℕ} (η L μ drift : ℝ)
    (gradient : ℕ → Parameters n → Parameters n)
    (encodeCue : (Fin 2 → Fin 2 → ℝ) → ℤ) (initial : OptimizerState n)
    (F : ℕ → Parameters n → ℝ) (floor : ℕ → ℝ)
    (hη : 0 < η) (hμ : 0 < μ) (hμL : μ ≤ L) (hηL : L * η ≤ 1)
    (hinit : ∀ b, (-32 ≤ initial.phase b ∧ initial.phase b ≤ 32) ∧
      (8 ≤ initial.reserve b ∧ initial.reserve b ≤ 16))
    (hcode : ∀ g, (-64 ≤ encodeCue g ∧ encodeCue g ≤ 64) ∧
      |(encodeCue g : ℝ) / 32 - relativeGradientCue g| ≤ (1 / 64 : ℝ))
    (hlower : ∀ t, floor t ≤ F t (optimizerPath η gradient encodeCue initial t).parameter)
    (hmodel : ∀ t,
      let s
```

## CGB.ConservativeCycleReview.filter_error_envelope

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1062)

解释：当递推增益、观测误差、漂移和舍入满足给定界时，滤波估计误差由声明的递推包络控制。

源码说明：The actual recursive estimate supplies its own error envelope. A zero gain is allowed: its envelope must then pay for drift and rounding. This is a pathwise world contract, not a claim that pooled empirical variances are known error bounds. It also covers gains chosen from the current observation.

形式陈述：

```lean
theorem filter_error_envelope (gain observed rounding actual radius drift noise quant : ℕ → ℝ)
    (initial : ℝ) (hinit : |initial - actual 0| ≤ radius 0)
    (hg : ∀ t, 0 ≤ gain t ∧ gain t ≤ 1)
    (hd : ∀ t, |actual (t + 1) - actual t| ≤ drift t)
    (hn : ∀ t, |observed (t + 1) - actual (t + 1)| ≤ noise t)
    (hq : ∀ t, |rounding t| ≤ quant t)
    (hb : ∀ t, (1 - gain t) * (radius t + drift t) +
      gain t * noise t + quant t ≤ radius (t + 1)) (t : ℕ) :
    |filterEstimate gain observed rounding initial t - actual t| ≤ radius t
```

## CGB.ConservativeCycleReview.preconditioned_filter_step

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1115)

解释：在指定尺度、估计误差及平滑条件下，滤波、逐坐标缩放和再生分配组成的最终参数步满足所列界。

源码说明：Estimation, diagonal scaling, regenerated allocation and the FINAL continuous parameter step share one certificate. Scale is the square root of the preconditioning denominator: theta changes by estimate / scale^2, whereas the cue measures work estimate^2 / scale^2. Multiplying an old cue by a new preconditioner afterward would not implement this law. The upper model is for the entire step and must include cross-coordinate curvature. Data-dependent scale and filter/phase correlation are allowed; the scaled estimation-error condition and upper model must actually hold. Scale is not asserted to be the Hessian or an accurate online estimate.

形式陈述：

```lean
theorem preconditioned_filter_step (m reserve noise η L before after : ℝ)
    (estimate actual scale : Fin 2 → Fin 2 → ℝ)
    (hm0 : -1 ≤ m) (hm1 : m ≤ 1)
    (hr0 : (1 / 2 : ℝ) ≤ reserve) (hr1 : reserve ≤ 1)
    (hn : |noise| ≤ (1 / 32 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1)
    (hs : ∀ i j, 0 < scale i j)
    (herror : normFour (fun i j => (actual i j - estimate i j) / scale i j) ≤
      normFour (fun i j => estimate i j / scale i j) / 128)
    (hmodel :
      let e
```

## CGB.ConservativeCycleReview.integer_preconditioned_step

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1159)

解释：当整数状态、线索误差、缩放和步长满足声明条件时，整数相位与连续预条件参数步满足联合证书。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem integer_preconditioned_step (K R C : ℤ) (η L before after : ℝ)
    (estimate actual scale : Fin 2 → Fin 2 → ℝ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32)
    (hr0 : 8 ≤ R) (hr1 : R ≤ 16) (hc0 : -64 ≤ C) (hc1 : C ≤ 64)
    (hcue : |(C : ℝ) / 32 - relativeGradientCue (fun i j => estimate i j / scale i j)| ≤
      (1 / 64 : ℝ))
    (hη : 0 ≤ η) (hL : 0 ≤ L) (hηL : L * η ≤ 1)
    (hs : ∀ i j, 0 < scale i j)
    (herror : normFour (fun i j => (actual i j - estimate i j) / scale i j) ≤
      normFour (fun i j => estimate i j / scale i j) / 128)
    (hmodel :
      let next
```

## CGB.ConservativeCycleReview.resource_filtered_tracking

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1218)

解释：若资源可用量有统一下界且观测误差有界，资源控制的滤波器满足给定目标跟踪误差界。

源码说明：Raw observations reach the existing filter at a resource-derived rate. The world condition is a bounded observation error about a fixed value, not an assumption that the computed cue already requests release. No counter is stored. Availability may depend on the entire joint phase/resource history.

形式陈述：

```lean
theorem resource_filtered_tracking (rate target error initial : ℝ)
    (available mass observed : ℕ → ℝ)
    (hr : 0 < rate) (hr1 : rate ≤ 1) (he : 0 ≤ error)
    (ha : ∀ t, (1 / 2 : ℝ) ≤ available t ∧ available t ≤ 1)
    (hm0 : 0 ≤ mass 0) (hm1 : mass 0 ≤ 1)
    (hm : ∀ t, mass (t + 1) = (1 - rate * available t) * mass t + rate * available t)
    (ho : ∀ t, |observed (t + 1) - target| ≤ error) :
    let gain
```

## CGB.ConservativeCycleReview.encoded_contrary_of_energy_ratio

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1278)

解释：当交叉方向梯度平方和严格超过对角方向的指定比例时，整数相对梯度线索取最强反向码 −64。

源码说明：Rounding supplies a finite release threshold even when an old filter component never becomes exactly zero. Strictness matters: equality gives code -63 because ties round upward. This is the EXISTING cue encoder.

形式陈述：

```lean
theorem encoded_contrary_of_energy_ratio (g : Fin 2 → Fin 2 → ℝ)
    (h : 255 * ((g 0 0)^2 + (g 1 1)^2) < (g 0 1)^2 + (g 1 0)^2) :
    encodedGradientCue g = -64
```

## CGB.ConservativeCycleReview.raw_stream_forces_release

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1308)

解释：在所声明恒定交叉梯度、资源、初始界及递推条件下，原始梯度流可经实际滤波与线索机制迫使相位释放。

源码说明：Joint raw-stream bridge. The world supplies a constant cross-mode gradient of amplitude A; it does NOT supply a desired phase or an encoded release cue. Every coordinate uses the actual generated observation gain, and the same integer phase/resource transition consumes the resulting filtered evidence. Any legal old phase is released in a finite time. A common positive RMS scale cancels in the cue; floating rounding is a separate implementation check. This proves learned allocation memory, not restoration of arbitrary weights.

形式陈述：

```lean
theorem raw_stream_forces_release (rate A B : ℝ) (available mass : ℕ → ℝ)
    (initial : Fin 2 → Fin 2 → ℝ) (z : ℕ → ℤ × ℤ)
    (hr : 0 < rate) (hr1 : rate ≤ 1) (hA : 0 < A) (hB : 0 ≤ B)
    (ha : ∀ t, (1 / 2 : ℝ) ≤ available t ∧ available t ≤ 1)
    (hm0 : 0 ≤ mass 0) (hm1 : mass 0 ≤ 1)
    (hm : ∀ t, mass (t + 1) = (1 - rate * available t) * mass t + rate * available t)
    (hinit : ∀ i j, |initial i j - (if i = j then 0 else A)| ≤ B)
    (hz0 : (-32 ≤ (z 0).1 ∧ (z 0).1 ≤ 32) ∧ (8 ≤ (z 0).2 ∧ (z 0).2 ≤ 16))
    (hz : ∀ t, z (t + 1) = integerCycleStep (z t).1 (z t).2
      (encodedGradientCue (fun i j => filterEstimate
        (fun t => rate * available t / mass (t + 1))
        (fun _ => if i = j then 0 else A) (fun _ => 0) (initial i j) (t + 1)))) :
    (∀ t, (-32 ≤ (z t).1 ∧ (z t).1 ≤ 32) ∧ (8 ≤ (z t).2 ∧ (z t).2 ≤ 16)) ∧
    ∃ N : ℕ, (1 - rate / 2)^N * B ≤ A / 32 ∧ ∀ t, (z (N + 8 + t)).1 = -32
```

## CGB.ConservativeCycleReview.relative_cue_shared_scale

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1428)

解释：任意非零共享尺度同时缩放 2×2 梯度时，指定相对梯度线索保持不变。

源码说明：A compartment's shared positive RMS changes physical step size, but not the exact allocation evidence. This connects the raw-stream result to the already implemented preconditioned observer without a new cue or state.

形式陈述：

```lean
theorem relative_cue_shared_scale (g : Fin 2 → Fin 2 → ℝ) (s : ℝ) (hs : s ≠ 0) :
    relativeGradientCue (fun i j => g i j / s) = relativeGradientCue g
```

## CGB.ConservativeCycleReview.integer_ignition_and_retention

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1441)

解释：指定整数递推可由正负强线索点燃两种模式，且在所列近端点范围和合法资源内无输入时恢复端点。

源码说明：Cold evidence ignites either mode in the SAME integer law. With input withdrawn, phases within eight codes of either endpoint regenerate to that endpoint in one tick for EVERY legal reserve. No frozen-state branch exists.

形式陈述：

```lean
theorem integer_ignition_and_retention :
    integerCycleStep 0 16 64 = (32, 16) ∧
    integerCycleStep 0 16 (-64) = (-32, 16) ∧
    ∀ K R : ℤ, -32 ≤ K → K ≤ 32 → 8 ≤ R → R ≤ 16 →
      (24 ≤ K → (integerCycleStep K R 0).1 = 32) ∧
      (K ≤ -24 → (integerCycleStep K R 0).1 = -32)
```

## CGB.ConservativeCycleReview.controlled_cycle_contract

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1464)

解释：对反馈开启或关闭两种配置，指定整数递推都保持相同状态域、资源账本及双预算分配约束。

源码说明：Both interventions use the SAME finite state domain, supply/use ledger and double-budget allocation. Turning off feedback does not remove phase, reserve, sensing, or its parameter readout; no inactive padding state is added.

形式陈述：

```lean
theorem controlled_cycle_contract (feedback : Bool) (K R C : ℤ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32) (hr0 : 8 ≤ R) (hr1 : R ≤ 16)
    (hc0 : -64 ≤ C) (hc1 : C ≤ 64) :
    let z
```

## CGB.ConservativeCycleReview.passive_quiet_quantization_band

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1499)

解释：无反馈的指定安静整数递推在六步内进入零到一的码区间，且零码与一码都是固定点。

源码说明：Do NOT claim exact integer forgetting: upward half-grid rounding leaves fixed codes 0 and 1. Nevertheless every legal quiet phase reaches this one-code band within six ticks, unlike the regenerated endpoints ±32. The finite proof is kernel reduction, not an empirical memory experiment.

形式陈述：

```lean
theorem passive_quiet_quantization_band (K : ℤ) (hk0 : -32 ≤ K) (hk1 : K ≤ 32) :
    (0 ≤ passiveQuietPhase 6 K ∧ passiveQuietPhase 6 K ≤ 1) ∧
    (∀ R : ℤ, (controlledCycleStep false 0 R 0).1 = 0 ∧
      (controlledCycleStep false 1 R 0).1 = 1)
```

## CGB.ConservativeCycleReview.controlled_final_readout

[源码位置](../../../../lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean#L1516)

解释：在给定状态、尺度、误差与平滑条件下，反馈开启和关闭两种配置的最终读出满足共同的下降证书。

源码说明：A common whole-readout certificate for BOTH arms, reusing the existing perturbed scalar lemma. t is the actual physical rate, including OLD resource, RMS denominator and tail correction. The global quadratic upper model must include cross-coordinate curvature. Estimation error is paid explicitly; RMS is not asserted to be a curvature estimate or to ensure this premise.

形式陈述：

```lean
theorem controlled_final_readout (feedback : Bool) (K R C : ℤ)
    (t L before after : ℝ) (estimate actual : Fin 2 → Fin 2 → ℝ)
    (hk0 : -32 ≤ K) (hk1 : K ≤ 32) (hr0 : 8 ≤ R) (hr1 : R ≤ 16)
    (hc0 : -64 ≤ C) (hc1 : C ≤ 64) (ht : 0 ≤ t) (hL : 0 ≤ L) (hst : L * t ≤ 1)
    (hmodel :
      let m
```
