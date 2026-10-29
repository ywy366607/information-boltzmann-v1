# unified_reference_20261025

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean) · [返回导航](../README.md)

模块源码说明：Algorithm verification only: export an executable integer representation of the EXISTING GraceRecovery world.

## UnifiedReferenceReview20261025.policy_table

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L39)

解释：指定六类候选单元状态按表映射到动作零、一或默认动作二。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem policy_table (q : Cell) : policy q = match q with
    | .aSpent | .aReady | .aHeld => 0
    | .bSpent | .bReady | .bHeld => 1
    | _ => 2
```

## UnifiedReferenceReview20261025.worldMass_correct

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L44)

解释：指定世界整数权重等于相应 GraceRecovery 实值转移权重的两倍。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem worldMass_correct (q : Cell) (w v : Env) :
    (worldMass q w v : Real) = 2 * GraceRecovery.moves w (fun _ => q) v
```

## UnifiedReferenceReview20261025.full_support_frozen

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L87)

解释：若每个候选都允许该字节观测，候选筛选更新保持原记忆不变。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem full_support_frozen {H : Type*} [Fintype H] [DecidableEq H]
    {B : Nat} (m : CausalLearning.Memory H B) (o : Fin 256) :
    CausalLearning.advance (fun (_ : H) (_ : Unit) (_ : Unit) (_ : Fin 256) => true)
      m () () o = m ∧
    CausalLearning.fee (fun (_ : H) (_ : Unit) (_ : Unit) (_ : Fin 256) => true)
      m () () o = 0
```

## UnifiedReferenceReview20261025.boundary_accept_balance

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L118)

解释：正权重 a、b 的双向 Metropolis 接受率满足 a·min(1,b/a)=b·min(1,a/b)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem boundary_accept_balance {a b : Real} (ha : 0 < a) (hb : 0 < b) :
    a * min 1 (b / a) = b * min 1 (a / b)
```

## UnifiedReferenceReview20261025.boundary_accept_minorization

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L124)

解释：若正权重 a、b 均不超过 m，则指定接受概率 min(1,b/a) 不小于 b/m。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem boundary_accept_minorization {a b m : Real}
    (ha : 0 < a) (hb : 0 < b) (ham : a ≤ m) (hbm : b ≤ m) :
    b / m ≤ min 1 (b / a)
```

## UnifiedReferenceReview20261025.local_cross_balance

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L149)

解释：对正参数 r、p、q，指定两个局部交互速率表达式满足给出的代数平衡等式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem local_cross_balance {r p q : Real}
    (hr : 0 < r) (hp : 0 < p) (hq : 0 < q) :
    r * p + r * q = (r * r * p * q) * (1 / r / p + 1 / r / q)
```

## UnifiedReferenceReview20261025.local_plastic_delta

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L160)

解释：给定局部能量模型中，连接量增加一且另一变量减少一的能量差等于指定代数表达式。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem local_plastic_delta (ε J p q w : Real) :
    (ε * (w + 1) - J * (w + 1) * p * (q - 1)) - (ε * w - J * w * p * q) =
      ε - J * p * (q - w - 1)
```

## UnifiedReferenceReview20261025.boundary_exchange_log_balance

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L166)

解释：指定边界正反指数之差化简为 −2(d+(c−μ)s)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem boundary_exchange_log_balance (d c μ s : Real) :
    (-d - (c - μ) * s) - (d + (c - μ) * s) = -2 * (d + (c - μ) * s)
```

## UnifiedReferenceReview20261025.boundary_contact_fraction

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L171)

解释：非零参数条件下，给定边界接触概率因子化简为 1/(FB)。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem boundary_contact_fraction {H B F : Real}
    (hH : H ≠ 0) (hB : B ≠ 0) (hF : F ≠ 0) :
    (1 / F) * ((2 * H) / (2 * H * B)) = 1 / (F * B)
```

## UnifiedReferenceReview20261025.boundary_contact_envelope

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L180)

解释：当单步接触率 q 位于零到一时，n 步至少一次接触概率位于零到一且不超过 nq。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem boundary_contact_envelope (q : Real) (h0 : 0 ≤ q) (h1 : q ≤ 1) (n : Nat) :
    0 ≤ 1 - (1 - q) ^ n ∧ 1 - (1 - q) ^ n ≤ 1 ∧
      1 - (1 - q) ^ n ≤ (n : Real) * q
```

## UnifiedReferenceReview20261025.boundary_response_budget

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L194)

解释：在单步接触率、费用、总预算和所需响应满足定理条件时，响应需求受有限预算界约束。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem boundary_response_budget (q fee budget required : Real) (n : Nat)
    (h0 : 0 ≤ q) (h1 : q ≤ 1) (hfee : 0 ≤ fee)
    (paid : (n : Real) * fee ≤ budget) (response : required ≤ 1 - (1 - q) ^ n) :
    required * fee ≤ budget * q
```

## UnifiedReferenceReview20261025.clique_total_update

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L207)

解释：替换有限群体中一个位置的自然数值后，新总量加旧值等于旧总量加新值。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem clique_total_update {ι : Type*} [Fintype ι] [DecidableEq ι]
    (f : ι → Nat) (i : ι) (value : Nat) :
    (∑ j, Function.update f i value j) + f i = (∑ j, f j) + value
```

## UnifiedReferenceReview20261025.family_conductance_keeps_balance

[源码位置](../../../../lean/CGBAxioms/reviews/unified_reference_20261025.lean#L218)

解释：若一对正反速率已满足加权平衡，则两者同时乘同一系数仍满足加权平衡。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
private theorem family_conductance_keeps_balance (πx πy κ rf rb : ℝ)
    (h : πx * rf = πy * rb) : πx * (κ * rf) = πy * (κ * rb)
```
