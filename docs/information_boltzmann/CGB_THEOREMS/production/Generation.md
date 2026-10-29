# Generation

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean) · [返回导航](../README.md)

模块源码说明：Generation, layer 1: what the axioms force on the equilibrium law The axioms constrain; this file starts deriving.

## CGB.JumpProcess.reversible_mul_invariant

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L46)

解释：若正函数在每个正速率转移两端相等，将细致平衡权重乘以该函数后仍为正并保持细致平衡。

源码说明：**G3**: a positive factor that is constant along every move preserves detailed balance.

形式陈述：

```lean
∀ {S : Type u_1} (P : CGB.JumpProcess S) {π w : S → ℝ},
  (∀ (x : S), 0 < π x) →
    (∀ (x y : S), π x * P.rate x y = π y * P.rate y x) →
      (∀ (x : S), 0 < w x) →
        (∀ (x y : S), 0 < P.rate x y → w x = w y) →
          (∀ (x : S), 0 < π x * w x) ∧ ∀ (x y : S), π x * w x * P.rate x y = π y * w y * P.rate y x
```

## CGB.JumpProcess.freezeSlow_conditional_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L74)

解释：联合过程有正细致平衡权重时，固定慢变量后，快变量上归一化的条件权重对冻结过程满足细致平衡。

源码说明：**G13 (prediction is the boundary's Gibbs conditional)**: if the joint calm dynamics has detailed-balance weight `π`, then with the slow interior `P` frozen, for each fixed `p` the conditional `π (·, p) / Σ π (·, p)` is a detailed-balance (hence stationary) law of the fast part's own moves. What an undriven fast boundary expresses about the next input, given its slow interior, is this conditional. Convergence to it (irreducibility, mixing time) and the time-scale separation itself are not claimed.

形式陈述：

```lean
∀ {Q : Type u_2} {P : Type u_3} [inst : Fintype Q] [inst_1 : DecidableEq P] (K : CGB.JumpProcess (Q × P))
  {π : Q × P → ℝ},
  (∀ (x : Q × P), 0 < π x) →
    (∀ (x y : Q × P), π x * K.rate x y = π y * K.rate y x) →
      ∀ (p : P) (q q' : Q),
        (π (q, p) / ∑ r, π (r, p)) * K.freezeSlow.rate (q, p) (q', p) =
          (π (q', p) / ∑ r, π (r, p)) * K.freezeSlow.rate (q', p) (q, p)
```

## CGB.Medium.parts_subset_scope

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L94)

解释：在 P1b 局部性条件下，若事件改变区域 D 的全部位置，则事件参与位置都在 D 的接触范围内。

源码说明：Under P1b an event that changes all of `D` reads only inside `scope D`.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I),
  M.P1b → ∀ {a : A} {D : Set U}, (∀ u ∈ D, u ∈ M.parts a) → ∀ v ∈ M.parts a, v ∈ M.scope D
```

## CGB.Medium.content_fixed_by_closed_single_unit

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L100)

解释：在 P2a 下，若正速率闭事件仅改变一个位置，则该位置的内容量保持不变。

源码说明：**G2**: under P2a a closed event that changes a single unit keeps that unit's content, so content moves inside the body only by events that change at least two units.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype U],
  M.P2a →
    ∀ {a : A} {s : U → I} {x : U → Λ} (u : U),
      M.Closed a →
        0 < M.rate a s x → (∀ (v : U), v ≠ u → M.act a s x v = x v) → M.content (M.act a s x u) = M.content (x u)
```

## CGB.Medium.jump_rate_local

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L118)

解释：在 P1b 条件下，具有相同改变区域及对应局部状态和刺激的两次转移，其总跳跃速率相同。

源码说明：**G1**: the total rate of a transition `x → y` with changed set `D` depends only on the new values on `D` and on the old state and stimulus on `D ∪ scope D`.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)],
  M.P1b →
    ∀ {s s' : U → I} {x y x' y' : U → Λ} (D : Set U),
      (∀ (u : U), u ∈ D ↔ x u ≠ y u) →
        (∀ u ∉ D, y' u = x' u) →
          (∀ u ∈ D, y' u = y u) →
            (∀ (u : U), u ∈ D ∨ u ∈ M.scope D → x' u = x u) →
              (∀ (u : U), u ∈ D ∨ u ∈ M.scope D → s' u = s u) → (M.jump s').rate x' y' = (M.jump s).rate x y
```

## CGB.Medium.equilibrium_ratio_local

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L171)

解释：在局部性、正细致平衡权重与正反向速率等条件下，平静转移的平衡权重比由对应局部数据决定。

源码说明：**G1E**: across one calm transition with a positive reverse rate, the equilibrium ratio `π y / π x` depends only on the same local data as G1.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)],
  M.P1b →
    ∀ {π : (U → Λ) → ℝ},
      (∀ (x : U → Λ), 0 < π x) →
        (∀ (x y : U → Λ), π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x) →
          ∀ {x y x' y' : U → Λ} (D : Set U),
            (∀ (u : U), u ∈ D ↔ x u ≠ y u) →
              (∀ u ∉ D, y' u = x' u) →
                (∀ u ∈ D, y' u = y u) →
                  (∀ (u : U), u ∈ D ∨ u ∈ M.scope D → x' u = x u) →
                    0 < (M.jump M.calm).rate y x → π y' / π x' = π y / π x
```

## CGB.Medium.local_markov_of_single_unit_moves

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L197)

解释：在指定局部性、正细致平衡和每个局部值可单点离开的条件下，单位置替换的平衡权重比只由该位置及其接触范围决定。

源码说明：**G4**: if every value of unit `u` can be left by a calm move that changes only `u`, the equilibrium law is locally Markov at `u`: the ratio `π (x with u := l) / π x` depends only on `x` on `{u} ∪ scope {u}`.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)] [inst_2 : DecidableEq U],
  M.P1b →
    ∀ {π : (U → Λ) → ℝ},
      (∀ (x : U → Λ), 0 < π x) →
        (∀ (x y : U → Λ), π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x) →
          ∀ (u : U),
            (∀ (x : U → Λ) (l : Λ), l ≠ x u → 0 < (M.jump M.calm).rate (Function.update x u l) x) →
              ∀ {x x' : U → Λ},
                (∀ (v : U), v = u ∨ v ∈ M.scope {u} → x' v = x v) →
                  ∀ (l : Λ), π (Function.update x' u l) / π x' = π (Function.update x u l) / π x
```

## CGB.Medium.p2b_of_micro_weighted

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L224)

解释：若正权重下每个有效平静事件都能由对合配对的逆事件恢复且满足逐事件加权平衡，则介质满足 P2b。

源码说明：**P2b from weighted micro-reversibility** (generalizes `p2b_of_micro`, which has uniform weights): if events pair up by an involution `rev` so that every effective calm event is undone by its partner and the pair satisfies detailed balance with a positive weight `π`, then P2b holds with that weight. This is the constructor an energy-based medium needs.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)] (π : (U → Λ) → ℝ),
  (∀ (x : U → Λ), 0 < π x) →
    ∀ (rev : A → A),
      Function.Involutive rev →
        (∀ (a : A) (x : U → Λ),
            M.act a M.calm x ≠ x →
              M.act (rev a) M.calm (M.act a M.calm x) = x ∧
                π x * M.rate a M.calm x = π (M.act a M.calm x) * M.rate (rev a) M.calm (M.act a M.calm x)) →
          M.P2b
```

## CGB.Medium.rate_interior_stimulus_free

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L255)

解释：在局部性和只有边界感知刺激的条件下，改变区域及其接触范围不触及边界的转移，其速率与外部刺激无关。

源码说明：**G6**: under P1b and P3 a transition whose changed set and its scope avoid the boundary has the same rate under every stimulus as in the calm environment.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)],
  M.P1b →
    M.OnlyBoundarySenses →
      ∀ (s : U → I) {x y : U → Λ} (D : Set U),
        (∀ (u : U), u ∈ D ↔ x u ≠ y u) →
          (∀ (u : U), u ∈ D ∨ u ∈ M.scope D → u ∉ M.boundary) → (M.jump s).rate x y = (M.jump M.calm).rate x y
```

## CGB.Medium.interior_detailed_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L269)

解释：在局部性、边界感知和指定平静细致平衡条件下，远离边界的转移对任意刺激仍满足该平衡关系。

源码说明：**G6DB**: hence, with P2b, interior transitions satisfy detailed balance with the calm weight under every stimulus: all irreversibility enters at the boundary.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)],
  M.P1b →
    M.OnlyBoundarySenses →
      ∀ {π : (U → Λ) → ℝ},
        (∀ (x y : U → Λ), π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x) →
          ∀ (s : U → I) {x y : U → Λ} (D : Set U),
            (∀ (u : U), u ∈ D ↔ x u ≠ y u) →
              (∀ (u : U), u ∈ D ∨ u ∈ M.scope D → u ∉ M.boundary) →
                π x * (M.jump s).rate x y = π y * (M.jump s).rate y x
```

## CGB.Medium.coupling_vanishes_beyond_scope

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L282)

解释：在定理给定的局部平衡前提下，区域 D 范围之外的第二个变化不改变 D 上转移的平衡偏好。

源码说明：**G7**: no equilibrium coupling beyond contact. If a second change (on `E`, e.g. a plastic variable) lies outside `D ∪ scope D`, it does not alter the equilibrium preference across the move on `D`; the mixed second difference of `log π` vanishes.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)],
  M.P1b →
    ∀ {π : (U → Λ) → ℝ},
      (∀ (x : U → Λ), 0 < π x) →
        (∀ (x y : U → Λ), π x * (M.jump M.calm).rate x y = π y * (M.jump M.calm).rate y x) →
          ∀ {x y x' y' : U → Λ} (D : Set U),
            (∀ (u : U), u ∈ D ↔ x u ≠ y u) →
              (∀ u ∉ D, y' u = x' u) →
                (∀ u ∈ D, y' u = y u) →
                  (∀ (u : U), x' u ≠ x u → u ∉ D ∧ u ∉ M.scope D) →
                    0 < (M.jump M.calm).rate y x → π y' / π x' = π y / π x
```

## CGB.Medium.modulation_needs_boundary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L298)

解释：在局部性与边界感知假设下，若同一转移的速率随刺激变化，则其改变区域或接触范围必触及边界。

源码说明：**G9 (modulation needs the boundary)**: if the rate of a transition (for example a plastic move: its learning rate or, through the forward/backward pair, its learning direction) differs between two stimuli, then some unit of the changed set or its contact scope is a boundary unit. Interior plasticity can therefore depend on the world only through local states it reads; world-driven gating, sign or rate control needs a boundary (modulator) unit at the plastic site. Contrapositive of G6.

形式陈述：

```lean
∀ {U : Type u_1} {Λ : Type u_2} {A : Type u_3} {I : Type u_4} (M : CGB.Medium U Λ A I) [inst : Fintype A]
  [inst_1 : DecidableEq (U → Λ)],
  M.P1b →
    M.OnlyBoundarySenses →
      ∀ {s s' : U → I} {x y : U → Λ} (D : Set U),
        (∀ (u : U), u ∈ D ↔ x u ≠ y u) →
          (M.jump s).rate x y ≠ (M.jump s').rate x y → ∃ u, (u ∈ D ∨ u ∈ M.scope D) ∧ u ∈ M.boundary
```

## CGB.Medium.fixed_gain_error

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L317)

解释：在观测噪声和先验方差均为正、增益处于 0 到 2 的随机游走估计模型中，固定增益的超额误差等于一个非负平方项；增益取 P/(P+r) 时该项为零。这里的增益是估计器参数，不能直接等同任意优化器的全局学习率。

源码说明：**G10a**: the fixed-gain error exceeds the optimum `P` by an exact square; the unique optimal fixed gain is the Kalman gain `P / (P + r)`.

形式陈述：

```lean
∀ (r P K : ℝ),
  0 < r →
    0 < P →
      0 < K →
        K < 2 → (P ^ 2 / (P + r) + K ^ 2 * r) / (K * (2 - K)) - P = ((P + r) * K - P) ^ 2 / (K * (2 - K) * (P + r))
```

## CGB.Medium.no_fixed_gain_for_two_volatilities

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L330)

解释：在相同观测噪声、两个正且不同的波动参数以及同一个固定增益下，至少一个环境的估计误差严格高于该环境的最优值。这说明该类估计问题跨环境保持逐点最优需要调节增益，并不证明某个自动学习率规则可在任意任务免调参。

源码说明：**G10b (necessity of modulation, minimal form)**: in two worlds with different volatility (`P₁ ≠ P₂`), no fixed gain is optimal in both: in at least one world its error is strictly above that world's optimum. A learner that persists across such worlds must change its learning rate with the world; by G9 that change needs boundary participation or local states informed by activity.

形式陈述：

```lean
∀ (r P₁ P₂ K : ℝ),
  0 < r →
    0 < P₁ →
      0 < P₂ →
        P₁ ≠ P₂ →
          0 < K →
            K < 2 →
              P₁ < (P₁ ^ 2 / (P₁ + r) + K ^ 2 * r) / (K * (2 - K)) ∨
                P₂ < (P₂ ^ 2 / (P₂ + r) + K ^ 2 * r) / (K * (2 - K))
```

## CGB.Medium.no_global_gain_for_heterogeneous_compartments

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L361)

解释：对多个隔室共享一个固定增益，如果至少两个隔室的正波动参数不同，那么至少有一个隔室的误差严格高于其局部最优。结论依赖随机游走估计模型及共享噪声设定，不排除其他类别的学习规则。

源码说明：**G10c (heterogeneous compartments need local gains)**: model two compartments by their local volatility parameters `P c`. If two compartments have different volatility, one globally shared gain is strictly above the local optimum in at least one compartment. This is the compartment form of G10b; it proves the necessity of local modulation only for this heterogeneous random-walk class, not for every possible learning rule.

形式陈述：

```lean
∀ {C : Type u_5} [Fintype C] [Nonempty C] (r : ℝ) (P : C → ℝ) (K : ℝ),
  0 < r →
    (∀ (c : C), 0 < P c) →
      (∃ c d, P c ≠ P d) → 0 < K → K < 2 → ∃ c, P c < (P c ^ 2 / (P c + r) + K ^ 2 * r) / (K * (2 - K))
```

## CGB.Medium.miss_posterior_bayes

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L385)

解释：给定正的先验与两类未命中概率，连续 n 次未命中后的贝叶斯后验可写为指定赔率形式。

源码说明：The miss posterior is Bayes' rule: with prior `π`, miss probabilities `qr < qp` (`qr = 1 - pr`, `qp = 1 - pp`), `n` misses give rich posterior `1 / (1 + a ρⁿ)` with `a = (1-π)/π` and `ρ = qp/qr`.

形式陈述：

```lean
∀ {π qr qp : ℝ},
  0 < π →
    π < 1 →
      0 < qr → 0 < qp → ∀ (n : ℕ), π * qr ^ n / (π * qr ^ n + (1 - π) * qp ^ n) = 1 / (1 + (1 - π) / π * (qp / qr) ^ n)
```

## CGB.Medium.missPrediction_strictAnti

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L395)

解释：在正先验赔率、似然比大于一且富源收益高于贫源的条件下，连续未命中次数越多，指定预测严格下降。

源码说明：**G11a**: each further miss strictly lowers the best prediction.

形式陈述：

```lean
∀ {a ρ pr pp : ℝ}, 0 < a → 1 < ρ → pp < pr → StrictAnti (CGB.Medium.missPrediction a ρ pr pp)
```

## CGB.Medium.intake_memory_capacity

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L414)

解释：若编码须以指定精度区分最多 N 次连续未命中的预测，且预测间隔大于两倍误差，则状态数至少为 N+1。

源码说明：**G11b (interoceptive memory)**: to predict intake within `ε` after every history of up to `N` consecutive misses, where `2ε` is below every gap between those predictions, an organism's state must take at least `N + 1` distinct values on those intake histories (via `CausalState.separated_histories_capacity`). In stage 1 the heard bytes carry no information about intake, so these distinctions must be stored from intake itself; by G6 the interior cannot re-read the stimulus later, so they must be held in states.

形式陈述：

```lean
∀ {S : Type u_5} [inst : Fintype S] {a ρ pr pp ε : ℝ} (N : ℕ) (code : List Bool → S) (decode : S → ℝ),
  (∀ (n : Fin (N + 1)), dist (CGB.Medium.missPrediction a ρ pr pp ↑n) (decode (code (List.replicate (↑n) false))) ≤ ε) →
    (∀ (i j : Fin (N + 1)),
        i ≠ j → ε + ε < dist (CGB.Medium.missPrediction a ρ pr pp ↑i) (CGB.Medium.missPrediction a ρ pr pp ↑j)) →
      N + 1 ≤ Fintype.card S
```

## CGB.Medium.intake_gap_exists

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L431)

解释：在指定严格单调预测模型中，任意有限个不同未命中次数之间存在共同的正分离精度。

源码说明：The gap condition is satisfiable: strict monotonicity separates any finite family, so some positive precision forces `N + 1` states.

形式陈述：

```lean
∀ {a ρ pr pp : ℝ},
  0 < a →
    1 < ρ →
      pp < pr →
        ∀ (N : ℕ),
          ∃ ε > 0,
            ∀ (i j : Fin (N + 1)),
              i ≠ j → ε + ε < dist (CGB.Medium.missPrediction a ρ pr pp ↑i) (CGB.Medium.missPrediction a ρ pr pp ↑j)
```

## CGB.Medium.exists_leave_threshold

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L452)

解释：当替代选择价值高于贫源收益且满足模型正性条件时，足够多次未命中会使预测降到该价值以下。

源码说明：The leave threshold exists: whenever going elsewhere is worth more than a known-poor source (`pp < c`), enough misses push the best prediction below `c`.

形式陈述：

```lean
∀ {a ρ pr pp c : ℝ}, 0 < a → 1 < ρ → pp < pr → pp < c → ∃ n, CGB.Medium.missPrediction a ρ pr pp n < c
```

## CGB.Medium.miss_counter_necessary

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L480)

解释：在指定首次越过离开阈值的模型里，若读出须对所列历史和未来续接均给出最优动作，则编码需要足够多的不同状态。

源码说明：**G12 (a miss counter is necessary for optimal foraging)**: let `nStar` be the first miss count whose best prediction falls below `c` (all earlier counts stay at or above it). Any state code whose readout gives the optimal stay/leave action, for histories of up to `nStar` misses and every continuation of further misses, takes at least `nStar + 1` distinct values: histories with `i < j ≤ nStar` misses must be told apart, because after `nStar - j` more misses one must leave and the other must stay. The precision is fixed by the world (`c`, `pr`, `pp`, prior), not chosen.

形式陈述：

```lean
∀ {S : Type u_5} [inst : Fintype S] {a ρ pr pp c : ℝ} (nStar : ℕ),
  CGB.Medium.missPrediction a ρ pr pp nStar < c →
    (∀ n < nStar, c ≤ CGB.Medium.missPrediction a ρ pr pp n) →
      ∀ (code : List Bool → S) (decode : S → List Bool → ℝ),
        (∀ (n : Fin (nStar + 1)) (k : ℕ),
            ↑n + k ≤ nStar →
              decode (code (List.replicate (↑n) false)) (List.replicate k false) =
                CGB.Medium.stayDecision a ρ pr pp c (List.replicate (↑n) false ++ List.replicate k false)) →
          nStar + 1 ≤ Fintype.card S
```

## CGB.Medium.merging_across_threshold_loses

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L508)

解释：两个正权重历史的预测分居决策阈值两侧时，给它们统一选择同一动作严格劣于分别选择。

源码说明：Merging two histories that lie on opposite sides of the threshold costs expected intake: with positive weights, a single shared action earns strictly less than acting optimally on each.

形式陈述：

```lean
∀ {w₁ w₂ p₁ p₂ c : ℝ},
  0 < w₁ →
    0 < w₂ →
      c < p₁ →
        p₂ < c →
          ∀ (stay : Bool),
            ((w₁ * if stay = true then p₁ else c) + w₂ * if stay = true then p₂ else c) < w₁ * p₁ + w₂ * c
```

## CGB.Medium.two_phase_drift

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L527)

解释：指定驱动相与自由相的预期净更新等于输入强度乘夹持标签项减 softmax 概率项。

源码说明：**G14a (two-phase drift)**: in the drive phase the boundary is clamped to the arrived value `b'` and the site moves by `+λ x` when the boundary shows `b`; in the free phase the boundary samples the readout law and the site moves by `-μ x` when it shows `b`. The expected drift per cycle is `x (λ [b = b'] - μ softmax z b)`.

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] [inst_1 : DecidableEq B] (z : B → ℝ) (b b' : B) (x lam mu : ℝ),
  (lam * x * if b' = b then 1 else 0) - mu * ∑ c, CGB.Medium.softmax z c * (x * if c = b then 1 else 0) =
    x * ((lam * if b' = b then 1 else 0) - mu * CGB.Medium.softmax z b)
```

## CGB.Medium.softmax_sum

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L535)

解释：非空有限标签集合上的 softmax 概率之和为一。

源码说明：The readout law sums to one.

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] [Nonempty B] (z : B → ℝ), ∑ c, CGB.Medium.softmax z c = 1
```

## CGB.Medium.log_softmax_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L542)

解释：对指定标签的 log-softmax 求单个耦合变量的导数，得到输入强度乘标签指示量减 softmax 概率。

源码说明：**G14b**: the log likelihood of the arrived value `b'` changes, per unit change of that coupling, by `x ([b' = b] - softmax z b)`.

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] [inst_1 : DecidableEq B] (z : B → ℝ) (b b' : B) (x : ℝ),
  HasDerivAt (fun t => Real.log (CGB.Medium.softmax (fun c => z c + if c = b then t * x else 0) b'))
    (x * ((if b' = b then 1 else 0) - CGB.Medium.softmax z b)) 0
```

## CGB.Medium.phases_must_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L579)

解释：若对所有读出状态和夹持标签，两相漂移向量都是似然梯度的正倍数，则驱动相与自由相权重相等。

源码说明：**G14c (the phases must balance)**: if, for every readout state and arrived value, the expected drift vector over boundary values is a positive multiple of the log-likelihood gradient vector, the drive and free phases carry equal weight (`λ = μ`): summed over boundary values the gradient vanishes while the drift sums to `λ - μ`. Sleep-like free sampling must balance waking clamped learning.

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] [inst_1 : DecidableEq B] [Nonempty B] (lam mu : ℝ),
  (∀ (z : B → ℝ) (b' : B),
      ∃ κ > 0,
        ∀ (b : B),
          (lam * if b' = b then 1 else 0) - mu * CGB.Medium.softmax z b =
            κ * ((if b' = b then 1 else 0) - CGB.Medium.softmax z b)) →
    lam = mu
```

## CGB.Medium.leakage_loses

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L600)

解释：当泄漏比例严格介于零和一且目标概率为正时，泄漏后的目标对数概率严格下降。

源码说明：**R2 (leakage loses on every outcome)**: if the free boundary puts mass `ℓ > 0` on configurations that are no symbol, every symbol's predicted probability is `(1 - ℓ) q b` instead of `q b`, and the log likelihood of whatever arrives is strictly lower. An optimal predictive boundary keeps its free dynamics inside the one-symbol sector.

形式陈述：

```lean
∀ {ℓ q : ℝ}, 0 < ℓ → ℓ < 1 → 0 < q → Real.log ((1 - ℓ) * q) < Real.log q
```

## CGB.Medium.one_unit_one_label

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L608)

解释：有限标签各占据量不超过一且总占据量为一时，恰好存在一个占据量为一的标签。

源码说明：**R1 (conservation makes the readout one-hot)**: if every label unit holds content `0` or `1` and the readout's total content is `1` (an invariant shell of the closed free dynamics under P2a, `total_of_reachable`), exactly one label is on.

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] [DecidableEq B] (n : B → ℕ), (∀ (b : B), n b ≤ 1) → ∑ b, n b = 1 → ∃! b, n b = 1
```

## CGB.Medium.sector_law_is_softmax

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L626)

解释：单粒子标签扇区中，将指数权重归一化得到所定义的 softmax 概率。

源码说明：In the one-particle sector a local label energy gives the softmax readout: with weights `exp (z b)` on the configurations `label b on`, the normalized law is `softmax z` (G13 restricted to the sector).

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] (z : B → ℝ) (b : B), Real.exp (z b) / ∑ c, Real.exp (z c) = CGB.Medium.softmax z b
```

## CGB.Medium.reward_two_phase_drift

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L642)

解释：按 softmax 采样动作并用奖励减基线调制指定两相更新时，其期望等于动作概率乘该动作奖励相对平均奖励的差。

源码说明：**G15a (reward-modulated two-phase drift is the policy gradient)**: if the motor particle was clamped at the action taken `a ~ softmax z`, the drive phase moved the site by `(R a - bl) x` when label `b` was on, and the free phase by `-(R a - bl) x softmax z b`, the expected drift equals `x softmax z b (R b - J)` with `J = Σ softmax z a * R a`, for every baseline `bl`.

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] [inst_1 : DecidableEq B] [Nonempty B] (z R : B → ℝ) (bl : ℝ) (b : B),
  ∑ a, CGB.Medium.softmax z a * ((R a - bl) * ((if a = b then 1 else 0) - CGB.Medium.softmax z b)) =
    CGB.Medium.softmax z b * (R b - ∑ a, CGB.Medium.softmax z a * R a)
```

## CGB.Medium.expected_reward_deriv

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L658)

解释：有限 softmax 策略的期望奖励对某个动作 logit 的导数，等于该动作概率乘其奖励与平均奖励的差。

源码说明：**G15b**: that drift is exactly the derivative of the expected reward `J` with respect to the logit of `b`.

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] [inst_1 : DecidableEq B] (z R : B → ℝ) (b : B),
  HasDerivAt (fun t => ∑ a, CGB.Medium.softmax (fun c => z c + if c = b then t else 0) a * R a)
    (CGB.Medium.softmax z b * (R b - ∑ a, CGB.Medium.softmax z a * R a)) 0
```

## CGB.Medium.outcome_blind_drift_zero

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L698)

解释：若调制因子与动作结果无关，则指定 softmax 两相更新的期望漂移为零。

源码说明：**G15c (outcome-blind modulation learns nothing)**: if the modulator's weight `m` does not depend on the action's outcome, the expected two-phase drift is zero for every site. So a policy can only climb through a modulator that reads intake, which by G9 is a boundary unit.

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] [inst_1 : DecidableEq B] [Nonempty B] (z : B → ℝ) (m : ℝ) (b : B),
  ∑ a, CGB.Medium.softmax z a * (m * ((if a = b then 1 else 0) - CGB.Medium.softmax z b)) = 0
```

## CGB.Medium.reward_phases_must_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L710)

解释：在定理规定的全奖励表梯度对应条件下，驱动相与自由相权重必须相等。

源码说明：**G15d (reward phases must balance)**: with drive weight `lam` and free weight `mu`, if the expected drift is a multiple of the expected-reward gradient for every reward table, then `lam = mu`.

形式陈述：

```lean
∀ {B : Type u_5} [inst : Fintype B] [inst_1 : DecidableEq B] [Nonempty B] (z : B → ℝ) (bl lam mu : ℝ) (b : B),
  (∀ (R : B → ℝ),
      ∃ κ,
        ∑ a, CGB.Medium.softmax z a * ((R a - bl) * ((lam * if a = b then 1 else 0) - mu * CGB.Medium.softmax z b)) =
          κ * (CGB.Medium.softmax z b * (R b - ∑ a, CGB.Medium.softmax z a * R a))) →
    lam = mu
```

## CGB.Medium.two_part_balance

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L751)

解释：对正的两个局部因子，指定两部分速率表达式满足给出的乘积权重代数平衡恒等式。

源码说明：**G8a** (two parts suffice): with positive part factors `φ₁, φ₂` and constants `a, b ≥ 0`, up rates `a φ₁ + b φ₂` and down rates `b / φ₁ + a / φ₂` (each term reads one part) have ratio exactly `φ₁ φ₂`.

形式陈述：

```lean
∀ {α : Type u_5} {β : Type u_6} (φ₁ : α → ℝ) (φ₂ : β → ℝ),
  (∀ (a : α), 0 < φ₁ a) →
    (∀ (b : β), 0 < φ₂ b) → ∀ (a b : ℝ) (x : α) (y : β), a * φ₁ x + b * φ₂ y = φ₁ x * φ₂ y * (b / φ₁ x + a / φ₂ y)
```

## CGB.Medium.three_part_balance_impossible

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/Generation.lean#L761)

解释：在指定三部分加性速率、乘积权重与各因子非恒定的条件下，平衡恒等式迫使选定状态的总反向速率为零。

源码说明：**G8b** (three parts are impossible): if up rates `f₁+f₂+f₃` and down rates `g₁+g₂+g₃` (each term reading one part) satisfy detailed balance with a product factor `φ₁ φ₂ φ₃` that is non-constant on every part, the total down rate vanishes at a chosen state. No sign assumption is needed; with nonnegative rates every part's down rate vanishes there.

形式陈述：

```lean
∀ {α : Type u_5} {β : Type u_6} {γ : Type u_7} (f₁ g₁ φ₁ : α → ℝ) (f₂ g₂ φ₂ : β → ℝ) (f₃ g₃ φ₃ : γ → ℝ),
  (∀ (a : α), 0 < φ₁ a) →
    (∀ (b : β), 0 < φ₂ b) →
      (∀ (c : γ), 0 < φ₃ c) →
        (∀ (a : α) (b : β) (c : γ), f₁ a + f₂ b + f₃ c = φ₁ a * φ₂ b * φ₃ c * (g₁ a + g₂ b + g₃ c)) →
          ∀ {a₀ a₁ : α} {b₀ b₁ : β} {c₀ c₁ : γ},
            φ₁ a₀ ≠ φ₁ a₁ → φ₂ b₀ ≠ φ₂ b₁ → φ₃ c₀ ≠ φ₃ c₁ → g₁ a₀ + g₂ b₀ + g₃ c₀ = 0
```
