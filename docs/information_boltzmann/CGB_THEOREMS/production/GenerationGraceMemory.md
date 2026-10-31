# GenerationGraceMemory

生产区：名称和完整形式陈述与通过覆盖及信任检查的库存逐条核对。

[源文件](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean) · [返回导航](../README.md)

模块源码说明：Historical memory in the unchanged grace process The natural law at tick four has one body state and three world states.

## _private.CGBAxioms.GenerationGraceMemory.0.CGB.GraceMemory.sum_cells

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L17)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : CGB.RenewableLearning.Cell → ℝ),
  ∑ q, f q =
    f CGB.RenewableLearning.Cell.vacuum + f CGB.RenewableLearning.Cell.search + f CGB.RenewableLearning.Cell.aSpent +
              f CGB.RenewableLearning.Cell.bSpent +
            f CGB.RenewableLearning.Cell.aReady +
          f CGB.RenewableLearning.Cell.bReady +
        f CGB.RenewableLearning.Cell.aHeld +
      f CGB.RenewableLearning.Cell.bHeld
```

## _private.CGBAxioms.GenerationGraceMemory.0.CGB.GraceMemory.sum_configs

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L24)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : (Unit → CGB.RenewableLearning.Cell) → ℝ),
  ∑ x, f x =
    (((((((f fun x => CGB.RenewableLearning.Cell.vacuum) + f fun x => CGB.RenewableLearning.Cell.search) +
                f fun x => CGB.RenewableLearning.Cell.aSpent) +
              f fun x => CGB.RenewableLearning.Cell.bSpent) +
            f fun x => CGB.RenewableLearning.Cell.aReady) +
          f fun x => CGB.RenewableLearning.Cell.bReady) +
        f fun x => CGB.RenewableLearning.Cell.aHeld) +
      f fun x => CGB.RenewableLearning.Cell.bHeld
```

## _private.CGBAxioms.GenerationGraceMemory.0.CGB.GraceMemory.sum_env

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L32)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (f : CGB.RenewableLearning.Env → ℝ),
  ∑ w, f w =
    f CGB.RenewableLearning.Env.feedA + f CGB.RenewableLearning.Env.feedB + f CGB.RenewableLearning.Env.missA +
                f CGB.RenewableLearning.Env.missB +
              f CGB.RenewableLearning.Env.probeA +
            f CGB.RenewableLearning.Env.probeB +
          f CGB.RenewableLearning.Env.answerA +
        f CGB.RenewableLearning.Env.answerB +
      f CGB.RenewableLearning.Env.trap
```

## CGB.GraceMemory.kernel_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L40)

源码说明：The original body kernel, obtained by forgetting its actual event label.

形式陈述：

```lean
∀ (s : Unit → CGB.RenewableLearning.Signal) (q r : CGB.RenewableLearning.Cell),
  (CGB.GraceRecovery.body.kernel 1 s (fun x => q) fun x => r) =
    if r = CGB.RenewableLearning.next true true (s ()) q then 1 else 0
```

## _private.CGBAxioms.GenerationGraceMemory.0.CGB.GraceMemory.policy_table

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L55)

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell),
  CGB.RenewableLearning.policy q =
    match q with
    | CGB.RenewableLearning.Cell.aSpent => 0
    | CGB.RenewableLearning.Cell.aReady => 0
    | CGB.RenewableLearning.Cell.aHeld => 0
    | CGB.RenewableLearning.Cell.bSpent => 1
    | CGB.RenewableLearning.Cell.bReady => 1
    | CGB.RenewableLearning.Cell.bHeld => 1
    | x => 2
```

## CGB.GraceMemory.natural_law_early

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L75)

源码说明：Each step rewrites the preceding sparse row before any finite summation; there is no expansion of a Cartesian product of histories.

形式陈述：

```lean
∀ n ≤ 4,
  ∀ (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
    CGB.GraceRecovery.body.law CGB.GraceRecovery.world (fun x => CGB.GraceRecovery.world.sense) n (fun x => q) w =
      CGB.GraceMemory.early n q w
```

## CGB.GraceMemory.actual_event_law_four

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L96)

源码说明：The same row is the law of the registered event process.

形式陈述：

```lean
∀ (q : CGB.RenewableLearning.Cell) (w : CGB.RenewableLearning.Env),
  CGB.Energetics.law (CGB.GraceRecovery.process false CGB.GraceRecovery.natural) CGB.RenewableLearning.initial 4
      (fun x => q, w) =
    CGB.GraceMemory.early 4 q w
```

## _private.CGBAxioms.GenerationGraceMemory.0.CGB.GraceMemory.one_write_marginal

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L110)

源码说明：Marginalize the actual world transition, whose row sums to one.

形式陈述：

```lean
∀ (n : ℕ) (x₀ x : Unit → CGB.RenewableLearning.Cell) (w₀ : CGB.RenewableLearning.Env),
  ∑ w, CGB.GraceRecovery.body.memLaw CGB.GraceRecovery.world n 1 x w x₀ w₀ =
    CGB.GraceRecovery.body.law CGB.GraceRecovery.world (fun x => CGB.GraceRecovery.world.sense) n x₀ w₀ *
      CGB.GraceRecovery.body.kernel 1 (CGB.RenewableLearning.sense w₀) x₀ x
```

## CGB.GraceMemory.one_write_joint

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L122)

源码说明：The three nonzero tagged outcomes have masses 1/4, 1/4 and 1/2.

形式陈述：

```lean
∀ (x₀ x : Unit → CGB.RenewableLearning.Cell) (w₀ : CGB.RenewableLearning.Env),
  ∑ w, CGB.GraceRecovery.body.memLaw CGB.GraceRecovery.world 4 1 x w x₀ w₀ = CGB.GraceMemory.writeJoint✝ x₀ w₀ x
```

## CGB.GraceMemory.one_write_information

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L139)

源码说明：Exact conditional mutual information in nats: one bit, not a support argument.

形式陈述：

```lean
(CGB.Medium.condMutualInfo fun x₀ w₀ x => ∑ w, CGB.GraceRecovery.body.memLaw CGB.GraceRecovery.world 4 1 x w x₀ w₀) =
  Real.log 2
```

## CGB.GraceMemory.remembers

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L155)

源码说明：The original `Life.Remembers` predicate holds for the unchanged grace body/world.

形式陈述：

```lean
CGB.GraceRecovery.body.Remembers CGB.GraceRecovery.world
```

## CGB.GraceMemory.retained_information

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L162)

源码说明：Quiet retains exactly the same bit at every later delay.

形式陈述：

```lean
∀ (m : ℕ),
  (CGB.Medium.condMutualInfo fun x₀ w₀ x =>
      ∑ w, CGB.GraceRecovery.body.memLaw CGB.GraceRecovery.world 4 (m + 1) x w x₀ w₀) =
    Real.log 2
```

## CGB.GraceMemory.remembers_with_reachable_action_change

[源码位置](../../../../lean/CGBAxioms/CGBAxioms/GenerationGraceMemory.lean#L177)

源码说明：Combine the numerical memory witness with the already-proved supported histories whose stored states select different actions.

形式陈述：

```lean
CGB.GraceRecovery.body.Remembers CGB.GraceRecovery.world ∧
  ∃ p q,
    0 < CGB.RenewableLearningControls.weight (CGB.GraceRecovery.process false CGB.GraceRecovery.natural) p 6 ∧
      0 < CGB.RenewableLearningControls.weight (CGB.GraceRecovery.process false CGB.GraceRecovery.natural) q 6 ∧
        (p 5).1 = (q 5).1 ∧ CGB.RenewableLearning.policy ((p 6).1 ()) ≠ CGB.RenewableLearning.policy ((q 6).1 ())
```
