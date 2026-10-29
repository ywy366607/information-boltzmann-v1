# active_loop_20261024

探索区：仅按源码盘点；不在生产区全量审计覆盖内，不能据此声称已通过当前构建或信任审计。

[源文件](../../../../lean/CGBAxioms/reviews/active_loop_20261024.lean) · [返回导航](../README.md)

## ActiveLoopReview20261024.sum_configs

[源码位置](../../../../lean/CGBAxioms/reviews/active_loop_20261024.lean#L10)

解释：对单位置四状态构型求任意实函数之和，等于函数在四个常值构型上的值之和。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem sum_configs (f : (Unit → State) → ℝ) :
    ∑ x, f x = f (fun _ => 0) + f (fun _ => 1) +
      f (fun _ => 2) + f (fun _ => 3)
```

## ActiveLoopReview20261024.body_kernel_table

[源码位置](../../../../lean/CGBAxioms/reviews/active_loop_20261024.lean#L23)

解释：指定四态身体的一步核由后继状态是否等于给定推进函数决定。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem body_kernel_table (s : Stim) (x y : State) :
    body.kernel 1 (fun _ => s) (fun _ => x) (fun _ => y) =
      if y = advance () s x then 1 else 0
```

## ActiveLoopReview20261024.natural_step_table

[源码位置](../../../../lean/CGBAxioms/reviews/active_loop_20261024.lean#L32)

解释：该见证的自然联合一步核等于世界转移权重乘身体确定性更新指示量。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem natural_step_table (t : ℕ) (x y : State) (w v : WorldState) :
    (jointMachine false).step t ((fun _ => x), w) () ((fun _ => y), v) =
      worldMoves w (fun _ => x) v *
        (if y = advance () (worldSense w ()) x then 1 else 0)
```

## ActiveLoopReview20261024.cut_step_table

[源码位置](../../../../lean/CGBAxioms/reviews/active_loop_20261024.lean#L41)

解释：该见证的断环联合一步核可展开为断环身体边缘混合世界转移与身体更新的乘积。

源码没有为此声明提供文字说明；下列形式陈述是可核对的条件与结论，需阅读上下文后才能给出可靠解释。

形式陈述：

```lean
theorem cut_step_table (t : ℕ) (x y : State) (w v : WorldState) :
    (jointMachine true).step t ((fun _ => x), w) () ((fun _ => y), v) =
      (∑ g : State, (∑ u : WorldState,
        body.lawCut world (fun _ => world.sense) t (fun _ => g) u) *
          worldMoves w (fun _ => g) v) *
        (if y = advance () (worldSense w ()) x then 1 else 0)
```
