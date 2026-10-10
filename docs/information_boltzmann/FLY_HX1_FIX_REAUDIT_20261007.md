# HX-1 修复复查与剩余交接

用户问题：这回行了不。范围：当前 Gemini 源码，只读检查和 CPU 接口验证；没有启动训练或修改其生产模型。

## 已有效修复

1. GraphNeuralTransition 返回 `cur_H + delta`，旧区域范数乘子已删除，零运动 latent 可以受消息激发。
2. 旧 `Linear(9,1)` 已拆成 E/I、adapt/STP、四档队列的独立区域投影，消除了原来强制共享一个通道标量的瓶颈。区域编码仍是学习压缩表征，任务响应误差是其保真验收对象。
3. 学生查询参考改为 `quiet_states[k+1]`，误差比较与扰动物理起点已对齐。
4. 缺失出生前历史改为零值；窗内一步预测已使用累积 posterior 历史。这部分延迟修复有效。

## 必须补完的一处：同源持存历史

当前 `fly_graph_observer.py:508` 在每个窗口重新建立 `[prior_state]`，`:604` 只返回一个 `next_prior`。`FlyPhysicalState` 仅有 `observer_prior`，没有前几拍 posterior 历史。这里的 prior 是当前拍状态的预测，不能当成上一拍真实/校正的历史值。

正式 `predict_fly_next` 每次调用长度为一的窗口。因此区域历史最多是“当前先验 + 当前 posterior”，而 tier3、tier4 的消息在实际一步预测中仍被置零。整窗训练后半段却拥有这些历史，形成训练/部署差别。

`:534` 的并行草稿从 `[Z_posterior]` 重新起步，`simulate_hops` 从 `[z_start]` 起步，query 的 `transition.forward(z_q)` 同样只收到一个状态。它们也需要各自起点已有的历史，而不只是窗内实际一步预测的 history。

CPU 窗口切分检查：相同权重、相同八拍输入，整窗调用与八次单拍调用（只传当前 API 返回的 prior）相比，默认 MLP 的 next_prior 最大差为 **0.00023928285**。用确定性的 tier4 消息更新隔离延迟语义时，差为 **0.33551604**。第二个结果是接口数值反例，不是成熟权重的性能或误差测量。

## 给 Gemini 的最小实现要求

- 将最近四拍区域 posterior 历史作为明确的持存状态，与 prior 分开保存；保持实际物理拍时戳一致。
- 实际一步、每个草稿起点、查询都接收同源历史。草稿追加自己的预测，查询扰动只修改指定参考拍。
- 窗口边界只 detach 历史的计算图，保存数值；checkpoint 恢复同时恢复历史。
- 验收同一流的整窗/拆窗/单拍/保存恢复的读出和末态等价，以及 tier4 脉冲确实在指定拍到达。

## 测试状态

`pytest tests/test_fly_graph_observer.py tests/test_fly_bptt_learning.py -q`：**35 passed in 9.82s**。

`test_query_timestamp_alignment_perfect_predictor_zero_error` 的末尾循环当前只有 `pass`，实际只断言查询数量；应补上 `z_q` 与对应参考编码的数值相等断言。源代码索引修复已看到，测试也应真正封住该错误。

结果与源码哈希：`results/published/fly_hx1_history_reaudit_20261007.json`。GPU 3564.2 MB / ~40 token/s 是 Gemini 提供的双窗口报告，本次没有重复 GPU 测量；尚未证明长训练、验证或保存时的峰值。

结论：轻量路线继续保留，三项修复成立；把跨窗口/单拍/草稿/查询的历史持存补完，再进入正式联合训练。NLL 收益继续由真实连续流评估判断。
