# CGB 低资源生成生命体接口合同

状态：历史接口与验收记录。当前架构推导以[CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md](CGB_LANGUAGE_LIFE_MODEL_V1_20261025.md)为唯一活规格；下文的状态分栏、稀疏索引和执行顺序用于追溯既有检查，不预设生产模型的模块、布局或即时行动后果。

这份合同把当前有限 CGB 证明收缩成一个可扩展的运行接口。它规定必须测量什么，不把有限状态见证写成 1B 语言世界已经具备的能力。

## 运行闭环

每个世界拍都执行同一条连续闭环：

`world input → sparse perception/update → local credit → prediction/generation → consequential action → world consequence`

输入持续到达。失联协议只影响感知内容，不暂停世界时钟、维护事件或行动后果。行动必须通过后果世界确认；内部生成或虚拟动作不能替代实际到达。

真实记录流接入时，`information_boltzmann/core/cgb_observation_stream.py` 将每条
JSONL 记录转换成带有 `tick`、`payload` 和来源的 `Observation`。适配层要求首拍为
0、后续拍严格递增一，并拒绝重复/跳拍、缺少 payload/token 以及 `arrived=false`。
这使“输入没有到达”和“输入内容为空”保持可区分，也不会把断流悄悄解释成暂停；
对应的无依赖验证入口是 `scripts/validate_cgb_observation_stream.py`。

现有真实 OWT `data/ib_owt_gpt2/train.npy` 通过同一文件中的
`token_observations` 接入；它只读整数 token，不复制或重排数据，并允许恢复时显式
传入已有的 `start_tick`。`scripts/audit_owt_observation_source.py` 已对真实文件的
前 1024 个 token 做边界验证；这证明输入接口可接入真实语言流，不等同于预测能力或
学习收益已经验证。

运行状态分成五个单一所有者：

- **控制状态**：有限离散生命阶段、储备、信用和失效标记；由局部守恒核更新。
- **感知状态**：输入 token/事件的稀疏编码和预测残差；保留到达时间与来源。
- **生成状态**：下一 token、下一事件和候选行动分布；生成头不控制物理时间。
- **信用状态**：在线责任分配、延迟结果和可撤销信用；不以 BPTT 作为理论前提。
- **账本状态**：输入、输出、写入、擦除、耗散和有效信息 bit，所有项分开记录。

## 最小可运行端口

实现必须提供以下端口，且每拍都有可追溯记录：

```text
observe(t, world_event) -> sparse_observation
update_local(state, sparse_observation) -> state'
predict(state') -> predictive_distribution
generate(state', predictive_distribution) -> candidate_output
act(state', candidate_output) -> consequential_action
credit(result, state', action) -> local_credit_update
measure(state, state', observation, result) -> ledger_row
```

`update_local` 必须满足局部预算合同；`predict` 和 `generate` 只能读状态和已到达输入；`act` 的验收来自世界后果；`measure` 不能把预测 bit、擦除 bit 和维护能量合并成一个虚拟指标。

## 量化和低显存约束

第一版实现沿用当前有限结构：稀疏边界用 bit-packed/目标索引，三值关联边界保留 `{-1,0,+1}`，小范围流量使用 3-bit 或 `int4` 容器，概率使用按任务验证过的定点格式，乘加使用更宽的 `int32` 累加器。参数规模增加时只允许按以下顺序扩展：

1. 增加稀疏索引或局部核覆盖；
2. 重新测量量化误差、溢出界和显存；
3. 重新测量 held-out 预测、行动后果和能耗；
4. 只有全部通过，才允许增加模型规模或时间窗口。

这保留无 BPTT 和代数稀疏优势，但不预先宣称 1B 参数下无损。

## 信息耗散合同

每个记录窗口提交：

```text
useful_bits = I(world_relevant_variable ; recorded_observation | declared_context)
erased_bits = measured irreversible state reductions
write_units = physical/logical write cost
dissipation = calibrated energy or explicitly labelled discrete proxy
survival/action_accuracy = held-out capability constraints
```

当前有限测量已经显示：`probeA/probeB` 都变成 `.probe` 时，probe 消耗离散维护单位但 useful bits 为 0；`answerA/answerB` 的平衡条件分支可以携带 1 bit，但这不是当前固定初态的状态律。未来语言世界必须用真实数据、非退化先验和 held-out 预测任务重新测量，不能用一次 probe 固定代表一个 bit。

## 失效边界

运行时必须显式输出以下边界：

- 感知连续到达但维护储备低于 guard；
- `GuardedTwoTickSupport` 不成立；
- 输入到达但预测互信息为零；
- 预测改善但行动后果未到达；
- 在线信用没有可归因结果；
- 量化溢出、累加器不足或状态账本不守恒；
- useful-bit 分母为零或物理能量尚未标定。

语言预测路径使用 `information_boltzmann/core/cgb_information_ledger.py` 记录每个
连续 token 的模型 NLL、声明的固定参考 NLL、正向预测增益 bit、负向预测损失 bit、
写入单位和耗散单位。`energy_per_useful_bit` 只有在提供标定能量且正向增益大于零时
才输出；否则保留为空。它是相对参考分布的操作性预测指标，不是由 CGB 公理推出的
普适互信息或焦耳定律。对应的数值边界验证是
`scripts/validate_cgb_information_ledger.py`。

`information_boltzmann/core/cgb_prequential_bridge.py` 将上述账本接到已有的在线
learner 接口：每拍先 `predict()`，再以 `external=True` 接收真实 token，最后提交
账本。它保留 learner 的状态和更新边界，不另造一个离线评估算法；跳拍、失联、非整数
目标和负的离散成本会在进入下一拍前拒绝。接口验证入口为
`scripts/validate_cgb_prequential_bridge.py`。

真实 held-out 测量入口为 `scripts/measure_cgb_prequential_heldout.py`。它要求显式
配置、checkpoint、训练参考流、validation 流和连续 `start_tick`，并检查 checkpoint
当前 token 是否与 validation 起点一致；没有合法 checkpoint 时直接失败，不会退化成
随机模型结果。它使用实际 `StreamRunner` 的预测/外部观察顺序，默认不创建 optimizer、
不 flush 更新，输出 checkpoint 哈希、数据 revision、参数量、事件数和信息账本。

在现有 5000 更新 byte-level checkpoint 上做的 128 个真实 validation 事件诊断记录了
14.3763 个正向预测增益 bit、24.3971 个负向预测损失 bit 和 128 个离散耗散单位，
净预测增益为 -10.0208 bit；正向事件的比值为 8.9035 个离散单位/有用 bit。该 checkpoint 的随机状态来自 CUDA，而当前运行环境
只有 CPU，因此报告明确标记 RNG 未精确复现；这是一条真实路径的预算受限账本证据，尚
不是收敛、完整生命能力或物理能耗结论。
默认情况下这种设备不匹配会直接拒绝；本次仅通过显式
`--allow-rng-device-mismatch` 进行诊断。

任何一个边界触发时，系统只能报告受限合同失效，不能报告“生命体仍然具有完整能力”。

## 验收顺序

先复用当前 CGB 区域合同和已有在线信用接口，建立真实语言流上的输入到达、预测、行动后果和账本记录；当前边界入口包括 `scripts/validate_cgb_observation_stream.py`、`scripts/audit_owt_observation_source.py`、`scripts/validate_cgb_information_ledger.py`、`scripts/validate_cgb_prequential_bridge.py` 和 `scripts/validate_cgb_contract.py`，分别通过 `cgb_observation_stream_ok`、`cgb_owt_source_ok`、`cgb_information_ledger_ok`、`cgb_prequential_bridge_ok` 和 `cgb_contract_ok`。再做充分训练的 held-out 评估；最后才比较 1B 规模、量化格式和显存占用。训练结果、能力结果和物理能耗分别归档，避免用 toy bit-memory 或短预算运行替代能力证据。
