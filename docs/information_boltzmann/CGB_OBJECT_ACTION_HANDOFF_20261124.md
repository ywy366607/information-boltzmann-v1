# 梦漓环世界：对象条件行动交接与闭环一致性修复合同

日期：2026-11-24  
基线检查点：`D:\mengli-umwelt-gaze-smoke5-20261124\body\last.pt.gz`（第 3284 轮，31.26M 连续语言生命体＋49K `GazeObjectPointer`）  
定位：24 小时双语长程成长训练前的闭环完整性修复、结构分工界定与吞吐复查。

---

## 1. 全链路闭环彻查发现的断口

沿完整闭环 `感知 → ALIF 节律 → 决策与对象指向 → 世界送达 → 回应边界捕获 → 因果结算 (metal_g) → 信用与优化器更新 → 存档恢复` 逐段核对，定位出 4 处会静默影响长期学习一致性的断口：

1. **[P0] 非妈妈来源缺少统一回应边界，导致 `metal_g` 静默归零且无法进入因果库**  
   `world_learning_cycle.observed_pieces()` 已放开支持 `('mother', 'parent', 'tool', 'object')`，但 `navigation_mother_boundary.mother_boundary_feature()` 仍硬编码只识别 `source == 'mother'` 与 `source == 'text'`（且依赖 `navigation_response_bytes`）。导致中文故事书（`source == 'chinese_storybook'`）、爸爸（`'parent'`）、工具（`'tool'`）和动态对象（`'object'`）全部返回 `_mother_boundary_target = None`，进而在 `finite_use_value.observed()` 中被判为无独立回应约束、`metal_g` 恒为 `0.0`（已在第 3277/3281/3283 轮实机日志复现）。

2. **[P0] 对象选择晚于行动起草与因果期望冻结**  
   此前 `GazeObjectPointer` 在 `world.take()` 内部（`_utter()` 与 `causal_use_value.begin()` 之后）才选对象，使因果模型无法直接比较“同一句表达发给妈妈”与“同一句表达用来检索书页”的预期后果差异。

3. **[P1] 指针独立优化器状态未纳入完整存档恢复合同**  
   `GazeObjectPointer` 使用独立 `torch.optim.Adam`，动态状态虽保存了对象指纹与计数，但未显式保存/恢复优化器的 `m / v / step` 状态；长期连续学习要求“重载后的权重一致”与“重载后下一次学习更新一致”同时成立。

4. **[P1] 外部即时打断泄漏进内部 ALIF 阈值归因 ＋ 中文书页偶发 UTF-8 半字节切片**  
   - 爸爸插话或紧邻对话拍通过 `has_pending_interrupt()` 强制置位 `pending_spike = True` 后，`record_executive_spike` 仍把这一拍的 `metal_g` 结算进内部 `LearnableAlifRhythm` 参数 `phi`；外部打断只应重置不应期，不污染内部节律阈值学习。
   - `SidecarCorpusStream._slice_from()` 按原始字节起止切片，偶发在共读书页首尾切出半个汉字。

---

## 2. 整体结构保留与合并表

| 模块 / 层次 | 架构判断 | 本轮动作 |
|---|---|---|
| **对象指向指针 ↔ 原行动与因果估值** | **合并决策与信用交接；保留快习惯＋慢估值分工。** 统一为 $a = (\text{目标对象}, \text{表达})$，由指针提供快偏好与动态感官指纹 $k_i$，因果库评估目标条件后果 $p(\text{回应}\mid Z, \text{目标对象}, u)$，共同决定目标并贯通结算 | **本轮完成** |
| **妈妈 / 书页 / 爸爸 / 工具回应处理** | **统一回应边界、来源身份与证据准入。** 所有合法来源均产出可结算回应边界；将“缺少有效证据”与“实测收益为零”显式区分 | **本轮完成** |
| **指针优化器、基线与恢复** | **补齐学习生命周期合同。** 保留轻量独立 Adam 以免扰动 31M `BudgetedAdamW` 参数组，但将其 `state_dict` 纳入检查点完整保存与重载，并验证重载后下一步更新逐项等价 | **本轮完成** |
| **外部打断 ↔ 内部 ALIF 节律** | **清晰边界隔离。** 外部人际/问答打断标 `external_interrupt=True`，跳过内部 `alif_rhythm.phi` 信用归因 | **本轮完成** |
| **`mother_response` 与 `body_transition` 两条程序** | **保留。** 分别表达“经由外界回应再续行”和“直接身体续行”，并非冗余；端口 `Port.MOTHER` 语义为“外界回应端口”，对象身份作为条件特征进入 | **保持不变** |
| **ALIF / 动量 / 工作场 / 长期因果库 / H-Net** | **保留正交分工。** 分别负责节律、相空间走势、当前工作记忆、长期共轭证据与动态分块 | **保持不变** |
| **22 层 `super().step()` 继承链** | **长跑后专项清理。** 涉及全链路历史兼容性，需在 24 小时数据之后单独做 1:1 数值等价重构 | **延后至下一里程碑** |

---

## 3. 目标条件行动 $a = (\text{target\_id},\ u)$ 闭环数据流

1. **边界统一（`navigation_mother_boundary.py` + `world_learning_cycle.py` + `unified_world_port.py`）**  
   - 前景互动源 `FOREGROUND_SOURCES = ('mother', 'parent', 'tool', 'object')`：前景区间 `[0, mother_bytes)`，回应边界在 `own + mother_bytes`；
   - 书页／文本源 `BOOK_SOURCES = ('text', 'chinese_storybook', 'matched', 'topic', 'parent_reply')`：回应边界在 `own + navigation_response_bytes`（所有书页返回统一填写 `navigation_response_bytes = len(resp)`，且 `SidecarCorpusStream` 严格对齐 UTF-8 首尾字符边界）。
   - 结算回执显式输出 `evidence_available`（区分“无有效边界证据”与“实测收益为 0”）与稳定 `target_object_id`。

2. **决策前移与目标条件预演（`gaze_pointer.py` + `world_learning_cycle.py`）**  
   - 在 `engine._utter()` 产出 `(utterance, record)` 之后、`causal_use_value.begin()` 和 `engine.world.take()` **之前**：
     - `GazeObjectPointer` 内部新增零初始化的 `target_to_action: Linear(d, d, bias=False)`，将每个在场对象 $i$ 的动态感官指纹 $k_i$ 映射为目标条件偏移 $\delta(k_i)$，构成 $\tilde{a}(i, u) = a(u) + \delta(k_i)$；
     - 用既有因果关系库（`causal_relations` / `mechanism` / `finite_use_value`）评估每个在场对象的慢预测价值，与指针快偏好 `habit_gaze` 融合后选定 `planned_target_id`；
     - 将选中对象的目标条件动作 $\tilde{a}(\text{chosen}, u)$ 同步写入本次预测查询，再进入 `causal_use_value.begin()` 冻结期望。

3. **送达、证据入库与双路学习**  
   - `world.take(utterance, ..., planned_target=planned_target_id)` 执行真实世界交互（真人即时插话或显式 `@对象` 优先覆盖），回传 `info['target_object_id'] = actual_target_id`；
   - `world_learning_cycle` 在构造最终 `action` 时应用同一个 `condition_action(action, actual_target_id)`，保证 `memory.action[slot]`、`identified_event_credit`、`finite_use_value.observed()` 与 `verify_commit()` 逐位一致（`rtol=0, atol=0`）；
   - `gaze_pointer.settle_credit(metal_g, ..., boundary_target=...)` 同时用真实 `metal_g` 更新指针策略权重、用真实回应边界预测误差更新 `target_to_action`（让同一句表达面向不同对象时能分化出不同的后果预期）。

4. **完整优化器恢复**  
   - `GazeObjectPointer.export_dynamic_state()` / `load_dynamic_state()` 完整序列化并恢复指纹、计数、`target_g_ema` 及 `self._opt.state_dict()`（所有 `exp_avg / exp_avg_sq / step` 张量）。

---

## 4. 验收门（长跑启动前全部通过）

1. **目标条件贯通单测**：同一句表达分别面向 `book` 与 `mother`（及 `parent`/`tool`），`target_object_id` 与 $\delta(k_i)$ 贯通预演、送达、回应边界与 `metal_g` 结算；
2. **全来源合法证据单测与实机验证**：中文故事书、英文书页、妈妈、爸爸、工具均产生合法 `_mother_boundary_target` 与非零真实 `metal_g`（`evidence_available = true`），不再出现中文书页恒为 `0.0`；
3. **重载后下一步学习更新一致**：保存检查点并重载到新实例后，执行同一步学习更新，更新后的参数与直接继续运行逐项一致（误差 `< 1e-6`）；
4. **外部打断隔离验证**：外部打断触发的脉冲不修改 `LearnableAlifRhythm.phi`；
5. **吞吐剖析与瓶颈复查**：在真实 31.26M GPU 身体上剖析每步耗时（冷静阅读、执行脉冲、因果结算、指针预演、存盘与 DSH 调用），消除新增热点后再启动 24 小时持续成长训练。
