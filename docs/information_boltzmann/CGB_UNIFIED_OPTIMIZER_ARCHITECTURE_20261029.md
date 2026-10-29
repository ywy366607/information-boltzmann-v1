# CGB 统一优化器架构：由已证接口组合

这份文档记录的是一个 Lean 机械契约，不是已经完成训练的生成脑。目标是把已有定理合并成一个最小可实现骨架，避免继续为每个指标增加独立模块。

## 现有机制的状态

| 机制 | Lean 状态 | 当前 ResourceCycle 状态 | 缺口 |
|---|---|---|---|
| 非可分耦合 | cgbRelationCarrier、pairwise_relation_requires_cross_term 已证 | phase 与矩阵分配产生运行时耦合 | 尚未证明真实轨迹保持给定的非零交叉项 |
| 局部资源抑制 | reservoirGain 的局部反优势已证 | 行列资源、疲劳池和竞争系数已实现 | 尚未证明代码增益等于该增益律 |
| 竞争与分化 | 双行列预算、排列极值和分化已证 | Sinkhorn 近似满足预算残差 | 只是分配层耦合，不等于关系状态 |
| 历史载体 | 延迟归因需要历史载体；两状态交换推出几何迹 | mean/rms/phase 只保存梯度和资源摘要 | 没有带来源类型的激活/行动历史 |
| 第三因子调制 | G9、G15c、fresh_decrease_unbiased 已证 | 现有反馈是方向代理，不是新证据上的后果 | 尚未接入独立后果调制 |
| 锁存与释放 | deferLatch 的保持、释放、感知已证；有益锁存还需漂移和风险条件 | 可选 anchor/latch 代码存在 | 尚未和资源池及第三因子统一 |
| 守恒 | ConservedCarrierPath、转移平衡和同调账本已证 | FP32 资源总量有小漂移 | 浮点实现与精确载体仍需对拍 |

## 合并后的最小架构

对每个局部关系单元，架构由以下五个状态通道组成：

1. **关系载体**

   Fκ(a,b) = κ*a*b，其中 κ ≠ 0 时混合差非零，因此不能被 Adam 式坐标可分状态表示。

2. **局部资源增益**

   G(s,m) = base * s / (s + m)。

   局部需求 m 增大时可用增益严格下降，给低需求隔室留下保护边际。

3. **双预算分配**

   A(p) = [[p, 1-p], [1-p, p]]，0 ≤ p ≤ 1。

   每行和每列都守恒。单独列预算允许赢家通吃，单独行预算允许复制；双预算把极值限制到排列分化。

4. **生成式历史载体**

   h(t+1) = h(t) + kp*(1-h(t)) - km*h(t)。

   历史不是额外命名的“资格迹模块”，而是局部交换动力学的当前状态。Lean 已证明其偏离固定点按 (1-kp-km)^t 衰减；需要更长尾时，再由多时间尺度池生成。

5. **结果调制与可释放锁存**

   步长界由有证据/无证据的 trust 规则更新；无证据时保持原界。锁存采用“短暂相反证据保持、达到阈值释放”的有限出口。真正的第三因子必须来自更新之后的新证据；当前梯度和历史梯度的对齐只能作为稳定性统计，不能替代它。

Adam 的 m,v 可以作为连续数值后端，负责尺度归一化；它不承担关系、历史来源或结果调制的理论角色。

## 对现有 ResourceCycle 的最小改造

不再新增一份 per-parameter 资格迹。把现有 phase 重新解释为边占据率 h，使用同一状态生成关系：

`Fκ,h(a,b) = κ * (2*h - 1) * a * b`。

当 h 偏离 1/2 且 κ 非零时，它携带非零交叉关系；h = 1/2 是对称、无关系的状态，符合“没有扰动就不自发分化”的定理。h 的递推直接采用两状态交换，因而历史衰减不是另加的资格迹机制。

现有行列 reserve 和 resource_pool 继续做资源账本，Adam 的 m/v 保持连续预条件；第三因子只增加每个隔室的独立后果调制 q，锁存作用在资源占用和 h 的释放上。这样新增的是关系守恒约束和后果输入，状态数量不会按参数再复制一份资格迹。

## Lean 证明入口

新增模块 [GenerationUnifiedOptimizer.lean](../../lean/CGBAxioms/CGBAxioms/GenerationUnifiedOptimizer.lean) 定义 CGB.Unified.build，并证明：

- build_is_cgb_candidate：关系载体和局部资源增益满足现有 CGB admission contract；
- balanced_allocation_contract：非负、行预算、列预算和竞争上界；
- balanced_max_differentiates：双预算极值分化；
- build_history_is_geometric：历史载体由两状态交换生成；
- build_history_balance：守恒载体按输入减输出的历史和恢复；
- build_modulation_stays_in_band：死区 trust 调制保持在稳定区间；
- build_latch_holds_and_releases：锁存保持并在出口阈值释放；
- uniform_allocation_is_budgeted、balanced_allocation_interaction、balanced_allocation_has_relation：双预算本身不保证关系项，均匀点会把交叉项压成零；
- ema_history_collision：不同有序梯度历史可以给出相同的一阶/二阶 EMA；
- phase_relation_interaction、phase_relation_has_relation、phase_history_is_generated：同一个 phase 状态同时承担关系读出和生成式历史；
- phase_step_stays_in_unit_interval、phase_history_stays_in_unit_interval：在交换率位于 [0,1] 时，phase 不会越界；
- phase_exchange_transfer_is_conserved、phase_exchange_transfer_reads_phase：连续 phase 更新可以写成单条带来源的源—目标守恒转移，并从目标读出净历史；
- fresh_outcome_measurement_unbiased、fresh_outcome_trust_stays_in_band：更新后的新证据可作为第三因子输入，同时保持 trust 边界；
- build_mechanical_admission：把上述机械条件合并成一个入口定理。

总模块已编译，完整信任/覆盖库存显示 1,200 条已审计源码定理，新增模块 27 条。审计的证明信任和模块覆盖通过；定理解释目录仍有一个与本次模块无关的旧解释源哈希不一致，暂不把目录刷新失败解释成 Lean 证明失败。

## 尚未满足的条件

这份架构仍有四个必须在生产实现前补上的连接：

1. 证明 phase → allocation → update 能保持非零关系，而不仅是每步出现矩阵耦合；当前新增反例已证明均匀分配会消除关系项；
2. 把现有状态与带来源类型的历史载体对拍，排除同一梯度序列对应不同激活/行动历史的碰撞；当前 EMA 碰撞定理证明一阶/二阶统计本身不够；
3. 用未见过更新的新数据形成第三因子，并证明它进入实际更新而非只记录日志；

### 2026-10-29 接口实测结论

在现有连续 `ResourceCycle` 上，`outcome_on` 与 `outcome_off` 使用同一初始化、同一 OWT A→B→A 数据预算，并以相同频率测量更新后的新批次。修正后的实现确认新鲜结果确实进入下一步有效步长；但当前把 `realized - predicted` 直接送入旧的 `gamma_down` 规则会使 `rho` 从 1.0 退化到 `rho_min=1e-4`，5000 步单种子 prequential BPB 为 4.1945，而观测成本匹配的关闭臂为 3.1259。该结果否决当前结果门进入主线，但保留“更新后独立后果必须能调制下一步”的接口；下一次设计必须先解决证据的尺度、基线和时间聚合，再进行长跑。
4. 把锁存的保持/释放与守恒资源转移接起来，证明保持期间仍感知，释放后资源重新可用。

在这四项完成前，ResourceCycle 只能称为“Adam 数值基底上的矩阵资源竞争优化器”，不能称为完整三因子生成生命体优化器。

## 下一步

先做关系保持审计和历史载体碰撞反例；两者任一失败，就回到 Lean 修改状态接口，不启动新的长训练。两者通过后，再在现有 ResourceCycle 上接入独立后果调制，并用同一 OWT 模型做单变量消融。
