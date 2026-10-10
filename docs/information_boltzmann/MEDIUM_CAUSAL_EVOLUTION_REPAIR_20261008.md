# 3D 介质因果演化修复与结构候选收尾报告

日期：2026-10-08。状态：功能实现、263 项 CPU 数值回归及 D768 单事件长区间 GPU 反传校准通过。本轮先修复并校准，原个体保持暂停。

此次工作的直接产出是让材料有完整的物理演化机会、让沿途观测保留这段历史，并使梯度与续训账本准确覆盖实际发生的过程。在此基础上提供有限接口、完整材料频带和预测驱动的结构资源候选。宏观神经通路是否从真实学习中形成、是否提高语言预测与重访保留，仍是待检验假设。

## 1. 暂停点与已知诊断

原连续个体已保存暂停：`step=1237`、fresh targets `39584`、实际 events `42272`、optimizer updates `1321`。fresh step 与所有阶段共同产生的优化器更新分开计数；已有物理态、经验、优化器历史与未决梯度按完整 checkpoint 保留。

真实 OWT 的零更新诊断见 [medium_objective_gradients_d768_20261008.json](../../results/published/medium_objective_gradients_d768_20261008.json)。它仅覆盖下一段未消费的 8 个 targets `[39585,39593)`，沿历史固定 `dt=.005` 路径，物理信用时距 `.04`，两个目标从相同完整初态分别前向，未改变 checkpoint、参数、初态或个体经验。材料系数的未裁剪梯度范数为：

| 材料参数组 | task 梯度范数 | port 梯度范数 | task–port cosine |
| --- | ---: | ---: | ---: |
| medium_material | 3.1658075 | 0.1495111 | +0.3417950 |

本窗口 task NLL 为 `3.8968711`，port objective 为 `2.8093839`；上表数值是梯度范数，不是损失。肯定结论是任务梯度清晰到达材料，且该窗口材料上的辅助梯度总体同向。辅助目标在长期、其他文本或其他坐标上的作用仍需分项记录；当前证据不支持把“辅助梯度冲突”定为根因，实际 Adam 更新还受全局裁剪及历史 moments 影响。原 32-target 双目标诊断曾超显存，成功结果只属于这个 8-target 前缀。

## 2. 按因果顺序完成的功能

### 2.1 先选择完整物理时长，再写入、演化与采样

`core/intrinsic_time.py` 与 `core/plastic_ports.py` 显式区分物理时长 `tau`、`solver_max_step` 和 `observer_max_step`。一个事件可以执行多个物理微步和多个观测区间；这两种数值分辨率独立声明，细化不新增数据、标签或优化器更新。时长参考可以依据当前材料传播标度与几何距离标定，不能把一个未校准常数称为最优时长。

内在时钟在新输入写入**之前**读取声明的有限 read aperture：局部 field/flux 能量、受体抑制和 STP 资源。它不读取新 token、当前 target 或孔径外瞬时全场来广播时序信息。正时长由参考尺度与可学函数产生，零初始化返回声明参考；没有固定 14 次循环或固定 14 拍的机制。执行上限超出时显式报错，而非暗中截短物理时间。

随后输入经既有局部端口写入并更新端口 precision，完整介质在整个 `tau` 中推进。每个 observer 区间完成实际物理演化后，再更新持存时间探针。field、三条 flux、conduction、receptors、STP、precision、temporal history 和各时钟一起保存。读出与重复读取不重复积分历史。

### 2.2 沿途测量与全区间信用

整数求积 schedule 由 detached 时长确定；各物理区间仍使用可微时长张量，clock、材料、写入及沿途状态的梯度穿过整个事件 `tau`。沿途 observer 采样解决只看末端而错过持续轨迹的问题；solver 细化控制动力学误差，observer 细化控制历史测量误差。

事件内的全区间反传保留在既有 token BPTT 窗口中。窗口边界仍显式 detach，所以它补足的是实际执行的事件内路径；跨 token 窗口的长期信用问题仍需用真实时距、响应延迟和充分训练证据评价。

root 新增 observer 内层非 reentrant checkpoint，配合事件外层重算，重放同一 schedule、同一显式区间选择和完整状态，避免长物理区间一次保留全部中间激活。该执行优化不截断路径、不重新选择时长，也不重复提交证据；它本身不扩大 token BPTT 窗口。最终等价范围以状态、损失和参数梯度回归为准。

### 2.3 有限接口预算保留可学几何

`core/local_ports.py` 和 readout 提供显式总口径预算：每侧按端口数乘物理 support box 体积计费，重叠端口仍分别收费。端口位置继续可学，读写及同侧端口允许重叠；不强制隔离端口，也不预画一条必须使用的运输路线。预算与真实 support/footprint diagnostics 一同报告。

有限口径约束的是接口观测与写入范围。重叠只说明近端路径可用，其实际任务贡献由相同完整初态下的 transport 反事实等诊断测量；不能由几何重叠单独推出捷径是语言瓶颈。网格细化沿用保存的物理半径，保持同一连续孔径。

### 2.4 材料频带与显式迁移

`core/plastic_medium.py` 保留历史 canonical `8×8×4` 基的严格加载行为，并允许 `material_reference_shape=None` 显式采用运行网格完整实 Fourier 基。8³ 上前者 `K=256`、基矩阵秩 256，后者 `K=512`；后者增加独立慢材料自由度，是模型扩容，不是单纯数值细化。

`ContinuousMaterial.load_expanded_state_dict` 按真实 Fourier 频率和正负号嵌入旧系数，新增频带 zero residual，在任意连续坐标保持原材料场。新增系数保留可学习梯度。权重迁移与完整续训分开声明，形状改变后的 optimizer moments、pending gradients 与候选出生历史需由外层显式处理，不能静默解释成原个体无变化续训。

频率坐标同时支持显式物理参考单位；旧状态严格加载保持物理滤波器，已有 Adam/AdamW moments 与 pending gradients 的坐标变换有单独迁移测试。坐标重参数化改变优化尺度，需要登记，即使迁移瞬间前向相同。

新内在时钟候选的滤波器初始半衰期与频率按其物理参考区间定义，频率优化坐标默认使用同一参考单位。它不再延用旧 `.005` 初始化来描述一个已显著延长的事件。历史固定节拍的默认物理滤波器保持原初始化，显式迁移仍需保留旧历史和优化器证据。

### 2.5 唯一慢结构资源候选

`core/structural_posterior.py` 复用上述材料连续基，不注册另一套空间字典。系数后验为 `q(c)=N(mu,diag(exp(2*log_std)))`，`mu/log_std` 形状 `[K,3]`。K=256 时新增 1536 个参数；runtime-full 8³ 的 K=512 时为 3072 个，实际参数账本随配置报告。

对一个整窗显式 noise epsilon，仅采样一次 `c=mu+exp(log_std)*epsilon`。同窗各 chunk 用保存的 epsilon 重建当前 autograd 图；checkpoint 不再随机抽样。无活动窗口的诊断与部署使用后验均值。保存 noise、固定窗先验、未决证据和标志，外层另存完整物理态、优化器及 pending gradients。

结构分配与传播为：

```text
a(x) = R * softmax([Psi(x)c, idle_logit=0])
sum_axis a_active(x) + a_idle(x) = R
B_slow(x) = v_ref * diag(a_active(x)) * unit_row_directions(x)
B_effective(x,s) = diag(fast_utilization(x,s)) * B_slow(x)
```

方向复用原下三角 shear 并归一化，避免独立幅值逃避资源收费；旧 log_speed 不再是结构模式的第二套自由幅值。快利用率使用 `sigmoid(conduction)` 与 STP 的 `x*u`，有界于 `[0,1]`，不再通过除以初始释放概率放大。维护量 `M=sum_x cell_volume_x*sum_axis a_active(x)` 按体积计费，idle 不收费。现有 field/flux 成对 skew 输运继续使用同一因子，保留活动能量与场和的数值守恒；这不等于生物代谢定律或任务通路已经涌现。

结构窗目标与继承先验为：

```text
J = mean_next_token_CE + [KL(q || fixed_window_prior) + lambda*(M-supply)] / L
rho = exp(-actual_window_physical_duration / tau_structure)
next_prior_mean = stationary_mean + rho*(updated_mean-stationary_mean)
next_prior_variance = rho^2*updated_variance + (1-rho^2)*stationary_variance
```

KL 是完整系数 Gaussian KL。每 chunk 都使用整窗 L，再乘 `chunk_count/L`，使整窗证据与 KL 只计一次。先评分、反传、记录实际事件总数和物理时距，再执行 optimizer step，最后 commit OU prior；前向和重算不会推进先验或账本。可选 dual 更新投影为非负。

端口 observed CE/action KL 继续训练本地 writer，但结构候选用精确 local writer/source VJP 显式路由辅助信用：q、材料及其余本体只接收主 next-token CE 与结构目标，writer/source 在主目标上再接本地辅助梯度。此为声明的 block objectives，不把重复观测的 port objective 当成 q 的另一份 likelihood，也不把 detached VJP 包装成直通估计器。它增加一次辅助反传成本，需要实际校准；历史模式保留原目标。

`R`、`v_ref`、时钟参考、`tau_structure`、prior/initial std、supply、dual 与口径预算均是声明的先验、材料标度或资源约束，需要给出来源与校准依据。它们不是由代码恒等式推出的 magic optimal 参数。

### 2.6 让执行和账本如实反映学习

新 optimizer 过滤当前未使用的 source 子网；旧 optimizer 保存的参数名、顺序、分组和 moments 按原策略恢复。AdamW 的 `grad=None` 与零梯度语义保持区别。健康快照只克隆实际可能更新的参数，缓存未变参数范数，仍保留全模型参数范数的监控分母。此处减少的是重复复制和计算，不虚构原本不存在的 Adam 状态节省。

训练账本新增 task NLL、port objective、joint objective，以及可得 port NLL、write action KL 与结构分项；按实际事件加权，旧保存没有的分项从新覆盖区间开始，并标明 component_events。端点 read 所需 field RHS 复用已算出的同一张量，避免重复材料/导数求值，保留原值与梯度。

精确执行优化（缓存、RHS 复用、checkpoint）与显式候选变化（事件时长、observer 求积、频率坐标、更宽材料基、结构资源和口径预算）分别登记。后者通过契约测试，不笼统声称与历史前向完全等价。

## 3. 数值证据与审核状态

所有以下构造均为数值或接口验收，不是合成能力训练：

| 契约 | 代码证据 |
| --- | --- |
| 实际模型分离端口：full tau 远端响应超过 tiny interval 的 1000 倍；transport 关闭为零；clock/write/material 梯度非零，时长梯度符合有限差分 | `tests/test_medium_interval_causality.py` |
| solver 与 observer 独立细化；全非线性 conductance、STP、adaptation、collision 开启时状态/历史/关键梯度随细化改善 | `tests/test_intrinsic_medium_time.py`、`tests/test_medium_interval_refinement.py` |
| target 不选择 schedule，孔径外状态不控制 clock；checkpoint 保持分数、完整状态与梯度 | `tests/test_medium_interval_causality.py`、`tests/test_active_medium_training.py` |
| 口径总量、重叠计费、位置梯度、局部梯度 support 和跨网格同物理半径 | `tests/test_port_aperture_budget.py` |
| 完整材料频带、连续 zero-residual 迁移、频率与 Adam/pending-gradient 坐标迁移 | `tests/test_medium_time_coordinates_and_material_bandwidth.py` |
| Gaussian KL 与解析梯度、整窗分摊、实际时间 OU、单次证据、pending sample/optimizer/gradient 精确重放、混合精度预算保护 | `tests/test_structural_posterior.py` |
| 结构有效增益不超安装容量、输运守恒、q/medium 辅助梯度隔离、writer VJP 精确性、pending 窗口下一更新完全一致 | `tests/test_medium_structural_integration.py` |
| 健康监控分母保持、冻结/无梯度快照避免复制、旧分项账本兼容、partial chunk 与恢复一致 | `tests/test_active_medium_accounting.py`、`tests/test_active_medium_training.py` |

独立审查依据：

- [原学习路径审查](../../scratch/medium_learning_review_20261008.md)、[时序验收计划审查](../../scratch/medium_timing_plan_acceptance_review_20261008.md)：定位需验证的路径与信用条件。
- [结构数学审查](../../scratch/medium_structure_math_review_20261008.md)、[旧结构代码审查](../../scratch/medium_structure_code_review_20261008.md)、[完整候选推导](MEDIUM_PATHWAY_VFE_20261008.md)：提出资源闭合与竞争解释；文档中历史“未实现”状态属于当时审查时点，本文件记录后续实现。
- [独立后验数值复核](../../scratch/structural_posterior_numerical_review_20261008.md)：17 项通过，修复 autocast 下方向预算逃逸、无效提交污染与初始方差下溢；与 resource/anisotropic 分项联合检查当时为 36 passed、1 GPU opt-in skipped。

后续独立集成 review 发现的四项问题已修复并复核通过：推理缓存同时检查参数与 buffers 的版本；schema9 校验结构法则、先验尺度与材料参考网格；结构窗强制单个持存个体 B=1；eager 原地恢复准确保留 None-gradient，固定 capture 无法表示该语义时在恢复前明确拒绝。新增 writer block-VJP 的逐参数精确梯度检查与 pending 恢复测试通过。

**最终联合 CPU 回归：263 passed、21 skipped。** 两批互不重复的受影响文件分别为 140 passed/7 skipped（63.89s）与 123 passed/14 skipped（19.99s）；skips 是未启用的 CUDA 专项测试。本次没有执行无关果蝇训练套件。完整文件清单、来源哈希与审查状态保存在 [验收清单](../../results/published/medium_causal_evolution_repair_20261008.json)。生产 Python 文件与入口语法检查通过。

**GPU 执行校准：D768/8³，真实 OWT 的一个下一 token，完整长区间前向与反传通过。** 由保存状态的实际传导因子与几何计算参考时长 `0.067733899`，为旧 `.005` 的 13.55 倍；21 个观察区间、84 个物理求解步。该全局最快速度参考仅定标，慢通道到达仍由实际响应与任务检验。物理/temporal 时钟完全一致；时间头梯度范数 `1.69269`、材料 `22.39879`，均有限非零。采样专用显存峰值 `2498.64 MiB`（约2.44GiB），分配峰值 `2362.61 MiB`。

两次尝试同模型、同状态、同 token、同物理区间：第一次触及预设2GiB allocator保护线；第二次仅将 allocator 调至2700MiB、总专用预算仍为3072MiB，完整反向通过。保留两次记录；原 checkpoint、模型参数、初态哈希不变，零 optimizer 更新、无新权重文件，GPU 退出后无计算进程。Native eager FP32 单事件前向4.365s、反向6.527s；这是数值资源校准，尚未验收 BPTT32 的生产速度或新增结构 auxiliary VJP 的 GPU 成本。见 [原始校准](../../results/published/medium_d768_long_interval_calibration_20261008.json)。下一步按同一3GB预算做融合执行和完整窗口校准，通过后才启动正式能力比较。

## 4. 三个竞争解释及后续预注册判据

| 解释 | 可区分的预测 | 公平比较与反证条件 |
| --- | --- | --- |
| 物理时间或测量信用不足 | 完整演化区间使运输形成可读响应，真实任务中材料/clock 影响随物理时距改变；solver/observer 各自细化后预测与梯度稳定 | 共享修复后的求解器、观测器与 BPTT/更新节奏，单独比较固定/内在时长；只增加数值步数而不改物理时距应趋于同解。若充分训练后任务依赖与曲线无对应改善，不能归因于时间不足 |
| 接口近端路径减少空间结构用途 | 同完整初态下局部重叠保留的任务响应可在 transport 关闭时持续；有限总口径下学习的位置/路径使用发生可测变化 | 保留允许重叠与可学位置，匹配端口数、总口径、读出容量和数据；不得强制分离制造必经运输。仅缩孔径导致信息减少或性能下降不足以证明原捷径是瓶颈 |
| 持久结构选择驱动不足 | 在相同时间、几何与有界快利用率下，可学 q 相比同出生与采样策略的冻结 q 改善真实预测、恢复或重访，并产生任务相关的持久资源变化 | 先共享因果修复 baseline，明确 aux routing/幅值本构/材料带宽等耦合变化，再做对应消融；匹配资源、参数/容量与优化预算，记录实际算力差异。漂亮通路、低 maintenance 或 KL 下降单独均不构成预测收益 |

能力验收使用专业真实 OWT，同 tokenizer、数据 revision/序列、出生来源、曝光预算、桥接 targets 和优化器节奏；matched joint 对照至少 3000 次实际 updates，并根据曲线与预先声明的停止/收敛准则决定是否需要继续，不能把固定短预算当平台。沿用 never-reset 个体、先评分再更新，分别报告首次预顺序预测、真实语境改变后的恢复、A→B→A 重访，以及主动四支柱监控中的实际能量账本、表示结构、时间粗糙度和完整状态条件响应。

当前功能与数值验收已完成，原训练保持暂停。后续先校准融合执行的完整 BPTT32 及结构候选成本，再进入真实数据比较；能力判据保持首次预测、恢复速度与平台、重访保留。这里已经接通的是可演化、可观测、可归因的路径，语言改善与通路涌现由正式训练决定。

## 5. 完整窗口的执行验收与重开

新出生使用 D768、8³、runtime-full 材料、有限口径、结构后验、时间探针及写前内在时钟。按出生材料标度声明参考区间 `0.8993329405784607`，求解上限 `0.008318093605339527`、观察上限 `0.03327237442135811`；初始事件为 28 个观察区间、112 个物理步。该参考与原 checkpoint 单事件校准的状态、材料及时间坐标不同，不能互换比较。

完整 BPTT32 校准暴露了单事件测试覆盖不到的执行缺陷：嵌套 dataclass checkpoint 强引用内部状态；将其张量化后，长反向重算仍累积中间缓存。几次仅调整重算粒度、引用生命周期与垃圾回收的尝试都在首次更新前触及 allocator 上限，配置和失败报告保留。所有资源尝试属于可丢弃个体，不提交正式研究状态，不保存权重。

最终执行候选为 `checkpoint_state_vjp`：每个事件独立重放并计算一阶状态/参数 VJP，将全部输入余切传回前一个事件；共享 `table`、`prepared` 和 `duration` 显式传入，由外层汇总后通过共享图反传。物理区间、事件数、窗口长度与目标均保持。内部 observer/physical-step checkpoint 继续复用同一编译单步；结构 writer 辅助目标与 task 目标串行执行，辅助梯度暂存 CPU。

数值验收：新增 helper/真实模型/原 checkpoint 共 15 项通过；真实 32 事件覆盖全部参数、初态、复数历史、FP64 时钟、内在时间、缩放和 pending 梯度。最新辅助路由及结构集成 11 项通过；active learner/accounting 16 项通过、2 项 CUDA opt-in 跳过。独立审查见 [执行审查](../../scratch/medium_event_vjp_execution_review_20261008.md)。此执行方式支持确定性、显式噪声、一阶反向；参数梯度 hook 与高阶导数不在契约内。数值正确性与资源可运行性分别验收。

完整 GPU 两窗口校准和正式重开状态随后记录于资源报告与出生计划，正式运行须从同配置新出生开始，不能继承校准更新。

### 5.1 第二窗口的常驻状态与精确暂存

独立事件 VJP 已使第一次完整 32-target 更新成功，暖执行约 `334.98 s/32 targets`。第二窗在辅助反向触及 2880 MiB allocator 限制；全局高水位日志不包含回落，不能据此断言跨事件泄漏。第一次更新后新建的 Adam 一、二阶 moments 约 574 MiB，构成第二窗增加的明确常驻项。释放已完成的 decoder 零梯度缓冲可省约 147 MiB，但单独仍不足。

新增显式 `--optimizer-state-offload`：仅在 eager 的完整 forward/backward 期间，将闲置非标量 CUDA optimizer state 同 dtype 同步拷到 CPU，退出作用域时恢复原设备，然后才进行 clipping、健康快照、AdamW step 与序列化。参数、pending gradients、scalar step 与物理状态均不移动或重置；captured 路径明确拒绝此开关。该方法改变存储驻留，不改变学习方程。

本次定向回归为 35 passed、2 CUDA opt-in skipped：31 项覆盖 active learner、accounting、结构集成与事件 VJP；另外 4 项覆盖 AdamW 暂存、None/zero 语义、异常可恢复路径与三次实际介质更新的逐位等价。既有非零 pending 恢复对照仍通过。独立审查见 [Adam 暂存复核](../../scratch/medium_optimizer_staging_review_20261008.md)。完整双窗口 GPU 校准分别验证实际设备传输、恢复后更新峰值、持续状态和真实耗时；通过后才释放正式运行。

若 OOM 导致暂存恢复本身失败，Python traceback 保留原错误，但失败进程内的部分恢复状态不作为新的提交点；从最后完整 checkpoint 恢复。原个体保持暂停，校准个体不生成 pt，正式个体从已登记的新出生重开。当前速度约每窗口五分半，代表恢复了完整物理区间后的真实计算成本；此前极短区间的 token/s 不再作为该候选吞吐量。后续速度优化应保持相同物理区间、观察采样和信用长度。

### 5.2 完整两窗口 GPU 验收通过

同一份源代码下，64 个真实 OWT targets、两次完整 AdamW 更新、pending=0 均完成。采样专用显存峰值2490.64MiB，分配峰值2331.38MiB，reserved峰值2816MiB；总专用预算3072MiB 未越线。共享采样80MiB属当前驱动/进程计数，不能由此断言新增模型共享为零。两窗口耗时344.66/321.28s，未裁剪梯度范数20.7543/16.9086均有限。时钟、temporal历史与结构提交按真实演化连续。校准不写pt、不污染正式个体。见 [完整窗口验收](../../results/published/medium_full32_execution_acceptance_20261008.json)。

该速度约0.10target/s，5000个新鲜窗口加主动评估约需20天；这是当前执行成本，不能沿用旧极短物理区间的吞吐率。正式重开仍保留完整演化与信用，后续提速以相同物理/观测/梯度契约为边界。

### 5.3 用户速度反馈后的执行决策修正

用户明确指出约五分半/32 targets不可用，要求重新讨论。正式新个体在第一次更新完成前停止；完整出生last.pt保留，GPU计算进程已退出。上节“资源通过后正式重开”的决定由本节取代，数值与资源结果保持原证据范围。

独立讨论确认两个不同问题：固定Householder读坐标被当成稠密矩阵反复计算，是可等价消除的执行浪费；每输入等待标称全域传播，则是可修改的刺激时间假设。后者应另登记新时序候选，允许旧信息跨后续输入持续传播，保留近期有限物理路径、全状态与完整32事件链。输入的物理参考需在出生时冻结，网格细化只改变数值精度。

采用[持续流水执行设计](MEDIUM_STREAMING_EXECUTION_DESIGN_20261008.md)作为下一唯一候选。当前约4个微步的出生定标替代112步/event仅说明名义工作数量变化，速度和能力都未验收。先做等价读出优化及流水接口的最小实现，再做数值与两个真实完整窗口的资源校准；分钟级执行不进入长训练。保持长等待轨迹换更好求解器作为比较候选暂不实施，避免重新支付当前约20天的成本。
