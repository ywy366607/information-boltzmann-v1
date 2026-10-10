# 第三代：容量付费的持存通量连接

日期：2026-10-09。状态：物理接线、CPU 数值/接口验收及存档契约已实现；GPU 数值、D768 实际开销与真实流收益待验收。第二代 Fourier 自然梯度容量更新继续作为既有结构学习器。

## 研究靶子与这一代的实际变化

目标是让有限持存介质根据任务价值形成、转向和重组通路，同时保留正在流动的信息。第二代回答“哪些位置和传播行值得分配容量”；第三代增加“同一位置的传播模式如何在已付费容量内交换信息”。这使学习到的分支能够改变后续传导，而后续传导的任务信用又能更新局部连接。

此前的 `LocalHopfBranchPathway` 作用在汇总后的读出 feature 上，每次从 `(feature, 0)` 开始，合并后只送出 main，residual 结束于该次调用。它可作为读出变换的历史对照。当前生产候选采用 `LocalFluxJunction`，位于 `PlasticMedium3D.transport` 内：直接耦合已经持存的三路通量 `q0/q1/q2`，之后由原来的局部读出观察整个介质的实际演化。

这一代保持 field、三路 flux、elapsed、conduction、receptors、STP transmission、时间探针及完整学习状态。新增活动状态元素为 **0**。D768 个体的材料宽度为 8，唯一新增可学模块是 `Linear(8 + 3, 3)`，共 **36 个参数**。没有每次读出求解 1536×1536 矩阵的步骤；连接计算由逐位置的少量投影与二维旋转组成，随网格点数和内容宽度线性增长。

## 固定状态坐标上的连接方程

在一个位置，以原有持存波变量表示状态：

\[
 s=(f,q_0,q_1,q_2),\qquad
 E=\frac12\sum_x w_x\left(\|f_x\|^2+\sum_{a=0}^{2}\|q_{a,x}\|^2\right).
\]

数值账本采用原实现的空间平均和通道求和；选择一致的体积权重可得到等价的体积分规范。结构改变发生在生成元上，存储坐标及其能量度量保持不变。

设 `B` 是已安装结构经过现有 conduction/STP 利用率后的传导因子，`b_a = ||B_a||` 是第 a 行已经付费的耦合幅值。三组局部模式对及其独立资源归属为：

| 连接的持存模式 | 资源 owner 行 |
| --- | --- |
| q0 ↔ q1 | 0 |
| q0 ↔ q2 | 1 |
| q1 ↔ q2 | 2 |

局部门控同时使用材料与当前通量能量分布：

\[
 e_a=\operatorname{mean}_c q_{a,c}^{2},\qquad
 u_a=\frac{e_a}{\sum_b e_b+\varepsilon},\qquad
 l=W_m m(x)+W_u u(x,t)+b.
\]

实现先以三个模式共同的最大绝对幅值缩放，再平方得到相对能量，从而让有限大幅值输入仍有有限的门控中间量。全零通量对应零能量份额。门控使用当下的局部状态和材料，目标 token 不进入这个条件输入。

由 CK 编译出的方向掩码 `d_a ∈ {-1, 0, 1}` 决定当前可执行连接：

\[
 \alpha_a=\frac{\pi}{2}\tanh(l_a)d_a,\qquad
 \kappa_a=\sin\alpha_a.
\]

把每行的付费容量在空间传导和局部连接之间分配：

\[
 B'_a=\cos\alpha_a\,B_a,\qquad
 j_a=b_a|\sin\alpha_a|,\qquad
 \omega_a=\frac{b_a\sin\alpha_a}{\ell_a}.
\]

于是有精确的平方耦合容量恒等式：

\[
 \|B'_a\|^2+j_a^2=b_a^2.
\]

这里的份额是 **κ² = sin² α**；角度 α 与容量份额在遥测中分开解释。这个恒等式约束有限的耦合能力，电功率、耗散热和波能量由各自真实账本衡量。

对于模式对 `(r, s)`，局部方程为

\[
 \dot q_r=-\omega_a q_s,\qquad
 \dot q_s=\omega_a q_r.
\]

其冻结速率的更新是精确二维旋转：

\[
 \begin{pmatrix}q_r'\\q_s'\end{pmatrix}
 =\begin{pmatrix}\cos(\omega_a\Delta t)&-\sin(\omega_a\Delta t)\\
 \sin(\omega_a\Delta t)&\cos(\omega_a\Delta t)\end{pmatrix}
 \begin{pmatrix}q_r\\q_s\end{pmatrix}.
\]

所有非零模式都参与并保留。旋转保持每个位置的三路通量平方和；完整 backward 同时传播门控对材料和状态的导数。能量保持和学习 Jacobian 的范数是两个验收对象，前者的恒等式已有数值证据，后者继续通过完整 VJP 与数值稳定性检查约束。

实际输运组合为半步连接 → 原有边输运 → 半步连接。每个半步内采用对称的五个模式对旋转。边输运仍使用原有成对反对称场/通量生成元。该组合是明确的算子分裂；它提供固定预算的可执行近似，积分误差随数值分辨率验收。

## CK 结构如何承接实际算子

`RootedTree` 的节点标签绑定真实 `q0/q1/q2` 存储。`compile_junction_forest` 遍历树边，编译三路有符号 `route_mask`。剪去树边会关闭相应物理耦合，重新嫁接会重新打开；反向装饰改变旋转方向。重复使用同一模式对会重复占用资源 owner，因此编译器拒绝该结构；单节点保留对应模式但不供应连接。

采用根树 CK 余乘法管理允许剪切：

\[
 \Delta T=T\otimes1+1\otimes T+
 \sum_c P_c(T)\otimes R_c(T).
\]

余结合性保证组合表示中的递归分解一致。物理 evaluator 把其中选定的森林转为掩码；相互重叠的物理旋转保留其实际次序。森林由三个已存在模式的连接原子组成，活动可以在输运和模式交换中形成反馈。根树是结构编辑的组合表示，完整反馈物理仍由持存介质执行。

当前门控幅值可通过任务梯度连续学习；离散剪切、嫁接通过显式 `set_topology` 接口执行。这一接口已经让结构编辑影响真实算子，自主的硬树搜索及其策略学习属于后续独立机制。余乘法表示的是结构分解，不将一份物理信息免费克隆到多个独立状态槽。

`prepare_evolution` 保存当前材料条件 logits，并 **clone 已编译掩码**。一个已准备的轨迹使用该拓扑快照；后续编辑通过重新 prepare 生效。存档同时保留森林 `_extra_state`、掩码和长度参数，加载前验证森林编译结果与掩码一致，避免静默恢复出不同拓扑。

## 材料长度与运行网格分开

`ℓ_a` 是保存于材料连接的固定构成长度；出生时由声明的材料参考形状确定。它与运行网格间距 `h_a = 1/n_a` 分开存储和使用。网格加密改变离散边输运的数值分辨率，局部连接的 `ω_a = b_a κ_a/ℓ_a` 则保持同一个材料时间尺度。重组与重采样均不得用新的运行 shape 悄悄重设已保存的 ℓ。

该分离让连续材料表示继续承接局部资源更新：Fourier 结构系数决定已安装容量及材料条件，局部活动条件决定当前利用和转向，分辨率只决定对该介质的求积与传播近似。

## 数值步长预算及完整信用

新增连接引入额外旋转速率，因此求解器公开保留相应角速率预算，物理输入时钟继续使用原来的持存历史机制。

对运行形状 `n_a`，设 `c_a = 2√2 n_a`，`d_a = 1/ℓ_a`。冻结生成元的逐 owner 速率上界可写成

\[
 \Lambda_{\rm new}\leq
 \sum_a\left[c_a\max_x b_a(x)\cos|\alpha_a(x)|
 +d_a\max_x b_a(x)\sin|\alpha_a(x)|\right].
\]

两个最大值分别取值，因为传导与连接的峰值可以发生在不同位置。由材料 logits 的最大绝对值和能量份额的凸组合界得到可达到的角度上界 `ᾱ_a`。令

\[
 r_a=\frac{d_a}{c_a},\qquad
 M=1+\max_a r_a\sin\bar\alpha_a.
\]

求解器使用 `effective_solver_max_step = solver_max_step / M`。未使用连接的零门控给出 **M = 1**；从零开始的门控变化产生连续的保留倍率，避免“只要非零就突然换一整套积分精度”。上界覆盖冻结模式耦合的频率；状态依赖门控的完整信用由实际 VJP 承担，长期全局积分误差另行验收。

原有 `ceil` 产生整数子步数。每个事件的已选择 schedule 在 primal 与 backward replay 间固定，因此验收的对象是 **该选择下的条件 VJP**；连续的事件时间、状态和材料参数仍沿完整物理轨迹求导。这个数值调度声明不会被包装成学习离散步数策略的 planner gradient。

## 容量学习、能量账本与可证伪预测

第二代结构学习仍是原有 Fourier 系数图表上的 simplex pullback 自然梯度：固定窗口噪声、冻结 posterior `log_std`，保留原有 likelihood、Gaussian KL、maintenance、OU prior 和已声明的 trust-region 更新。结构 mean 继续由独立容量 updater 持有；36 个连接参数进入已有普通优化器组。此版本没有附加“流量大就奖励生长”的规则。

每步分别报告 transport、junction、collision、bath 的总能量及非 DC 能量变化。junction 半步产生的变化从输运合并账本中分出，避免重复计数：

\[
 E_{\rm after}-E_{\rm before}
 =\Delta E_{\rm transport}+\Delta E_{\rm junction}
 +\Delta E_{\rm collision}+\Delta E_{\rm bath}.
\]

纯局部旋转的总波能量变化应仅为舍入误差；局部方向不同可改变 DC/空间能量分布。浴的真实 source work 和 dissipated heat 继续单独记录。连接改变信息如何继续传播，稳定的瞬时流线或材料各向异性都要结合实际状态与能流解释。

这一代的可证伪预测是：

1. 完整零出生迁移的 logits、全部 belief、物理时钟与基线一致；以零门控为对照可分辨结构机制和其他初始化收益。
2. 在非零局部活动下，现有任务梯度可更新连接 gate；随后相同输入可以产生不同的实际传播和读出，容量恒等式始终闭合。
3. 非零 gate 下剪切、嫁接会改变对应算子的响应，单纯规范排序或未修改可执行掩码的表示变化保持零效应。
4. 改变运行分辨率时，保存的 ℓ 和固定材料连接速率不变；差异归于求积/输运近似并随精度验收。
5. 若真实预测改善依赖连接，连续体配对的禁用/剪切应改变对应结果。指标仍是同源真实 OWT 的 prequential NLL、恢复速度与确认平台构成的 AG、实际重访保留，以及真实计算/显存开销。

## 从第二代完整个体接续

本次已确认的 Gen2 完整来源是：

- `results/medium_d768_capacity_growth_gen2/last.pt`
- step **2813**，fresh training tokens **90,016**。

日志中的 93,824 token 终态另行保留为未完整存档的记录；接续使用实际权重中的物理状态、学习器及优化器计数。`--adopt-hopf-branch --hopf-recomposition` 要求独立输出目录和完成的信用窗口，新 gate 精确置零，旧权重、belief、OU 状态、RNG、cursors、pending gradient 键与学习/评估节奏逐项保留。

源码迁移记录新增参数、结构字段及 before/after hashes。active learner、health、training 三条兼容变更已经通过逐字节反变换验证为声明的差异；它们必须匹配已审阅 hash 对才能通过迁移。证据见 [source continuation audit](../../results/published/medium_gen3_source_continuation_audit_20261009.json)。

保存采用真实完整保存回执：权重内嵌 checkpoint identity，`last.pt.receipt.json` 记录成功替换、文件大小及精确 counters，progress 显示当前与最后成功存档的位置。`completed/stopped` 必须与最新回执一致。恢复保留旧日志，`resume_journal.jsonl` 记录 attempt ancestry 和未存档尾段；统计通过已提交分支过滤并去重源数据区间。

## 当前验收与剩余验收

已完成独立源码审查。CPU 局部连接数值/接口验收 **18 项通过**，涵盖容量恒等式、完整非零模式保留、冻结旋转逆、零门控迁移、任务梯度、剪切/嫁接的实际作用、参考长度、极端有限幅值、prepared 拓扑快照、结构存档一致性、阶段能量闭合及数值倍率。文件契约测试 **11 项通过**，涵盖替换锁重试、失败后的终态拒绝、回执一致性、真实恢复分支及精确源码兼容。

`tests/test_state_checkpoint_vjp_model.py` 包含带持存内在时钟和非零连接的 **32-event** 检查，比较原生 autograd 与 checkpoint VJP 的参数、初态和连续时间信用，并覆盖已有 pending gradients。其用途是验证实现的完整信用，真实语言能力由专业语料的持续学习评估决定。

可重复的 CPU 验收命令：

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_medium_junction.py tests/test_checkpoint_receipt.py
python -B -m pytest -q -p no:cacheprovider tests/test_state_checkpoint_vjp_model.py
```

GPU 浮点/重放一致性、D768 单更新耗时、低于 3900 MiB 的峰值与真实流收益均 **pending**，由后续验收填入实际结果。本文件完成数学、状态和生产接口闭合；比较训练继承既有真实数据、完整个体接续和四柱持续评估，不以 CPU 恒等式替代能力结果。
