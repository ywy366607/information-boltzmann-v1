# 给编码 AI 的执行清单

> 2026-09-11 更新：局部训练已接通并在执行 3000 步短训；先读 `INFORMATION_BOLTZMANN_LOCAL_3000_BASELINE.md`。新实现为 device_collision.py / sampling.py / window.py / block_graph.py / train.py。42 项 CPU 检查及 GPU 跨块梯度审计通过。旧全局对照在2500步保存后暂停。当前不得并发占用GPU；完整能量账和临界性测量仍是后续任务。下文保留原任务规格，不代表这些执行项仍未开始。

先读 `INFORMATION_BOLTZMANN_LOCAL_OPERATOR_SPEC.md`。只执行被分配的一项；不要重设计模型。遇到OPEN条目提交问题，不自行补规则。现有全局Slice训练不得中断。

## 通用交付格式

每次提交只包含：(1)改动文件；(2)实现的公式/规格节号；(3)测试命令和实际结果；(4)最大数值误差；(5)尚未处理的问题。禁止用“已实现”代替运行验证，禁止用短跑CE作能力结论。CPU测试设torch线程数1，避免争抢现有训练CPU。

所有新增模块在 `scripts/ib_local/`，测试在 `tests/test_ib_local_*.py`。禁止改 `fine_grain/information_boltzmann/` 公共实现、当前BPE/Slice训练文件或运行目录。没有GPU独占窗口时，仅做CPU测试；GPU性能任务记为未测。

## T01 候选表与串行参考（依赖：无）

文件：`__init__.py`, `types.py`, `reference.py`, `tests/test_ib_local_reference.py`。

实现规格§3–4、§10的CandidateTable/FrozenContext/CollisionResult。使用dataclass，形状检查；注入原CollisionKernel参数，不重新随机初始化一个不同核做等价测试。支持`legacy_eps`和`strict_local_v1`两个显式模式，配置序列化包含模式。M=0返回原值及可正确接入loss的零log_prob。

复制原数学函数而不是直接调用包含随机抽样的forward；候选表决定法向和uniform。上下文与工作速度不能alias导致原地改写，也不能detach。采样函数与执行函数分离，禁止候选cap。

验收：反射、局部性、对称性、空表、所有inactive、原legacy函数的rate逐项一致、状态与logprob有限。严格mask分支测试远处context输入的梯度为零；pair本身变化的梯度另算。

## T02 调度与批处理（依赖：T01）

文件：`schedule.py`, `batched.py`, `tests/test_ib_local_schedule.py`。

先CPU构造依赖level；每层batched Attention可用einsum/bmm。不共享粒子的写回scatter索引必须唯一，assert在测试路径启用。保留event_id恢复accepted顺序。只在同层并行；ctx值整个子步冻结。

验收：链、星形、重复pair、互不相交pair、混合inactive；serial与batched用同一个模块权重、同一张表。比较入口x/v、context与每个可学习参数梯度。先以L2型可微终点损失做数值测试，不能称记忆实验。

## T03 精确梯度审计（依赖：T01/T02）

文件：`tests/test_ib_local_score_gradient.py`。

做规格§7/§10的1与2事件分支枚举。每条路径重新计算依赖历史的概率和终点loss；真目标是ΣP(path)L(path)。surrogate的外层路径权重要detach，否则把概率梯度加两次。内部logprob必须可微，CE系数detach。常数baseline改变不得改变枚举期望梯度。

验收：碰撞率参数、输入速度、位置、context参数梯度都close；纯pathwise估计在构造例中应能显示漏项。测试数据只为数值推导，不做toy能力结论。

## T04 GPU执行组织（依赖：T02/T03通过）

文件：`cuda_schedule.py`, `cuda_collision.py`, `tests/test_ib_local_cuda.py`。

先移除逐候选CPU同步并缓存K/V。记录提交数、event count、level数及耗时。无证据不写Triton。若profiling显示DAG构造值得融合，再实现GPU整数调度核。整数调度不是可学习过程；连续几何权重仍保留梯度。

容量bucket必须有overflow路径并可复现；溢出不能丢候选/邻居。稀疏邻域若实际不稀疏，可保留masked dense计算。改变proposal为只抽邻居需要重新推导采样率，本任务禁止。

验收：规格FP32阈值；独占GPU才能计时；不得把CPU版慢基线与不同N/H版本作等价加速比。若某些accept阈值附近因浮点换序发生分支变化，报告数量，不隐去。

## T05 局部BPE窗口与生命周期入口（依赖：T03，生产GPU依赖T04）

文件：`window.py`, `train.py`, `tests/test_ib_local_window.py`。

参考现有BPEWindow的bulk token projection/readout，不继承其Slice碰撞。每token保留碰撞logprob前缀，返回原CE和surrogate两种量。L=128起步，N512,d4,H128,S4；tie embedding显式写config。

必须保存：x/v、真实事件cursor、物理clock、optimizer update数、BOS/previous token、全部RNG与候选算法版本、mask模式、config/源码/数据hash。step与events独立，不用events=step*当前L恢复混合历史。

验收：顺序单事件与窗口loss/梯度一致；全程因果；首次出生梯度；真实OWT短窗口resume一致。验证RNG不影响训练RNG。状态checkpoint在observed update boundary保存。

## T06 能量账与测量接口（依赖：T01/T05）

文件：`diagnostics.py`, `tests/test_ib_local_accounting.py`。

实现§8离散能量恒等式；记录驱动/阻尼/热浴的定义与单位，禁止连续热注入和OU端点残差混用。测量副本不能修改主个体或optimizer。实现§9响应、多ε、分支分歧、方差trace及标准化有效秩；冻结参数与在线学习两类报告分开。

本任务只输出指标，不创建临界controller，不把边界快照计算为高频谱。固定标准化尺度写入结果，不每帧自适应尺度掩盖幅度变化。

## T07 性能验收（依赖：T04–T06）

文件：`benchmark.py`，精简结果放`results/published/`。

执行规格§11。最少L128，完整更新p95<=1.5秒；10预热+100计时，不掺验证，但另外列每250步验证摊销成本与监控成本。GPU不能与当前5000步对照争抢。失败交付profile，不启动新长训练，不擅自降低模型规模或截断梯度。

## T08 长训练与阶段审核（依赖：T07通过）

沿固定OWT真实流执行；预算使用用户授权值，比较按真实tokens对齐。输出best/last/阶段权重及验证曲线。能力与临界结论需要独立审核，编码AI只报告事实。当前Slice的5000授权不自动等于任意数量新实验的授权。

## 可直接发送给便宜模型的任务消息

```text
工作目录 D:/information-boltzmann-v1。
阅读 AGENTS.md、docs/INFORMATION_BOLTZMANN_LOCAL_OPERATOR_SPEC.md 和
docs/INFORMATION_BOLTZMANN_CODING_TASKS.md。
本轮只执行任务 T01。严格遵守数学定义、允许修改的文件与测试要求。
不要修改/停止正在运行的Slice训练，不用GPU，不启动新训练。
只实现候选表、显式模式的串行参考和对应CPU数值测试。
不要实现BGK、全局Slice、controller，也不要优化尚未验收的部分。
结束时列出文件、测试结果、数值误差和未解决问题。
```

后续任务只替换任务编号，并确认依赖已通过；不要把所有任务一次交给执行模型。
