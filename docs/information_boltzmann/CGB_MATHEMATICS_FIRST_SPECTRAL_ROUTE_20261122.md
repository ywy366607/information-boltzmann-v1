# 从数学结构向身体生长：谱、轨迹与临界边界

2026-11-22，按用户本轮修订确立研究起点。本文负责本轮数学路线与新增工作判定。

## 执行顺序

数学问题与原始文献 → 已知定理及适用条件 → 能区分解释的新推论 → 项目结果去重 → 身体映射 → 所需绑定的Lean验证 → 实例化与测量。

数学部分先使用自身的对象、符号和问题。项目材料在去重和建立对应时进入；已有接口用于承接结果。新推导若要求改变表示或增加记录，先交代数学必要性及获得的信息。

证明分工：文献定理列来源及前提；仓库已有证明列引用；新证明限于尚缺的推论或身体绑定。数值工作对应已经明确的新数学量及其判别用途。

## 数学问题

核心问题：一个持续受驱动、具有内部耦合与适应过程的系统，如何形成长寿命而可切换的动力学组织？怎样从整段轨迹的数学结构识别这种组织、其临界边界及其生成机制？

先分别研究下列性质，再推导它们之间的联系：

| 数学起点 | 对象及性质 | 本轮要取得的东西 |
|---|---|---|
| 自伴谱、酉群与收缩半群 | 正定内积、生成元、谱与不变量 | 旋转、耗散、保持各自的条件；可作为其他结构的精确组成部分 |
| Ruelle/转移算子 | 轨迹权重、多步传播、共振、闭合轨道 | 集体持存与切换的算子表述，以及所需函数空间/表示条件 |
| 轨迹Lee–Yang理论 | 累积活动的生成函数、复零点、谱分支竞争 | 有限时间零点、累积量和时间/容量标度之间的关系 |
| de Bruijn–Newman热流 | 一族整函数的实零点保持阈值 | 学习怎样为“临界”指定明确的形变族与被保持性质 |
| 生物临界性研究 | 灵敏度、动态范围、适应及稳健性 | 按功能选择需要解释的性质，为数学量的实际用途建立后续问题 |

主要工作路线优先采用转移算子与轨迹生成函数。Newman热流提供明确阈值问题的范例；其形变参数与生命体的物理时间分别定义。

## 第一项具体数学对象

从路径空间开始。设系统状态为x，原始一步演化核为P_t(x,dy)，一步可加观测为g_t(x,y)，定义

\[
A_N=\sum_{t=0}^{N-1}g_t(X_t,X_{t+1}),\qquad
Z_N(s)=\mathbb E[\exp(-sA_N)].
\]

g在数学阶段是声明了取值范围及单位的观测函数。s为统计计数场。两者的具体内容在身体映射阶段确定。

相应加权转移算子为

\[
(K_{s,t}f)(x)=\int P_t(x,dy)e^{-s g_t(x,y)}f(y).
\]

采用行分布约定，标准Feynman–Kac表示为

\[
Z_N(s)=\mu_0 K_{s,0}K_{s,1}\cdots K_{s,N-1}\mathbf1.
\]

这是要引用和应用的标准表示。时变核以有序乘积承接适应和外部变化；若将生成下一步所需的变量纳入状态，也可研究扩展状态上的固定演化核。

在有限维、固定算子的情形，可同时引用

\[
\det(I-zK_s)^{-1}
=\exp\left(\sum_{n\ge1}\frac{z^n}{n}\operatorname{tr}(K_s^n)\right),
\]

等式先在收敛域内使用。Z_N统计指定起点分布和终点求和下的路径权重；trace统计加权闭合路径。它们共享算子，同时保留各自的端点条件。一般动力学zeta与传递算子的关系需引用相应trace-class或正则化定理。

由此形成一个可推进的数学问题：哪些系统具有相互竞争的长寿命轨迹组织，它们怎样表现为加权传播的谱分支、轨迹生成函数的零点与活动涨落？

## 第一阶段需要区分的数学情形

下列标准情形用作推论的判别条件，引用基本概率恒等式即可。

1. 一个确定路径值A_N=a给出Z_N(s)=exp(-sa)，该函数零点集合为空。研究轨迹零点首先需要定义路径概率对象及其估计方式。
2. 独立同分布增量给出Z_N(s)=z_1(s)^N。单步分布已经可以产生复零点；随N增加，零点的重数增加。该结构提供单步分布贡献的精确参照。
3. 时变但独立的增量给出Z_N(s)=prod_t z_t(s)。这个对象承接各时刻边缘分布的变化，用来区分驱动/适应引起的漂移与时间关联。
4. 有关联的路径以多时刻联合累积量刻画。多变量log生成函数的混合导数给出连接累积量，提供时间和通道组织的信息。
5. 有限系统交付复零点、分支竞争及有限时间/容量标度；相应极限中的非解析性定义动力学相变。

每种情形对应一个可区分的结构，不把复零点的存在单独指定为临界判据。后续选择统计估计方法时，先推导样本需求、估计稳定性和非平稳情况下保留的解释。

## 向有限表示生长时的数学要求

设q把完整状态映射到一个候选表示。需要检查的对象是

\[
\int_{q^{-1}(C)}P_t(x,dy)e^{-s g_t(x,y)}.
\]

一个标准的充分闭合条件是：对同一q纤维中的x与x'，上式在所有目标集合C及所需s上相同。这样可定义表示空间上的加权核，并保留相应的多步生成函数。

这里先寻找并引用加权lumpability/Markov-additive过程的已有结果，再检查候选表示满足哪些前提。近似表示则需要目标生成函数或零点稳定性的误差控制。最终身体绑定验证的是这些目标是否由实际记录及表示确定。

## 已有工作去重

检索范围：CGB_PROVEN_INDEX.md、docs/information_boltzmann、lean/CGBAxioms/CGBAxioms、lean/CGBAxioms/reviews、scripts/ib_local。检索词包含Lee–Yang、Ruelle、de Bruijn/Newman、Feynman–Kac、Koopman、lumpability、轨迹配分、谱隙、伪谱、亚稳与倾斜算子。该目录检索用于已有结果去重。

| 已有对象 | 处理 |
|---|---|
| 真实依赖复形、边界、Hodge结构与5082维Dirac及D²核对 | 继承原结果和证书 |
| 实际关系响应的Hodge活动分解、Dirac读出 | 继承原数据、映射和数值 |
| 原生S/P相位、X⁻、有向环路Xo | 继承原结果与相位绑定 |
| 连接变化、R(t)、最大关系群与整圈选择性干预 | 继承原作用链和对照 |
| 木—火—金选择/资格/后到支付与最终旧事实用途账 | 继承原实测与因果区分 |
| 局部feedback latch临界减速及保持/纠错定理 | 直接引用GenerationHoldCorrect，保留其作用域 |
| Koopman/受控算子 | CGB_CAUSAL_LEARNING_UNIFICATION_20261110.md第197、393行已讨论过候选参数化，归入已有思想 |
| 亚稳结构与断驱动持存 | 物理核心及旧驱动维护文档已有讨论，作为已有机制认识 |

旧测量总入口：CGB_NATIVE_BODY_LEAN_MEASUREMENT_20261117.md §9.5—§9.12；整合见CGB_NATIVE_TOPOLOGY_FINAL_ACCEPTANCE_20261117.md。

本轮新增工作对象明确为：实际轨迹概率对象及加权传播、零点与时间关联的可区分预测、适应过程中的有序传播，以及保留这些目标的表示条件。是否新增Lean声明和采样任务，由这些对象的剩余数学缺口决定。

## 后续交付顺序

2026-11-22全系统展开见`CGB_MATHEMATICAL_FIVE_PHASE_MAP_20261122.md`：数学对象向木火土金水和内外交接分别建立对应，新增带时间标记的轨迹核、主动交话首达核及目标闭合检查；对应源码/AST/SHA256在`cgb_mathematical_five_phase_map_20261122.json`。既有因果资格压缩等证明继续复用。

1. 数学来源表：定理、前提、结论、适用情形，标明哪些属于引用。
2. 数学判别表：上述情形各自预言什么轨迹/谱/零点行为，所需概率对象与规模如何。
3. 项目差额表：逐项列已有材料如何承接，新增信息具体是什么。
4. 身体映射与所需验证：保留学习、环境反馈和原状态身份，选取满足数学要求的表示。
5. 先复用完整记录计算新量；确需新采样时，由已推导的信息需求给出最小采样方案。

## 原始来源

- Rodgers–Tao, The de Bruijn–Newman constant is non-negative: https://arxiv.org/abs/1801.05914
- Berry–Keating, The Riemann zeros and eigenvalue asymptotics: https://doi.org/10.1137/S0036144598347497
- Ruelle, Dynamical zeta functions for maps of the interval: https://arxiv.org/abs/math/9404235
- Hilgert–Rilke, Meromorphic continuation of dynamical zeta functions via transfer operators: https://arxiv.org/abs/math/0701097
- Flindt–Garrahan, Trajectory phase transitions, Lee–Yang zeros, and high-order cumulants in full counting statistics: https://arxiv.org/abs/1209.2524 （主方程、生成函数及谱分支见原文式1—7）
- Yoshida–Takahashi, Dynamical Lee–Yang zeros for continuous-time and discrete-time stochastic processes: https://arxiv.org/abs/2201.00946 （含周期时变转移）
- Chetrite–Touchette, Nonequilibrium Markov processes conditioned on large deviations: https://arxiv.org/abs/1405.5157
- Kinouchi–Copelli, Optimal dynamical range of excitable networks at criticality: https://arxiv.org/abs/q-bio/0601037
- Khajehabdollahi et al., When to be critical?: https://arxiv.org/abs/2303.16195
