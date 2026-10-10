# 可塑 3D 持存介质：最小方程、实现与学习接口

日期：2026-10-03。状态：独立数值候选，尚无语料训练或吞吐成绩。

目标是让经验塑造计算介质的局部性质，而不只优化均匀介质里的活动。
果蝇提供局部异质性、持续突触响应、延迟和调控的机制实例；这里不复制
感觉/运动脑区、神经元类别或固定生物连接组。现有语言与果蝇训练保持独立。

介质实现：`information_boltzmann/core/plastic_medium.py`。
完整端口图：`information_boltzmann/core/plastic_ports.py`，复用 W4 和可学习探针。
验证：原介质/端口 17 项、可行性 6 项、空间谱 2 项与既有时间回归 7 项通过，共 32 项。
运行计时：`scripts/ib/benchmark_plastic_medium.py`。没有 toy 能力实验。

## 1. 两种时间尺度与三个状态对象

单位周期域 \(\Omega=\mathbb T^3\)，每个位置有 D 个带符号内容分量。

- 快状态 \(f_i\in\mathbb R^D\)：当前局部内容。
- 快状态 \(j_{i,a}\in\mathbb R^D\)：正向轴 a 的边上储存的传递响应，a=x,y,z。
- 慢结构 \(a_\theta(x)\)：连续周期介质场，决定传导速度、散射偏好和释放速率。

这里的 j 是波介质的通量/共轭响应，不是突触电流的逐项生物仿真。
局部 f 受到更新时，已经在传播的 j 仍然保留。无输入期间也可持续推进。

介质用有限 Fourier 基表示：

\[
a_\theta(x)=A_0+\sum_{k\in\mathcal K}
[A_k\cos(2\pi k\cdot x)+B_k\sin(2\pi k\cdot x)].
\]

基的带宽是结构表达容量；网格只是求值分辨率，参数形状不依赖网格。
当前默认使用 8×8×4 参考格上的完整实 Fourier 基，256 列、秩 256。
Nyquist 自共轭模只保留 cosine，避免采样格上恒零的 sine 参数。
原先 13 列低频基只够表达平滑大尺度偏好，已从默认核心配置移除；显式指定
低频 modes 仍可建立受限带宽对照。改变求值网格时保持参考带宽，参数可严格加载。
系数从零开始，形成均匀介质；非均匀经验可以提供非均匀的结构梯度。
这允许形成结构分工，但不保证一定形成神经元形态或最优拓扑。

## 2. 局部传导：线性、互易、带传递状态

对每条有向表示的相邻边 i→l，h 为轴向网格间距：

\[
c_{il,a}=c_{ref}\exp\{\ell_a(a_i)\}>0,
\]
\[
\dot f_i=-c_{il,a}j_{il}/h,\quad
\dot f_l=+c_{il,a}j_{il}/h,\quad
\dot j_{il}=c_{il,a}(f_i-f_l)/h.
\]

每条正向边只存一份系数，两端以相反符号共用；i 是边属性的索引，反向传播
不会重新计算另一个系数。去掉端点平均，是因为偶数周期格上的平均算子
会精确抹去沿该轴的棋盘格 log-conductance 模。
给定介质，传播对 f,j 严格线性，不从当前语义临时生成
另一套输运网络。原有内容可以在边状态中继续存在并在后续时刻影响节点。

离散能量（V 是单元体积）：

\[
E=\frac V2\sum_i[\|f_i\|^2+\sum_a\|j_{i,a}\|^2].
\]

每条边的能量导数为零；每个内容分量的全空间 f 积分也守恒。
令 d=(f_i-f_l)/sqrt(2)，则 (d,j) 是角速度 sqrt(2)c/h 的二维旋转，
两节点的和保持不变。代码对这个三对象耦合做解析旋转。

每个轴分偶/奇边两色，共六个静态层；同层边互不共享节点，可批量计算。
实现使用切片、roll 和逐元素运算，无全局 FFT、全连接矩阵或线性方程求解。
六层乘积是局部算子分裂，不等于任意长时间的精确全场解。
颜色层是无写冲突的计算调度，不是六次认知决策。

第一版支持三轴均为偶数的周期网格。严格保能量不意味着大时间步仍然准确，
精度由固定物理时长下的积分细化检查。它也不保证整个外部事件内只有一跳传播。

## 3. 内容与空间共同条件化的非线性散射

每个位置组合 g=(f,j_x,j_y,j_z)，共享同一介质 a_i：

\[
\omega_{i}=\mathrm{MLP}_{C}(\mathrm{RMSNorm}(g_i),a_i),
\quad \theta_i=\Delta\tau\omega_i.
\]

对四个 D 维组分别用 Householder 变换，把各组内容均值映射到第一个坐标。
保留这四个坐标，对其余 4(D−1) 坐标做状态/介质条件化 Givens 旋转，再变回。
各轴的边响应参与同一局部自由空间，因此散射可以改变传播方向的分配。

解析不变量：局部四个内容组的线性和、局部总二次能量。
这些是当前波状态的明确线性约束，不冒称 D3Q8 的物理质量和三维动量。
这个候选是 kinetic-inspired 波介质，不是非负 f(x,v) 的完整 Boltzmann 碰撞律。
状态相关旋转虽保范数，其 Jacobian 仍可能放大扰动；临界性尚未证明。

## 4. 选择性二次浴：空间、内容共同决定释放

\[
\rho_i=\|g_i\|^2/(4D),\qquad
r_{i,d}=\mathrm{softplus}(\mathrm{MLP}_{B}(\mathrm{RMSNorm}(g_i),a_i)_d+b),
\]
\[
\dot g_{i,d}|_B=-\rho_i r_{i,d}g_{i,d}.
\]

同一个介质同时参与传导、碰撞和浴；浴并不按空间频率预判哪个模式有用。
语义价值由真实观测学习目标训练，不由正率本身证明。

第一版冻结一次更新中的 rho,r，采用：

\[
g^+_{i,d}=g_{i,d}/\sqrt{1+2\Delta\tau\rho_i r_{i,d}}.
\]

这在各方向速率相等时是径向二次耗散 ODE 的精确解；一般状态相关方向速率
需要数值细化。所有分量绝对值都不增，释放能量记录为 E_before−E_after。

固定有限参数下，RMSNorm 输入有界，连续介质在紧域上有界，所以实数算术中
r 有正下界 m。连续方程满足 dE/dtau ≤ −m E²/D（无边界输入）。
若外界净注入功率受控为 P，则 E 的上界由 dE/dtau ≤ P−m E²/D 推导。
部署中的 m 可以极小；这既不保证理想记忆寿命，也不保证训练参数演化时
存在同一个统一下界。浮点溢出/下溢与条件扰动增长需要额外验证。

## 5. 外界事件与随时读出

端口沿用正交交换原则：

\[
f^+=\cos\alpha\,f^-+\sin\alpha\,p_{in},
\quad p_{out}=-\sin\alpha\,f^-+\cos\alpha\,p_{in}.
\]

f 与出射包的能量闭合；j 和物理时间不被端口事件清空。
写入 agent 负责生成入射包和接纳角。模块也提供 `with_field` 以供既有 W4
创新写入替换 f，并保留全部 j。第一版没有更改现有 W4 端口账本。

```python
medium = PlasticMedium3D(shape=(8, 8, 4), channels=128)
state = medium.initial_state()
# 外部事件：已有 write agent 接收真实观测，只替换 field。
state = state.with_field(posterior_field)
# 无需新输入，也无需等状态收敛。
state, diagnostics = medium.advance(state, elapsed_physical_time, substeps=resolution)
# 在实际输出请求时读取快状态；生成输出之后继续保留 state。
features = read_agent(state.field)  # 此处为接口示意
```

实际端口接口已实现：

```python
from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D

model = PlasticMediumPorts3D(shape=(8, 8, 4), channels=128)
belief = model.initial_belief()
belief, write_info = model.assimilate(belief, observed_token_ids)
belief, evolution_info = model.advance(belief, elapsed_physical_time, substeps=resolution)
next_token_logits, read_info = model.read(belief, diagnostics=True)
# 下一事件继续使用 belief；detach 只切断计算图，保留 field、flux、时间与 precision。
belief = belief.detach()
```

`forward(input_ids, targets, belief, event_duration=..., substeps=...)` 用下一词
CE 与既有 W4 端口目标联合训练。目标标签只参与损失，W4 先验在观测到达前计算。
这是已接通的真实语言似然接口；现有端口目标不额外宣称完整潜变量 ELBO。
precision 当前随 W4 事件更新，推进期间保持原值；后续可增加动力学不确定性传播。
W4 的原有精度度量账本与本介质的欧氏能量账本分别记录，尚未建立统一度量闭合。

`elapsed_physical_time` 是实际请求的模型时间，`substeps` 是积分分辨率。
参考速度与浴时间是明确的模型单位/参数，不是临界常数。
第一版每个子步顺序为传播→碰撞→浴；这是低阶分裂实现，不主张机制在真实时间
中依次等待，也不保证随时插入事件就自动降低墙钟延迟。
长事件可以拆成短片段交错读/写；之后根据误差与运行负载选择求解方法。

## 6. 慢介质怎样学习

第一阶段：与写入、状态算子、读出共同优化真实观测的先验预测似然。
一个训练片段内固定参数，只在完成该片段的反向传播后更新；保持快状态值。
局部参数学习示意：theta ← theta − eta ∂L_observed/∂theta。
当前数值模块提供完整梯度，不自动执行优化器，不把梯度下降称为已实现的
生物可塑性或完整主动推理。

第二阶段学习接口：由新证据检索相关旧事件，重新编码实际观测，计算解释责任
后验，以当前短计算图更新介质和生成模型。事件记忆、责任后验与在线变分 EM
尚未实现。这是长期信用分配的独立工作，不靠把本模块时间推进改名来获得。

快状态与慢结构是功能与更新时间的区别，不人为划定存储区、思考区或脑区。
不同位置只有学习后才可能形成不同偏好。

## 7. 集成与性能验收次序

1. 验证本模块的能量、线性、局部散射约束、方向重组、时间、梯度与续接。
2. 已连接既有 W4 写入与可学习读出，验证了因果性、全部算子梯度和流式分段续接。
   新检查点必须额外保存三个边状态和 elapsed。
   不能静默将已有 f-only 检查点称为完整状态续训。
3. 独占 GPU 窗口时测完整 forward/backward/optimizer，与当前同数据预算基线比。
   六色边旋转和局部反应是融合核候选，CPU fullgraph 检查不等于 CUDA Graph 验收。
4. GPU/状态预算通过后，再真实 OWT 至少 3000 次联合更新及独立暖启动验证。
   之后判断 NLL、长程利用、空间分工和吞吐；现在无正负能力结论。

基线：均匀介质与可学习介质共享状态、端口与数据预算，按结构能力而非更多
自由计算公平比较。网格 strict-load 仅证明参数接口兼容；跨网格轨迹还需精度收敛。

计时入口包括 forward、backward、AdamW 与持存状态续接，不以单核计时代替整步：

```powershell
# 空闲 GPU 窗口执行；时间单位为本候选的模型单位，需另做积分细化校准。
python scripts/ib/benchmark_plastic_medium.py --scope core --device cuda --event-duration 0.01 --tokens 128
python scripts/ib/benchmark_plastic_medium.py --scope ports --device cuda --event-duration 0.01 --tokens 128 --compile
```

完整空间基的默认介质核 135,060 个参数。B=1、8×8×4、D=128、FP32 下四组快状态加时间
占 524,296 字节（时间使用 FP64）；这仅是状态存储量，不包含端口、优化器和反向激活。
当前 GPU 有独立果蝇训练，未运行这两个 GPU 命令，也未取得新候选的速度成绩。

## 8. 自发节律与内容相关锁相：允许形成，保持机制闭合

既有边响应是共轭动力学自由度。固定介质时，传导写成
`d(f,j)/dt = L(a)(f,j)`，其中 L 在上述能量度量下反自伴；非零振荡模态
的本征值为 ±iω。ω 由域尺度、介质传导与耦合结构共同决定，代码无需
再加入独立的时间正弦驱动或预设振荡频率。

非线性碰撞依赖当前内容与位置介质，因而不同输入能产生不同的瞬时耦合响应。
慢介质 a 当前由联合优化更新；一次推进内固定。当前实现尚未维护独立的
在线快介质调制状态。这样已允许内容依赖的功能网络，同时保留给定介质下
线性传导与非线性碰撞的职责分离。

相锁的目标是：某些区域的相位差趋向稳定，受到扰动后能够恢复；各区域
不必同相，也不必全场共享一个节律。相位差恢复要求相对相位方向具有吸引性。
仅保范数的传导没有这种收缩保证；非线性散射、正率选择性释放与持续输入
共同构成产生这种吸引性的候选机制，尚未证明或测得。

只有在实际形成稳定振荡轨道、并满足弱耦合等条件后，才可以从原系统做相位
约化，得到 `dφ_i/dt = ω_i + Σ_j H_ij(φ_j−φ_i)`；H 由动力学的相位响应决定。
Kuramoto 正弦耦合是其中一种近似，当前代码不把它另加为一个强制模块。

正率二次浴使无输入能量单调下降，因此当前被动介质无法在完全断能时维持
非零耗能极限环；可形成衰减振荡。无限输入流提供持续供能，支持研究受驱动
节律、选择性锁相与分工。这不需要把正能量反馈偷偷塞进保守碰撞。
训练前的构造性锁相证明、可达条件与尚未闭合的 W4 接口，见
`PLASTIC_MEDIUM_FEASIBILITY.md`。它取代“只能等训练看是否锁相”的验收方式。

机制观察应从现有状态提取区域振荡相位，比较相位差轨迹、相位锁定值
`PLV_ij = |mean_t exp(i(φ_i−φ_j))|` 与时滞；对照共同输入与传播耦合的作用。
频谱峰集中只是起点，不能单独区分共同驱动、共振与吸引性的相锁。
介质时间使用模型单位；转换为 Hz 需要额外物理时间标定。积分细化应保持
本征节律与相位关系稳定，避免把数值更新频率当成物理节律。
# Runtime conduction upgrade

See [ADAPTIVE_CONDUCTION.md](ADAPTIVE_CONDUCTION.md) for the persistent local
conduction state, bounded autonomous structural law, extended phase-locking
certificate and full W4/read integration. The adaptive port candidate uses
438 additional parameters and 3 KiB FP32 state at8x8x4/D128. Static core
certificates remain reproducible through `adaptive_conduction=False`.

See [CONTINUOUS_RUNTIME.md](CONTINUOUS_RUNTIME.md) for independent observation,
evolution and readout clocks, timestamped BPTT, complete continuation state and
the matched execution benchmark. Idle intervals evolve without external tokens.

The opt-in `bath_type="conductance"` candidate upgrades local response with
persistent receptor kinetics, learnable C/L/g/R and reversal-source work/heat
accounting. See [CONDUCTANCE_RESPONSE.md](CONDUCTANCE_RESPONSE.md) for equation
provenance, explicit approximations and its separate feasibility certificate.
