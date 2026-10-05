# 以太空间统一介质场与机械辐射规范 (USEF-v1)

状态：**以太沙盒与具身动力学唯一来源规范**。  
核心原则：**严禁为具体行为编写私有音效或触觉补丁；所有感官、声音、触觉与运动全部由连续介质第一性原理自发涌现。**

---

## 1. 物理宇宙度规 (Cosmic Metric)

1. **统一欧拉空间 $\Omega \subset \mathbb{R}^3$**：
   - 全空间具有唯一连续度规，时间以连续物理微步 $dt$ 推进；
   - 介质参数：标准空气密度 $\rho_0 = 1.225 \text{ kg/m}^3$，声速 $c_s = 343 \text{ m/s}$，空间衰减与阻尼 $\gamma$；
   - 恒定重力场：$\vec{g} = [0, 0, -9.81] \text{ m/s}^2$。

2. **实体的不可分割性 (Co-located Entity Invariance)**：
   - 任何实体（智能体躯干、四肢、地面、障碍物、人）在空间中占用确定体积 $V_k$，边界为 $\partial \Omega_k(t)$；
   - 任何物理事件发生在确定空间坐标 $\vec{x}_0$，绝不存在“视觉在一个坐标、声音在另一个坐标”的 API 割裂。

---

## 2. 机械波第一性原理 (Unified Mechanical Wave Emission)

真实物理世界中不存在孤立的“声音”与“触觉”，两者皆为**机械应力张量 $\boldsymbol{\sigma}$ 与边界运动在固相与流相介质中的传导**。

根据气动声学与连续介质声学的 **Ffowcs Williams-Hawkings (FW-H) 方程**：

\[
\left(\frac{1}{c_s^2} \frac{\partial^2}{\partial t^2} - \nabla^2\right) p(\vec{x}, t) = 
\frac{\partial}{\partial t} \Big[ \rho_0 v_n \delta(f) \Big] - \nabla \cdot \Big[ \vec{F}_{\text{surf}} \delta(f) \Big]
\]

### 2.1 表面接触偶极子辐射 (Dipole Stress Radiation)
全空间中任意两物体发生接触碰撞时，接触面产生的接触力 $\vec{F}_{\text{contact}}(t)$ 产生双向机械波：
1. **固相传导（触觉与骨骼冲击）**：
   \[
   \vec{T}_{\text{tactile}}(\vec{x}) = \boldsymbol{\sigma}_{\text{solid}}(\vec{x}) \cdot \vec{n}
   \]
2. **流相辐射（空间声波）**：
   力的时间变化率 $\dot{\vec{F}}_{\text{contact}} = \frac{d\vec{F}}{dt}$ 向空气中辐射偶极压力波：
   \[
   p(\vec{x}, t) = \frac{1}{4\pi r c_s} \left( \dot{\vec{F}}_{\text{contact}}\left(t - \frac{r}{c_s}\right) \cdot \frac{\vec{x} - \vec{x}_0}{r} \right)
   \]

> **涌现推论**：
> - **轻踩 vs 重踏**：同一套公式。踩踏越猛烈，$\dot{\vec{F}}$ 越大，脚底触觉冲击越强，辐射声波越响亮；
> - **摔倒撞击**：躯干跌落撞地，碰撞点产生剧烈应力激增，触觉承受撞击，同时自发辐射出闷响，位置天然锁定在碰撞点；
> - **刮擦与摩擦**：切向摩擦力的高频微震动自发辐射刮擦声。

### 2.2 表面法向加速度单极子辐射 (Monopole Flux Radiation)
物体表面法向加速度 $a_n = \frac{\partial v_n}{\partial t}$ 排开流体体积，辐射单极子波：
\[
p(\vec{x}, t) = \frac{\rho_0}{4\pi r} a_n\left(t - \frac{r}{c_s}\right)
\]
- 肢体高速挥动产生空气流体切变声；
- 发声器官（发音膜）的周期速度振荡自发辐射语音。

---

## 3. 智能体物理边界积分 (Agent Boundary Observations)

智能体不再从环境读取人工拼凑的字典，感官是其几何外壳 $\partial \Omega_{\text{agent}}$ 上的物理量积分：

1. **听觉（双耳天线）**：
   左耳天线 $\vec{x}_L(t)$ 与右耳天线 $\vec{x}_R(t)$ 处采集空间以太场声压的连续标量叠加：
   \[
   s_L(t) = p(\vec{x}_L(t), t), \quad s_R(t) = p(\vec{x}_R(t), t)
   \]
   天线物理间距天然产生到达时间差（ITD）与声级差（ILD），自发形成声源空间定位。

2. **触觉（脚底与肢体表面）**：
   各肢体接触面直接读取法向支持力与切向摩擦力：
   \[
   \vec{F}_{\text{tactile}} = \sum_{k \in \text{contacts}} \vec{F}_k
   \]

3. **本体感觉（Proprioception / Vestibular）**：
   - 前庭感觉：躯干重力倾角 $\theta_{\text{pitch}}, \theta_{\text{roll}}$ 与角速度 $\vec{\omega}$；
   - 关节肌肉感受：各关节当前角度 $\vec{q}$、角速度 $\dot{\vec{q}}$、以及抗重力所需的电机力矩 $\vec{\tau}$。

---

## 4. 自主运动与第一步探索 (Locomotion & Exploration)

1. **中枢模式发生器 (CPG) 与 Kuramoto 相位共鸣**：
   - 运动节律由内部动力场（如三维环面极限环）的非线性振荡驱动：
     \[
     \dot{\phi} = \omega_0 + K_{\text{tactile}} \cdot (\vec{F}_{\text{foot}, L} - \vec{F}_{\text{foot}, R})
     \]
   - 触地反作用力通过相位反馈实现自适应相位重置（Phase Resetting），使步频与地面动态耦合。

2. **被动动态倒立摆与主动自救**：
   - 躯干受自重重力向前倾斜；
   - 前倾使变分自由能上升，促使前摆腿落地支撑；
   - 触地瞬间应力激发机械波，脚底获得触觉确认，空间辐射出一步清脆的脚步声，被自身天线实时监听。
