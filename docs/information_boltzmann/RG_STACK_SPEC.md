# RG 堆叠语言模型实现规格（RGStackLM）

> 目标读者：负责实现的编码 AI。本规格自包含：不依赖对话历史，所有引用的类与文件都在本仓库中真实存在。
> 一句话：在已验证的 W4 端口语言线上，叠加一个**粗粒化层**（实空间重整化 RG），使细层的空间结构有了"去处"，解决已确诊的"场塌缩到 100% DC"问题。

---

## 0. 背景与动机（为什么做）

1. 现有主力线：`CBIMTorus3D`（W4 创新写入 + belief 读出 + K=16 微步/dt=4 + quadratic 浴，shape 8×8×4，d=128），在 OpenWebText 上以 IB-warm-local-language-v1 协议注册训练，best NLL 7.4203。
2. 已确诊的缺陷（见 research_tree.json 节点 `IB_Q8_SPATIAL_COLLAPSE_20260930`）：训练后的场 100.00% 谱能量在 DC 模（空间常数），256 个站点向量完全相同——场退化为一个复制的 128 维循环态，输运/空间物理未被行使。原因已解析：统一浴的谱黏性在每个事件内把非 DC 模杀到 10⁻⁷；且粗尺度信息没有自己的动力学居所。
3. 本方案（用户提出的重整化思路）：叠加第二层环面动力学——细层（层 1）的 8×8×4 场经**粗粒化提升算子**驱动一个 4×4×2 的粗层（层 2），粗层有自己的输运/碰撞/浴，**承担长程慢变量**；细层保留空间细节。两层各自持续演化、永不重置。
4. 与 UNet 的区别（不要照抄 UNet）：没有"输入图像"概念——这是无限 token 流驱动的**耦合动力系统**；层间耦合是**每事件持续进行的 RG 提升**（积分掉细自由度 → 重整化耦合流入粗层），不是一次性池化；粗层是长期记忆本体；各层是不同时间尺度的不同有效理论（细=快模态，粗=慢模态）。

**v1 范围限定**（必须遵守）：只做**上行提升**（细→粗）与双层读出；**不做**粗→细的下行条件化（留作 v2 注册）。原因：保持因果归因干净——v1 回答"给模型一个粗粒度积分器是否有益"。

---

## 1. 新增文件

### 1.1 `information_boltzmann/core/rg_stack.py`（新建）

三个类 + 一个组合模块。全部双精度兼容（`.double()` 可用）、无数据依赖的控制流（CUDA Graph 可捕获）。

#### `LiftOperator(nn.Module)`
细层场 → 粗层驱动的重整化提升。
- `__init__(self, d: int, block: int = 2, hidden: int = 128)`：
  - `self.block = block`
  - `self.project = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, d))`
  - 末层权重 `nn.init.normal_(std=1e-4)`、bias 零——**初始化时提升≈块平均的小修正，不注入噪声**。
- `forward(self, fine_field: torch.Tensor) -> torch.Tensor`：
  - 输入 `[B, 8, 8, 4, d]`；按 `block=2` 分成 2×2×2 块（8 站点/块），对块内站点取均值 → `[B, 4, 4, 2, d]`；
  - `out = block_mean + self.project(block_mean)`（残差形式）；
  - 返回 `[B, 4, 4, 2, d]`。
  - 块均值实现建议：`fine.view(B, 4, 2, 4, 2, 2, d).mean(dim=(2, 4, 5))`（轴顺序自验，写确定性测试锁定）。
- 注意：`LiftOperator` **不得**接触 token 嵌入——粗层只观察细层状态（单写入路径纪律的粗层版）。

#### `CoarseTorusDynamics(nn.Module)`
粗层自身的动力学（复用现有算子类，直接实例化，不要改它们）。
- `__init__(self, d: int = 128, shape: tuple = (4, 4, 2), velocities: int = 8,
           collision_layers: int = 2, micro_steps: int = 8)`：
  - `self.transport = VelocityCayleyTransport3D(shape, velocities, 16)`
  - `self.collision = LocalInvariantCollision3D(shape, velocities, 16, layers=collision_layers)`
  - `self.bath = QuadraticTorusBath(shape, d, position_conditioned=False)`
  - `self.shape, self.d, self.micro_steps = shape, d, micro_steps`
- `forward(self, coarse_field: torch.Tensor) -> tuple[torch.Tensor, dict]`：
  - 输入/输出 `[B, 4, 4, 2, d]`；
  - 执行 `micro_steps` 次循环：`transport.apply_multiplier(field, mult)` → `collision(field, 1.0)` → `bath(field, 1.0)`；
  - `mult` 每次调用计算（粗层小，不值得提升优化）；`dt=1.0`；
  - 返回 `(coarse_field_next, diagnostics)`，diagnostics 至少含 `coarse_energy`（`0.5*field.square().sum(-1).mean()`，detach）。

#### `RGStackLM(nn.Module)`
组合模块：持有细层 `CBIMTorus3D`、粗层动力学、提升算子、读出合并头。
- `__init__(self, fine_kwargs: dict, coarse_shape=(4,4,2), coarse_steps=8, lift_hidden=128)`：
  - `self.fine = CBIMTorus3D(**fine_kwargs)` —— fine_kwargs 即当前主力配置：
    `dict(shape=(8,8,4), velocities=8, content_dim=16, collision_layers=2,
    relative_address=True, readout_type="belief_agent", write_type="w4_predictive_agent",
    micro_steps=16, dissipation_type="quadratic", tau_0=4.0)`
    **浴必须是 `quadratic`（结构保持），禁止 unified（已证明的空间歼灭器）。**
  - `self.lift = LiftOperator(d=self.fine.d)`
  - `self.coarse = CoarseTorusDynamics(d=self.fine.d, shape=coarse_shape, micro_steps=coarse_steps)`
  - `self.injection_gain = nn.Parameter(torch.zeros(()))`——粗层注入增益，**初始化为 0**（粗层从纯观察者开始，NLL 有收益才学会注入；防初始扰动）。
  - `self.read_merge = nn.Sequential(nn.Linear(2 * self.fine.d, self.fine.d), nn.SiLU(),
                                    nn.Linear(self.fine.d, self.fine.d))`
    （合并细读出特征与粗池化特征；末层小初始化，保证初始行为接近原单尺度线。）
  - 解码器沿用 `self.fine.decoder`（其权重与 `self.fine.source.embedding` 绑定，**不得破坏绑定**）。
- `initial_states(self, batch_size, device) -> tuple[KineticBeliefState, torch.Tensor]`：
  - 返回 `(self.fine.initial_belief(batch_size, device=device, warm_start=False), zeros([B,4,4,2,d]))`。
- `forward_chunk(self, input_ids: torch.Tensor, targets: torch.Tensor,
                belief: KineticBeliefState, coarse_field: torch.Tensor) ->
                tuple[torch.Tensor, torch.Tensor, KineticBeliefState, torch.Tensor, dict]`：
  逐 token 循环（镜像 `CBIMTorus3D.forward_belief` 的结构，见 torus3d.py:1715）：
  1. `logits, belief, diagnostics = self.fine.belief_step(belief, input_ids[:, i], include_private=True, micro_steps=self.fine.micro_steps)`
  2. `likelihood = F.cross_entropy(logits, targets[:, i])`，累加 `total_likelihood`
  3. `total_free_energy += diagnostics.pop("_write_free_energy")`
  4. **提升**（每事件一次，在细层演化之后）：`drive = self.lift(belief.field)`；
     `coarse_field = coarse_field + torch.tanh(self.injection_gain) * drive`
     （tanh 有界注入，防粗层能量爆炸）
  5. `coarse_field, c_diag = self.coarse(coarse_field)`
  6. 循环结束后：`fine_feat = self.fine.readout(belief.field, belief.precision, return_diag=False)`
     （注意：此处读出作用在**终态** belief 上，与现有 forward_belief 的逐步读出不同——这是有意为之：读出读事件后的双层状态。
     若实现时发现逐 token 读出更贴合原训练行为，可改为每 token 读出并把合并特征直接进该 token 的解码——二选一，写明所选即可。）
  7. `coarse_feat = coarse_field.mean(dim=(1,2,3))`；`feature = self.read_merge(cat(fine_feat, coarse_feat))`；
     `logits_final = self.fine.decoder(feature)`
  8. **损失**：`loss = total_likelihood + port_free_energy_weight * total_free_energy`（与 forward_belief 相同，权重 1.0），
     加上 `final_logits` 对最后一个 target 的交叉熵（`F.cross_entropy(logits_final, targets[:, -1])`，权重 1.0，记入诊断 `final_token_nll`）。
  9. 返回 `(loss, final_logits, belief, coarse_field, diagnostics)`。
- `diagnostics` 必须包含：`fine_dc_share`、`coarse_dc_share`、`injection_gain_abs`、`coarse_energy`、
  以及细层 diagnostics 里所有 trainer 已记录的键（token_nll、write_free_energy 等）。DC 份额函数：

```python
def dc_share(x: torch.Tensor) -> torch.Tensor:
    spec = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
    mag2 = spec.abs().square()
    return mag2[:, 0, 0, 0].sum() / mag2.sum()
```

### 1.2 `scripts/ib/train_rg_stack.py`（新建）

以 `scripts/ib/train_q8_port_agents.py` 为模板复制修改（保持其全部纪律）：
- 用 `RGStackLM` 替代裸 `CBIMTorus3D`；`TruncatedBeliefGraphTrainer` 改造为携带**三重持久状态**：
  `field`、`precision`、`coarse_field`——三者在每次图重放后 `copy_(next.detach())`，**永不重置**；
  CUDA 图捕获的 `backward_chunk` 覆盖 `forward_chunk` 全部（细层+提升+粗层+合并读出）。
- 检查点：在现有键之外保存 `"coarse_state"`；resume 恢复它，且 resume 一致性键表加入
  `("coarse_shape", "coarse_steps")`。缺 `coarse_state` 的旧检查点报错拒绝。
- 训练循环、验证协议（IB-warm-local-language-v1、四站点 8192/12288/16384/20480、warm 256、score 128）、
  每 500 步验证、BBest/last 保存、metrics.jsonl 行格式——全部照搬原训练器。
- 验证行额外键：`fine_dc_share`、`coarse_dc_share`（**每次验证必须记录**——这是本实验的第一判据）。
- 注册配置写入 `configs/information_boltzmann/q8_rg_stack_k16.json`（结构照抄
  `q8_predictive_ports_k16_unified_bath.json`，注明 RG 堆叠、粗层形状与浴类型）。

### 1.3 `tests/test_rg_stack.py`（新建，确定性测试，禁止收敛性主张）

1. `test_lift_block_mean_shapes_and_residual_init`：LiftOperator 输出形状 `[1,4,4,2,128]`；
   末层权重 std 极小 ⇒ 输出 ≈ 块均值（atol 1e-6 级）。
2. `test_coarse_dynamics_norm_and_shapes`：CoarseTorusDynamics 前后形状一致、有限、能量有限。
3. `test_forward_chunk_gradients_reach_all_levels`：`forward_chunk` 后 `loss.backward()`，
   断言 lift.project、coarse.transport/collision/bath、fine.write_agent、fine.readout、
   read_merge 的参数梯度存在且有限。
4. `test_states_persist_and_checkpoint_roundtrip`：两次 forward_chunk 后粗层状态非零且被携带；
   保存/加载含 `coarse_state` 的检查点后状态逐位一致。
5. `test_dc_share_helper`：构造已知场（常数场 dc_share=1；单站点脉冲场 dc_share≈1/256）验证函数。
6. `test_injection_gain_starts_at_zero`：初始时粗层注入为零 ⇒ 一次 forward_chunk 前后
   coarse_field 只因粗层动力学变化、不含 drive 项。

---

## 2. 训练启动命令（实现者验收用）

```
D:\conda_envs\vox\python.exe scripts\ib\train_rg_stack.py ^
  --output results\q8_rg_stack_k16_3000 --steps 3000 --tokens 128 ^
  --chunk-tokens 32 --micro-steps 16 --tau-0 4.0 ^
  --coarse-shape 4 4 2 --coarse-steps 8 --validate-every 500
```

- 工作目录必须是仓库根；python 必须是 `D:\conda_envs\vox\python.exe`（PATH 上的 `python` 是 Windows Store 别名）。
- 训练前跑 `pytest tests/test_rg_stack.py -q` 必须全绿。

---

## 3. 验收判据（可证伪，写入 research_tree 注册节点）

1. **存活判据（主）**：训练全程各验证点 `fine_dc_share < 0.95`（对照：单尺度线为 1.0000）。
   细层空间结构在 RG 堆叠下存活 ⇒ "塌缩是单尺度架构的属性"成立。
2. **不退步判据**：四站点 best NLL ≤ 7.4203（单尺度 best）。
3. **粗层职能判据（后续）**：评估时将 coarse_state 置零，长程记分窗口的 NLL 恶化应大于短程。
4. 若判据 1 失败（细层仍塌缩）：塌缩是 NLL 压力的内在属性，与浴无关——结论升级，写入 tree，
   后续转向写入特异性/读取稀缺性修复。

---

## 4. 实现者必须知道的坑（全部来自本仓库已发生的真实事故）

1. **禁止 unified 浴**：其谱黏性每事件把非 DC 模压到 10⁻⁷（解析已闭合，见
   `IB_Q8_SPATIAL_COLLAPSE_20260930`）。全栈（细层+粗层）都用 quadratic。
2. **W4 单写入路径纪律**：粗层与提升算子不得接触 token 嵌入；token 进细层的唯一通道是创新端口。
   粗层观察的是细层**状态**，不是 token。
3. **CUDA Graph**：所有形状静态；三重状态（field/precision/coarse_field）在重放间复制；
   禁止数据依赖分支；tanh 增益是有界标量乘子，图安全。
4. **解码器权重绑定**：`fine.decoder.weight` 与 `fine.source.embedding.weight` 是同一个张量
   （torus3d.py 构造函数里绑定），不要用 `read_merge` 替代解码器，也不要复制权重。
5. **resume 防呆**：`--resume` 必须校验 `coarse_shape`、`coarse_steps`、`K`、`tau_0`、
   `dissipation_type`（quadratic）——历史上发生过 tau_0 不匹配的静默错误。
6. **python 解释器**：仓库的 git 钩子与脚本都要求真实解释器；提交时
   `PATH="/d/conda_envs/vox:$PATH" git commit ...`。
7. **不要相信 NLL 单指标**：每次验证必须同时记录 `fine_dc_share`——本次会话的最大教训是
   NLL 正常而场的空间自由度已经死了。
8. **research_tree.json 注册**：新节点 id ≤32 字符；status 只能取
   active/confirmed/noisy/partially_reusable/pruned/superseded；confidence 单字母。
   `confirmed` 需要预注册预测块与独立评审，首次注册用 `active`。

---

## 5. 参考文件（实现者必读）

- `information_boltzmann/core/torus3d.py`：`CBIMTorus3D`（step/belief_step/forward_belief）、
  `VelocityCayleyTransport3D`、`LocalInvariantCollision3D`、`QuadraticTorusBath`、
  `KineticBeliefState`、`PredictiveImpedanceWriteAgent`（注意 `_packet_chart`）。
- `scripts/ib/train_q8_port_agents.py`：`TruncatedBeliefGraphTrainer`（三重状态的改造母体）、
  验证协议、metrics 行格式。
- `research_tree.json` 节点：`IB_Q8_SPATIAL_COLLAPSE_20260930`（塌缩诊断全文）、
  `IB_Q8_MAIN_ARCH_20260930`（主力线与已排除项）。
- `docs/information_boltzmann/Q8_PORT_AGENCY_SPEC.md`：端口律规格（事件范数、单写入路径）。
