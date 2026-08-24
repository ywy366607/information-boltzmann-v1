# Slice–NaViT 双流 MoT：接口与设计草图

> **定位：可选辅助观察流，不是产品主状态。** 架构不变量与生成契约以 [`NORTH_STAR.md`](NORTH_STAR.md) 为准；若 patch 路径绕过 Slice/Deslice 写回，则不属于规范主图。

Status: **partially implemented** in `fine_grain/native_mot.py`  
  - `dual_patch=True`: `PointPatchEmbed(X) → P`, `SliceRead → S`, MoT concat `[P;S]` with text,  
    `Deslice(S')` + optional `PointUnpatch(P')` on the **same** point field X.  
  - Mini arm: `python scripts/run_slicemot_mini_gate.py --arm dual_ps`  
Related: `docs/native_slice_mot_vlm_experiment.md`, `fine_grain/native_mot.py`  
Date: 2026-08

## 0. 一句话

用 **NaViT 式变长 patch pack** 保住局部性与任意分辨率序列接口，用 **全分辨率点场 + Slice 写回** 保住细结构可编辑记忆；二者与语言一起进入 **MoT 共享注意力空间**，且 **下一层 pack 必须吃写回后的点场特征**，从而闭合「Slice 强调细节 → patch 能看见」。

本文件只定接口与不变量，不改正在跑的 SliceMoT-Mini 烟测 / 四旋钮消融。

---

## 1. 动机与对照

### 1.1 现有两条路各自缺什么

| 路线 | 强项 | 弱项 |
|------|------|------|
| 纯 NaViT / MoonViT → projector → LLM | 原生 res、pack 高效、预训练生态（SigLIP 等） | 单向编码；无全分辨率可写状态；细结构依赖高分 patch 堆算力 |
| 纯点场 Slice–MoT（当前 `NativeMoT`） | 逐点状态、Deslice 可写回、M 固定与 N 解耦 | 缺显式 patch 局部归纳；固定 res 协议；大 N 的 assignment 需扫描优化 |
| **双流（本文）** | patch 局部 + 点场细写 + 跨层回流 | 实现与算力更复杂；需 cap \(P\) 与稳定写回 |

### 1.2 外部参考（设计归属，非代码依赖）

**NaViT（Patch n’ Pack）** — Dehghani et al., arXiv:2307.06304

- 不强制固定方图 resize；按原生分辨率 / 长宽比切 patch。
- **Sequence packing**：不同大小图像的 patch 拼成定长训练序列，attention mask 禁止跨图。
- **分解式 2D 位置编码**（factorized y/x PE），避免绑死单一 \(H\times W\) 位置表。
- 推理可调分辨率做 cost–quality 权衡。

**MoonViT / MoonViT-3D（Kimi-VL / Kimi K2.5）**

- 原生分辨率视觉塔；**显式采用 NaViT packing**。
- 常从 SigLIP-SO-400M 类权重续训；经 MLP projector 接 LLM。
- MoonViT-3D：多帧时空 pack + patch 级时间压缩，图视频权共享。
- 与本文差异：MoonViT 是 **encoder→LLM 单向**；本文在塔内保留 **可写点场 + 跨层回流**。

---

## 2. 状态与符号

每层 \(\ell = 0,\ldots,L-1\)：

| 符号 | 形状 | 含义 |
|------|------|------|
| \(X_\ell\) | \([B, N, d_x]\)，\(N=H\cdot W\) | 全分辨率视觉点场（主记忆） |
| \(S_\ell\) | \([B, M, d]\) | 瞬态 Slice（**层私有参数**；M 固定） |
| \(P_\ell\) | \([B, P, d]\) | NaViT 式 patch tokens（**P 随分辨率变**） |
| \(H_\ell\) | \([B, T, d]\) | 语言状态 |
| \(w_\ell\) | \([B, N, M]\) | Slice 分配（读软、写可稀疏） |

约束（注册默认，可改 experiment id）：

- \(d_x = d\)（点场与 MoT 宽对齐；与 mini 用户锁定一致）。
- SliceRead / Deslice / MoT **按层独立**，禁止跨层权重 tying（除非新实验）。
- 坐标一律 **归一化** \((y,x)\in[-1,1]^2\)（与现有 `coords(R)` 一致）。

---

## 3. 单层数据流（核心契约）

```text
X_ℓ  (point field)
 │
 ├─► SliceRead(X_ℓ) ──────────────► S_ℓ , w_ℓ
 │
 └─► PatchPack(X_ℓ) ──────────────► P_ℓ     # 吃特征场，不是冻结 RGB
           │
           ▼
    MoT_ℓ( S_ℓ , P_ℓ , H_ℓ )
           │
           ├─► S'_ℓ , P'_ℓ , H_{ℓ+1}
           │
           ▼
    X_{ℓ+1} = LocalMix( X_ℓ + Deslice(S'_ℓ, w_ℓ) )
           │
           ▼
    下一层 PatchPack(X_{ℓ+1})  ……  回流闭合
```

### 3.1 不变量（MUST）

1. **回流闭合**：\(\ell+1\) 的 `PatchPack` 输入是 \(X_{\ell+1}\)（或由其线性投影的特征图），**禁止**全程只 pack 原始 RGB 而忽略写回。  
2. **写回主通道是 Slice**：默认 \(\Delta X\) 仅来自 `Deslice(S')`；\(P'\) 不直接 scatter 到每个像素（避免第二套软 MoE 泄漏）。若需 patch→点，须另开实验 id。  
3. **M 固定、P 可变**：分辨率变化改变 \(N\) 与 \(P\)，不改变 slice 槽位数 \(M\)。  
4. **因果 / prompt**：视觉 query 不得读答案 span（与现有 `prompt_mask` 一致）；patch 与 slice 同属视觉侧。  
5. **层参数独立**：`SliceRead_ℓ`、`PatchPack_ℓ`（若含可训 patch embed）、`MoT_ℓ`、`Deslice_ℓ`、`LocalMix_ℓ` 默认不共享。

### 3.2 可选（MAY）

- `LocalMix`：DWConv 3×3 / ConvNeXt 式大核 block / 无 local（消融）。  
- 仅最后 \(K\) 层启用写回；前几层 read-only Slice 以稳训。  
- 推理时动态降 \(P\)（更大 patch 或 token merge）做 test-time 算力滑动。

---

## 4. 模块接口

### 4.1 `PatchPack`（NaViT 风格，点场版）

**输入**：\(X\in\mathbb{R}^{B\times N\times d_x}\)，元数据 `H, W`（或 `res`），可选 `patch_size=p`。

**步骤**：

1. Reshape → \([B, d_x, H, W]\)（或 channels-last 等价）。  
2. 非重叠（或 stride=\(p\)） unfold → 每块 \([p,p,d_x]\)。  
3. `Linear(p·p·d_x → d)` 或 DW+ PW 得到 patch token。  
4. 加 **factorized 2D PE**：`pe = pe_y[iy] + pe_x[ix]`，`iy,ix` 为 patch 网格索引；训练多种 \((H,W)\) 时 PE 用插值或相对坐标。  
5. 输出 \(P\in\mathbb{R}^{B\times P\times d}\)，\(P=\lceil H/p\rceil\cdot\lceil W/p\rceil\)。

**Packing（训练，对齐 NaViT）**：

- 同 batch 不同 \((H_i,W_i)\) 时，将各图 \(P_i\) 拼成一条序列或 pad 到 `max_P`，`attn_mask` 禁跨样本。  
- 4GB mini 可先 **单图变长、不做 multi-image pack**。

**禁止**：

- 把 `PatchPack` 做成会 **stride 下采样丢点场** 的完整 ConvNeXt/ViT stage 栈，却仍声称「全分辨率可写」——下采样后 Deslice 的 \(N\) 对不齐。

### 4.2 `SliceRead` / `Deslice`

与 `fine_grain/native_mot.py` 同语义：

- 读：mass-norm soft pool → \(S\)。  
- 写：默认 soft scatter；消融 `deslice_topk`。  
- 旋钮归属不变：Ada-Temp / Gumbel ← Transolver++；topk / Stiefel ← 本仓库。

双流下：assignment 仍在 **点** 上算，不在 patch 上算（细结构地址空间保持 \(N\)）。

### 4.3 `MoT` 块（三路专家，共享 K/V 空间）

相对当前两路（vision slice / text），扩展为可选三路：

| 模态 expert | QKV / FFN | 序列角色 |
|-------------|-----------|----------|
| Slice expert | 私有 | \(S\)，长度 \(M\) |
| Patch expert | 私有 | \(P\)，长度 \(P\) |
| Text expert | 私有 | \(H\)，长度 \(T\) |

**共享**：仅拼接后的注意力空间  
\(K=[K_S; K_P; K_H],\quad V=[V_S; V_P; V_H]\)。

**掩码**：

- Slice / Patch query：可读全部 \(S,P\) + **prompt 文本**；不可读 target 文本。  
- Text query：可读 \(S,P\) + 因果文本。  
- 可选：限制 Patch 只对局部窗口的其他 patch 可见（Swin 式）——默认 **全局 patch**，窗口为后续优化。

**实现分期**：

- v1：把 \(S\) 与 \(P\) 在视觉侧 concat 成一条视觉序列，共用现有 vision expert（改动小，语义弱）。  
- v2：真正三路私有 expert（推荐产品形态）。

### 4.4 `LocalMix`

点场局部 refine，**不下采样**。候选：

- `dw3`：现状  
- `convnext_block`：大核 DW + LN + 1×1（可选后续）  
- `none`：消融

### 4.5 读出到 LLM

与现有一致可选：

- 最终 \(S_L\) 经 projector → 视觉 prefix tokens；或  
- \(P_L\) 经 projector（更像 MoonViT）；或  
- 二者 concat（需 cap 长度）。

机制实验优先：**同一读出协议下** 比「有无写回 / 有无双流」。

---

## 5. 「强调细节 → patch 注意到」的机制命题

**命题.** 若任务需要突出区域 \(R\subset\Omega\)，Slice 路径在 MoT 中写入与 \(R\) 相关的信息到 \(S'\)，Deslice 使 \(X'|_R\) 相对背景可分；则下一层 `PatchPack(X')` 得到的 patch tokens 在 \(R\) 上的表征分布应显著偏离 `PatchPack(X)`，且下游 NLL / 注意力质量改善。

**必要结构条件**：回流闭合（§3.1.1）。否则命题在计算图上为假。

**推荐探针（实现后）**：

1. **人工写回**：在 ROI 上加可控 \(\Delta X\)，测 patch 范数 / 注意力质量是否集中到 ROI。  
2. **matched vs shuffle**：双流+写回 vs 双流无写回 vs 仅 NaViT pack。  
3. **细结构子集**：1px / 小字；写回开关的 Δ。  
4. **跨分辨率**：同一权重，`res∈{32,48,64}` 扫 TF / ΔNLL（坐标已归一化）。

---

## 6. 与当前代码的映射

| 现有 | 双流增量 |
|------|----------|
| `NativeMoTLayer`: Read→MoT(S,H)→Deslice→Local | + `PatchPack`；MoT 吃 `(S,P,H)` |
| `NativeMoTStack.encode_X` stem | 保留；\(X_0\) 仍 RGB+xy→\(d_x\) |
| `prompt_mask` | 延伸到 P 侧视觉 mask |
| mini `res=32` 固定 | 双流阶段解除 assert；动态 `H,W` |
| 四旋钮消融 | 仍只作用在 Slice 分配/写回；与 pack 正交 |

**非目标（本设计明确不做）**：

- 用完整预训练 MoonViT **替换** 点场并丢弃 Slice。  
- 在未完成 SliceMoT mini 1×/消融前并进主表。  
- 全因子与 Agent Swarm 等 K2.5 系统项。

---

## 7. 复杂度与工程

设 \(N=HW\)，\(P=N/p^2\)，\(M\) 固定，文本长 \(T\)。

| 部分 | 粗算 |
|------|------|
| Slice assignment | \(O(N M)\) 每层；大图需 **tiled scan**（预注册 doc 已要求） |
| Patch self-attn in MoT | \(O((M+P+T)^2 d)\)；需 **cap P** 或窗口 |
| Deslice | \(O(N M)\) 或 topk 后更稀 |
| Pack | \(O(N d)\) 级 unfold+linear |

**4GB 建议上限（草案）**：`res≤64`，`p=8` 或 `16`，`M≤32`，`L≤2`，batch=1；先不做 multi-image pack。

**算子优化方向**（需要时）：

- 融合 soft pool：`w,X → S` 的 matmul/bmm 代替 Python einsum 路径。  
- topk deslice 用 gather/scatter。  
- 可选 `torch.compile` 包住 `PatchPack`+`SliceRead`。

---

## 8. 分阶段落地（建议）

| 阶段 | 内容 | 完成标准 |
|------|------|----------|
| **D0** | 本文接口冻结；与 mini 解耦 | 文档入仓 |
| **D1** | `PatchPack` 单元测试：变 `H,W` shape；PE 插值 | pytest 绿 |
| **D2** | `NativeMoTLayer` 双流 v1（视觉 concat S\|P） | 与单流 smoke 可比 |
| **D3** | 强制 `PatchPack(X_{ℓ+1})`；写回开关消融 | 命题探针 1–2 有表 |
| **D4** | 真·三路 expert；可选 multi-res 训练 | 跨 res 曲线 |
| **D5** | 预训练 patch stem 初始化（可选 SigLIP 子集） | 仅当 D3 有信号 |

当前仓库进度：SliceMoT-Mini **clean 烟测 / 计划中的四旋钮** 属于 **D0 之前的机制基线**，不被 D1+ 抢跑。

---

## 9. 张量契约小结（实现检查单）

```text
assert X.shape == (B, H*W, d_x) and d_x == d
assert S.shape == (B, M, d)
assert P.shape == (B, (H//p)*(W//p), d)   # 或 ceil 规则写死
assert H_txt.shape == (B, T, d)

# 回流
X_next = layer(X, ...)
P_next = patch_pack(X_next)   # MUST use X_next, not rgb0

# 掩码
visual_keys_see_text_targets == False
```

---

## 10. 决策记录

| 问题 | 决定 |
|------|------|
| 写回是否来自 patch？ | 默认否；仅 Slice deslice |
| pack 吃 RGB 还是 X？ | **吃 X（特征）** 以闭合回流 |
| 是否整模 MoonViT？ | 否；只借 pack + 2D PE + 变长序列思想 |
| 与 ConvNeXt | 可选仅替换 `LocalMix`，与双流正交 |
| 与当前 git 主实验 | 本文设计稿；实现另开 PR/提交 |

---

## 11. 参考文献

1. Dehghani et al., *Patch n’ Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution*, arXiv:2307.06304.  
2. Kimi Team, *Kimi K2.5: Visual Agentic Intelligence*, arXiv:2602.02276（MoonViT-3D + NaViT packing）.  
3. 本仓库 `docs/native_slice_mot_vlm_experiment.md`（点场 + 瞬态 Slice + MoT 预注册）.  
4. Transolver / Transolver++（Physics-Attention；Ada-Temp / Gumbel 归属 ++）.
