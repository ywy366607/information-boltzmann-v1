# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库概览与核心架构

本仓库探索**内容自适应切片池化（Content-adaptive slice pooling，基于 Transolver++ / Native MoT）**与**固定 patch 网格（ViT / NaViT）**在细粒度视觉结构与多模态架构中的对比。实验覆盖从合成微基准测试（Synthetic micro-benchmarks）到小型 VLM/OCR 架构（如 Gemma-3-270M、Pythia-160M、TinyLM）。

### 核心架构与演进路线

1. **Transolver3 / Native MoT（原生多模态产品路线）**
   - **全分辨率可写点场（Point Field）**：不将视觉坍缩为单向传给 LLM 的静态 tokens，而是全程保留全分辨率特征场 $X \in \mathbb{R}^{B \times N \times d_x}$（$N = H \cdot W$）作为工作记忆。
   - **瞬态切片（Ephemeral Slices）**：切片 $S \in \mathbb{R}^{B \times M \times d}$ 是层内局部的临时交互视角，通过质量归一化软读取（`SliceRead`）生成。
   - **MoT 共享注意力空间**：视觉切片 $S$ 与语言状态 $H \in \mathbb{R}^{B \times T \times d}$ 进入共享注意力空间交互，配有模态专有投影与 FFN 专家（`fine_grain/native_mot.py`、`fine_grain/mot_coevolve.py`）。
   - **Deslice 写回与局部混合**：切片更新后通过 Deslice 写回点场（$X' = \text{LocalMix}(X + \text{Deslice}(S'))$），下一层再从更新后的点场提取特征，实现记忆跨层回流（`fine_grain/cross_modal_slice_loop.py`）。
   - **Slice–NaViT 双流 MoT**：在点场上结合 NaViT 式变长 patch packing（$P$）与 Slice（$S$），与文本（$H$）共同在 MoT 空间交互，写回后闭合回流（`docs/slice_navit_dual_stream_mot.md`）。

2. **传统前端 Baseline（消融/历史对照，非产品主路线）**
   - `PatchFrontend`（A 路）：固定 patch 网格 $\to$ 投影层 $\to$ LLM。
   - `SliceFrontend`（B 路）：静态切片 tokens $\to$ LLM（丢弃了 live deslice 点场，已废弃）。
   - `HybridFrontend`（C 路）：Patch 与 Slice 拼接。

3. **关键数学与架构不变量**
   - **质量归一化（Mass Normalization，`norm="mass"`）**：除以分配质量总和（$\text{tok} = \sum w x / \sum w$），使 1px 细结构的激活幅度与目标面积解耦（尺寸不变性机制）。
   - **稀疏 Deslice 写（Sparse Deslice Write，`sparse_deslice_weights` / `deslice_topk`）**：读路径保持软分配以传导梯度；写路径使用 top-$k$ 或阈值截断，彻底消除 Transolver 软 MoE 散布带来的背景能量泄漏（主噪源修复）。
   - **Newton–Schulz Stiefel 正交化（`newton_schulz` / `stiefel_ns`）**：切片方向在质量归一化后采用 Muon 五阶极分解系数（$A=3.4445, B=-4.7750, C=2.0315$）进行 Stiefel 投影，作为核心抗坍塌（anti-collapse）机制。
   - **$G \le C$ 秩约束**：切片数量 $G$ 不得大于通道维度 $C = \text{heads} \times \text{dim\_head}$，否则在数学上必然发生秩强制坍塌。默认设置：$G=32, C=64$。
   - **Qwen 式 SDPA 后门控（`qwen_sdpa_gate`，arXiv:2505.06708）**：门控置于 SDPA 之后（$\sigma(W X) \odot \text{AttnOut}$），作用于残差流特征，而非 QK softmax 或任务输出头。

## 环境与缓存配置

为了避免模型下载占满系统盘 `C:`，所有 ModelScope、HuggingFace、Transformers 与 Torch 缓存必须路由到 `D:` 盘。

```bash
# Windows Bash:
export ML_CACHE_ROOT="D:\ml_cache"

# Windows Cmd:
set ML_CACHE_ROOT=D:\ml_cache
```

### 安装环境

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install transformers modelscope tokenizers peft
pip install -e .
```

## 常用命令

### 运行测试

```bash
# 运行完整测试套件（80 个单元测试）
pytest

# 安静模式运行
pytest -q

# 运行指定测试文件
pytest tests/test_native_mot.py
pytest tests/test_cross_modal_slice_loop.py
pytest tests/test_deslice_scatter_and_gate.py

# 运行单个测试用例
pytest tests/test_native_mot.py -k "test_native_mot_block_forward"

# 独立脚本运行测试
python tests/test_deslice_scatter_and_gate.py
```

### 基准测试与训练脚本

```bash
# 合成任务分类评估（needle / glyph / lines / connect / kinks / angles）
python scripts/train_benchmark.py --task needle --arms patch4,slice --steps 500 --seeds 3
python scripts/train_benchmark.py --task glyph --arms patch4,slice_loc_nogumbel --steps 1500

# CUDA 吞吐量微基准测试
python scripts/train_benchmark.py --bench --device cuda

# 密集 1px 折线掩码重建（Line Reconstruction）
python scripts/line_recon.py --arms slice_loc_nogumbel,patch16 --res 64 --steps 600 --rgb

# VLM 前端 A/B/C 扫描（接冻结小语言模型）
python scripts/train_vlm_frontends.py --mode sweep --prefer local --res 32 --T_list 64 32 16 --steps 60 --amp

# Native MoT / SliceMoT-Mini 机制门控与消融实验
python scripts/run_slicemot_mini_gate.py --arm base
python scripts/run_slicemot_mini_gate.py --arm topk2
python scripts/run_slicemot_mini_gate.py --arm dual_ps

# Transolver3 多模态 OCR 协同演进训练
python scripts/train_transolver3_ocr.py --steps 200 --amp

# 合成数据集生成
python scripts/build_kinks_dataset.py --out data/kinks256 --n_train 6000 --n_val 600
python scripts/build_ocr_dataset.py --out data/ocr1px --n_train 5000 --n_val 500
```

## 代码库结构索引

- `fine_grain/models.py`：核心编码器 `AdaTempSlice`、`SliceNet`、`PatchNet`、`Block`、`AttnPool`、`ARMS` 注册表、`newton_schulz`、`sparse_deslice_weights`。
- `fine_grain/native_mot.py`：原生 MoT 规范实现（`NativeMoT`、`SliceRead`、`Deslice`、`PointPatchEmbed`、`LocalVisual`）。
- `fine_grain/cross_modal_slice_loop.py`：多模态场循环演进（$X_{t+1}, H_{t+1}$）与临时工作区交互（`CrossModalSliceFrontend`）。
- `fine_grain/mot_coevolve.py`：切片与文本序列联合自注意力机制。
- `fine_grain/transolver3_vlm.py`：Transolver3 原生 VLM OCR 桥接器（支持语言模型冻结或联合训练）。
- `fine_grain/frontends.py`：视觉前端抽象（`PatchFrontend`、`SliceFrontend`、`HybridFrontend`）。
- `fine_grain/llm_backend.py`：ModelScope 与本地 TinyLM 加载器，强制缓存至 `D:\ml_cache`。
- `fine_grain/lora_llm.py`：因果语言模型的 PEFT LoRA 适配。
- `fine_grain/mm_projector.py`：多模态投影层（Linear / MLP / GeGLU）。
- `fine_grain/tasks.py`：合成数据生成器（`make_needle`、`make_glyph`、`make_lines`、`make_connect`、`make_kinks`、`make_angles`）。
- `fine_grain/ocr_1px.py`：1px Bresenham 笔画合成数字 OCR 生成器。
- `fine_grain/vlm_data.py` & `fine_grain/hf_caption_data.py`：VLM 批数据构造与 HuggingFace 数据加载。
- `fine_grain/train_utils.py`：优化器构造与切片坍塌探针（`collapse_stats`、`pr_obj`）。
- `results/published/`：已发布的基准测试 JSON 快照、结论对比表与架构规范文档。
- `docs/`：Slice–NaViT 双流 MoT 与 VLM 前端的详细设计文档。
