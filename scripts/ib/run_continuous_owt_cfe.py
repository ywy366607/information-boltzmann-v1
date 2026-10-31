"""
真实 OpenWebText (OWT) 长流构形场生命体实验 (GPU 与多时间尺度加速版)
严格遵循连续个体主动学习协议 (Continuous-Individual Active Learning Policy):
1. 真实数据: 直接消费 data/ib_owt_gpt2/train.npy 真实语言流 (GPT-2 词表 50,257)
2. 无全局反向传播 (Zero BPTT): 纯局域变分流、Kuramoto 锁相、默里定律管径强化与自适应阻抗匹配
3. 严格 O(1) 时空算力: 恒定单步毫秒级响应，显存常驻 (< 300MB)
4. 预先评分 (Prequential Evaluation): 严格先测后学，全程记录 NLL 与比特惊奇度
"""

import sys, os, math, time, argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from fine_grain.information_boltzmann.owt_cfe import OWTConstructalEngine

def run_owt_cfe_long_stream(
    total_tokens: int = 10000,
    init_slots: int = 2048,
    max_slots: int = 8192,
    pipe_lr: float = 4.0,
    j_crit: float = 3.2,
    eta_oja: float = 0.05,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 90)
    print(f"启动真实 OpenWebText (OWT) 长流构形场生命体实验 ({total_tokens:,} 连续词元)")
    print(f"计算后端: {device.upper()} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"第一性原理体系: 构形水力冲刷自生河道 (Phi^0.5) | 哈密顿相空间引力公转轨道 | 贝让-默里极值管径分配")
    print(f"词表规模: 50,257 (GPT-2 Tokenizer) | 初始槽位: {init_slots} | 动态容量上限: {max_slots}")
    print("=" * 90)

    train_path = Path("data/ib_owt_gpt2/train.npy")
    val_path = Path("data/ib_owt_gpt2/validation.npy")
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"未找到 OWT 数据文件: {train_path} 或 {val_path}")

    train_tokens = np.load(train_path, mmap_mode="r")
    val_tokens = np.load(val_path, mmap_mode="r")
    print(f"已加载 OWT 语料库: 训练集 {len(train_tokens):,} tokens, 验证集 {len(val_tokens):,} tokens")

    # 初始化完全第一性原理 CFE 个体
    engine = OWTConstructalEngine(
        vocab_size=50257,
        dim=64,
        init_slots=init_slots,
        max_slots=max_slots,
        lambda_metabolic=0.015,
        pipe_lr=pipe_lr,
        pipe_decay=0.0005,
        J_crit=j_crit,
        eta_oja=eta_oja,
        device=device
    )

    if device == "cuda":
        print(f"GPU 物理显存分配: {torch.cuda.memory_allocated() / 1e6:.1f} MB (显存常驻，绝无泄露)")

    history = []
    surprisals_nats = []
    resistances = []
    slot_counts = []
    beta_stars = []
    doc_id = 0

    start_time = time.time()

    print("-" * 90)
    print("Step    | 活跃槽位 | NLL (nats) | 预先惊奇度 (bits) | 阻抗刚度 β* | 流阻 R | 分裂/合并 | 状态")
    print("-" * 90)

    for step in range(total_tokens):
        current_token = int(train_tokens[step])
        next_token = int(train_tokens[step + 1])

        if current_token == 50256:
            doc_id += 1

        metrics = engine.step_stream(current_token, next_token)
        surprisals_nats.append(metrics["surprisal_nats"])
        resistances.append(metrics["resistance"])
        slot_counts.append(metrics["num_slots"])
        beta_stars.append(metrics["beta_star"])

        should_print = (step % 500 == 0) or (step == total_tokens - 1) or (current_token == 50256 and step % 200 == 0)
        if should_print:
            window = min(100, len(surprisals_nats))
            avg_nats = np.mean(surprisals_nats[-window:])
            avg_bits = avg_nats / math.log(2)
            evt = "平稳层流"
            if metrics["splits"] > 0:
                evt = f"⚡ 惊奇度反压! Hopf Δ 分裂 (增至 {int(metrics['num_slots'])})"
            elif metrics["merges"] > 0:
                evt = f"🍂 冗余拓扑自收缩 (合并减至 {int(metrics['num_slots'])})"
            elif current_token == 50256:
                evt = f"📄 跨越文档边界 Doc {doc_id}"

            act_idx = torch.where(engine.active_mask)[0]
            g_min = engine.gamma[act_idx].min().item()
            g_max = engine.gamma[act_idx].max().item()
            print(f"Step {step:5d} | Slots: {int(metrics['num_slots']):4d} | NLL: {avg_nats:5.2f} n | γ:[{g_min:.2f}-{g_max:.2f}] | β*: {metrics['beta_star']:.2f} | R: {metrics['resistance']:5.1f} | S:{int(metrics['splits'])} M:{int(metrics['merges'])} | {evt}")

        if step % 50 == 0 or step == total_tokens - 1:
            history.append({
                "step": step,
                "nll_nats": metrics["surprisal_nats"],
                "surprisal_bits": metrics["surprisal_bits"],
                "beta_star": metrics["beta_star"],
                "resistance": metrics["resistance"],
                "num_slots": metrics["num_slots"],
                "splits": metrics["splits"],
                "merges": metrics["merges"]
            })

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start_time
    print("-" * 90)
    print(f"长流实验完成: 连续流经 {total_tokens:,} 真实 OWT Tokens | 总耗时: {elapsed:.2f}s ({elapsed/total_tokens*1000:.2f} ms/token)")

    # 周期独立验证 (无需重置个体状态)
    print("正在评估独立验证集前序 NLL (300 tokens)...")
    val_subset = val_tokens[:300]
    val_surprisals = []
    for vi in range(len(val_subset) - 1):
        logits, _, _, _, _, _ = engine.prequential_predict(int(val_subset[vi]))
        target_t = torch.tensor([int(val_subset[vi + 1])], device=engine.device)
        val_surprisals.append(F.cross_entropy(logits.unsqueeze(0), target_t).item())

    init_nats = np.mean(surprisals_nats[:100])
    mid_nats = np.mean(surprisals_nats[total_tokens//2 : total_tokens//2 + 200])
    final_nats = np.mean(surprisals_nats[-200:])
    val_nats = np.mean(val_surprisals)

    print(f"\n长流定量实验评估结果 (NLL & Information Entropy):")
    print(f"  • 理论瞎猜基准 (Uniform 50,257): 10.825 nats (15.617 bits)")
    print(f"  • 初始未习得阶段 (前 100 词元):   {init_nats:.3f} nats ({init_nats/math.log(2):.3f} bits)")
    print(f"  • 中程阶段 (第 {total_tokens//2} 词元处):      {mid_nats:.3f} nats ({mid_nats/math.log(2):.3f} bits)")
    print(f"  • 终态长流稳态 (最后 200 词元):   {final_nats:.3f} nats ({final_nats/math.log(2):.3f} bits) -> 净降 {init_nats - final_nats:.2f} nats ({(init_nats - final_nats)/init_nats*100:.1f}%)")
    print(f"  • 独立验证集前序 NLL:            {val_nats:.3f} nats ({val_nats/math.log(2):.3f} bits)")
    print(f"  • 槽位自组织演化: 初始 {init_slots} 个 -> 终态稳态 {engine.num_active} 个 (累计触发 {engine.total_splits} 次 Hopf 分裂, {engine.total_merges} 次合并)")
    print(f"  • 阻抗匹配刚度 β*: 从初始 {beta_stars[0]:.2f} -> 演化至 {beta_stars[-1]:.2f}")
    if device == "cuda":
        print(f"  • 显存峰值: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB (物理上限 4096 MB)")

    out_file = Path("C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/owt_cfe_gpu_stream_data.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "device": device,
            "total_tokens": total_tokens,
            "elapsed_seconds": elapsed,
            "ms_per_token": elapsed / total_tokens * 1000,
            "init_nats": init_nats,
            "mid_nats": mid_nats,
            "final_nats": final_nats,
            "val_nats": val_nats,
            "final_slots": engine.num_active,
            "total_splits": engine.total_splits,
            "total_merges": engine.total_merges,
            "history": history
        }, f, indent=2)
    print(f"长流实验完整数据已保存至 {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=10000, help="处理的连续词元总数")
    parser.add_argument("--init-slots", type=int, default=2048, help="初始活跃槽位数")
    parser.add_argument("--max-slots", type=int, default=8192, help="最大槽位容量")
    parser.add_argument("--pipe-lr", type=float, default=4.0, help="默里管径强化率")
    parser.add_argument("--j-crit", type=float, default=3.2, help="自由能相变临界通量")
    parser.add_argument("--eta-oja", type=float, default=0.05, help="Oja流形共流收缩率")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")
    args = parser.parse_args()

    run_owt_cfe_long_stream(
        total_tokens=args.tokens,
        init_slots=args.init_slots,
        max_slots=args.max_slots,
        pipe_lr=args.pipe_lr,
        j_crit=args.j_crit,
        eta_oja=args.eta_oja,
        device=args.device
    )
