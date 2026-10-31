"""
构形场引擎交互式推断工具 (Interactive Constructal Inference)
用户可以直接输入任意 A 与 B，由 300 轮演化成熟的构形神经网络实时给出计算结果。
"""

import sys, os, math
import json
import argparse
import numpy as np

DATA_FILE = 'C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/extended_source_sink_evolution_data.json'

def load_mature_engine_weights():
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    # 取最后一轮成熟快照
    snap = data['snapshots'][-1]
    w_a = {}
    w_b = {}
    w_out = {}
    for t in snap['tubes']:
        if t['source'] == 0:
            w_a[t['target']] = t['weight']
        elif t['source'] == 1:
            w_b[t['target']] = t['weight']
        elif t['target'] == 2:
            w_out[t['source']] = t['weight']

    internal_ids = [n['id'] for n in snap['nodes'] if not n['is_source'] and not n['is_sink']]
    return internal_ids, w_a, w_b, w_out, snap['epoch']

def infer(a: float, b: float, internal_ids, w_a, w_b, w_out):
    u_list = []
    v_list = []
    out = 0.0
    for k in internal_ids:
        wa = w_a.get(k, 0.0)
        wb = w_b.get(k, 0.0)
        u = wa * a + wb * b
        v = 0.25 * (u ** 2)
        wo = w_out.get(k, 0.0)
        out += v * wo
        u_list.append(u)
        v_list.append(v)

    target = a * b
    error = abs(target - out)
    rel_acc = max(0.0, 1.0 - error / max(0.01, abs(target))) * 100.0
    return out, target, error, rel_acc

def main():
    parser = argparse.ArgumentParser(description="构形场引擎交互式乘法推断")
    parser.add_argument("--a", type=float, default=None, help="输入变量 A")
    parser.add_argument("--b", type=float, default=None, help="输入变量 B")
    args = parser.parse_args()

    internal_ids, w_a, w_b, w_out, epoch = load_mature_engine_weights()

    print("=" * 65)
    print(f"★ 构形场引擎 (CFE) 实时交互推断台 (已加载 Epoch {epoch} 成熟脑模型)")
    print(f"★ 模型架构: 2 源端输入 (A, B) -> {len(internal_ids)} 极化中继板层 -> 1 任务汇端出口")
    print("=" * 65)

    if args.a is not None and args.b is not None:
        out, target, error, rel_acc = infer(args.a, args.b, internal_ids, w_a, w_b, w_out)
        print(f"输入: A = {args.a:.4f}, B = {args.b:.4f}")
        print(f"  -> 目标真值 (A × B):    {target:.4f}")
        print(f"  -> 构形网络推断输出:   {out:.4f}")
        print(f"  -> 绝对误差:           {error:.4f}")
        print(f"  -> 物理相对精度:       {rel_acc:.2f}%\n")
        return

    print("进入实时交互控制台 (输入 'q' 退出):")
    while True:
        try:
            line_a = input("\n请输入 A (例如 1.8): ").strip()
            if line_a.lower() in ('q', 'exit', 'quit'):
                break
            line_b = input("请输入 B (例如 2.2): ").strip()
            if line_b.lower() in ('q', 'exit', 'quit'):
                break
            val_a = float(line_a)
            val_b = float(line_b)
        except ValueError:
            print("输入格式错误，请输入浮点数。")
            continue

        out, target, error, rel_acc = infer(val_a, val_b, internal_ids, w_a, w_b, w_out)
        print("-" * 50)
        print(f"  输入参数:               A = {val_a:.4f}, B = {val_b:.4f}")
        print(f"  外部目标真值 (A × B):   {target:.4f}")
        print(f"  神经网络流出结果:       {out:.4f}")
        print(f"  瞬时流阻残差 (误差):   {error:.4f}")
        print(f"  计算匹配度 (准确率):   {rel_acc:.2f}%")
        print("-" * 50)

if __name__ == '__main__':
    main()
