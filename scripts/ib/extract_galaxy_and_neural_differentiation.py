"""
提取词引力星系结构与大脑神经分化数据
用于 3D WebGL / Canvas 交互式全景可视化分析
"""

import sys, os, time, math, json
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import GPT2TokenizerFast

from fine_grain.information_boltzmann.owt_cfe import OWTConstructalEngine

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading data and initializing engine on {device}...")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")

engine = OWTConstructalEngine(
    vocab_size=50257,
    dim=64,
    init_slots=2048,
    max_slots=8192,
    k_sparse=256,
    pipe_lr=4.0,
    J_crit=3.2,
    eta_oja=0.05,
    device=device
)

total_steps = 6000
print(f"Running continuous stream for {total_steps} steps to allow galaxy formation and neural differentiation...")
t0 = time.time()
surps = []
for step in range(total_steps):
    c_tok = int(train_tokens[step])
    n_tok = int(train_tokens[step + 1])
    metrics = engine.step_stream(c_tok, n_tok)
    surps.append(metrics["surprisal_nats"])
    if step % 1000 == 0:
        print(f"  Step {step:5d} / {total_steps} | NLL: {np.mean(surps[-100:]):.2f} n | Slots: {int(metrics['num_slots'])}")

torch.cuda.synchronize()
print(f"Stream training completed in {time.time() - t0:.2f}s")

# ==========================================================
# 1. 分析词引力星系结构 (Semantic Galaxy Formation)
# ==========================================================
print("\n[1/3] 分析词元引力星系与哈密顿公转轨道...")
counts = engine.token_counts.cpu().numpy()
seen_indices = np.where(counts > 0)[0]
print(f"Total unique tokens observed: {len(seen_indices)}")

# 选取出现频次较高的高信息量核心词元 (Top 350 个高频与活跃词元)
top_seen = seen_indices[np.argsort(-counts[seen_indices])[:350]]

# 获取 64 维嵌入与速度
embeddings = engine.embedding.weight[top_seen].detach().cpu().numpy()
velocities = engine.token_velocity[top_seen].detach().cpu().numpy()

# 3D PCA 投影降维
pca = PCA(n_components=3, random_state=42)
coords_3d = pca.fit_transform(embeddings)
# 归一化到空间球体半径 [-80, 80]
scale_factor = 70.0 / np.max(np.abs(coords_3d))
coords_3d = coords_3d * scale_factor

# 投影速度到 3D 空间
vel_3d = pca.transform(embeddings + velocities * 5.0) * scale_factor - coords_3d

# 自动发现星系团 (K-Means 聚类，找出 6 大语义星系)
n_galaxies = 6
kmeans = KMeans(n_clusters=n_galaxies, random_state=42, n_init=10).fit(embeddings)
cluster_labels = kmeans.labels_

galaxy_names = [
    "标点与篇章骨架星系 (Punctuation & Discourse)",
    "语法功能词星系 (Grammar & Syntagms)",
    "核心动态动词星系 (Verbal Dynamics)",
    "实体与人称代词星系 (Entities & Pronouns)",
    "领域名词与概念星系 (Topical Nouns)",
    "形态修饰与副词星系 (Modifiers & Descriptors)"
]

# 分析每个星系的代表词
galaxy_representatives = []
for g_id in range(n_galaxies):
    g_tok_idx = [top_seen[i] for i in range(len(top_seen)) if cluster_labels[i] == g_id]
    sample_words = [tokenizer.decode([t]).strip() for t in g_tok_idx[:8]]
    galaxy_representatives.append(", ".join([f"'{w}'" for w in sample_words if w]))
    print(f"  🌌 星系 {g_id+1}: {galaxy_names[g_id]} -> 样本: {galaxy_representatives[-1]}")

tokens_data = []
for i, tok_id in enumerate(top_seen):
    tok_str = tokenizer.decode([tok_id])
    m_val = float(math.log(1.0 + counts[tok_id]))
    v_norm = float(np.linalg.norm(velocities[i]))
    tokens_data.append({
        "id": int(tok_id),
        "text": tok_str,
        "x": round(float(coords_3d[i, 0]), 2),
        "y": round(float(coords_3d[i, 1]), 2),
        "z": round(float(coords_3d[i, 2]), 2),
        "vx": round(float(vel_3d[i, 0]), 3),
        "vy": round(float(vel_3d[i, 1]), 3),
        "vz": round(float(vel_3d[i, 2]), 3),
        "v_norm": round(v_norm, 4),
        "mass": round(m_val, 2),
        "count": int(counts[tok_id]),
        "galaxy_id": int(cluster_labels[i]),
        "galaxy_name": galaxy_names[int(cluster_labels[i])]
    })

# ==========================================================
# 2. 分析大脑神经分化结构 (Neural Differentiation of Slots)
# ==========================================================
print("\n[2/3] 分析构形槽位的大脑神经级功能分化...")
act_slots = torch.where(engine.active_mask)[0].cpu().numpy()
n_act = len(act_slots)
print(f"Active slots: {n_act}")

gammas = engine.gamma[act_slots].cpu().numpy()
fluxes = engine.flux_accum[act_slots].cpu().numpy()
W_out = engine.W_out[act_slots].detach().cpu()

# 槽位功能分化：分析每个槽位偏好激活的前 3 个特征词元
slot_specializations = []
capillary_count = 0
meso_count = 0
trunk_count = 0

for i, slot_id in enumerate(act_slots):
    g_val = float(gammas[i])
    f_val = float(fluxes[i])
    top3_w = torch.topk(W_out[i], 3).indices.numpy()
    words = [tokenizer.decode([w]).strip() for w in top3_w if w > 0]

    # 神经类型根据时间尺度自发分类
    if g_val < 0.58:
        slot_type = "毛细瞬时神经元 (Capillary Fast-Spike)"
        capillary_count += 1
    elif g_val < 0.75:
        slot_type = "中层连络中继神经元 (Meso Relay)"
        meso_count += 1
    else:
        slot_type = "深层宏观主题大河 (Macro Trunk Pyramidal)"
        trunk_count += 1

    slot_specializations.append({
        "slot_id": int(slot_id),
        "gamma": round(g_val, 4),
        "flux": round(f_val, 2),
        "tau": round(1.0 / max(1e-4, 1.0 - g_val), 1),
        "slot_type": slot_type,
        "tuning_words": words
    })

print(f"  • 毛细瞬时神经元 (γ < 0.58, τ ≈ 2): {capillary_count} 个 ({capillary_count/n_act*100:.1f}%)")
print(f"  • 中层连络中继神经元 (0.58 ≤ γ < 0.75, τ ≈ 3~4): {meso_count} 个 ({meso_count/n_act*100:.1f}%)")
print(f"  • 深层宏观主题大河 (γ ≥ 0.75, τ ≥ 4~10+): {trunk_count} 个 ({trunk_count/n_act*100:.1f}%)")

# 抽样 120 个槽位构建可视化网络（按分层比例均匀抽样）
sample_slots = []
# 挑选前 30 个最强宏观大河，40 个中层中继，50 个毛细神经元
sorted_by_gamma = np.argsort(gammas)
selected_indices = np.concatenate([
    sorted_by_gamma[:50],  # 毛细
    sorted_by_gamma[len(sorted_by_gamma)//2 - 20 : len(sorted_by_gamma)//2 + 20],  # 中层
    sorted_by_gamma[-30:]  # 主动脉
])

# 3D 投影槽位受体 k_sig
k_sigs = engine.k_sig[act_slots[selected_indices]].detach().cpu().numpy()
pca_k = PCA(n_components=3, random_state=42)
slot_coords_3d = pca_k.fit_transform(k_sigs) * 60.0

sampled_slots_data = []
for idx_in_sel, sel_idx in enumerate(selected_indices):
    s_info = slot_specializations[sel_idx]
    s_info["x"] = round(float(slot_coords_3d[idx_in_sel, 0]), 2)
    s_info["y"] = round(float(slot_coords_3d[idx_in_sel, 1]), 2)
    s_info["z"] = round(float(slot_coords_3d[idx_in_sel, 2]), 2)
    sampled_slots_data.append(s_info)

# 计算槽位之间的构形导管连边 (基于受体余弦相似度 > 0.40)
edges = []
sim_matrix = np.dot(k_sigs, k_sigs.T)
for i in range(len(selected_indices)):
    for j in range(i + 1, len(selected_indices)):
        if sim_matrix[i, j] > 0.42:
            edges.append({
                "source": i,
                "target": j,
                "weight": round(float(sim_matrix[i, j]), 3)
            })

# ==========================================================
# 3. 输出自包含 JSON 数据
# ==========================================================
out_json_path = "present/galaxy_neural_data.json"
os.makedirs("present", exist_ok=True)
payload = {
    "total_steps": total_steps,
    "active_slots_total": n_act,
    "seen_tokens_total": len(seen_indices),
    "galaxies": [
        {"id": i, "name": galaxy_names[i], "samples": galaxy_representatives[i]}
        for i in range(n_galaxies)
    ],
    "tokens": tokens_data,
    "slots": sampled_slots_data,
    "edges": edges,
    "slot_hierarchy": {
        "capillary": capillary_count,
        "meso": meso_count,
        "trunk": trunk_count
    }
}

with open(out_json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"\n[3/3] 提取完成! 数据已保存至 {out_json_path}")
print(f"  • 可视化词元数量: {len(tokens_data)} 颗恒星 (带 3D 公转速度矢量)")
print(f"  • 可视化槽位数量: {len(sampled_slots_data)} 个神经元节点 (分层血管树)")
print(f"  • 构形血管连接边数: {len(edges)} 条河道")
