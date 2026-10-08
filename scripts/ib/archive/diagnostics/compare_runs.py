import json
import os

runs = {
    "Centered Baseline (解冻前)": "artifacts/active_training/q8_fly_bptt32_centered_100k/lifelong_evaluation.jsonl",
    "Learned Gamma-2333 (解冻后当前)": "artifacts/active_training/q8_fly_bptt32_gamma_2333_100k/lifelong_evaluation.jsonl",
    "AdamW Continuous (更早基准)": "results/q8_fly_bptt32_adamw_continuous_100k/lifelong_evaluation.jsonl"
}

def analyze_jsonl(path):
    if not os.path.exists(path):
        return None
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

print("=" * 100)
print(f"{'Run / Stage':<32} | {'Tokens Span':<18} | {'Live B NLL (Mean/Best)':<22} | {'A1 (First)':<10} | {'A2 (Revisit)':<12} | {'A2 Saving':<10}")
print("=" * 100)

for name, path in runs.items():
    records = analyze_jsonl(path)
    if not records:
        print(f"{name:<32} | Not found")
        continue
    
    start_tok = records[0].get("bptt_train_tokens", 0)
    end_tok = records[-1].get("bptt_train_tokens", 0)
    token_span = f"{start_tok:,} -> {end_tok:,}"
    
    live_b_list = [r.get("live_prequential_nll", 0.0) for r in records]
    live_mean = sum(live_b_list) / len(live_b_list)
    live_best = min(live_b_list)
    live_str = f"{live_mean:.4f} / {live_best:.4f}"
    
    a1_list = [sum(r.get("A1_curve", [0]))/len(r.get("A1_curve", [1])) for r in records if r.get("A1_curve")]
    a2_list = [sum(r.get("A2_replay_curve", [0]))/len(r.get("A2_replay_curve", [1])) for r in records if r.get("A2_replay_curve")]
    saving_list = [r.get("savings", {}).get("nll_saving", 0.0) for r in records]
    
    a1_mean = sum(a1_list) / len(a1_list) if a1_list else 0.0
    a2_mean = sum(a2_list) / len(a2_list) if a2_list else 0.0
    saving_mean = sum(saving_list) / len(saving_list) if saving_list else 0.0
    
    print(f"{name:<32} | {token_span:<18} | {live_str:<22} | {a1_mean:<10.4f} | {a2_mean:<12.4f} | {saving_mean:<+10.4f}")

print("=" * 100)

# Detailed per-checkpoint progression for Centered Baseline vs Current Gamma run
print("\n" + "=" * 100)
print("DETAILED CHRONOLOGICAL COMPARISON: 解冻前 (Centered) vs 解冻后 (Gamma-2333)")
print("=" * 100)

for name, path in [("Centered Baseline (解冻前)", runs["Centered Baseline (解冻前)"]), 
                   ("Learned Gamma-2333 (解冻后当前)", runs["Learned Gamma-2333 (解冻后当前)"])]:
    records = analyze_jsonl(path)
    print(f"\n--- {name} ---")
    print(f"{'Checkpoint Token':<18} | {'Live B NLL':<12} | {'A1 (First)':<12} | {'A2 (Revisit)':<12} | {'A2 Saving':<10} | {'Rank':<6}")
    print("-" * 80)
    for r in records:
        tok = r.get("bptt_train_tokens", 0)
        live = r.get("live_prequential_nll", 0.0)
        a1 = sum(r.get("A1_curve", [0]))/len(r.get("A1_curve", [1]))
        a2 = sum(r.get("A2_replay_curve", [0]))/len(r.get("A2_replay_curve", [1]))
        sav = r.get("savings", {}).get("nll_saving", 0.0)
        rnk = r.get("centered_effective_rank", 0.0)
        print(f"{tok:<18} | {live:<12.4f} | {a1:<12.4f} | {a2:<12.4f} | {sav:<+10.4f} | {rnk:<6.2f}")
