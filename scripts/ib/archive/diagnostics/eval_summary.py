import json
import os

path = "artifacts/active_training/q8_fly_bptt32_gamma_2333_100k/lifelong_evaluation.jsonl"
if not os.path.exists(path):
    print("File not found")
    exit(0)

with open(path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Total evaluation checkpoints recorded: {len(lines)}")
print("-" * 90)
print(f"{'BPTT Tokens':<12} | {'Live Prequential NLL':<20} | {'A1 (First)':<10} | {'A2 (Revisit)':<12} | {'A2 Saving':<10} | {'Rank':<6}")
print("-" * 90)

for line in lines:
    d = json.loads(line)
    tok = d.get("bptt_train_tokens", 0)
    live_nll = d.get("live_prequential_nll", 0.0)
    a1 = d.get("A1_curve", [])
    a2 = d.get("A2_replay_curve", [])
    a1_mean = sum(a1) / len(a1) if a1 else 0.0
    a2_mean = sum(a2) / len(a2) if a2 else 0.0
    savings = d.get("savings", {}).get("nll_saving", 0.0)
    rank = d.get("centered_effective_rank", 0.0)
    print(f"{tok:<12} | {live_nll:<20.4f} | {a1_mean:<10.4f} | {a2_mean:<12.4f} | {savings:<10.4f} | {rank:<6.2f}")
print("-" * 90)
