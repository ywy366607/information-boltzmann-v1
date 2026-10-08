import json

path = 'results/q8_fly_bptt32_adamw_continuous_100k/lifelong_evaluation.jsonl'
lines = open(path, 'r', encoding='utf-8').readlines()

print(f"{'Index':<6} | {'Tokens':<10} | {'Cursor':<10} | {'Live B NLL':<12} | {'A1 (First)':<12} | {'A2 (Revisit)':<12} | {'A2 Saving':<10}")
print("-" * 85)

for i, line in enumerate(lines):
    d = json.loads(line)
    tok = d.get("bptt_train_tokens", 0)
    cur = d.get("train_cursor", 0)
    live = d.get("live_prequential_nll", 0.0)
    a1 = sum(d.get("A1_curve", [0])) / len(d.get("A1_curve", [1]))
    a2 = sum(d.get("A2_replay_curve", [0])) / len(d.get("A2_replay_curve", [1]))
    sav = d.get("savings", {}).get("nll_saving", 0.0)
    print(f"{i:<6} | {tok:<10} | {cur:<10} | {live:<12.4f} | {a1:<12.4f} | {a2:<12.4f} | {sav:<+10.4f}")
