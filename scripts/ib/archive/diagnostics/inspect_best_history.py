import json

def inspect_file(path):
    print(f"\n=== {path} ===")
    lines = [json.loads(l) for l in open(path, 'r', encoding='utf-8') if l.strip()]
    for i, l in enumerate(lines):
        b = l.get('live_prequential_nll', 99)
        a1 = sum(l.get('A1_curve', [99])) / len(l.get('A1_curve', [1]))
        a2 = sum(l.get('A2_replay_curve', [99])) / len(l.get('A2_replay_curve', [1]))
        if b < 7.5 or a1 < 7.2 or a2 < 7.0:
            tok = l.get('bptt_train_tokens')
            cur = l.get('train_cursor')
            sav = l.get('savings', {}).get('nll_saving', 0)
            print(f"[{i:2d}] tok={tok} | cur={cur} | Live B={b:.4f} | A1={a1:.4f} | A2={a2:.4f} | Saving={sav:+.4f}")

inspect_file('results/q8_fly_bptt32_continuous_100k/lifelong_evaluation.jsonl')
inspect_file('results/q8_fly_bptt32_adamw_continuous_100k/lifelong_evaluation.jsonl')
inspect_file('results/q8_fly_bptt32_centered_100k/lifelong_evaluation.jsonl')
inspect_file('artifacts/active_training/q8_fly_bptt32_gamma_2333_100k/lifelong_evaluation.jsonl')
