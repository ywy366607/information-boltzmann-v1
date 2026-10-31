import json, numpy as np
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
with open(r'C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/owt_cfe_gpu_stream_data.json') as f:
    data = json.load(f)

train_tokens = np.load('data/ib_owt_gpt2/train.npy', mmap_mode='r')
hist = data['history']

low_nll = [h for h in hist if h['nll_nats'] < 3.0]
high_nll = [h for h in hist if h['nll_nats'] > 10.5]

print('=== Tokens with LOW NLL (< 3.0 nats) ===')
for h in low_nll[:10]:
    step = h['step']
    c_w = tokenizer.decode([int(train_tokens[step])])
    n_w = tokenizer.decode([int(train_tokens[step+1])])
    print(f"Step {step:5d}: [{c_w!r} -> {n_w!r}] NLL = {h['nll_nats']:.3f} nats")

print('\n=== Tokens with HIGH NLL (> 10.5 nats) ===')
for h in high_nll[:10]:
    step = h['step']
    c_w = tokenizer.decode([int(train_tokens[step])])
    n_w = tokenizer.decode([int(train_tokens[step+1])])
    print(f"Step {step:5d}: [{c_w!r} -> {n_w!r}] NLL = {h['nll_nats']:.3f} nats")

late_hist = [h for h in hist if h['step'] >= 10000]
late_nlls = [h['nll_nats'] for h in late_hist]
print(f"\n=== LATE STAGE (10,000 ~ 15,000 steps) ===")
print(f"Late stage mean NLL: {np.mean(late_nlls):.3f} nats")
print(f"Late stage median NLL: {np.median(late_nlls):.3f} nats")
print(f"Late stage min NLL: {min(late_nlls):.3f} nats")
print(f"Late stage < 7 nats percentage: {np.mean([1 if x < 7.0 else 0 for x in late_nlls])*100:.1f}%")
print(f"Late stage < 8 nats percentage: {np.mean([1 if x < 8.0 else 0 for x in late_nlls])*100:.1f}%")

late_high = [h for h in late_hist if h['nll_nats'] > 10.0]
print(f"Late high (>10 nats): {len(late_high)} / {len(late_hist)} ({len(late_high)/len(late_hist)*100:.1f}%)")
for h in late_high[:5]:
    step = h['step']
    c_w = tokenizer.decode([int(train_tokens[step])])
    n_w = tokenizer.decode([int(train_tokens[step+1])])
    print(f"Step {step:5d}: [{c_w!r} -> {n_w!r}] NLL = {h['nll_nats']:.3f} nats")
