import json, numpy as np
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
with open(r'C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/owt_cfe_gpu_stream_data.json') as f:
    data = json.load(f)

train_tokens = np.load('data/ib_owt_gpt2/train.npy', mmap_mode='r')
hist = data['history']

late_hist = [h for h in hist if h['step'] >= 10000]
print(f'Late stage total points: {len(late_hist)}')
sub6 = [h for h in late_hist if h['nll_nats'] < 6.0]
print(f'Late stage < 6.0 nats: {len(sub6)} ({len(sub6)/len(late_hist)*100:.1f}%)')
sub4 = [h for h in late_hist if h['nll_nats'] < 4.0]
print(f'Late stage < 4.0 nats: {len(sub4)} ({len(sub4)/len(late_hist)*100:.1f}%)')

late_high = [h for h in late_hist if h['nll_nats'] > 11.0]
print(f'Late high (> 11 nats): {len(late_high)} ({len(late_high)/len(late_hist)*100:.1f}%)')
print('\n=== Sample Late Stage < 4.0 nats ===')
for h in sub4[:8]:
    s = h['step']
    c_w = tokenizer.decode([int(train_tokens[s])])
    n_w = tokenizer.decode([int(train_tokens[s+1])])
    print(f"Step {s:5d}: [{c_w!r} -> {n_w!r}] NLL = {h['nll_nats']:.3f} nats")

print('\n=== Sample Late Stage > 11.0 nats ===')
for h in late_high[:8]:
    s = h['step']
    c_w = tokenizer.decode([int(train_tokens[s])])
    n_w = tokenizer.decode([int(train_tokens[s+1])])
    print(f"Step {s:5d}: [{c_w!r} -> {n_w!r}] NLL = {h['nll_nats']:.3f} nats")
