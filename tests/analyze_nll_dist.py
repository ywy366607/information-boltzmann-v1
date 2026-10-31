import numpy as np

import json
with open("C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/owt_cfe_gpu_stream_data.json") as f:
    surps = json.load(f)
history = surps["history"]
nlls = [h["nll_nats"] for h in history]

nlls = np.array(nlls)
print(f"Total tokens recorded: {len(nlls)}")
print(f"Min NLL:    {nlls.min():.3f} nats")
print(f"Median NLL: {np.median(nlls):.3f} nats")
print(f"Mean NLL:   {nlls.mean():.3f} nats")
print(f"Max NLL:    {nlls.max():.3f} nats")

print("\nDistribution breakdown:")
print(f"  < 3.0 nats:      {np.mean(nlls < 3.0)*100:5.1f}%  (High confidence predictions)")
print(f"  3.0 - 5.0 nats:  {np.mean((nlls >= 3.0) & (nlls < 5.0))*100:5.1f}%  (Strong associative predictions)")
print(f"  5.0 - 7.0 nats:  {np.mean((nlls >= 5.0) & (nlls < 7.0))*100:5.1f}%  (Frequent syntactic & bigram transitions)")
print(f"  7.0 - 9.0 nats:  {np.mean((nlls >= 7.0) & (nlls < 9.0))*100:5.1f}%  (Unigram baseline distribution)")
print(f"  > 9.0 nats:      {np.mean(nlls >= 9.0)*100:5.1f}%  (Rare words, proper nouns, domain shifts)")
