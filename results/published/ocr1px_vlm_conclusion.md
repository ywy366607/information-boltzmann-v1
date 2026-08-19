# 1px OCR digits → frozen LLM (A/B/C)

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- Task: single stick-digit **0–9**, **1px Bresenham** strokes on noisy canvas
- res=32 projector=mlp topk=2 answer_only=True
- Train stream ≈ **3200** samples/cell (steps×batch); eval n=200
- Metrics: free-gen exact-match + constrained ranking over digits
- Seeds: [0] (single-seed default)
- Chance = 0.10 (10-way)

| kind | T | seed | loss | free_acc | cons_acc | notes |
|------|---|------|------|----------|----------|-------|
| A | 64 | 0 | 1.165 | 0.110 | 0.110 | |
| B | 64 | 0 | 1.162 | 0.160 | 0.160 | |
| C | 48 | 0 | 1.222 | 0.125 | 0.125 | |
| A | 32 | 0 | 1.182 | 0.110 | 0.110 | |
| B | 32 | 0 | 1.163 | 0.115 | 0.115 | |
| A | 16 | 0 | 1.224 | 0.110 | 0.110 | |
| B | 16 | 0 | 1.270 | 0.110 | 0.110 | |
| C | 12 | 0 | 1.191 | 0.105 | 0.105 | |
| C | 32 | 0 | 1.147 | 0.140 | 0.140 | |

## Conclusion

- Best **free-gen**: B@T64 s0 acc=0.160
- Best **constrained**: B@T64 s0 acc=0.160
- **A**: best free=0.110 (T64), best cons=0.110 (T64)
- **B**: best free=0.160 (T64), best cons=0.160 (T64)
- **C**: best free=0.140 (T32), best cons=0.140 (T32)
- B vs A (best cons): B=0.160 vs A=0.110 (Δ=+0.050)
- Mean free=0.121, mean cons=0.121 (chance=0.10); cons−free≈+0.000
