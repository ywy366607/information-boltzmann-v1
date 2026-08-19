# VLM frontend A/B/C comparison (task probe)

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- Cache: `D:\ml_cache` (D: drive)
- res=32 patch=4 steps=80 topk=2 ST=on for B/C
- **Probe**: held-out synthetic VQA exact-match (val_seed=90001); tasks=color + kinks; **no angles**
- Train: `Question: … Answer: …` on-the-fly mix (train seed=0)

| kind | T | loss | acc | color | kinks | notes |
|------|---|------|-----|-------|-------|-------|
| A | 64 | 1.952 | 0.156 | 0.294 | 0.000 |  |
| B | 64 | 2.945 | 0.094 | 0.176 | 0.000 |  PR=49.7 r99=62.5 sup=2.0 |
| C | 48 | 2.748 | 0.156 | 0.294 | 0.000 | 16+32 PR=18.9 r99=31.1 sup=2.0 |
| A | 32 | 1.785 | 0.156 | 0.294 | 0.000 |  |
| B | 32 | 3.805 | 0.016 | 0.029 | 0.000 |  PR=16.4 r99=30.2 sup=2.0 |
| C | 32 | 3.187 | 0.062 | 0.118 | 0.000 | 16+16 PR=11.1 r99=15.5 sup=2.0 |
| A | 16 | 1.869 | 0.156 | 0.294 | 0.000 |  |
| B | 16 | 3.258 | 0.109 | 0.206 | 0.000 |  PR=7.0 r99=15.6 sup=2.0 |
| C | 12 | 2.661 | 0.234 | 0.441 | 0.000 | 4+8 PR=3.7 r99=7.2 sup=2.0 |

## Short conclusion

- Best **held-out VQA exact-match** by kind: C=T12 acc=0.234 (color=0.44, kinks=0.00), A=T64 acc=0.156 (color=0.29, kinks=0.00), B=T16 acc=0.109 (color=0.21, kinks=0.00)
- B token compression (overall acc): T64:0.094 → T32:0.016 → T16:0.109
- Slice: ST + topk=2; support/PR/r99 logged when available.
- Synthetic VQA only (no third-party VQAv2 yet); protocol is task exact-match, not teacher-forced caption token acc.
