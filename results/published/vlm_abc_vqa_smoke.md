# VLM frontend A/B/C comparison (task probe)

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- Cache: `D:\ml_cache` (D: drive)
- res=32 patch=4 steps=20 topk=2 ST=on for B/C
- **Probe**: held-out synthetic VQA exact-match (val_seed=90001); tasks=color + kinks; **no angles**
- Train: `Question: … Answer: …` on-the-fly mix (train seed=0)

| kind | T | loss | acc | color | kinks | notes |
|------|---|------|-----|-------|-------|-------|
| A | 32 | 2.904 | 0.062 | 0.111 | 0.000 |  |
| B | 32 | 4.326 | 0.000 | 0.000 | 0.000 |  PR=16.9 r99=30.2 sup=2.0 |
| C | 32 | 4.320 | 0.000 | 0.000 | 0.000 | 16+16 PR=11.4 r99=15.5 sup=2.0 |

## Short conclusion

- Best **held-out VQA exact-match** by kind: A=T32 acc=0.062 (color=0.11, kinks=0.00), B=T32 acc=0.000 (color=0.00, kinks=0.00), C=T32 acc=0.000 (color=0.00, kinks=0.00)
- Slice: ST + topk=2; support/PR/r99 logged when available.
- Synthetic VQA only (no third-party VQAv2 yet); protocol is task exact-match, not teacher-forced caption token acc.
