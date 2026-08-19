# VLM-OCR frontend @ 4GB (frozen small LM)

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- Profile: batch=1 accum=8 eff_batch=8 amp=True bf16=False
- Task: **string** 1px strokes; multi-digit strings len 2-4
- Train: frontend+MLP only, LLM frozen, answer-only CE
- Eval: free-gen exact string match + CER (n=64)
- steps=1200 res=32 projector=mlp

| kind | T | seed | loss | exact_acc | CER | peak_VRAM_MB |
|------|---|------|------|-----------|-----|--------------|
| A | 32 | 0 | 1.518 | 0.000 | 0.979 | 1754 |
| B | 32 | 0 | 1.511 | 0.000 | 0.993 | 1761 |
| A | 16 | 0 | 1.631 | 0.000 | 0.979 | 1695 |
| B | 16 | 0 | 1.534 | 0.000 | 1.030 | 1700 |

## Conclusion

- Best CER: A@T32 CER=0.979 exact=0.000
- **A**: best CER=0.979 exact=0.000 (T32)
- **B**: best CER=0.993 exact=0.000 (T32)
- B vs A (best CER): B=0.993 vs A=0.979 (ΔCER=+0.015; lower better)
- Peak VRAM observed: 1761 MB
- 4GB profile: frozen LM + train vision only; string OCR = real short-sequence readout metric (CER), not 10-way classification.
