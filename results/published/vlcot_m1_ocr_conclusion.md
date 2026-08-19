# M1 Visual Latent CoT + LLM two-look OCR

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- B_vlcot: K=2 latent + **LLM h → second look**
- Eval: **digit-constrained** gen; refine_every=1 for B
- steps=800 batch=1 accum=8

| kind | two_look | exact | CER | VRAM | loss |
|------|----------|-------|-----|------|------|
| A | False | 0.021 | 0.847 | 1747 | 1.740 |
| B_vlcot | True | 0.000 | 0.903 | 2400 | 2.357 |

## Conclusion

- CER A=0.847 vs B=0.903 (Δ=+0.056)
- exact A=0.021 vs B=0.000
- No clear B>A advantage on this run (absolute OCR may still be weak).
- Decode constrained to digits reduces open-vocab collapse; free-gen garbage less dominant.
