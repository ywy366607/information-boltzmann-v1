# Transolver3 native multimodal VLM-OCR (vision–language co-evolution)

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- Thesis: full-res field X + shared U; **not** permanent vision slice tokens
- Profile: steps=400 batch=1 accum=8 res=32 T=32 n_layers=2 coevolve_rounds=2
- Metrics: TF answer-span + digit-constrained CER/exact (n=48)

| kind | T3 | tf_acc | exact | CER | loss | VRAM | status |
|------|----|--------|-------|-----|------|------|--------|
| A | False | 0.333 | 0.000 | 0.948 | 1.707 | 1747 | ok |
| B_xmodal | True | 0.339 | 0.000 | 0.906 | 2.193 | 2795 | ok |

## Comparison

- TF: T3=0.339 vs A=0.333 (Δ=+0.006)
- CER: T3=0.906 vs A=0.948 (Δ=-0.042; lower better)
- exact: T3=0.000 vs A=0.000 (Δ=+0.000)
- **Transolver3 co-evolve beats static patch A** on at least one primary margin.

Non-claim: open free-gen alone is not product proof. Legacy B slice-tokens frontend is not used here.
