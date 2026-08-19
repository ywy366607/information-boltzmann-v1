# Transolver3 native multimodal VLM-OCR (vision–language co-evolution)

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- Thesis: full-res field X + shared U; **not** permanent vision slice tokens
- Profile: steps=400 batch=1 accum=8 res=32 T=32 n_layers=2 coevolve_rounds=2
- Metrics: TF answer-span + digit-constrained CER/exact (n=48)

| kind | mot | tf_acc | exact | CER | loss | VRAM | status |
|------|-----|--------|-------|-----|------|------|--------|
| A | False | 0.333 | 0.000 | 0.948 | 1.707 | 1747 | ok |
| B_xmodal | False | 0.339 | 0.000 | 0.906 | 2.193 | 2795 | ok |
| B_mot | True | 0.339 | 0.000 | 0.941 | 1.711 | 2137 | ok |

## Comparison (vs A)

- **B_xmodal** vs A: TF Δ=+0.006, CER Δ=-0.042, exact Δ=+0.000
- **B_mot** vs A: TF Δ=+0.006, CER Δ=-0.007, exact Δ=+0.000

## MoT (joint self-attn) vs H-path (B_xmodal)
- TF: MoT=0.339 vs H-path=0.339 (Δ=+0.000)
- CER: MoT=0.941 vs H-path=0.906 (Δ=+0.035)
- MoT ≈ H-path under short budget (architecture comparison).

MoT = slices‖text in **one** self-attn + deslice to point field. Non-claim: open free-gen alone is not product proof.
