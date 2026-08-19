# Visual Latent CoT OCR (4GB, frozen Gemma)

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- A = static patch snapshot; B_vlcot = K=2 thoughts + W_q + damped deslice (β=0.5, γ=0.9)
- NextLat aux weight=0.1; string OCR len 2-3
- batch=1 accum=8 steps=600

| kind | T | K | exact | CER | VRAM_MB | loss |
|------|---|---|-------|-----|---------|------|
| A | 32 | 1 | 0.000 | 2.698 | 1747 | 1.718 |
| B_vlcot | 32 | 2 | 0.000 | 2.547 | 1761 | 1.791 |

## Notes

- CER A=2.698 vs B_vlcot=2.547 (Δ=-0.151, lower better)
- exact A=0.000 vs B_vlcot=0.000
- This is M0: visual latent CoT inside frontend; LLM still frozen snapshot reader.
- Full design: LLM last-h → h_to_z / W_q as control (hooked via external_h API).
