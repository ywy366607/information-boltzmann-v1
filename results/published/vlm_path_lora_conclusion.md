# Phase 3: frozen LM vs LoRA (trainable language side)

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- LoRA: r=8 α=16 last_n=4 (layers 14:18) targets=`q_proj,v_proj` (~82k adapter params)
- Profile: batch=1 accum=8 steps=400 res=32 T=32 amp=True; seed=0
- Task: digit strings len 2–3; eval n=48
- Metrics: **TF answer-token acc** + **digit-constrained** CER/exact (not open free-gen)

| kind | mode | tf_acc | exact | CER | loss | train_params | VRAM |
|------|------|--------|-------|-----|------|--------------|------|
| A | frozen | 0.322 | 0.000 | 0.927 | 1.876 | 533952 | 1747 |
| B | frozen | 0.322 | 0.000 | 0.948 | 1.503 | 519468 | 2137 |
| A | lora | 0.333 | 0.000 | 0.906 | 1.580 | 615872 | 2515 |
| B | lora | **0.356** | 0.000 | **0.906** | 1.536 | 601388 | 2905 |

## LoRA vs frozen

| kind | Δ TF | Δ CER (↓ better) | Δ exact |
|------|------|------------------|---------|
| A | +0.011 | −0.021 | 0.000 |
| B | **+0.034** | −0.042 | 0.000 |

- Mean LoRA vs frozen: TF **0.345 vs 0.322** (Δ=+0.023); CER **0.906 vs 0.938** (Δ=−0.032); exact **0.000 vs 0.000**
- **B + LoRA** clears the TF threshold (ΔTF≥0.03) → small but real **trainable-language lift** on teacher-forced answer tokens.
- CER improves modestly; still high (~0.91).

## B vs A

| mode | ΔTF (B−A) | ΔCER | Δexact |
|------|-----------|------|--------|
| frozen | +0.000 | +0.021 (B worse) | 0 |
| lora | **+0.023** | 0.000 | 0 |

Under LoRA, B edges A on TF; CER tied. Absolute **exact remains 0** for all cells.

## Conclusion

1. **LoRA path works on 4GB** (peak ~2.9GB for B+LoRA): FE + last-4-layer q/v adapters train jointly.
2. **Partial unlock:** LoRA improves TF (esp. B: 0.322→0.356) and slightly lowers CER; matches Phase-3 hypothesis that a *trainable* language side moves VLM-path metrics where frozen was stuck-flat.
3. **Not product OCR yet:** constrained **exact=0** everywhere under this short budget / 270M / synthetic 1px strings. Need longer curriculum, stronger adapter (more layers / r), or tiny reasoner head (3B), then multi-font product data (Phase 4).
4. Non-claim: open free-gen exact≈0 is still not a success metric.

## Artifacts

- Table: `results/published/vlm_path_lora_table.json`
- Code: `fine_grain/lora_llm.py`, `scripts/run_vlm_path_lora.py`
