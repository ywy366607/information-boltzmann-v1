# VLM-path Phase 2: TF answer-span + digit-constrained CER

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- Train: frontend (+ VL-CoT if B_vlcot) + projector; **LLM frozen**
- Task: multi-digit 1px strings len 2-3
- Profile: batch=1 accum=8 steps=400 res=32 T=32 amp=True
- Eval n=48: **TF answer-token acc** + **digit-constrained** CER/exact (not open free-gen as primary)

| kind | T | two_look | tf_acc | exact | CER | loss | VRAM | status |
|------|---|---------|--------|-------|-----|------|------|--------|
| A | 32 | False | 0.333 | 0.000 | 0.927 | 1.276 | 1747 | ok |
| B | 32 | False | 0.333 | 0.000 | 0.924 | 1.251 | 1755 | ok |
| B_vlcot | 32 | True | 0.333 | 0.000 | 0.924 | 1.814 | 2400 | ok |

## Relative A vs B / B_vlcot

- TF: A=**0.333**, B=**0.333**, B_vlcot=**0.333** (ΔB−A=**+0.000**, ΔB_vlcot−A=**+0.000**)
- constrained CER: A=**0.927**, B=**0.924**, B_vlcot=**0.924** (ΔB−A=**−0.003**, lower better; noise-scale)
- constrained exact: A=**0.000**, B=**0.000**, B_vlcot=**0.000**

## Conclusion

- **No meaningful B (or B_vlcot) win over A** under frozen Gemma-3-270M: TF identical; CER difference (~0.003) is not a product margin; exact remains **0** for all arms.
- **Honest stuck under frozen small LM:** vision-only training + frozen 270M does not unlock string OCR readout (constrained exact=0, CER≈0.92). TF answer-token acc≈0.33 is above pure chance on a multi-digit token span but **does not separate** frontends and does not yield correct free-path strings.
- Phase-3 implication: need a **trainable** language side (LoRA / small reasoner head), not longer open free-gen marathons.
- Non-claim: open free-gen exact≈0 is not product OCR success.
