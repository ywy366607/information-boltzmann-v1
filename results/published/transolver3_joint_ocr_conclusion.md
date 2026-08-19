# Joint MoT OCR (vision + language trainable via LoRA)

## Why full unfreeze hit "shared memory"

On GTX 1650 **4GB**:

| Mode | Lang params | Peak VRAM (obs.) | Result |
|------|-------------|------------------|--------|
| Freeze LM | 0 | ~1.7–2.8GB | OK |
| Full unfreeze + AdamW | ~268M | **~5.4GB** | spill → Windows **shared RAM** |
| Last-N full weights | tens of M | still heavy | often spill |
| **`--joint` = LoRA r=8 last-4** | **~82k** | **~1.53GB** | **OK, no shared spill** |

AdamW keeps 2 momentum buffers per trainable param. Full 270M × fp32 × 3 ≈ multi-GB optimizer alone — exceeds 4GB even before activations.

## This run (LoRA joint, 4GB-safe)

- Backend: Gemma-3-270M fp32 backbone + **LoRA r=8** on layers 14:18 `q_proj,v_proj`
- Vision: **B_mot** (slice‖text same self-attn + deslice field)
- steps=400 batch=1 accum=8 res=32 T=32 n_layers=2
- lr_vis=3e-4 lr_lang=1e-4; gradient checkpointing on

| kind | joint | mot | tf_acc | exact | CER | loss | VRAM |
|------|-------|-----|--------|-------|-----|------|------|
| B_mot | LoRA | True | 0.339 | 0.000 | **0.892** | 1.965 | **1530** |

## vs earlier frozen runs (same task, 400 steps)

| setup | CER | exact | VRAM |
|-------|-----|-------|------|
| A frozen | 0.948 | 0 | 1747 |
| B_xmodal frozen | 0.906 | 0 | 2795 |
| B_mot frozen | 0.941 | 0 | 2137 |
| **B_mot + LoRA joint** | **0.892** | 0 | **1530** |

CER is the best among these short runs (lower better). exact still 0 under 400 steps — need longer curriculum / more LoRA capacity for string exact.

## How to run

```bash
# recommended joint on 4GB (LoRA)
python scripts/train_transolver3_ocr.py --joint --kinds B_mot --steps 400 --amp

# full LLM unfreeze — will hit shared memory on 4GB; avoid
python scripts/train_transolver3_ocr.py --joint_full --kinds B_mot

# last-N full weights (still heavy)
python scripts/train_transolver3_ocr.py --joint_lastn --llm_last_n 2 --kinds B_mot
```
