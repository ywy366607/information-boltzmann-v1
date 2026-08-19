# Native MoT (spec) on Flickr8k

- Spec: private QKV/FFN per modality; K/V concat shared attention space; SliceRead→MoT→Deslice→LocalVisual
- Dims: d_x=128 d=256 M=32 L=2 res=32
- LLM: lora_r8_14:18; stack_params=4545587
- Token gate (1× params): need ≥4645587, approx_seen~40000, **eligible=False**
- steps=400 max_rows=2000

| kind | tf_acc | eval_loss | CER-N/A | VRAM | eligible |
|------|--------|-----------|---------|------|----------|
| native_mot | 0.356 | 3.436 | — | 1932 | False |

## Notes
- This smoke proves the **architecture path** on real captions.
- Full **validation eligibility** needs open multimodal data with token count ≥ parameter count (scale Flickr30k / COCO shards / LLaVA-558k).
- Full unfreeze of 270M on 4GB spills to shared RAM; joint uses LoRA.
