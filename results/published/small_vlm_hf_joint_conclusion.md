# Small VLM on HF captions: slice vs patch encoder

- Dataset: **jxie/flickr8k** (train max_rows=2000)
- LLM: Gemma-3-270M; **joint=True** (LoRA r=8 last-4)
- Vision+MLP always trained; language via LoRA when joint (full unfreeze OOMs 4GB)
- res=64 T=64 steps=400 batch=1 accum=8
- Metric: teacher-forced caption-span token acc

| encoder | joint | lang | tf_acc | eval_loss | final_loss | VRAM | status |
|---------|-------|------|--------|-----------|------------|------|--------|
| slice | True | lora_r8_14:18 | 0.352 | 3.357 | 2.869 | 1733 | ok |
| patch | True | lora_r8_14:18 | 0.264 | 3.880 | 3.291 | 1726 | ok |

## slice vs patch
- TF: slice=0.352 vs patch=0.264 (Δ=+0.088)
- eval_loss: slice=3.357 vs patch=3.880
- **Slice looks competitive as a vision encoder** under this budget.

## HF datasets for next scale
- `jxie/flickr8k` — this run
- `nlphuji/flickr30k` — medium
- `liuhaotian/LLaVA-Pretrain` (558k) — LLaVA stage-1 projector pretrain
- `Multimodal-Fatima/COCO_captions_train` — COCO shards

Synthetic 1px OCR remains diagnostic; real caption data answers encoder utility.
