# Small VLM on HF captions: slice vs patch encoder

- Dataset: **jxie/flickr8k** (train max_rows=2000)
- LLM: frozen Gemma-3-270M; train vision frontend + MLP projector
- res=64 T=64 steps=400 batch=1 accum=8
- Metric: teacher-forced caption-span token acc (not free-gen BLEU)

| encoder | tf_acc | eval_loss | final_loss | VRAM | status |
|---------|--------|-----------|------------|------|--------|
| slice | 0.127 | 5.335 | 5.858 | 2014 | ok |
| patch | 0.306 | 3.981 | 3.249 | 1959 | ok |

## slice vs patch
- TF: slice=0.127 vs patch=0.306 (Δ=-0.179)
- eval_loss: slice=5.335 vs patch=3.981
- No clear slice win yet; still a valid encoder path to scale (more data / steps / unfreeze projector curriculum).

## HF datasets for next scale
- `jxie/flickr8k` — this run
- `nlphuji/flickr30k` — medium
- `liuhaotian/LLaVA-Pretrain` (558k) — LLaVA stage-1 projector pretrain
- `Multimodal-Fatima/COCO_captions_train` — COCO shards

Synthetic 1px OCR remains diagnostic; real caption data answers encoder utility.
