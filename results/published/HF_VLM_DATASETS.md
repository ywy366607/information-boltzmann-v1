# HuggingFace datasets for a tiny VLM (this repo)

Goal: stop leaning only on synthetic 1px OCR; train a **small VLM**  
`slice/patch encoder → MLP → Gemma-3-270M` on real image–text pairs and see if
**slice works as a vision encoder** at this scale.

## Recommended (practical on one 4GB box)

| Priority | HF id | Size | Notes |
|----------|-------|------|--------|
| **1 (use now)** | [`jxie/flickr8k`](https://huggingface.co/datasets/jxie/flickr8k) | ~8k imgs, 5 caps | Free; images **inside parquet**; script already loads it |
| 2 | [`nlphuji/flickr30k`](https://huggingface.co/datasets/nlphuji/flickr30k) | ~31k | zip+csv; larger download |
| 3 | [`Multimodal-Fatima/COCO_captions_train`](https://huggingface.co/datasets/Multimodal-Fatima/COCO_captions_train) | COCO-scale | Multi-shard parquet; take 1–2 shards first |
| 4 | [`liuhaotian/LLaVA-Pretrain`](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | 558k | Classic **LLaVA stage-1** projector pretrain; heavy |
| alt | `svjack/pokemon-blip-captions-en-zh` | tiny | Ungated pokemon-style captions |
| gated | `lambdalabs/pokemon-blip-captions` | ~800 | Needs HF login |

## Recipe (matches LLaVA spirit, small)

1. **Stage A (this script):** freeze LLM; train **vision encoder + MLP** on captions  
   (`Question: Describe… Answer: <caption>` answer-only CE).  
2. **Stage B (later):** LoRA / light LM finetune on instruct pairs if Stage A works.  
3. Compare **slice vs patch** under same T / res / steps.

## Run

```bash
set ML_CACHE_ROOT=D:\ml_cache
python scripts/train_small_vlm_hf.py --kinds slice patch --steps 800 --res 64 --T 64 --max_rows 4000 --amp
```

Loader: `fine_grain/hf_caption_data.py` (no `datasets` lib — hub download + parquet).

## What this is *not*

- Not a claim that frozen free-gen OCR on stick digits is the product task.  
- Not full LLaVA-1.5 7B instruct.  
- Slice **as encoder** = content-adaptive tokens from the image field; product Transolver3 field+U story can stack later once encoder utility is clear on real captions.
