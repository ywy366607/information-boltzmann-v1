"""Load small HF image-caption datasets without ``datasets`` (SSL-broken here).

Uses ``huggingface_hub.hf_hub_download`` + parquet + PIL.
Cache under ``D:\\ml_cache\\huggingface`` (or HF_HOME).
"""
from __future__ import annotations

import io
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# Prefer D: cache
os.environ.setdefault("HF_HOME", r"D:\ml_cache\huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"D:\ml_cache\huggingface\hub")


def _ssl_env():
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except Exception:
        pass


def download_parquet(repo_id: str, filename: str, cache_dir: Optional[str] = None) -> Path:
    _ssl_env()
    from huggingface_hub import hf_hub_download
    cache = cache_dir or os.environ.get("HUGGINGFACE_HUB_CACHE", r"D:\ml_cache\huggingface\hub")
    p = hf_hub_download(
        repo_id, filename, repo_type="dataset", cache_dir=cache,
    )
    return Path(p)


def pil_from_hf_image(cell: Any) -> Image.Image:
    if isinstance(cell, dict):
        b = cell.get("bytes")
        if b is None and cell.get("path"):
            return Image.open(cell["path"]).convert("RGB")
        return Image.open(io.BytesIO(b)).convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    raise TypeError(type(cell))


def pil_to_tensor(img: Image.Image, res: int = 64) -> torch.Tensor:
    """Resize short side / center-ish to res×res, float [0,1] CHW."""
    img = img.convert("RGB").resize((res, res), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # 3,H,W


class Flickr8kCaptionStore:
    """jxie/flickr8k: columns image, caption_0..4. ~6k train images in 2 shards."""

    REPO = "jxie/flickr8k"
    TRAIN_FILES = (
        "data/train-00000-of-00002-2f8f6bfa852eac4b.parquet",
        "data/train-00001-of-00002-2173151d8cd6c7fb.parquet",
    )
    TEST_FILES = (
        "data/test-00000-of-00001-42a2661d12c73e48.parquet",
    )

    def __init__(self, split: str = "train", max_rows: Optional[int] = None):
        import pyarrow.parquet as pq
        files = self.TRAIN_FILES if split == "train" else self.TEST_FILES
        tables = []
        for f in files:
            p = download_parquet(self.REPO, f)
            t = pq.read_table(p)
            tables.append(t)
        # concat
        import pyarrow as pa
        self.table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        if max_rows is not None:
            self.table = self.table.slice(0, int(max_rows))
        self.n = self.table.num_rows
        self.caption_cols = [c for c in self.table.column_names if c.startswith("caption")]
        assert "image" in self.table.column_names
        assert self.caption_cols

    def __len__(self) -> int:
        return self.n

    def get(self, i: int) -> Tuple[Image.Image, str]:
        row = self.table.slice(i, 1).to_pydict()
        img = pil_from_hf_image(row["image"][0])
        caps = [row[c][0] for c in self.caption_cols if row[c][0]]
        cap = random.choice(caps) if caps else ""
        return img, str(cap).strip()


def format_caption_prompt(caption: str, instruct: Optional[str] = None) -> Tuple[str, str]:
    """Return (prompt, full_text) for answer-only style training."""
    q = instruct or "Describe this image briefly."
    prompt = f"Question: {q}\nAnswer:"
    full = f"{prompt} {caption}"
    return prompt, full


def sample_batch(
    store: Flickr8kCaptionStore,
    idxs: Sequence[int],
    res: int = 64,
) -> Dict[str, Any]:
    imgs, prompts, texts, caps = [], [], [], []
    for i in idxs:
        im, cap = store.get(int(i))
        pr, full = format_caption_prompt(cap)
        imgs.append(pil_to_tensor(im, res=res))
        prompts.append(pr)
        texts.append(full)
        caps.append(cap)
    return {
        "image": torch.stack(imgs, dim=0),
        "prompt": prompts,
        "text": texts,
        "caption": caps,
    }


# Registry of recommended HF sources for this repo's scale
HF_VLM_DATASETS = {
    "flickr8k": {
        "id": "jxie/flickr8k",
        "task": "image-caption",
        "size": "~8k images, 5 caps each",
        "note": "Best first target: small, free, parquet with embedded images",
        "loader": "Flickr8kCaptionStore",
    },
    "flickr30k": {
        "id": "nlphuji/flickr30k",
        "task": "image-caption",
        "size": "~31k",
        "note": "zip images + csv; larger download",
    },
    "llava_pretrain_558k": {
        "id": "liuhaotian/LLaVA-Pretrain",
        "task": "image-caption / pretrain",
        "size": "558k",
        "note": "LLaVA stage-1 style; heavy download; projector pretrain SOTA recipe",
    },
    "coco_caps": {
        "id": "Multimodal-Fatima/COCO_captions_train",
        "task": "image-caption",
        "size": "large multi-shard parquet",
        "note": "Use 1–2 shards for experiments",
    },
    "pokemon_blip": {
        "id": "lambdalabs/pokemon-blip-captions",
        "task": "image-caption",
        "size": "~800",
        "note": "Gated — needs HF login; alt: svjack/pokemon-blip-captions-en-zh",
    },
}
