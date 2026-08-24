"""Real apple photos + captions. Cache on D:\\ml_cache. No digit/stroke hacks."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image

from fine_grain.hf_caption_data import Flickr8kCaptionStore, pil_to_tensor

CACHE = Path(os.environ.get("ML_CACHE_ROOT", r"D:\ml_cache")) / "apple_photos"

# Direct Wikimedia / commons stills (real apples, not drawings).
_WIKI = [
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Red_Apple.jpg/640px-Red_Apple.jpg",
        "a red apple",
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Honeycrisp.jpg/640px-Honeycrisp.jpg",
        "a red apple",
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Red_Delicious_apple.jpg/640px-Red_Delicious_apple.jpg",
        "a red delicious apple",
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Granny_smith.jpg/640px-Granny_smith.jpg",
        "a green apple",
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/Apple_01.jpg/640px-Apple_01.jpg",
        "an apple",
    ),
    (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Pink_lady_and_cross_section.jpg/640px-Pink_lady_and_cross_section.jpg",
        "a pink apple",
    ),
]


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        return True
    try:
        import certifi
        import ssl
        ctx = ssl.create_default_context(cafile=certifi.where())
        req = urllib.request.Request(url, headers={"User-Agent": "fine-grain-vision/1.0"})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest.stat().st_size > 1000
    except Exception as e:
        print(f"  apple download fail {url}: {e}", flush=True)
        return False


def flickr_apples(max_n: int = 80) -> List[Tuple[Image.Image, str]]:
    out: List[Tuple[Image.Image, str]] = []
    try:
        store = Flickr8kCaptionStore(split="train")
    except Exception as e:
        print(f"  flickr8k skip: {e}", flush=True)
        return out
    import re
    for i in range(len(store)):
        img, cap = store.get(i)
        c = cap.lower()
        if "applebee" in c:
            continue
        if re.search(r"\bapples?\b", c) is None:
            continue
        if "leap" in c:
            continue
        out.append((img.convert("RGB"), cap))
        if len(out) >= max_n:
            break
    return out


def wiki_apples() -> List[Tuple[Image.Image, str]]:
    out = []
    for i, (url, cap) in enumerate(_WIKI):
        dest = CACHE / f"wiki_{i}.jpg"
        if not _download(url, dest):
            continue
        try:
            img = Image.open(dest).convert("RGB")
        except Exception:
            continue
        out.append((img, cap))
        out.append((img, "an apple"))
        out.append((img, "a photo of an apple"))
    return out


def load_apple_pairs(res: int = 64, max_n: int = 96) -> dict:
    """image [N,3,R,R] in [0,1], list of captions."""
    pairs = flickr_apples(max_n=max_n) + wiki_apples()
    if not pairs:
        raise RuntimeError("no apple photos (flickr+wiki failed)")
    imgs, caps = [], []
    for im, cap in pairs:
        imgs.append(pil_to_tensor(im, res=res))
        caps.append(cap)
    x = torch.stack(imgs, dim=0)
    print(f"[apple] {len(caps)} (image, caption) pairs res={res}", flush=True)
    for c in caps[:8]:
        print(f"  - {c}", flush=True)
    return {"image": x, "caption": caps}


def sample_batch(store: dict, idxs, signed: bool = True) -> dict:
    imgs = store["image"][list(idxs)]
    caps = [store["caption"][int(i)] for i in idxs]
    if signed:
        imgs = imgs * 2.0 - 1.0
    return {
        "image": imgs,
        "target_rgb": imgs.clone(),
        "prompt": list(caps),
        "need_pix": [True] * len(idxs),
        "need_text": [False] * len(idxs),
        "answer": ["0"] * len(idxs),
        "stroke": torch.zeros(len(idxs), imgs.shape[2], imgs.shape[3]),
    }
