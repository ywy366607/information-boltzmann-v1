from pathlib import Path
import os, sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

from fine_grain.omni_model import DualStreamOmni
from fine_grain.unified_arch import UNIFIED_OMNI_SIZE
from scripts.train_omni_probe import eval_ports, render_gallery

dev = torch.device("cuda")
m = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE).to(dev)
m.load_state_dict(torch.load(ROOT / "checkpoints" / "omni_d256_unified_fm_t2i_where_best.pt", map_location="cpu"))
ev = eval_ports(m, np.random.default_rng(9001), 32, dev, n=48, kinds=["t2i"], flow_steps=8)["t2i"]
print(
    f"t2i-only n=48 psnr={ev['psnr']:.2f} bg={ev['bg_psnr']:.2f} "
    f"flood={ev['flood']*100:.1f} ink={ev['ink']*100:.1f}",
    flush=True,
)
render_gallery(
    m, np.random.default_rng(7), 32, dev,
    ROOT / "present" / "figs" / "omni_unified_fm_t2i_where_t2ionly_gallery.png",
    kinds=["t2i"] * 5, flow_steps=8,
)
