"""Paper-trained x-pred ckpt, sampled from Gaussian (OOD). Then you train for real."""
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

CKPT = ROOT / "checkpoints" / "omni_d256_unified_jit_t2i_best.pt"


def main() -> None:
    dev = torch.device("cuda")
    m = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE, fm_pred="x").to(dev)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    ev = eval_ports(
        m, np.random.default_rng(9001), 32, dev, n=32, kinds=["t2i"],
        flow_steps=8, flow_method="heun", fm_x0="noise",
    )["t2i"]
    print(
        f"OOD paper-ckpt from NOISE  psnr={ev['psnr']:.2f} bg={ev['bg_psnr']:.2f} "
        f"flood={ev['flood']*100:.1f} ink={ev['ink']*100:.1f}  "
        f"(PSNR vs a *different* random paper — expected low)",
        flush=True,
    )
    render_gallery(
        m, np.random.default_rng(7), 32, dev,
        ROOT / "present" / "figs" / "omni_jit_paperckpt_from_noise_gallery.png",
        kinds=["t2i"] * 5, flow_steps=8, flow_method="heun", fm_x0="noise",
    )


if __name__ == "__main__":
    main()
