"""Compare Euler vs JiT-Heun on the x-pred t2i ckpt."""
from pathlib import Path
import json, os, sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ML_CACHE_ROOT", r"D:\ml_cache")

from fine_grain.omni_model import DualStreamOmni
from fine_grain.unified_arch import UNIFIED_OMNI_SIZE
from scripts.train_omni_probe import eval_ports, render_gallery

CKPT = ROOT / "checkpoints" / "omni_d256_unified_jit_t2i_best.pt"
OUT = ROOT / "results" / "published" / "omni_unified_jit_heun_vs_euler.json"


def main() -> None:
    dev = torch.device("cuda")
    m = DualStreamOmni.unified(**UNIFIED_OMNI_SIZE, fm_pred="x").to(dev)
    m.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    recs = {}
    for method, steps in (("euler", 8), ("heun", 8), ("heun", 20)):
        ev = eval_ports(
            m, np.random.default_rng(9001), 32, dev, n=48, kinds=["t2i"],
            flow_steps=steps, flow_method=method,
        )["t2i"]
        key = f"{method}_{steps}"
        recs[key] = ev
        print(
            f"  {key:10s} psnr={ev['psnr']:.2f} bg={ev['bg_psnr']:.2f} "
            f"str={ev['stroke_psnr']:.2f} flood={ev['flood']*100:.1f} ink={ev['ink']*100:.1f}",
            flush=True,
        )
    OUT.write_text(json.dumps({"ckpt": str(CKPT), "eval": recs}, indent=2), encoding="utf-8")
    render_gallery(
        m, np.random.default_rng(7), 32, dev,
        ROOT / "present" / "figs" / "omni_unified_jit_heun8_gallery.png",
        kinds=["t2i"] * 5, flow_steps=8, flow_method="heun",
    )
    print(f"saved {OUT}", flush=True)


if __name__ == "__main__":
    main()
