"""Long run of the written-birth bodies: registers them, then runs `cgb_language_long_run`.

Usage: python scripts/cgb_born_long_run.py gdn2_born --bytes 125000 --segment 125000 --json out.json [--set k=v ...]
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cgb_born_gdn2 as born  # noqa: E402
import cgb_language_long_run  # noqa: E402

if __name__ == "__main__":
    torch.set_num_threads(1)
    if "--set" in sys.argv:
        at = sys.argv.index("--set")
        for item in sys.argv[at + 1:]:
            key, value = item.split("=")
            born.GDN2Born.birth[key] = float(value)
        del sys.argv[at:]
    cgb_language_long_run.main()
