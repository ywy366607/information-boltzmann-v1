"""Generate configs for particle count scaling study N in [64, 128, 256, 512, 1024]."""
import json
from pathlib import Path

base_cfg = json.loads(Path("configs/information_boltzmann/smoke_owt.json").read_text(encoding="utf-8"))

for n in [64, 128, 256, 512, 1024]:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["model"]["particles"] = n
    cfg["train"]["supervised_targets"] = 5000
    cfg_file = Path(f"configs/information_boltzmann/owt_n{n}.json")
    cfg_file.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Created {cfg_file}")
