from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from fine_grain.active_gdn2 import ActiveInferenceGDN2


def run(seed: int, duplicate_address: bool = False) -> dict[str, object]:
    torch.manual_seed(seed)
    memory = ActiveInferenceGDN2(d_model=4, n_slots=4, res=4)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    tokens = torch.tensor(
        [[[1.0, -0.5, 0.25, 0.75], [-0.25, 0.5, 1.0, -1.0],
          [0.5, 0.25, -0.75, 0.125], [0.0, 1.0, -0.25, 0.5]]]
    )
    centers = memory.atlas.unsqueeze(0).clone()
    if duplicate_address:
        centers[:, 1] = centers[:, 0]
    keys = memory.address(tokens, centers)
    written = memory.assimilate(state, tokens, centers, torch.ones(1), update_motion=False)
    overwritten = memory.assimilate(
        written, -tokens, centers, torch.full((1,), 8.0), update_motion=False,
    )
    pairwise = torch.pdist(keys[0].detach())
    written_error = float(memory.association_probe_error(written, keys, tokens).detach().item())
    overwritten_error = float(memory.association_probe_error(overwritten, keys, tokens).detach().item())
    return {
        "seed": seed,
        "duplicate_address": duplicate_address,
        "address_rank": int(torch.linalg.matrix_rank(keys[0].detach()).item()),
        "address_pairwise_min_distance": float(pairwise.min().item()),
        "written_probe_error": written_error,
        "overwritten_probe_error": overwritten_error,
        "bridge_conditions": {
            "probe_injective": bool(pairwise.min() > 1e-6),
            "written_probe_recoverable": written_error <= 0.2,
            "overwritten_probe_unrecoverable": overwritten_error > 0.2,
        },
    }


rows = [run(seed) for seed in range(8)] + [run(0, duplicate_address=True)]
summary = {
    "positive_passes": sum(all(r["bridge_conditions"].values()) for r in rows[:8]),
    "positive_count": 8,
    "negative_duplicate_probe_injective": rows[-1]["bridge_conditions"]["probe_injective"],
}
out = {"experiment": "gdn_probe_bridge_sweep_20261030", "status": "mechanism_diagnostic_only", "summary": summary, "rows": rows}
p = ROOT / "results/published/gdn_probe_bridge_sweep_20261030.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
