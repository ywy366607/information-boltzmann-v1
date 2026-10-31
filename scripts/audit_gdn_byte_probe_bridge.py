from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from fine_grain.active_gdn2 import ActiveInferenceGDN2


def byte_features(text: str) -> tuple[torch.Tensor, list[int]]:
    raw = list(text.encode("utf-8"))
    raw = raw[:8]
    bits = [[1.0 if (value >> bit) & 1 else -1.0 for bit in range(8)] for value in raw]
    return torch.tensor(bits, dtype=torch.float32).unsqueeze(0), raw


def run(seed: int = 0, text: str = "language") -> dict[str, object]:
    torch.manual_seed(seed)
    memory = ActiveInferenceGDN2(d_model=8, n_slots=8, res=8)
    state = memory.initial_state(1, torch.device("cpu"), torch.float32)
    tokens, raw = byte_features(text)
    centers = memory.atlas.unsqueeze(0)
    keys = memory.address(tokens, centers)
    written = memory.assimilate(state, tokens, centers, torch.ones(1), update_motion=False)
    erased = memory.dynamic_prior(written, torch.tensor([64.0]))
    overwritten = memory.assimilate(
        written, -tokens, centers, torch.full((1,), 8.0), update_motion=False,
    )
    def metrics(st: object) -> dict[str, float]:
        return {
            "probe_error": float(memory.association_probe_error(st, keys, tokens).detach().item()),
            "mean_probe_variance": float(memory.association_probe_uncertainty(st, keys).detach().mean().item()),
        }
    key_matrix = keys[0].detach()
    gram = key_matrix @ key_matrix.transpose(-1, -2)
    gram_offdiag = gram - torch.diag(torch.diagonal(gram))
    pairwise = torch.pdist(key_matrix)
    written_m = metrics(written)
    dual_written_error = float((memory.association_dual_readout(written, keys) - tokens).abs().amax().detach().item())
    overwritten_m = metrics(overwritten)
    return {
        "seed": seed,
        "bytes": raw,
        "text": text,
        "address_rank": int(torch.linalg.matrix_rank(keys[0].detach()).item()),
        "address_pairwise_min_distance": float(pairwise.min().item()),
        "gram_max_offdiag_abs": float(gram_offdiag.abs().max().item()),
        "gram_condition_number": float(torch.linalg.cond(gram).item()),
        "written": written_m,
        "dual_written_probe_error": dual_written_error,
        "erased": metrics(erased),
        "overwritten": overwritten_m,
        "bridge_conditions": {
            "probe_injective": bool(pairwise.min() > 1e-6),
            "written_probe_recoverable": written_m["probe_error"] <= 0.2,
            "dual_written_probe_recoverable": dual_written_error <= 0.2,
            "overwritten_probe_unrecoverable": overwritten_m["probe_error"] > 0.2,
        },
    }

out = {"experiment": "gdn_byte_probe_bridge_20261030", "status": "mechanism_diagnostic_only", "result": run()}
p = ROOT / "results/published/gdn_byte_probe_bridge_20261030.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
