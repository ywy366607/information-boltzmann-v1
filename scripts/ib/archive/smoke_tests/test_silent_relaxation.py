"""Test single-injection autonomous relaxation and delayed information metrics.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:64"

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core import fly_reservoir as reservoir
from information_boltzmann.core.fly_bptt_learning import advance_fly_input_event, FlyPhysicalState
from scripts.ib.run_full_diagnosis import load_checkpoint, clone_state

def quiet_step(model, s, dummy_token, options):
    h, spike, ring, ge, gi, b, x, u = model.step(
        s.h, dummy_token, spike_ring=s.ring, ge=s.ge, gi=s.gi, b=s.b,
        x=s.x, u=s.u, sensory_drive=torch.zeros_like(s.h), **options
    )
    hm = 0.99 * s.h_mean + 0.01 * h
    read = (h - hm)[:, model.read_indices] if model.read_centering else h[:, model.read_indices]
    a = model.get_read_gamma_decay()
    z1 = a * s.gamma_z1 + (1.0 - a) * read
    z2 = a * s.gamma_z2 + (1.0 - a) * z1
    return FlyPhysicalState(h, ring, ge, gi, b, x, u, s.baseline, hm, s.dan_gate, z1, z2), spike

def test_quick():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("artifacts/active_training/q8_fly_bptt32_gamma_2333_wide115_100k/last.pt")
    saved, learner = load_checkpoint(ckpt_path, device=device)
    model = learner.model
    s_init = clone_state(learner.state)

    graph_data = np.load(ROOT / saved["config"]["graph"])
    s_names = list(graph_data["superclass_names"])
    superclass_id = graph_data["superclass_id"]

    motor_indices = np.flatnonzero(np.isin(superclass_id, [s_names.index(c) for c in model.OUTPUT_CLASSES if c in s_names]))
    sensory_indices = model.topographic_writer.injection_index.cpu().numpy()
    intrinsic_indices = np.setdiff1d(np.arange(model.n_neurons), np.union1d(motor_indices, sensory_indices))

    motor_idx_t = torch.as_tensor(motor_indices, device=device)
    sensory_idx_t = torch.as_tensor(sensory_indices, device=device)
    intrinsic_idx_t = torch.as_tensor(intrinsic_indices, device=device)

    rates, thresholds = model.get_decay_rates(), model.get_thresholds()
    gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()
    options = dict(base_rates=rates, thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)

    # Let's inspect the static unigram distribution from decoder bias
    with torch.no_grad():
        bias_logits = model.decoder.bias.unsqueeze(0)
        bias_probs = F.softmax(bias_logits, dim=-1)
        bias_entropy = -(bias_probs * F.log_softmax(bias_logits, dim=-1)).sum().item()
        print(f"Decoder Bias Unigram Entropy: {bias_entropy:.4f} nats ({bias_entropy / np.log(2):.2f} bits)")

    # Test an injection of token x0, followed by 32 ticks of silence
    data = np.load(ROOT / saved["config"]["data"] / "train.npy", mmap_mode="r")
    cursor = int(saved["train_cursor"])
    x0_val = int(data[cursor])
    x1_val = int(data[cursor + 1])
    print(f"Testing OWT sequence at cursor {cursor}: x0={x0_val}, x1={x1_val}")
    print(f"Static unigram NLL for x0: {-F.log_softmax(bias_logits, dim=-1)[0, x0_val].item():.4f}")
    print(f"Static unigram NLL for x1: {-F.log_softmax(bias_logits, dim=-1)[0, x1_val].item():.4f}")

    x0 = torch.tensor([x0_val], device=device)
    dummy = torch.tensor([x0_val], device=device)

    # First, run a quiet control (no injection) to get baseline trajectory
    s_control = clone_state(s_init)
    control_h = []
    control_z2 = []
    with torch.no_grad():
        for tau in range(33):
            s_control, sp = quiet_step(model, s_control, dummy, options)
            control_h.append(s_control.h.clone())
            control_z2.append(s_control.gamma_z2.clone())

    # Now, run injection at tau=0, followed by quiet
    s_inj = clone_state(s_init)
    with torch.no_grad():
        # Inject at tau=0
        s_inj = advance_fly_input_event(model, s_inj, x0, settle_ticks=0, writer_baseline_clock="input", **options)

        print("\nLag tau | Sensory dH | Intrins dH | Motor dH | Gamma dZ2 | NLL(x0) | NLL(x1) | Logit Diff")
        print("-" * 80)
        for tau in range(33):
            if tau > 0:
                s_inj, sp = quiet_step(model, s_inj, dummy, options)

            diff_h = s_inj.h - control_h[tau]
            diff_z2 = s_inj.gamma_z2 - control_z2[tau]

            dh_sensory = diff_h[:, sensory_idx_t].norm().item()
            dh_intrins = diff_h[:, intrinsic_idx_t].norm().item()
            dh_motor = diff_h[:, motor_idx_t].norm().item()
            dz2 = diff_z2.norm().item()

            feat = model.output_read(s_inj.gamma_z2)
            logits = model.decoder(model.read_norm(feat))
            nll_x0 = F.cross_entropy(logits, torch.tensor([x0_val], device=device)).item()
            nll_x1 = F.cross_entropy(logits, torch.tensor([x1_val], device=device)).item()

            feat_ctrl = model.output_read(control_z2[tau])
            logits_ctrl = model.decoder(model.read_norm(feat_ctrl))
            d_logit = (logits - logits_ctrl).norm().item()

            print(f"{tau:7d} | {dh_sensory:10.2f} | {dh_intrins:10.2f} | {dh_motor:8.2f} | {dz2:9.3f} | {nll_x0:7.4f} | {nll_x1:7.4f} | {d_logit:10.3f}")

if __name__ == "__main__":
    test_quick()
