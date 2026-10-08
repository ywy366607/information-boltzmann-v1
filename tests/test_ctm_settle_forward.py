import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState, FlyBPTTLearner, step_fly_physical_tick, extract_fly_motor_latent
)

def test_ctm_forward():
    print("Testing CTM dual-anchor logic...")
    ckpt_path = Path('E:/ib_checkpoints/q8_fly_bptt32_hx1_residual_100k/last.pt')
    saved = torch.load(ckpt_path, map_location='cuda')
    cfg = saved['config']
    old = saved['learner']
    model = FlyReservoirLM(
        Path(cfg['graph']), vocab_size=50257, d_model=cfg['d_model'],
        injection='topographic', read_surface='output', synapse_model='coba',
        use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias'],
        read_centering=cfg.get('read_centering', False),
        use_read_gamma_trace=cfg.get('use_read_gamma_trace', False),
        init_read_gamma=cfg.get('init_read_gamma', 'anatomical')
    ).cuda()
    model.load_state_dict({k: v for k, v in saved['model'].items() if not k.startswith('graph_observer.') and not k.startswith('latent_predictor.')}, strict=False)
    state = FlyPhysicalState(**{key: tuple(t.to('cuda') for t in value) if key == 'ring' else value.to('cuda') for key, value in old['physical'].items()})
    del saved
    torch.cuda.empty_cache()

    W = 8
    S = 4
    K = 1 + S
    learner = FlyBPTTLearner(model, state, lr=2e-4, settle_ticks=S, writer_baseline_clock='physical', learn_stp=True)
    tokens = torch.randint(0, 50257, (1, W), device='cuda')
    targets = torch.randint(0, 50257, (1, W), device='cuda')
    learner.previous_token = 100

    rates, thresholds = model.get_decay_rates(), model.get_thresholds()
    gains, alif, stp = model.get_conductance_gains(), model.get_alif_params(), model.get_stp_params()
    options = dict(base_rates=rates, thresholds=thresholds, conductance_gains=gains, alif_params=alif, stp_params=stp)

    latents = []
    st = learner.state
    quiet_source = torch.zeros_like(st.h)

    for token in tokens.unbind(1):
        # Tick 0: external pulse
        drive, baseline = model.topographic_writer.forward_with_state(
            model.embedding(token), st.h, st.baseline)
        st = step_fly_physical_tick(model, st, token, drive, baseline, options,
                                    h_mean_decay=0.99, base_rates=rates)
        latents.append(extract_fly_motor_latent(model, st))

        # Quiet settling ticks 1..S
        for _ in range(S):
            quiet_baseline = model.topographic_writer.lambda_adapt * st.baseline
            st = step_fly_physical_tick(model, st, None, quiet_source, quiet_baseline, options,
                                        h_mean_decay=0.99, base_rates=rates)
            latents.append(extract_fly_motor_latent(model, st))

    # All latents: [W * K, d_model]
    all_latents = torch.cat(latents, dim=0)
    normed_features = model.read_norm(all_latents)
    logits = model.decoder(normed_features)  # [W * K, V]
    logits_window = logits.view(W, K, -1)     # [W, K, V]
    V = logits.size(-1)

    target_expanded = targets.view(W, 1).expand(W, K)
    loss_all = F.cross_entropy(logits_window.reshape(W * K, V), target_expanded.reshape(-1), reduction='none').view(W, K)

    log_p = F.log_softmax(logits_window, dim=-1)
    p = torch.exp(log_p)
    entropy = -(p * log_p).sum(dim=-1)
    log_V = math.log(V)
    certainty = 1.0 - entropy / log_V

    k_min_loss = loss_all.argmin(dim=1, keepdim=True)
    k_max_cert = certainty.argmax(dim=1, keepdim=True)

    loss_min = loss_all.gather(1, k_min_loss).squeeze(1)
    loss_cert = loss_all.gather(1, k_max_cert).squeeze(1)
    loss_dual = 0.5 * (loss_min + loss_cert)

    total_loss = loss_dual.mean()
    print("Total dual loss:", total_loss.item())
    print("k_min_loss:", k_min_loss.squeeze(1).tolist())
    print("k_max_cert:", k_max_cert.squeeze(1).tolist())
    print("Loss tick 0:", loss_all[:, 0].mean().item())
    print("Loss tick 4:", loss_all[:, 4].mean().item())
    print("Loss cert:", loss_cert.mean().item())
    print("Loss min:", loss_min.mean().item())

    total_loss.backward()
    active_grads = [p.grad is not None for p in learner.trainable if p.requires_grad]
    print(f"All {len(active_grads)} trainable parameters received gradients: {all(active_grads)}")

if __name__ == '__main__':
    test_ctm_forward()
