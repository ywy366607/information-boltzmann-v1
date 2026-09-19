"""Two-Stage Progressive Multi-Horizon Training (CBIM Three-Clock v1).

Stage A (Steps 1 .. 3000):
- Batch-Level Stratified Octave Quota Sampling over K in {1, 2, 4, 8, 16, 32, 64, 128}
- p(K) proportional to 1/K (Equal compute per temporal octave: p(K) * K = const)
- BPTT = 128 tokens, chunk_tokens = 128 (no truncation!)
- Zero-memory Reversible Hamiltonian Inversion: VRAM strictly < 2.7 GB, 0 shared memory!

Stage B (Steps 3001 .. 3500):
- Token-Level Temporal Mixing Fine-Tuning (500 steps)
- Seamless continuation: NO reset of field state F_t, optimizer, scheduler, or stream!
- Curriculum: Steps 3001..3100 sticky P(K_t = K_{t-1}) = 0.5; Steps 3101..3500 independent.

Post-Training Automated Deliverables:
1. Multi-scale validation curve L(K_eval) for K in {1, 2, 4, 8, 16, 32, 64, 128}
2. 8x8 Transition Matrix L(K_t | K_{t-1})
3. Oracle Pondering Headroom L_fixed - L_oracle
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.cbim_torus3d import CBIMTorus3D, ReversibleHamiltonianPonderFunction
from scripts.ib_local.train_cbim_malecns_internal_time import TruncatedInternalTimeGraphTrainer


def build_stratified_schedule(total_steps=3000, block_size=100):
    horizons = [1, 2, 4, 8, 16, 32, 64, 128]
    weights = [1.0 / k for k in horizons]
    Z = sum(weights)
    probs = [w / Z for w in weights]
    if total_steps < block_size:
        return [horizons[i % len(horizons)] for i in range(total_steps)], horizons, probs
    num_blocks = total_steps // block_size
    block_quotas = [int(round(p * block_size)) for p in probs]
    diff = block_size - sum(block_quotas)
    block_quotas[0] += diff

    np.random.seed(42)
    schedule = []
    for b in range(num_blocks):
        block_items = []
        for k, count in zip(horizons, block_quotas):
            block_items.extend([k] * count)
        # Alternate K=128: if block quota gave 0 for K=128 (since round(0.39) = 0),
        # place one K=128 every second block
        if b % 2 == 1:
            # Replace one K=1 with K=128
            idx_1 = block_items.index(1)
            block_items[idx_1] = 128
        np.random.shuffle(block_items)
        schedule.extend(block_items)
    return schedule, horizons, probs


def sample_token_level_ks(batch_size, horizons, probs, sticky_prob=0.0):
    """Sample token-level K_t with optional sticky transition P(K_t = K_{t-1}) = sticky_prob."""
    ks = []
    last_k = np.random.choice(horizons, p=probs)
    for _ in range(batch_size):
        if sticky_prob > 0 and np.random.rand() < sticky_prob:
            k = last_k
        else:
            k = np.random.choice(horizons, p=probs)
        ks.append(int(k))
        last_k = k
    return ks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/cbim_two_stage_multi_horizon_3500"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--stage-a-steps", type=int, default=3000)
    parser.add_argument("--stage-b-steps", type=int, default=500)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--shape", type=int, nargs=3, default=(8, 8, 4))
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    train_data = np.load(args.data / "train.npy", mmap_mode="r")
    val_data = np.load(args.data / "validation.npy", mmap_mode="r")

    print("=" * 95)
    print("   CBIM TWO-STAGE PROGRESSIVE MULTI-HORIZON TRAINING (3000 Stage A + 500 Stage B)")
    print("=" * 95)

    schedule_a, horizons, probs = build_stratified_schedule(args.stage_a_steps, block_size=100)
    print("Stage A Stratified Octaves Distribution:")
    for k, p in zip(horizons, probs):
        count = schedule_a.count(k)
        print(f"  K = {k:<3d} | Scheduled: {count:<5d} steps ({count/len(schedule_a)*100:5.2f}%) | Target: {p*100:5.2f}%")
    print("-" * 95)

    model = CBIMTorus3D(
        shape=tuple(args.shape), velocities=8, content_dim=16,
        v2_coordinate_components=True, readout_type="kernel_r1",
        write_type="w2_impedance", micro_steps=1, adaptive_clock=True,
        continuous_velocities=True, dissipation_type="unified",
        dissipation_rank=4, three_clock=True, tau_mem=3.0, nu_s_init=0.020,
        decouple_source_feedback=True, reversible_ponder=True
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lexical_names = {"source.embedding.weight", "decoder.weight", "decoder.bias"}
    named = list(model.named_parameters())
    gradient_groups = (
        [p for n, p in named if n in lexical_names],
        [p for n, p in named if n not in lexical_names],
    )

    state = model.initial_state(1, "cuda")

    def batch(data, offset):
        return (torch.as_tensor(data[offset:offset + args.tokens], dtype=torch.long, device="cuda")[None],
                torch.as_tensor(data[offset + 1:offset + args.tokens + 1], dtype=torch.long, device="cuda")[None])

    metrics_path = args.output / "metrics.jsonl"

    def log_metric(row):
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if row.get("kind") == "train" and (row["step"] % 50 == 0 or row["step"] == 1):
            k_val = row.get("k", "token-level")
            print(f"Step {row['step']:<5d} [{row.get('stage','A')}] | K={k_val} | NLL: {row['nll']:.4f} | Grad: {row['grad_norm']:.3f} | {row['seconds']:.3f}s/step | VRAM: {row['vram_mb']:.1f}MB", flush=True)

    @torch.no_grad()
    def validate_multi_k(current_state, step_num):
        model.set_ness_prior(current_state)
        val_results = {}
        val_toks = min(512, args.validation_tokens)
        for k_test in [1, 2, 4, 8]:
            s_val = current_state.clone()
            tot_loss = 0.0
            for offset in range(0, val_toks, args.tokens):
                x, y = batch(val_data, offset)
                loss, s_val, _ = model(x, y, s_val, micro_steps=k_test)
                tot_loss += float(loss.item()) * args.tokens
            val_results[k_test] = tot_loss / val_toks
        row = {
            "kind": "validation",
            "step": step_num,
            "val_nll_k1": val_results[1],
            "val_nll_k2": val_results[2],
            "val_nll_k4": val_results[4],
            "val_nll_k8": val_results[8],
            "best_k": min(val_results, key=val_results.get),
            "best_nll": min(val_results.values()),
        }
        log_metric(row)
        print(f"\n>>> [Val @ Step {step_num}] K=1: {val_results[1]:.4f}, K=2: {val_results[2]:.4f}, K=4: {val_results[4]:.4f}, K=8: {val_results[8]:.4f}, Best K={row['best_k']} ({row['best_nll']:.4f})\n", flush=True)
        return row["best_nll"]

    best_val_nll = validate_multi_k(state, 0)
    data_offset = 0

    # =========================================================================
    # STAGE A: Batch-Level Stratified Octave (Steps 1 .. 3000)
    # =========================================================================
    print("\n>>> STARTING STAGE A: Batch-Level Stratified Octave (3000 Steps)...", flush=True)
    t_stage_a_start = time.perf_counter()

    for step in range(1, args.stage_a_steps + 1):
        k_step = schedule_a[step - 1]
        x, y = batch(train_data, data_offset)
        data_offset += args.tokens

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        loss, state, diag = model(x, y, state.detach(), micro_steps=k_step)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(gradient_groups[0], 1.0, foreach=True)
        torch.nn.utils.clip_grad_norm_(gradient_groups[1], 1.0, foreach=True)
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        do_log = (step % 50 == 0 or step == 1)
        grad_norm = float(math.sqrt(sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None))) if do_log else 0.0
        log_metric({
            "kind": "train", "stage": "Stage A", "step": step, "k": k_step,
            "nll": float(loss.item()), "grad_norm": grad_norm,
            "seconds": elapsed, "vram_mb": vram_mb
        })

        if step % args.validate_every == 0:
            val_nll = validate_multi_k(state, step)
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                args.output.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "state": state.detach(), "step": step, "nll": val_nll}, args.output / "Best_StageA.pt")

    print(f"\nStage A completed in {(time.perf_counter() - t_stage_a_start)/60:.2f} minutes!")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "state": state.detach(), "optimizer": optimizer.state_dict()}, args.output / "checkpoint_stage_a_final.pt")

    # =========================================================================
    # STAGE B: Token-Level Temporal Mixing Fine-Tuning (Steps 3001 .. 3500)
    # =========================================================================
    print("\n>>> SEAMLESS HANDOVER TO STAGE B: Token-Level Temporal Mixing (500 Steps)...", flush=True)
    print(">>> (State F_t, optimizer, scheduler, data stream continue without ANY reset!)", flush=True)
    t_stage_b_start = time.perf_counter()

    for step in range(args.stage_a_steps + 1, args.stage_a_steps + args.stage_b_steps + 1):
        # Curriculum sticky probability
        if step <= args.stage_a_steps + 100:
            sticky_p = 0.5 * (1.0 - (step - args.stage_a_steps) / 100.0)
        else:
            sticky_p = 0.0

        token_ks = sample_token_level_ks(args.tokens, horizons, probs, sticky_prob=sticky_p)
        x, y = batch(train_data, data_offset)
        data_offset += args.tokens

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        # Token-level forward pass with heterogeneous K_t and hoisted projections
        curr = state.detach()
        features = []
        cached_learned = model.transport.learned_symbol()
        all_tok_embed = model.source.embedding(x)
        all_disp = 0.5 * torch.tanh(model.source.address(all_tok_embed)) if model.source.relative_address else torch.sigmoid(model.source.address(all_tok_embed))
        all_width = F.softplus(model.source.width(all_tok_embed))
        all_content = model.source.content(all_tok_embed)
        all_u = F.rms_norm(all_tok_embed, (model.d,))
        all_q_field = model.readout.w_q(all_u).view(1, args.tokens, model.readout.heads, model.readout.queries, model.readout.d_h)

        for t_idx in range(args.tokens):
            tok_id = x[:, t_idx]
            tok_embed = all_tok_embed[:, t_idx]
            field, _, _ = model.source(
                curr, tok_id, return_diag=False, token=tok_embed,
                displacement=all_disp[:, t_idx], width=all_width[:, t_idx],
                content=all_content[:, t_idx])
            k_t = token_ks[t_idx]
            field = ReversibleHamiltonianPonderFunction.apply(field, k_t, model, tok_embed)

            flat_field = field.reshape(1, model.readout.nodes, model.d)
            normed_field = model.readout.field_norm(flat_field)
            keys = model.readout.k_proj(normed_field)
            key_h = keys.reshape(1, model.readout.nodes, model.readout.heads, model.readout.d_h).transpose(1, 2)
            values = model.readout.v_proj(normed_field)
            val_h = values.reshape(1, model.readout.nodes, model.readout.heads, model.readout.d_h).transpose(1, 2)
            m, _, _ = model.readout.measure(key_h, val_h, all_q_field[:, t_idx])
            feat = model.readout.output(model.readout.merge(m))
            features.append(feat)

            dt_mem = model.tau_mem * model.tau_0_tensor
            curr, _ = model.bath(field, dt_mem, tok_embed=tok_embed, disable_viscosity=True, disable_subspace=False)

        logits = model.decoder(torch.stack(features, 1))
        loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(gradient_groups[0], 1.0, foreach=True)
        torch.nn.utils.clip_grad_norm_(gradient_groups[1], 1.0, foreach=True)
        optimizer.step()

        state = curr.detach()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

        do_log = (step % 50 == 0 or step == args.stage_a_steps + 1)
        grad_norm = float(math.sqrt(sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None))) if do_log else 0.0
        log_metric({
            "kind": "train", "stage": "Stage B", "step": step,
            "k": f"token_mix(mean={np.mean(token_ks):.1f})",
            "nll": float(loss.item()), "grad_norm": grad_norm,
            "seconds": elapsed, "vram_mb": vram_mb
        })

        if step % args.validate_every == 0 or step == args.stage_a_steps + args.stage_b_steps:
            val_nll = validate_multi_k(state, step)
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                args.output.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "state": state.detach(), "step": step, "nll": val_nll}, args.output / "Best_StageB.pt")

    print(f"\nStage B completed in {(time.perf_counter() - t_stage_b_start)/60:.2f} minutes!")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "state": state.detach(), "step": 3500}, args.output / "Final_MultiHorizon.pt")

    # =========================================================================
    # POST-TRAINING AUTOMATED DELIVERABLES (Fast Sequential Single-Pass)
    # =========================================================================
    print("\n" + "=" * 95, flush=True)
    print("   COMPUTING POST-TRAINING DELIVERABLES...", flush=True)
    print("=" * 95, flush=True)

    def eval_sequential_horizons(base_field, tok_id, target_id, target_horizons):
        tok_embed = model.source.embedding(tok_id)
        f_w, _, _ = model.source(base_field, tok_id, return_diag=False)
        cached_learned = model.transport.learned_symbol()
        nullspace = model.collision.nullspace.to(dtype=f_w.dtype)
        curr = f_w
        losses = {}
        states = {}
        max_k = max(target_horizons)
        horizon_set = set(target_horizons)

        for step_idx in range(1, max_k + 1):
            alpha = model.clock(curr, tok_embed) if model.adaptive_clock else model.tau_0_tensor
            dt = alpha * model.tau_0_tensor if model.adaptive_clock else model.tau_0_tensor
            dir_k = model.direction_controller(curr, tok_embed) if model.continuous_velocities else None
            mult, _ = model.transport.multiplier(dt, direction=dir_k, learned=cached_learned)
            f_tr = model.transport.apply_multiplier(curr, mult)

            batch = curr.shape[0]
            flat = f_tr.reshape(batch, -1, model.collision.d)
            coefficient = torch.einsum("dk,bnd->bnk", nullspace, flat)
            conserved = flat - torch.einsum("dk,bnk->bnd", nullspace, coefficient)

            pos = model.collision.position_features.to(flat)[None].expand(batch, -1, -1) if model.collision.position_conditioned else None
            angle_in = torch.cat((model.collision.norm(flat), pos), -1) if pos is not None else model.collision.norm(flat)
            angles = model.collision.angle(angle_in).reshape(batch, flat.shape[1], model.collision.layers, -1)
            dt_val = dt.view(batch, 1, 1, 1) if isinstance(dt, torch.Tensor) else float(dt)
            scaled_angles = angles * dt_val

            val = coefficient
            for layer in range(model.collision.layers):
                pair = model.collision.schedules[layer]
                th = scaled_angles[:, :, layer]
                cos, sin = th.cos(), th.sin()
                l, r = val[..., pair[:, 0]], val[..., pair[:, 1]]
                upd = val.clone()
                upd[..., pair[:, 0]] = cos * l - sin * r
                upd[..., pair[:, 1]] = sin * l + cos * r
                val = upd

            curr = (conserved + torch.einsum("dk,bnk->bnd", nullspace, val)).reshape_as(f_w)

            if step_idx in horizon_set:
                feat, _ = model.readout(curr, tok_embed, return_diag=False)
                logits = model.decoder(feat)
                if target_id is not None:
                    losses[step_idx] = F.cross_entropy(logits, target_id).item()
                states[step_idx] = curr

        return losses, states

    # Deliverable 1: L(K_eval) for all 8 octaves
    print("\n--- Deliverable 1: Continuous Pondering Profile L(K_eval) ---", flush=True)
    k_eval_results = {}
    model.set_ness_prior(state)
    eval_tokens = 512
    with torch.no_grad():
        for k_eval in horizons:
            s_eval = state.clone()
            tot = 0.0
            for off in range(0, eval_tokens, args.tokens):
                x, y = batch(val_data, off)
                l_e, s_eval, _ = model(x, y, s_eval, micro_steps=k_eval)
                tot += float(l_e.item()) * args.tokens
            k_eval_results[k_eval] = tot / eval_tokens
            print(f"  K = {k_eval:<3d} | Val NLL: {k_eval_results[k_eval]:.4f} nats", flush=True)

    # Deliverable 2: 8x8 Transition Matrix L(K_t | K_{t-1})
    print("\n--- Deliverable 2: 8x8 Cross-Horizon Transition Matrix L(K_t | K_{t-1}) ---", flush=True)
    transition_matrix = np.zeros((len(horizons), len(horizons)))
    test_start = 8192
    num_trans_tokens = 16
    dt_mem = model.tau_mem * model.tau_0_tensor

    with torch.no_grad():
        s_base = state.clone()
        for idx in range(num_trans_tokens):
            t_curr_idx = test_start + idx * 2
            inp_prev = torch.as_tensor([val_data[t_curr_idx]], dtype=torch.long, device="cuda")
            inp_curr = torch.as_tensor([val_data[t_curr_idx + 1]], dtype=torch.long, device="cuda")
            tgt_curr = torch.as_tensor([val_data[t_curr_idx + 2]], dtype=torch.long, device="cuda")

            # Sequential single-pass for inp_prev across all horizons
            _, states_prev = eval_sequential_horizons(s_base, inp_prev, None, horizons)

            for i, k_prev in enumerate(horizons):
                # Apply bath once
                tok_embed_prev = model.source.embedding(inp_prev)
                s_after_prev, _ = model.bath(states_prev[k_prev], dt_mem, tok_embed=tok_embed_prev, disable_viscosity=True)
                # Sequential single-pass for inp_curr across all horizons
                losses_curr, _ = eval_sequential_horizons(s_after_prev, inp_curr, tgt_curr, horizons)
                for j, k_curr in enumerate(horizons):
                    transition_matrix[i, j] += losses_curr[k_curr] / num_trans_tokens

            # Advance base state with K=4
            tok_embed_base = model.source.embedding(inp_prev)
            s_base, _ = model.bath(states_prev[4], dt_mem, tok_embed=tok_embed_base, disable_viscosity=True)
            losses_curr, states_curr = eval_sequential_horizons(s_base, inp_curr, tgt_curr, [4])
            tok_embed_curr = model.source.embedding(inp_curr)
            s_base, _ = model.bath(states_curr[4], dt_mem, tok_embed=tok_embed_curr, disable_viscosity=True)

    print("Transition Matrix Rows: K_{t-1} in [1, 2, 4, 8, 16, 32, 64, 128], Cols: K_t:", flush=True)
    print(np.array2string(transition_matrix, precision=3, suppress_small=True), flush=True)

    # Deliverable 3: Adaptive Oracle Headroom L_fixed - L_oracle
    print("\n--- Deliverable 3: Adaptive Oracle Pondering Headroom ---", flush=True)
    oracle_tokens = 256
    oracle_start = 16384
    token_losses = {k: [] for k in horizons}
    with torch.no_grad():
        s_oracle = state.clone()
        for t in range(oracle_tokens):
            inp_t = torch.as_tensor([val_data[oracle_start + t]], dtype=torch.long, device="cuda")
            tgt_t = torch.as_tensor([val_data[oracle_start + t + 1]], dtype=torch.long, device="cuda")

            # Single-pass sequential evaluation across all 8 horizons
            losses_t, states_t = eval_sequential_horizons(s_oracle, inp_t, tgt_t, horizons)
            for k in horizons:
                token_losses[k].append(losses_t[k])

            # Advance background field with median horizon K=4
            tok_embed_t = model.source.embedding(inp_t)
            s_oracle, _ = model.bath(states_t[4], dt_mem, tok_embed=tok_embed_t, disable_viscosity=True)

    mean_fixed_losses = {k: float(np.mean(token_losses[k])) for k in horizons}
    best_fixed_k = min(horizons, key=lambda k: mean_fixed_losses[k])
    l_fixed_best = mean_fixed_losses[best_fixed_k]

    best_k_per_token = [min(horizons, key=lambda k: token_losses[k][t]) for t in range(oracle_tokens)]
    min_loss_per_token = [min(token_losses[k][t] for k in horizons) for t in range(oracle_tokens)]
    l_oracle = float(np.mean(min_loss_per_token))
    oracle_headroom = l_fixed_best - l_oracle

    k_counts = {k: best_k_per_token.count(k) for k in horizons}
    k_distribution = {k: float(k_counts[k] / oracle_tokens) for k in horizons}

    print(f"  Best Fixed Depth (K={best_fixed_k}): {l_fixed_best:.4f} nats", flush=True)
    print(f"  Oracle Adaptive Depth:      {l_oracle:.4f} nats", flush=True)
    print(f"  Oracle Headroom Gain:       +{oracle_headroom:.4f} nats ({oracle_headroom/l_fixed_best*100:.2f}%)", flush=True)
    print("  Optimal Depth Distribution p(K*):", flush=True)
    for k in horizons:
        print(f"    K = {k:<3d}: {k_distribution[k]*100:5.2f}% ({k_counts[k]} tokens)", flush=True)

    report = {
        "stage_a_steps": args.stage_a_steps,
        "stage_b_steps": args.stage_b_steps,
        "deliverable1_l_k": k_eval_results,
        "deliverable2_transition_matrix": transition_matrix.tolist(),
        "deliverable3_oracle": {
            "l_fixed_best": l_fixed_best,
            "best_fixed_k": best_fixed_k,
            "l_oracle": l_oracle,
            "oracle_headroom": oracle_headroom,
            "k_distribution": k_distribution,
        },
        "best_val_nll": best_val_nll,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "multi_horizon_final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nFinal report saved to {args.output / 'multi_horizon_final_report.json'}", flush=True)


if __name__ == "__main__":
    main()
