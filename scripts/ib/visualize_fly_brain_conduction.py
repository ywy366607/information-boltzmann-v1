"""Spatiotemporal Signal Conduction and 3D Neuropil Visualization for Fly CNS.

Loads the MaleCNS v1.0 connectome (165,122 neurons, 25.3M synapses) and trained
FlyReservoirLM physical state (S=14 settling ticks, COBA ALIF+STP), simulates
signal propagation following an input token pulse, and produces:
1. present/fly_conduction_spatiotemporal.png: High-resolution multi-panel publication figure
2. present/fly_brain_3d_conduction.html: Standalone interactive 3D WebGL viewer with time playback
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import tiktoken
import plotly.graph_objects as go

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    advance_fly_input_event,
    extract_fly_motor_latent,
)


# Canonical Drosophila anatomical & functional divisions
SYSTEM_CONFIG = [
    {
        'name': 'Sensory: Visual',
        'color': '#00E5FF',       # Electric Cyan
        'match': lambda df, cl, sc: cl == 'visual',
        'order': 0,
        'category': 'Sensory',
    },
    {
        'name': 'Sensory: Olfactory',
        'color': '#FF9100',       # Vibrant Orange
        'match': lambda df, cl, sc: cl == 'olfactory',
        'order': 1,
        'category': 'Sensory',
    },
    {
        'name': 'Sensory: Mechanosensory',
        'color': '#76FF03',       # Bright Lime
        'match': lambda df, cl, sc: np.char.startswith(cl.astype(str), 'mechano'),
        'order': 2,
        'category': 'Sensory',
    },
    {
        'name': 'Relay: Antennal Lobe PN',
        'color': '#FF6E40',       # Coral
        'match': lambda df, cl, sc: cl == 'ALPN',
        'order': 3,
        'category': 'Relay',
    },
    {
        'name': 'Relay: Visual Projection',
        'color': '#2979FF',       # Deep Sky Blue
        'match': lambda df, cl, sc: sc == 'visual_projection',
        'order': 4,
        'category': 'Relay',
    },
    {
        'name': 'Relay: Ascending (VNC->Brain)',
        'color': '#00B0FF',       # Light Blue
        'match': lambda df, cl, sc: sc == 'ascending_neuron',
        'order': 5,
        'category': 'Relay',
    },
    {
        'name': 'Associative: Mushroom Body (KC)',
        'color': '#FFD600',       # Gold
        'match': lambda df, cl, sc: cl == 'Kenyon_Cell',
        'order': 6,
        'category': 'Associative',
    },
    {
        'name': 'Associative: MB Output (MBON)',
        'color': '#FF1744',       # Magenta/Red
        'match': lambda df, cl, sc: cl == 'MBON',
        'order': 7,
        'category': 'Associative',
    },
    {
        'name': 'Central Complex: Navigation (CX)',
        'color': '#E040FB',       # Violet
        'match': lambda df, cl, sc: cl == 'CX',
        'order': 8,
        'category': 'Central',
    },
    {
        'name': 'Neuromodulatory: Dopaminergic (DAN)',
        'color': '#F50057',       # Hot Pink
        'match': lambda df, cl, sc: cl == 'DAN',
        'order': 9,
        'category': 'Central',
    },
    {
        'name': 'Central Brain Intrinsic',
        'color': '#1DE9B6',       # Turquoise
        'match': lambda df, cl, sc: (sc == 'cb_intrinsic') & (~np.isin(cl, ['Kenyon_Cell', 'MBON', 'CX', 'DAN'])),
        'order': 10,
        'category': 'Central',
    },
    {
        'name': 'Optic Lobe Intrinsic',
        'color': '#78909C',       # Slate Gray-Blue
        'match': lambda df, cl, sc: sc == 'ol_intrinsic',
        'order': 11,
        'category': 'Central',
    },
    {
        'name': 'Descending Command (DN)',
        'color': '#FF3D00',       # Red-Orange
        'match': lambda df, cl, sc: sc == 'descending_neuron',
        'order': 12,
        'category': 'Motor',
    },
    {
        'name': 'VNC Motor Neurons',
        'color': '#D50000',       # Crimson
        'match': lambda df, cl, sc: sc == 'vnc_motor',
        'order': 13,
        'category': 'Motor',
    },
    {
        'name': 'VNC Intrinsic Circuits',
        'color': '#546E7A',       # Slate Blue
        'match': lambda df, cl, sc: sc == 'vnc_intrinsic',
        'order': 14,
        'category': 'Motor',
    },
]


def load_model_and_connectome(checkpoint_path, graph_path, annotations_path, device):
    print(f"Loading MaleCNS connectome from {graph_path}...")
    npz = np.load(graph_path, allow_pickle=True)
    coords_um = npz['coords_um']  # (165122, 3)
    body_ids = npz['neuron_body_ids']
    sc_names = npz['superclass_names']
    sc_ids = npz['superclass_id']
    neuron_superclasses = sc_names[sc_ids]

    print(f"Loading annotations from {annotations_path}...")
    df_raw = pd.read_feather(annotations_path).set_index('bodyId')
    df = df_raw.reindex(body_ids)

    classes = df['class'].fillna('').values
    superclasses = df['superclass'].fillna('').values
    types = df['type'].fillna('').values

    # Assign each neuron to a functional group
    group_indices = {}
    neuron_group_name = np.array(['Other CNS'] * len(body_ids), dtype=object)
    neuron_color = np.array(['#37474F'] * len(body_ids), dtype=object)

    for cfg in SYSTEM_CONFIG:
        mask = cfg['match'](df, classes, superclasses)
        idx = np.where(mask)[0]
        group_indices[cfg['name']] = idx
        neuron_group_name[idx] = cfg['name']
        neuron_color[idx] = cfg['color']

    # Any remaining
    other_idx = np.where(neuron_group_name == 'Other CNS')[0]
    group_indices['Other CNS'] = other_idx

    print(f"Total neurons: {len(body_ids):,}")
    for name, idx in group_indices.items():
        print(f"  {name:<36}: {len(idx):6,d} neurons")

    # Load model checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = saved['config']
    model = FlyReservoirLM(
        graph_path,
        vocab_size=50257,
        d_model=cfg['d_model'],
        injection='topographic',
        read_surface='output',
        synapse_model='coba',
        use_alif=True,
        use_stp=True,
        decoder_bias=cfg.get('decoder_bias', True),
        read_centering=cfg.get('read_centering', False),
        use_read_gamma_trace=cfg.get('use_read_gamma_trace', False),
    ).to(device)

    # Load parameters safely
    weights = {k: v for k, v in saved['model'].items() if k not in ('edge_weight_e', 'edge_weight_i')}
    with torch.no_grad():
        for name in ('edge_weight_e', 'edge_weight_i'):
            getattr(model, name).copy_(saved['model'][name])
        model.load_state_dict(weights, strict=False)
    model.eval()

    # Restore physical state
    phys = saved['learner']['physical']
    state_dict = {}
    for k, v in phys.items():
        if isinstance(v, torch.Tensor):
            state_dict[k] = v.to(device)
        elif isinstance(v, tuple):
            state_dict[k] = tuple(t.to(device) for t in v)
        else:
            state_dict[k] = v
    initial_state = FlyPhysicalState(**state_dict)

    connectome_meta = {
        'coords_um': coords_um,
        'body_ids': body_ids,
        'types': types,
        'classes': classes,
        'superclasses': superclasses,
        'group_indices': group_indices,
        'neuron_group_name': neuron_group_name,
        'neuron_color': neuron_color,
    }
    return model, initial_state, connectome_meta


def simulate_stream_conduction(model, initial_state, prompt, device, settle_ticks=14):
    print(f"\nSimulating stream conduction for prompt: '{prompt}'...")
    enc = tiktoken.get_encoding('gpt2')
    token_ids = enc.encode(prompt)
    if len(token_ids) < 3:
        token_ids = token_ids * 3

    rates = model.get_decay_rates()
    thresholds = model.get_thresholds()
    gains = model.get_conductance_gains()
    alif = model.get_alif_params()
    stp = model.get_stp_params()

    current_state = initial_state

    # Warm up / settle through initial context tokens
    print(f"Warming up context across {len(token_ids)-1} tokens...")
    with torch.no_grad():
        for i in range(len(token_ids) - 1):
            tok = torch.tensor([token_ids[i]], dtype=torch.long, device=device)
            current_state = advance_fly_input_event(
                model, current_state, tok,
                settle_ticks=settle_ticks,
                writer_baseline_clock='input',
                base_rates=rates, thresholds=thresholds,
                conductance_gains=gains, alif_params=alif,
                stp_params=stp, return_ticks=False,
            )

        # Now capture the full 15-tick settling trajectory of the target token
        target_token = torch.tensor([token_ids[-2]], dtype=torch.long, device=device)
        next_target = torch.tensor([token_ids[-1]], dtype=torch.long, device=device)
        print(f"Capturing settling trajectory for token {target_token.item()} ('{enc.decode([target_token.item()])}') -> target '{enc.decode([next_target.item()])}'...")

        final_state, tick_states = advance_fly_input_event(
            model, current_state, target_token,
            settle_ticks=settle_ticks,
            writer_baseline_clock='input',
            base_rates=rates, thresholds=thresholds,
            conductance_gains=gains, alif_params=alif,
            stp_params=stp, return_ticks=True,
        )

    # Extract metrics across ticks
    num_ticks = len(tick_states)
    trajectory_data = []

    print(f"Processing {num_ticks} physical ticks (t = 0 .. {num_ticks-1})...")
    for t, st in enumerate(tick_states):
        spk = (st.ring[0][0] > 0).cpu().numpy().astype(bool)
        h = st.h[0].cpu().numpy()
        ge = st.ge[0].cpu().numpy()
        gi = st.gi[0].cpu().numpy()

        lat = extract_fly_motor_latent(model, st)
        logits = model.decoder(lat)
        nll = torch.nn.functional.cross_entropy(logits, next_target).item()
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).item()
        norm_entropy = entropy / np.log(50257)
        certainty = 1.0 - norm_entropy

        trajectory_data.append({
            'tick': t,
            'spikes': spk,
            'h': h,
            'ge_mean': float(ge.mean()),
            'gi_mean': float(gi.mean()),
            'certainty': certainty,
            'nll': nll,
            'motor_norm': float(lat.norm().item()),
            'active_count': int(spk.sum()),
        })

    return trajectory_data


def generate_static_figure(connectome_meta, trajectory_data, output_path):
    print(f"\nGenerating publication-grade static figure: {output_path}...")
    coords = connectome_meta['coords_um']
    group_indices = connectome_meta['group_indices']
    num_ticks = len(trajectory_data)

    # Compute regional firing rate matrix [N_regions, N_ticks]
    active_regions = [cfg for cfg in SYSTEM_CONFIG if cfg['name'] in group_indices and len(group_indices[cfg['name']]) > 0]
    region_names = [cfg['name'] for cfg in active_regions]
    region_colors = [cfg['color'] for cfg in active_regions]
    rate_matrix = np.zeros((len(active_regions), num_ticks), dtype=np.float32)
    h_matrix = np.zeros((len(active_regions), num_ticks), dtype=np.float32)

    for t_idx, d in enumerate(trajectory_data):
        spk = d['spikes']
        h = d['h']
        for r_idx, cfg in enumerate(active_regions):
            idx = group_indices[cfg['name']]
            rate_matrix[r_idx, t_idx] = spk[idx].mean() * 100.0  # percent
            h_matrix[r_idx, t_idx] = h[idx].mean()

    # Dark scientific publication theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 15), facecolor='#071018')
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.1, 1.2, 0.8], hspace=0.32, wspace=0.28)

    # Header title
    fig.suptitle(
        "Spatiotemporal Signal Conduction Across the Adult Drosophila Connectome (MaleCNS v1.0)\n"
        r"Physical Settling Dynamics ($S=14$, COBA ALIF+STP) Following Sensory Injection Pulse ($t=0$)",
        fontsize=16, fontweight='bold', color='#E0F2FE', y=0.97
    )

    # -------------------------------------------------------------
    # Panel A: Spatiotemporal Conduction Cascade (Heatmap)
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, :2])
    ax_a.set_facecolor('#0B1924')

    # Normalize each region's rate relative to its max or plot absolute firing %
    im = ax_a.imshow(rate_matrix, aspect='auto', cmap='plasma', interpolation='nearest', origin='upper')
    cbar = plt.colorbar(im, ax=ax_a, pad=0.02, fraction=0.03)
    cbar.set_label('Mean Firing Rate (%)', color='#94A3B8', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='#94A3B8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94A3B8')

    ax_a.set_xticks(range(num_ticks))
    ax_a.set_xticklabels([f"t={t}" for t in range(num_ticks)], color='#CBD5E1', fontsize=9)
    ax_a.set_yticks(range(len(region_names)))
    ax_a.set_yticklabels(region_names, color='#CBD5E1', fontsize=9.5)
    ax_a.set_xlabel("Physical Settling Ticks (Discrete Connectome Clocks)", color='#94A3B8', fontsize=11, labelpad=6)
    ax_a.set_title("A. Spatiotemporal Conduction Cascade Across 15 Functional Neuropils", color='#38BDF8', fontsize=12, fontweight='bold', loc='left', pad=8)

    # Annotate significant milestones
    ax_a.axvline(x=0.5, color='#00E5FF', linestyle='--', alpha=0.6, linewidth=1.2)
    ax_a.text(0.1, len(region_names)-0.6, "Input Pulse\n(t=0)", color='#00E5FF', fontsize=8, fontweight='bold', ha='center', va='bottom')

    ax_a.axvline(x=2.5, color='#FF6E40', linestyle='--', alpha=0.5, linewidth=1.0)
    ax_a.text(2.5, len(region_names)-0.6, "Relay Projection\n(t=1..2)", color='#FF6E40', fontsize=8, ha='center', va='bottom')

    ax_a.axvline(x=9.5, color='#E040FB', linestyle='--', alpha=0.5, linewidth=1.0)
    ax_a.text(7.0, len(region_names)-0.6, "Recurrent Integration\n(t=3..9)", color='#E040FB', fontsize=8, ha='center', va='bottom')

    ax_a.text(12.0, len(region_names)-0.6, "Motor Convergence\n(t=10..14)", color='#FF3D00', fontsize=8, ha='center', va='bottom')

    # -------------------------------------------------------------
    # Panel B: Dynamic Regional Activation Waveforms
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 2])
    ax_b.set_facecolor('#0B1924')
    ax_b.grid(True, color='#1E293B', linestyle=':', alpha=0.7)

    waveform_keys = [
        ('Sensory: Olfactory', '#FF9100', 'Sensory In'),
        ('Relay: Antennal Lobe PN', '#FF6E40', 'ALPN Relay'),
        ('Associative: MB Output (MBON)', '#FF1744', 'MBON Associative'),
        ('Central Complex: Navigation (CX)', '#E040FB', 'CX Attractor'),
        ('Descending Command (DN)', '#FF3D00', 'Descending Readout'),
    ]

    ticks = np.arange(num_ticks)
    for name, col, lbl in waveform_keys:
        if name in region_names:
            r_idx = region_names.index(name)
            ax_b.plot(ticks, rate_matrix[r_idx], color=col, linewidth=2.2, label=lbl, alpha=0.95)

    ax_b.set_xlabel("Physical Ticks", color='#94A3B8', fontsize=10)
    ax_b.set_ylabel("Firing Rate (%)", color='#94A3B8', fontsize=10)
    ax_b.set_xticks([0, 3, 6, 9, 12, 14])
    ax_b.tick_params(colors='#CBD5E1', labelsize=9)
    ax_b.set_title("B. Dynamic Conduction Waveforms", color='#38BDF8', fontsize=12, fontweight='bold', loc='left', pad=8)
    leg = ax_b.legend(facecolor='#071018', edgecolor='#1E293B', fontsize=8.5, loc='upper right')
    for text in leg.get_texts():
        text.set_color('#CBD5E1')

    # -------------------------------------------------------------
    # Panel C: Tri-phase Whole-Brain Spatial Projections
    # -------------------------------------------------------------
    # Sample background landmark neurons for brain silhouette
    bg_subsample = 12
    bg_idx = np.arange(0, len(coords), bg_subsample)
    bg_x = coords[bg_idx, 0]
    bg_z = coords[bg_idx, 2]

    phase_configs = [
        (0, "C1. Early Phase: Sensory Injection (t=0)", gs[1, 0]),
        (4, "C2. Mid Phase: Recurrent Processing (t=4)", gs[1, 1]),
        (11, "C3. Late Phase: Motor Convergence (t=11)", gs[1, 2]),
    ]

    for p_tick, p_title, p_loc in phase_configs:
        ax_p = fig.add_subplot(p_loc)
        ax_p.set_facecolor('#050C12')
        ax_p.set_aspect('equal')

        # Background silhouette
        ax_p.scatter(bg_x, bg_z, c='#1E293B', s=1.0, alpha=0.25, edgecolors='none', rasterized=True)

        # Active spiking neurons at this tick
        st_data = trajectory_data[p_tick]
        spk_indices = np.where(st_data['spikes'])[0]

        # Subsample if too dense for rasterization
        if len(spk_indices) > 3000:
            spk_indices = np.random.choice(spk_indices, 3000, replace=False)

        spk_x = coords[spk_indices, 0]
        spk_z = coords[spk_indices, 2]
        spk_cols = connectome_meta['neuron_color'][spk_indices]

        ax_p.scatter(spk_x, spk_z, c=spk_cols, s=5.0, alpha=0.85, edgecolors='none', rasterized=True)

        ax_p.set_xlim(0, 770)
        ax_p.set_ylim(1100, 50)  # Inverted so head is up, VNC down
        ax_p.set_xlabel("Lateral X (μm)", color='#94A3B8', fontsize=9)
        if p_loc == gs[1, 0]:
            ax_p.set_ylabel("Longitudinal Z (μm)\nHead [Brain] -> Body [VNC]", color='#94A3B8', fontsize=9)
        else:
            ax_p.set_yticklabels([])
        ax_p.tick_params(colors='#64748B', labelsize=8)
        ax_p.set_title(p_title, color='#38BDF8', fontsize=11, fontweight='bold', loc='left', pad=6)

        # Region label overlays
        ax_p.text(385, 90, "Central Brain", color='#64748B', fontsize=7.5, ha='center', va='center', style='italic')
        ax_p.text(385, 450, "Cervical Connective", color='#64748B', fontsize=7, ha='center', va='center', style='italic')
        ax_p.text(385, 800, "Ventral Nerve Cord (VNC)", color='#64748B', fontsize=7.5, ha='center', va='center', style='italic')

    # -------------------------------------------------------------
    # Panel D: Computational & Thermodynamic Convergence
    # -------------------------------------------------------------
    # D1: Language Predictive Loss (NLL)
    ax_d1 = fig.add_subplot(gs[2, 0])
    ax_d1.set_facecolor('#0B1924')
    ax_d1.grid(True, color='#1E293B', linestyle=':', alpha=0.7)
    nll_vals = [d['nll'] for d in trajectory_data]
    ax_d1.plot(ticks, nll_vals, color='#38BDF8', linewidth=2.5, marker='o', markersize=5)
    best_tick_nll = np.argmin(nll_vals)
    ax_d1.scatter([best_tick_nll], [nll_vals[best_tick_nll]], color='#FF0055', s=90, zorder=5)
    ax_d1.text(best_tick_nll, nll_vals[best_tick_nll]-0.05, f"Min NLL\nt={best_tick_nll}", color='#FF0055', fontsize=8, ha='center', va='top', fontweight='bold')
    ax_d1.set_xlabel("Physical Ticks", color='#94A3B8', fontsize=10)
    ax_d1.set_ylabel("Cross-Entropy Loss (NLL)", color='#38BDF8', fontsize=10)
    ax_d1.set_xticks([0, 3, 6, 9, 12, 14])
    ax_d1.tick_params(colors='#CBD5E1', labelsize=9)
    ax_d1.set_title("D1. Language Predictive Loss", color='#38BDF8', fontsize=11, fontweight='bold', loc='left', pad=6)

    # D2: Decision Certainty C(k)
    ax_d2 = fig.add_subplot(gs[2, 1])
    ax_d2.set_facecolor('#0B1924')
    ax_d2.grid(True, color='#1E293B', linestyle=':', alpha=0.7)
    cert_vals = [d['certainty'] for d in trajectory_data]
    ax_d2.plot(ticks, cert_vals, color='#4ADE80', linewidth=2.5, marker='s', markersize=5)
    best_tick_cert = np.argmax(cert_vals)
    ax_d2.scatter([best_tick_cert], [cert_vals[best_tick_cert]], color='#FACC15', s=90, zorder=5)
    ax_d2.text(best_tick_cert, cert_vals[best_tick_cert]+0.01, f"Peak Cert\nt={best_tick_cert}", color='#FACC15', fontsize=8, ha='center', va='bottom', fontweight='bold')
    ax_d2.set_xlabel("Physical Ticks", color='#94A3B8', fontsize=10)
    ax_d2.set_ylabel(r"Certainty $C(k) = 1 - \tilde{H}$", color='#4ADE80', fontsize=10)
    ax_d2.set_xticks([0, 3, 6, 9, 12, 14])
    ax_d2.tick_params(colors='#CBD5E1', labelsize=9)
    ax_d2.set_title("D2. Motor Readout Certainty", color='#4ADE80', fontsize=11, fontweight='bold', loc='left', pad=6)

    # D3: E/I Conductance Homeostasis
    ax_d3 = fig.add_subplot(gs[2, 2])
    ax_d3.set_facecolor('#0B1924')
    ax_d3.grid(True, color='#1E293B', linestyle=':', alpha=0.7)
    ge_vals = [d['ge_mean'] * 1e3 for d in trajectory_data]
    gi_vals = [d['gi_mean'] * 1e3 for d in trajectory_data]
    ax_d3.plot(ticks, ge_vals, color='#38BDF8', linewidth=2.0, label='AMPA Excitatory (g_E)')
    ax_d3.plot(ticks, gi_vals, color='#F43F5E', linewidth=2.0, label='GABA Inhibitory (g_I)')
    ax_d3.set_xlabel("Physical Ticks", color='#94A3B8', fontsize=10)
    ax_d3.set_ylabel("Conductance (×10⁻³)", color='#94A3B8', fontsize=10)
    ax_d3.set_xticks([0, 3, 6, 9, 12, 14])
    ax_d3.tick_params(colors='#CBD5E1', labelsize=9)
    ax_d3.set_title("D3. E/I Synaptic Conductance Balance", color='#38BDF8', fontsize=11, fontweight='bold', loc='left', pad=6)
    leg3 = ax_d3.legend(facecolor='#071018', edgecolor='#1E293B', fontsize=8.5, loc='lower right')
    for text in leg3.get_texts():
        text.set_color('#CBD5E1')

    # Save high-resolution PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Static figure successfully saved to: {output_path}")


def generate_interactive_3d_html(connectome_meta, trajectory_data, output_path, sample_bg=12000, max_active_points=3500):
    print(f"\nBuilding interactive 3D WebGL viewer: {output_path}...")
    coords = connectome_meta['coords_um']
    body_ids = connectome_meta['body_ids']
    neuron_group = connectome_meta['neuron_group_name']
    neuron_color = connectome_meta['neuron_color']
    classes = connectome_meta['classes']
    num_ticks = len(trajectory_data)

    # 1. Background anatomy silhouette (subsampled across all 165,122 neurons)
    n_total = len(coords)
    step = max(1, n_total // sample_bg)
    bg_indices = np.arange(0, n_total, step)

    bg_x = coords[bg_indices, 0]
    bg_y = coords[bg_indices, 1]
    bg_z = -coords[bg_indices, 2]  # Invert Z so head (low Z) is upward, VNC (high Z) downward

    # Dim colors for background
    bg_trace = go.Scatter3d(
        x=bg_x, y=bg_y, z=bg_z,
        mode='markers',
        marker=dict(
            size=1.6,
            color=neuron_color[bg_indices],
            opacity=0.10,
        ),
        hoverinfo='none',
        name='CNS Background Morphology',
    )

    # 2. Extract active spiking points per frame
    frames = []
    initial_active_trace = None

    for t, st in enumerate(trajectory_data):
        spk_mask = st['spikes']
        active_idx = np.where(spk_mask)[0]

        if len(active_idx) > max_active_points:
            # Prioritize higher membrane potential
            h_vals = st['h'][active_idx]
            top_k = np.argsort(h_vals)[-max_active_points:]
            active_idx = active_idx[top_k]

        act_x = coords[active_idx, 0]
        act_y = coords[active_idx, 1]
        act_z = -coords[active_idx, 2]
        act_color = neuron_color[active_idx]
        act_group = neuron_group[active_idx]
        act_class = classes[active_idx]
        act_body = body_ids[active_idx]
        act_h = st['h'][active_idx]

        hover_texts = [
            f"<b>Neuron #{b}</b><br>"
            f"System: {g}<br>"
            f"Class: {c or 'Intrinsic'}<br>"
            f"Pos: ({x:.1f}, {y:.1f}, {-z:.1f}) μm<br>"
            f"Membrane h: {h:.3f}"
            for b, g, c, x, y, z, h in zip(act_body, act_group, act_class, act_x, act_y, act_z, act_h)
        ]

        active_trace = go.Scatter3d(
            x=act_x, y=act_y, z=act_z,
            mode='markers',
            marker=dict(
                size=4.2,
                color=act_color,
                opacity=0.88,
            ),
            text=hover_texts,
            hoverinfo='text',
            name=f'Active Spikes (t={t})',
        )

        if t == 0:
            initial_active_trace = active_trace

        frames.append(go.Frame(
            data=[active_trace],
            traces=[1],
            name=f"tick_{t}",
            layout=dict(
                title_text=f"Fruit Fly CNS Connectome Conduction | Physical Tick t={t} (Active Spikes: {st['active_count']:,} | Readout Certainty: {st['certainty']:.3f} | NLL: {st['nll']:.3f})"
            )
        ))

    # Layout & Camera Controls
    layout = go.Layout(
        template='plotly_dark',
        title=dict(
            text=f"Fruit Fly CNS Connectome Conduction | Physical Tick t=0 (Active Spikes: {trajectory_data[0]['active_count']:,} | Readout Certainty: {trajectory_data[0]['certainty']:.3f})",
            font=dict(size=14, color='#38BDF8'),
            x=0.03, y=0.97
        ),
        paper_bgcolor='#071018',
        plot_bgcolor='#071018',
        scene=dict(
            bgcolor='#071018',
            xaxis=dict(title='Lateral X (μm)', showgrid=True, gridcolor='#1E293B', zeroline=False, color='#94A3B8'),
            yaxis=dict(title='Dorsoventral Y (μm)', showgrid=True, gridcolor='#1E293B', zeroline=False, color='#94A3B8'),
            zaxis=dict(title='Longitudinal Z (μm) [Head -> VNC]', showgrid=True, gridcolor='#1E293B', zeroline=False, color='#94A3B8'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.35, y=-1.55, z=0.75),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.1),
            )
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        label="▶ Play Conduction",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 350, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 50, "easing": "linear"},
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        }]
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.03,
                y=0.06,
                xanchor="left",
                yanchor="bottom",
                bgcolor="#0F172A",
                font=dict(color="#38BDF8"),
            ),
            # Camera View Preset buttons
            dict(
                type="buttons",
                direction="down",
                buttons=[
                    dict(
                        label="Coronal (Front)",
                        method="relayout",
                        args=[{"scene.camera": dict(eye=dict(x=0.0, y=-2.4, z=0.0), up=dict(x=0, y=0, z=1))}]
                    ),
                    dict(
                        label="Dorsal (Top)",
                        method="relayout",
                        args=[{"scene.camera": dict(eye=dict(x=0.0, y=0.0, z=2.4), up=dict(x=0, y=1, z=0))}]
                    ),
                    dict(
                        label="Sagittal (Side)",
                        method="relayout",
                        args=[{"scene.camera": dict(eye=dict(x=2.4, y=0.0, z=0.0), up=dict(x=0, y=0, z=1))}]
                    ),
                    dict(
                        label="3D Isometric",
                        method="relayout",
                        args=[{"scene.camera": dict(eye=dict(x=1.35, y=-1.55, z=0.75), up=dict(x=0, y=0, z=1))}]
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.97,
                y=0.95,
                xanchor="right",
                yanchor="top",
                bgcolor="#0F172A",
                font=dict(color="#CBD5E1", size=10),
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="bottom",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=12, color="#38BDF8"),
                    prefix="Physical Settle Tick: ",
                    visible=True,
                    xanchor="right"
                ),
                transition=dict(duration=50, easing="linear"),
                pad=dict(b=10, t=20),
                len=0.72,
                x=0.24,
                y=0.06,
                steps=[
                    dict(
                        method="animate",
                        args=[[f"tick_{t}"], {
                            "frame": {"duration": 150, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }],
                        label=f"{t}"
                    )
                    for t in range(num_ticks)
                ]
            )
        ],
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig = go.Figure(data=[bg_trace, initial_active_trace], layout=layout, frames=frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"Interactive 3D WebGL HTML viewer successfully saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt'),
                        help='Path to trained model checkpoint')
    parser.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba.npz'),
                        help='Path to MaleCNS v1 connectome NPZ')
    parser.add_argument('--annotations', type=Path, default=Path('data/malecns_v1/body-annotations.feather'),
                        help='Path to body annotations feather file')
    parser.add_argument('--out-dir', type=Path, default=Path('present'),
                        help='Output directory for generated figures and HTML')
    parser.add_argument('--prompt', type=str,
                        default='The neural connectome of the fruit fly brain orchestrates complex sensory processing, spatial navigation, and motor behavior.',
                        help='Prompt string to stream through the connectome')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--settle-ticks', type=int, default=14)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Executing on device: {device}")

    # Step 1: Load connectome & model
    model, initial_state, connectome_meta = load_model_and_connectome(
        args.checkpoint, args.graph, args.annotations, device
    )

    # Step 2: Simulate conduction across settling ticks
    trajectory_data = simulate_stream_conduction(
        model, initial_state, args.prompt, device, settle_ticks=args.settle_ticks
    )

    # Step 3: Generate publication static figure
    static_fig_path = args.out_dir / 'fly_conduction_spatiotemporal.png'
    generate_static_figure(connectome_meta, trajectory_data, static_fig_path)

    # Step 4: Generate interactive 3D WebGL viewer
    interactive_html_path = args.out_dir / 'fly_brain_3d_conduction.html'
    generate_interactive_3d_html(connectome_meta, trajectory_data, interactive_html_path)

    # Step 5: Save execution JSON report
    summary = {
        'checkpoint': str(args.checkpoint),
        'graph': str(args.graph),
        'settle_ticks': args.settle_ticks,
        'prompt': args.prompt,
        'static_figure': str(static_fig_path),
        'interactive_viewer': str(interactive_html_path),
        'ticks': [
            {
                'tick': d['tick'],
                'active_count': d['active_count'],
                'certainty': d['certainty'],
                'nll': d['nll'],
                'ge_mean': d['ge_mean'],
                'gi_mean': d['gi_mean'],
            }
            for d in trajectory_data
        ]
    }
    report_path = args.out_dir / 'fly_conduction_report.json'
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary report saved to: {report_path}")
    print("\nVisualization generation complete!")


if __name__ == '__main__':
    main()
