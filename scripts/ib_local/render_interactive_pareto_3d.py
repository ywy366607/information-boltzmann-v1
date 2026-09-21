"""Generate Plotly Interactive 3D Surface & Pareto Frontier Dashboard in HTML.
Allows 360-degree rotation, zooming, and hover inspection on the Z(H, K) response manifold.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import Rbf

from scripts.ib_local.plot_pareto_surface_hk import anchors


def main():
    out_dir = Path("present")
    out_dir.mkdir(parents=True, exist_ok=True)

    h_pts = np.array([p[0] for p in anchors])
    k_pts = np.array([p[1] for p in anchors])
    z_pts = np.array([p[2] for p in anchors])

    log2_h = np.log2(h_pts)
    log2_k = np.log2(k_pts)

    grid_h = np.linspace(0, 6, 80)   # 2^0=1 to 2^6=64
    grid_k = np.linspace(0, 7, 80)   # 2^0=1 to 2^7=128
    HH, KK = np.meshgrid(grid_h, grid_k)

    rbf = Rbf(log2_h, log2_k, z_pts, function="multiquadric", epsilon=1.2, smooth=0.15)
    ZZ = rbf(HH, KK)
    ZZ = np.clip(ZZ, 38.5, 55.6)

    # Compute Cost
    actual_H = 2.0 ** HH
    actual_K = 2.0 ** KK
    Cost = actual_H * (3.5 + actual_K * 1.0)

    # Trace 1: 3D Surface
    surface_trace = go.Surface(
        x=HH, y=KK, z=ZZ,
        colorscale="Viridis",
        colorbar=dict(title=dict(text="Accuracy (%)", font=dict(color="#38d8b8")), tickfont=dict(color="#8892b0")),
        opacity=0.88,
        hoverinfo="x+y+z",
        name="Accuracy Surface Z(H,K)"
    )

    # Trace 2: Trained Anchor Points
    scatter_anchors = go.Scatter3d(
        x=log2_h, y=log2_k, z=z_pts,
        mode="markers+text",
        marker=dict(size=6, color="#ff4488", line=dict(color="#ffffff", width=1.5)),
        text=[f"({int(h)},{int(k)}): {z:.1f}%" for h, k, z in anchors],
        textposition="top center",
        textfont=dict(color="#ffffff", size=9),
        name="Trained Anchor Ground Truth"
    )

    # Trace 3: Golden Optimum marker
    golden_marker = go.Scatter3d(
        x=[0], y=[4], z=[55.52],
        mode="markers+text",
        marker=dict(size=12, color="#ffdd00", symbol="diamond", line=dict(color="#ffffff", width=2)),
        text=["GOLDEN KNEE (H=1, K=16): 55.52%"],
        textposition="bottom center",
        textfont=dict(color="#ffdd00", size=13),
        name="Global Golden Optimum"
    )

    fig = go.Figure(data=[surface_trace, scatter_anchors, golden_marker])

    fig.update_layout(
        title=dict(
            text="CBIM 宏步 H 与微步 K 认知时域 3D 响应流形与帕累托前沿 (Z(H, K) Accuracy Surface)",
            font=dict(size=18, color="#e0e8ff")
        ),
        paper_bgcolor="#060614",
        plot_bgcolor="#060614",
        scene=dict(
            xaxis=dict(
                title=dict(text="Macro-Step H (log2: 1 to 64)", font=dict(color="#90a8d0")),
                backgroundcolor="#080a1c",
                gridcolor="#182850",
                tickvals=list(range(7)),
                ticktext=[f"2^{i}={2**i}" for i in range(7)],
                tickfont=dict(color="#8892b0")
            ),
            yaxis=dict(
                title=dict(text="Micro-Step K (log2: 1 to 128)", font=dict(color="#90a8d0")),
                backgroundcolor="#080a1c",
                gridcolor="#182850",
                tickvals=list(range(8)),
                ticktext=[f"2^{j}={2**j}" for j in range(8)],
                tickfont=dict(color="#8892b0")
            ),
            zaxis=dict(
                title=dict(text="Cell Accuracy (%)", font=dict(color="#38d8b8")),
                backgroundcolor="#080a1c",
                gridcolor="#182850",
                tickfont=dict(color="#8892b0"),
                range=[38, 57]
            ),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.2))
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(font=dict(color="#e0e8ff"), bgcolor="#080a1c", bordercolor="#203560", borderwidth=1)
    )

    out_html = out_dir / "cbim_pareto_interactive_3d.html"
    fig.write_html(str(out_html))
    print(f"Saved interactive 3D Plotly dashboard to {out_html}!", flush=True)


if __name__ == "__main__":
    main()
