# SliceMoT mini: base (LocalVisual dw3) vs nolocal

Shared: ST ✓, Ada-Temp ✓, Gumbel ✗, soft deslice, 100k smoke, Flickr8k.

| | **base** (local=dw3) | **nolocal** (local=none) |
|--|---------------------:|-------------------------:|
| steps | 100000 | 100000 |
| tokens | 1.206M (0.244×) | 1.205M (0.251× gate) |
| eligible_1x | false | false |
| matched NLL | 2.887 | **2.538** |
| shuffled NLL | 3.136 | 2.816 |
| **Δ (shuffle−matched)** | +0.249 | **+0.278** |
| frac matched better | **0.792** | 0.771 |
| TF acc | 0.419 | **0.426** |

## Decision (caption smoke only)

- Both weak-positive causality; **not 1× eligible**.
- On 32px Flickr caption, **removing layer LocalVisual does not kill the signal**; Δ even slightly higher, frac slightly lower.
- Does **not** refute “local needed for thin-structure recon”; that remains a different task.
- Default for **this caption recipe** may keep dw3 or none; fine-structure stacks still prefer loc.

Next: `--arm dual_ps` (PatchEmbed(X)+MoT(P,S)+Deslice+Unpatch).
