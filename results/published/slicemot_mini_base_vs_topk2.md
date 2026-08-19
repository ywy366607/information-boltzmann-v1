# SliceMoT mini: base (soft deslice) vs topk2

Shared: ST ✓, Ada-Temp ✓, Gumbel ✗, 2L/M=32/d=d_x=256, Gemma+LoRA, Flickr8k, warmup+cosine, 100k smoke.

| | **base** (topk=0 soft) | **topk2** |
|--|------------------------|-----------|
| steps | 100000 | 100000 |
| loss tokens | 1,205,716 (0.244×) | 1,203,611 (0.243×) |
| eligible_1x | false | false |
| final_loss (noisy) | 3.77 | 2.99 |
| peak VRAM | ~1958 MB | ~1959 MB |
| wall time | ~9.1 h | ~9.4 h |

## Lite causality (n=48)

| metric | base | topk2 | topk − base |
|--------|-----:|------:|------------:|
| matched NLL | 2.887 | 2.986 | +0.10 |
| shuffled NLL | 3.136 | 3.204 | +0.07 |
| **Δ (shuffle−matched)** | **+0.249** | **+0.218** | **−0.031** |
| frac matched better | **0.792** | **0.667** | **−0.125** |
| TF acc | 0.419 | 0.403 | −0.016 |

Both clear soft positive (matched beats shuffle). **Soft deslice base slightly stronger** on Δ and especially frac.

## Decision (smoke, not 1×)

- **Default deslice: soft (topk=0)** under ST+Ada-Temp is fine.
- **topk=2** remains a write-leakage option for recon/thin-structure contexts; **not required** for this caption causality smoke, and **did not improve** lite Gate signal here.
- Neither arm is 1×-eligible; no full efficacy claim.

## Related

- Gumbel early-stop negative: `slicemot_mini_gumbel_conclusion.md`
- Default stack: **Stiefel + Ada-Temp**, Gumbel off, soft deslice default.
