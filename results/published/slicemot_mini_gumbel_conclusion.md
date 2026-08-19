# SliceMoT-Mini-2L arm=`gumbel` — early stop

- note: base (ST + Ada-Temp) **+** Transolver++ Gumbel (train only)
- layers=2 M=32 d=d_x=256; Gemma-3-270M + LoRA; Flickr8k
- **Early-stopped** (~step 5300 / 100000); no full 1× gate; **no** end-of-run causality probes

## Setup

| flag | value |
|------|-------|
| Stiefel | **on** (default) |
| Ada-Temp | **on** (default) |
| **Gumbel** | **on** (this arm) |
| deslice topk | 0 (soft) |

## Training (partial)

| metric | value |
|--------|------:|
| steps | **5300** (planned 100k) |
| loss tokens | **~64k / 4.94M** (~0.013×) |
| eligible_1x | **false** |
| last raw loss | ~**12.1** |
| last EMA loss (log pts, α≈0.1) | ~**8.2** |
| fraction log pts with loss ≥ 8 | ~**33%** |

## Same-stage vs `base` (no Gumbel)

Under identical ST+Ada-Temp, schedule, data, seed protocol:

| step (approx) | base EMA loss | gumbel EMA loss |
|--------------:|--------------:|----------------:|
| 1000 | ~3.5 | ~7.2 |
| 2000 | ~3.4 | ~6.7 |
| 2300 | ~3.3 | ~7.7 |

- **base** after full 100k smoke: weak positive visual causality  
  (matched NLL 2.89, shuffled 3.14, **Δ=+0.25**, frac matched better **0.79**, TF **0.42**; still `eligible_1x=false`).
- **gumbel**: chronic spikes (loss 10–16), EMA stuck ~7–9; does not approach base training quality.

## Decision

1. **Default `use_gumbel=False`** for this repo’s SliceMoT / VLM mini recipe.
2. **Stiefel is enough** as the primary anti-collapse mechanism; Gumbel is not a substitute and is not required once ST (+ mild Ada-Temp) is on.
3. Gumbel remains an optional ablation flag only; **no need to finish 100k** for a default decision.
4. Aligns with prior recon / NS experience: Gumbel often hurts optimization; Transolver’s default Gumbel is historical discrete-routing habit, not a validated must for vision fine-structure here.

## Non-claims

- Not a full 1×-eligible efficacy claim for either arm.
- Does not prove Gumbel is useless on all PDE / official Transolver benchmarks.

## Next

- Run **`topk2`** under ST+Ada-Temp base (write-path ablation only).
