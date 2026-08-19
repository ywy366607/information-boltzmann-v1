# Phase 1 recurrence conclusion (encoder / probe, no frozen-LM free-gen)

## Setup

- **Goal:** relative gain from more recursive steps under matched compute, without frozen tiny LM OCR noise.
- **Exp 1.1:** line recon, `slice_loc_nogumbel_recur{T}` with `recur_T ∈ {1,2,4}`, res=32, steps=150, RGB, batch=16, hard_frac=0.35, seed=0.
- **Exp 1.2:** closed-set digit probe (mean-pool → Linear(10)), `VisualLatentCoTSlice` with `K ∈ {1,2,4}` vs patch baseline, res=32, T=32, steps=400, batch=32, hard_frac=0.35, seed=0, val_seed=90001.
- Artifacts: `recur_recon_table.json`, `recur_recon_conclusion.md`, `vlcot_probe_K_table.json`, `vlcot_probe_K_conclusion.md`.

## Measured numbers

### Line recon (`recur_T`)

| recur_T | arm | final dice | final iou | note |
|---------|-----|------------|-----------|------|
| 1 | slice_loc_nogumbel_recur1 | **1.000** | 1.000 | ceiling by step 50 |
| 2 | slice_loc_nogumbel_recur2 | **1.000** | 1.000 | ceiling by step 50 |
| 4 | slice_loc_nogumbel_recur4 | **1.000** | 1.000 | ceiling by step 50 |

History (dice): T=1 s0=0.231 → s50=1.000; T=2 s0=0.177 → s50=1.000; T=4 s0=0.001 → s50=1.000.

**Relative claim:** under this matched budget, **no positive gain from `recur_T↑`** — all cells hit Dice=1.0 by the first post-init probe (step 50). Honest **ceiling null** on final Dice (Δ vs T=1: 0.000). Init at step 0 is slightly worse for higher T; not a win for multi-pass under this easy recon regime (same pattern as prior fair line_recon where slice already saturates).

### Digit closed-set probe (`K`)

| kind | K | acc_digit | final_loss | vs chance (0.10) |
|------|---|-----------|------------|------------------|
| A_patch | 1 | **0.334** | 1.494 | +0.234 |
| B_vlcot | 1 | **0.234** | 1.735 | +0.134 |
| B_vlcot | 2 | **0.074** | 2.293 | −0.026 (≤ chance) |
| B_vlcot | 4 | **0.166** | 1.913 | +0.066 |

- K=1 → K=2: **Δacc = −0.160**
- K=1 → K=4: **Δacc = −0.068**
- Best B (K=1) vs A: **0.234 vs 0.334 (Δ = −0.100)**

**Relative claim:** under this matched short budget, **`K↑` does not help** — best B is K=1; K=2 is worst and near/below chance. Honest **null / negative** for latent multi-step on closed-set digit probe at steps=400. (Does **not** claim frozen-LM free-gen OCR success or failure.)

## Phase-1 bottom line

| Axis | More steps help? | Numbers |
|------|------------------|---------|
| `recur_T` (recon) | **Null (ceiling)** | all Dice=1.000; ΔT=0 |
| `K` (digit probe) | **Null / worse** | K1=0.234 > K4=0.166 > K2=0.074 |

Recurrence knobs are **wired and exercised** (ARMS `recur_T`, VL-CoT `latent_steps`), but **this Phase-1 budget does not show a relative multi-step win**. Encoder single-pass slice remains strong on recon; probe path prefers K=1 over deeper latent refinement under equal short training. Next product-relevant work is KPI ladder L3–L4 (teacher-forced / constrained CER) under a **trainable** readout — not more free-gen exact on frozen 270M.

## Non-claims

- Not a product OCR claim.
- Not “recurrence never works” on harder long-horizon tasks; only **measured null under these matched cells**.
- Open free-gen remains diagnostic only (see prior VQA/OCR tables).
