# Digit closed-set probe: latent steps K sweep (no LLM)

- Task: 1px stick digits 0–9; hard_frac=0.35; res=32
- Readout: mean-pool tokens → Linear(10); **no free-gen**
- steps=400 batch=32 T=32 chance=0.10

| kind | K | acc_digit | final_loss | seconds | status |
|------|---|-----------|------------|---------|--------|
| A_patch | 1 | 0.334 | 1.494 | 5.4 | ok |
| B_vlcot_K1 | 1 | 0.234 | 1.735 | 25.0 | ok |
| B_vlcot_K2 | 2 | 0.074 | 2.293 | 35.6 | ok |
| B_vlcot_K4 | 4 | 0.166 | 1.913 | 57.4 | ok |

## Relative K

- K=1: acc=0.234 (Δ vs K=1: +0.000)
- K=2: acc=0.074 (Δ vs K=1: -0.160)
- K=4: acc=0.166 (Δ vs K=1: -0.068)
- Best B: **K=1** acc=0.234
- Best B vs A: 0.234 vs 0.334 (Δ=-0.100)

Phase-1 closed-set probe only; not frozen-LM free-gen OCR.
