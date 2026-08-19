# Feature vs alignment: linear probe vs long VQA (A/B/C)

- Cache: `D:\ml_cache`
- Backend (VQA): `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- res=32 patch=4 topk=2 projector=mlp ST=on for B/C; **no angles**
- Vision→LLM connector: LLaVA-1.5 style **mlp** (mlp = Linear→GELU→Linear)
- Linear probe steps=400 d_feat=640 (mean-pool tokens → linear heads; **no LLM generation**)
- VQA steps=400 seeds=[0, 1] protocol=`Question:`/`Answer:` greedy exact-match
- val_seed=90001

## Linear probe (classification)

| kind | T | seed | acc | color | kinks | notes |
|------|---|------|-----|-------|-------|-------|
| A | 64 | 0 | 0.188 | 0.176 | 0.200 |  |
| B | 64 | 0 | 0.188 | 0.176 | 0.200 |  PR=44.6 sup=2.0 |
| C | 48 | 0 | 0.156 | 0.118 | 0.200 | 16+32 |
| A | 32 | 0 | 0.188 | 0.176 | 0.200 |  |
| B | 32 | 0 | 0.188 | 0.176 | 0.200 |  PR=18.1 sup=2.0 |
| C | 32 | 0 | 0.344 | 0.471 | 0.200 | 16+16 |
| A | 16 | 0 | 0.188 | 0.176 | 0.200 |  |
| B | 16 | 0 | 0.562 | 0.882 | 0.200 |  PR=8.1 sup=2.0 |
| C | 12 | 0 | 0.188 | 0.176 | 0.200 | 4+8 |
| A | 64 | 1 | 0.328 | 0.412 | 0.233 |  |
| B | 64 | 1 | 0.266 | 0.294 | 0.233 |  PR=49.7 sup=2.0 |
| C | 48 | 1 | 0.297 | 0.412 | 0.167 | 16+32 |
| A | 32 | 1 | 0.328 | 0.412 | 0.233 |  |
| B | 32 | 1 | 0.281 | 0.324 | 0.233 |  PR=17.4 sup=2.0 |
| C | 32 | 1 | 0.297 | 0.412 | 0.167 | 16+16 |
| A | 16 | 1 | 0.328 | 0.412 | 0.233 |  |
| B | 16 | 1 | 0.266 | 0.294 | 0.233 |  PR=3.9 sup=2.0 |
| C | 12 | 1 | 0.391 | 0.529 | 0.233 | 4+8 |

## Long VQA (exact-match free-gen)

| kind | T | seed | acc | color | kinks | notes |
|------|---|------|-----|-------|-------|-------|
| A | 64 | 0 | 0.203 | 0.294 | 0.100 |  |
| B | 64 | 0 | 0.141 | 0.176 | 0.100 |  |
| C | 48 | 0 | 0.156 | 0.294 | 0.000 | 16+32 |
| A | 32 | 0 | 0.125 | 0.235 | 0.000 |  |
| B | 32 | 0 | 0.156 | 0.265 | 0.033 |  |
| C | 32 | 0 | 0.219 | 0.324 | 0.100 | 16+16 |
| A | 16 | 0 | 0.156 | 0.294 | 0.000 |  |
| B | 16 | 0 | 0.016 | 0.029 | 0.000 |  |
| C | 12 | 0 | 0.156 | 0.294 | 0.000 | 4+8 |
| A | 64 | 1 | 0.219 | 0.412 | 0.000 |  |
| B | 64 | 1 | 0.328 | 0.500 | 0.133 |  |
| C | 48 | 1 | 0.344 | 0.529 | 0.133 | 16+32 |
| A | 32 | 1 | 0.172 | 0.118 | 0.233 |  |
| B | 32 | 1 | 0.219 | 0.412 | 0.000 |  |
| C | 32 | 1 | 0.328 | 0.471 | 0.167 | 16+16 |
| A | 16 | 1 | 0.344 | 0.412 | 0.267 |  |
| B | 16 | 1 | 0.281 | 0.382 | 0.167 |  |
| C | 12 | 1 | 0.234 | 0.382 | 0.067 | 4+8 |

## Diagnosis

- **Class**: alignment-weak (still primary under **MLP projector**)
- **B probe vs VQA flip=YES**: B−A probe=**+0.156** (B=0.414@T16 mean, A=0.258); B−A vqa=**−0.016** (B=0.234@T64 mean, A=0.250) — flip remains but **VQA gap shrinks a lot** vs old single-Linear run (was about −0.11).
- Max probe color=**0.882** (B@T16 seed0) kinks=0.233; max VQA color=**0.529** (C@T48 seed1) kinks=0.267
- B best probe acc=0.414 (T=16) vs best seed-mean VQA 0.234 (T=64); **best single-cell B VQA** = seed1 T64 acc=**0.328** color=**0.50**
- A best seed-mean probe 0.258 / VQA 0.250; peak A VQA seed1 T16 acc=0.344
- C best seed-mean probe 0.320 / VQA 0.273; peak C VQA seed1 T48 acc=0.344 color=0.529
- Kinks **>0** on both protocols (probe_max=0.233, vqa_max=0.267)

### vs previous single-Linear connector

| | Linear (prior table) | **MLP (this run)** |
|--|----------------------|---------------------|
| B−A probe | +0.227 | +0.156 |
| B−A VQA | −0.109 | **−0.016** |
| B peak VQA color (cell) | ~0.38 | **0.50** (B@64 s1) |
| Flip | YES | YES (weaker on VQA side) |

**Takeaway:** 2-layer MM MLP **helps** slice tokens talk to frozen Gemma (B/C VQA less catastrophic, B@64 seed1 color 0.5), but does **not** remove the probe≫VQA gap for compact B (B@T16 probe color 0.88 vs VQA still poor on seed0). Story is still **alignment/interface**, not “frontend empty”; MLP is necessary hygiene, not a full fix.

## Method notes

- Connector: LLaVA-1.5 **Linear→GELU→Linear** (`--projector mlp`) on A/B/C token outputs.
- Linear probe: mean-pool vision tokens → CE heads; **no LLM generation**.
- VQA: frozen Gemma-3-270M @ `D:\ml_cache`; train frontend+MLP only; greedy exact-match.
- Same synthetic mix (color 4-way, kinks 5–8); seeds 0/1; **no angles**.

