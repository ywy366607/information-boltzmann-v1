# Feature vs alignment: linear probe vs long VQA (A/B/C)

- Cache: `D:\ml_cache`
- Backend (VQA): `n/a (probe-only; no LLM)`
- res=32 patch=4 topk=2 ST=on for B/C; **no angles**
- Linear probe steps=400 d_feat=640 (mean-pool tokens → linear heads; **no LLM generation**)
- VQA steps=None seeds=[0, 1] protocol=`Question:`/`Answer:` greedy exact-match
- val_seed=90001

## Linear probe (classification)

| kind | T | seed | acc | color | kinks | notes |
|------|---|------|-----|-------|-------|-------|
| A | 64 | 0 | 0.312 | 0.412 | 0.200 |  |
| B | 64 | 0 | 0.328 | 0.412 | 0.233 |  PR=49.4 sup=2.0 |
| C | 48 | 0 | 0.312 | 0.412 | 0.200 | 16+32 |
| A | 32 | 0 | 0.312 | 0.412 | 0.200 |  |
| B | 32 | 0 | 0.328 | 0.412 | 0.233 |  PR=19.1 sup=2.0 |
| C | 32 | 0 | 0.328 | 0.412 | 0.233 | 16+16 |
| A | 16 | 0 | 0.312 | 0.412 | 0.200 |  |
| B | 16 | 0 | 0.562 | 0.882 | 0.200 |  PR=2.9 sup=2.0 |
| C | 12 | 0 | 0.391 | 0.559 | 0.200 | 4+8 |
| A | 64 | 1 | 0.312 | 0.382 | 0.233 |  |
| B | 64 | 1 | 0.312 | 0.176 | 0.467 |  PR=52.3 sup=2.0 |
| C | 48 | 1 | 0.234 | 0.265 | 0.200 | 16+32 |
| A | 32 | 1 | 0.312 | 0.382 | 0.233 |  |
| B | 32 | 1 | 0.312 | 0.176 | 0.467 |  PR=14.7 sup=2.0 |
| C | 32 | 1 | 0.328 | 0.441 | 0.200 | 16+16 |
| A | 16 | 1 | 0.312 | 0.382 | 0.233 |  |
| B | 16 | 1 | 0.469 | 0.471 | 0.467 |  PR=3.5 sup=2.0 |
| C | 12 | 1 | 0.266 | 0.294 | 0.233 | 4+8 |

## Long VQA (exact-match free-gen)

| kind | T | seed | acc | color | kinks | notes |
|------|---|------|-----|-------|-------|-------|

## Diagnosis

- **Class**: alignment-weak, alignment-weak
- **B probe vs VQA**: n/a
- Max probe color=0.882 kinks=0.467; max VQA color=0.000 kinks=0.000
- Kinks learnable on at least one protocol (probe_max=0.467, vqa_max=0.000).

## Method notes

- Linear probe: freeze-free frontend + mean-pool over T vision tokens + CE heads; LLM unused.
- VQA: frozen Gemma-3-270M @ D:\ml_cache; train frontend only; greedy exact-match.
- Same on-the-fly synthetic mix (color 4-way, kinks 5–8 corners); train/val seed split.
