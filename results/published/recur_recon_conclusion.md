# Line recon recurrence sweep (`recur_T`)

- Task: 1px polyline mask recon; rgb=True res=32 steps=150 batch=16 hard_frac=0.35
- Arms: `slice_loc_nogumbel_recur1,slice_loc_nogumbel_recur2,slice_loc_nogumbel_recur4` (same family as `slice_loc_nogumbel`)
- Seed=0; matched budget across cells

| recur_T | arm | dice | iou | recall | final_step | status |
|---------|-----|------|-----|--------|------------|--------|
| 1 | slice_loc_nogumbel_recur1 | 1.000 | 1.000 | 1.000 | 150 | ok |
| 2 | slice_loc_nogumbel_recur2 | 1.000 | 1.000 | 1.000 | 150 | ok |
| 4 | slice_loc_nogumbel_recur4 | 1.000 | 1.000 | 1.000 | 150 | ok |

## Relative recurrence

- recur_T=1: dice=1.000 (Δ vs T=1: +0.000)
- recur_T=2: dice=1.000 (Δ vs T=1: +0.000)
- recur_T=4: dice=1.000 (Δ vs T=1: +0.000)
- Best: **slice_loc_nogumbel_recur1** recur_T=1 dice=1.000

### Learning speed (history dice)
- recur_T=1: s0=0.231, s50=1.000, s100=1.000, s150=1.000
- recur_T=2: s0=0.177, s50=1.000, s100=1.000, s150=1.000
- recur_T=4: s0=0.001, s50=1.000, s100=1.000, s150=1.000

Note: this is Phase-1 recurrence relative gain (encoder), not frozen-LM free-gen OCR.
