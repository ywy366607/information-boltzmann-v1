# ShareGPT-4o Real-Data Pilot

Status: **pipeline passed; unified capability not admitted**. The pilot puts
natural image/text observations into the existing X–Slice–H graph. It adds no
diffusion model, private vision backbone, trainable Pythia, or `lm.generate()`.

## Dataset boundary

- `FreedomIntelligence/ShareGPT-4o-Image` supplies T2I targets and one-source
  IT2I edits. The local bank has 134 T2I and 142 IT2I records.
- `OpenGVLab/ShareGPT-4o` supplies image-caption I2T. A complete scan of 57,289
  first-turn, one-image conversations found only generic caption requests and
  **no genuine question/instruction-conditioned IT2T examples**.
- Earlier reports of 128 IT2T records were a classifier error: caption
  paraphrases were mislabeled as IT2T. Those IT2T metrics are invalidated.

The corrected bank contains 134 T2I, 142 IT2I, and 512 I2T records. All 930
referenced images decode successfully. Raw data, credentials, and checkpoints
remain under `D:\ml_cache` or ignored directories.

`fine_grain/sharegpt4o_data.py` preserves aspect ratio by letterboxing, supports
partial tar and remote ZIP64 extraction, masks answer tokens from visual
evidence, and builds matched/shuffled controls. Natural RGB uses the graph's
heteroscedastic Gaussian likelihood; token NLL is accumulated in FP32 while
the frozen Pythia forward remains FP16.

## Reproduction

```powershell
python scripts/prepare_sharegpt4o_pilot.py --max-each 512
python scripts/train_sharegpt4o_pilot.py --resolution 64 --steps 300 `
  --batch 4 --phase token_reader --tasks i2t --holdout-fraction 0.2 `
  --max-answer-tokens 64 --shuffle-coef 2 --shuffle-margin 0.1 `
  --init checkpoints/omni_d64_pythia_sharegpt4o_unified_init.pt
```

## Evidence

The earlier safe RGB calibration updated only 646 mean/log-variance head
parameters. On its deterministic 64-sample audit, T2I NLL improved
`0.0831→0.0614`, IT2I `0.1127→0.0893`, and reconstruction `0.0913→0.0681`;
all protected synthetic T2I/current/edit gates remained passing.

The corrected I2T run uses 410 training and 102 held-out captions. Held-out
NLL improved `9.977→4.602`; matched-vs-shuffled mean gap remained positive
(`+0.639→+0.173`) and 52.9% of samples had positive gaps. This proves the
pretrained language model is usable as a frozen prior and that the terminal
visual-to-language path is active, but it is not yet a robust I2T admission.
The candidate
`omni_d64_pythia_sharegpt4o_i2t_candidate.pt` preserves every B3 static gate.

### Five-sample overfit gate

A diagnostic bank of five real images uses short, image-specific captions and
explicit EOS supervision. With frozen Pythia and only `token_reader` trainable,
teacher token accuracy reaches 100% and graph-greedy exact reaches 5/5 by step
200 (`NLL=0.0845`), remaining 5/5 at step 600 (`NLL=0.0158`). Rotating the five
images rotates all five answers correctly and every answer stops at EOS. A
16×16 reload still passes T2I, current reconstruction/segmentation, and edit.
This establishes finite-sample expressivity and a working visual-language
training path; it is deliberately not evidence of held-out generalization.

The same gate now scales through a deterministic nested curriculum. Sixteen
real images reach token/graph-greedy/rotated-image exact
`100%/16-of-16/16-of-16`. Expanding that checkpoint to 32 images initially
retains exactly the first 16 mappings; balanced training, lower-rate finishing,
and paired hard-example replay then reach NLL `0.0339`, token accuracy `99.7%`,
graph-greedy `32/32`, rotated-image `32/32`, and EOS `32/32`. All 32 targets are
unique short prefixes derived from real captions. This is a stronger capacity
proof, but targets are intentionally compact diagnostic labels rather than
open-ended descriptions. The 32-image candidate again passes all three B3
static gates.

### Joint 1px OCR and generation gate

The 32-image candidate is extended with ten fixed noisy hard OCR samples, one
per digit, rendered as true 1px Bresenham strokes at 64×64. A 3:1 real-I2T/OCR
rehearsal plus low-rate paired replay reaches graph-greedy exact `42/42`:
real I2T `32/32`, OCR `10/10`, rotated-image `42/42`, and EOS `42/42`. Pythia
and every visual write path remain frozen. The same checkpoint independently
retains native 1px digit generation on 117 fixed prompts: digit `1.000`, color
`0.969`, paired IoU `0.941`, digit/color shuffles `0.000/0.000`, and flood
`0.00165`. Reconstruction/segmentation and next-color edit gates also pass.
This proves the requested fixed-bank OCR and generation coexistence, not font,
layout, or natural-image generalization.

### Natural-image T2I content gate

`train_sharegpt4o_t2i_overfit.py` uses two visually distinct Freedom T2I
targets at 16×16. The input is always an all-zero field with explicit
`image_precision=0`; frozen Pythia supplies prompt embeddings only, its causal
decoder is skipped, and the existing Slice–MoT–Deslice path emits RGB in one
pass. At step 480, prompt retrieval is `2/2`, PSNR is `20.16 dB`, and rotating
the prompts increases MSE from `0.00964` to `0.19477` (gap `0.18513`; output
RMS change `0.42735`). This rejects an unconditional-average shortcut, but the
original-PNG gallery exposed that it mainly matches low-frequency color. A
post-hoc edge audit gives relative edge MSE `0.803` and correlation `0.426`, so
the old pixel-only capacity decision is withdrawn.

At 64×64, the 16-Slice write remains a smooth field (PSNR `19.04 dB`, edge
correlation `0.133`). Opening the existing visual stem/KV/FFN/local path,
increasing to 64 slices, removing Gaussian variance-head incentive, and adding
a strong edge loss produces a coarse mask outline but still reaches only PSNR
`15.64 dB` and edge correlation `0.190`; the island and boat structures are
absent. The current graph therefore has prompt-selective low-frequency RGB
capacity, but has not demonstrated natural-image content overfit. It also fails
the protected static gates and is not admitted. See the original-target gallery
and `results/published/sharegpt4o_natural_t2i_r64_m64_capacity.json`.

Reproduce the finite-bank result with:

```powershell
python scripts/train_sharegpt4o_t2i_overfit.py --ids freedom-t2i-34407,freedom-t2i-3191 --resolution 64 --n-slices 64 --phase generation_capacity --nll-coef 0 --mse-coef 10 --edge-coef 200 --steps 600 --rehearsal-coef 0
```

## Next admission gate

Do not invent IT2T labels from caption data. Add a genuine image-conditioned
question/instruction dataset, retain a deterministic holdout, and require
improved matched likelihood plus a clearly positive per-sample shuffle gap.
Then train I2T and IT2T with T2T and static-capability rehearsal in the same
graph. Keep every real checkpoint candidate-only until text decode, natural
image geometry, and all protected gates pass together across multiple seeds.
