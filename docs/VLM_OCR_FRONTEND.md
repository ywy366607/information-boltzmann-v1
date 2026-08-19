# Fine-grain vision frontend for VLM-OCR

## Product goal — Transolver3 multimodal (final)

**Transolver3 in multimodal** = native fields + shared workspace \(U\) via
symmetric Read/Write (see `NATIVE_MULTIMODAL_WORKSPACE.md`).  
**Not** “stack encoders → permanent vision **slice tokens** → LLM.”

**Abandoned as product vision:** legacy **B** (`SliceFrontend`) that turns the
image into a bag of slice tokens for the language model. That path keeps slice
as a *token product*, not a *field write* system — deslice becomes optional
decor, and Transolver is discounted. Code remains only for historical A/B
tables; new work uses `CrossModalSliceFrontend` / `B_xmodal`.

**Not** “text reasons first, then sprinkles vision tokens.”  
**Not** a pure vision-token pipeline that discards the point field.

**Final form:** bidirectional co-evolution of a **full-resolution point field**
\(X \in \mathbb{R}^{B \times N \times C}\) (\(N=H\cdot W\), always kept) and a
text/control state \(H\):

\[
X_{t+1} = F_{\mathrm{vision}}(X_t, H_t)
\]
\[
H_{t+1} = F_{\mathrm{language}}(H_t, \mathrm{Read}(X_{t+1}))
\]

- Soft mass **read** pools points → slices (amplitude-safe, content-adaptive).
- Sparse **deslice write** scatters slice updates **back to all points** —
  this is the Transolver property we refuse to give up.
- Language only **reads** and **writes through** that field; it does not
  replace \(X\) with a bag of LLM visual tokens mid-loop.

**Spec-accurate native MoT (current product core):** `fine_grain/native_mot.py`  
(`NativeMoTBlock`: private QKV/FFN; shared K/V space; SliceRead→MoT→Deslice→LocalVisual)

Earlier prototypes: `cross_modal_slice_loop.py`, `mot_coevolve.py` (shared MHA — **not** the full expert spec).

Thesis notes:

- Spec impl: `results/published/NATIVE_MOT_SPEC.md`
- Point field + bidir: `results/published/FINAL_BIDIR_POINT_FIELD.md`
- Shared workspace \(U\): `results/published/NATIVE_MULTIMODAL_WORKSPACE.md`

### Shared workspace \(U\) (native multimodal)

Communication is **symmetric**, not “all sensors → language tokens forever”:

| Port | Direction |
|------|-----------|
| Read_visual | visual field → \(U\) |
| Write_visual | \(U\) → visual field (deslice / field write) |
| Read_language | language state → \(U\) |
| Write_language | \(U\) → language state |

**Each modality keeps its own topology:**

- Image → **2D pixel / full-res point field**
- Audio → **high-frequency temporal field**
- Point cloud → **geometric point states**
- Language → **causal sequence**

Shared is **information** in temporary \(U\); future modalities plug in via
Read/Write to \(U\), without only stacking more encoders in front of an LLM.

This is **not** a claim that a frozen tiny text LM (e.g. Gemma-3-270M free-gen)
is already a product OCR system. Absolute open free-gen OCR scores under that
recipe are **out of scope** as a product proof.

## In scope

- Full-res point stream \(X\) maintained every step (deslice write target)
- Slice vs patch under matched **token/readout** budgets (slices are a *view*, not a replacement for \(X\))
- Soft mass **read** + sparse **write** (topk deslice)
- Bidirectional \((X,H)\) stack (`CrossModalSliceFrontend`)
- Optional recurrence / VL-CoT experiments as ablations toward the same thesis
- Metrics on the ladder below
- 4GB-class local experiments (`batch=1` / small res where needed; cache on `D:\ml_cache`)

## What we reject

| Anti-pattern | Why it fails the thesis |
|--------------|-------------------------|
| Encode once → fixed vision tokens → only text TF | No re-look; no deslice into points; Transolver wasted |
| “Language predicts, vision is just conditioning tokens” | Vision is not a live field; thin structure dies in the token bag |
| Dropping \(X\) after first slice pool | Cannot write judgments back to full-res geometry |
| Force every modality into text tokens as working memory | Destroys native topology (2D / time / geometry / causal seq) |
| New modality = only another encoder in front of LLM | No Write back into that modality’s own field |
| **Permanent vision = slice tokens (legacy B)** | No live field; Transolver3 multimodal abandoned |

## Bidirectional loop (locked)

Narrative per step: field \(X\) → text **reads** → stage **judgment** →
judgment **writes back** (deslice into points) → field **reorganizes**
(Slice→interact→deslice) → **re-read** next step.

`F_vision` = (H-condition + H→deslice write onto points) then global slice reorg.  
`F_language` = residual fuse of \(H_t\) with `Read(X_{t+1})`.  
After \(L\) steps, slices may be projected for an external LLM **readout** —
that projection is an **interface**, not the working visual memory.

## Out of scope (near term)

- Using **frozen tiny free-gen exact≈0** tables as “we can do OCR”
- Full finetune of large VLMs on 4GB
- Mandatory GDN-2; multi-seed marathons for every ablations
- Inventing metrics when a run OOMs

## KPI ladder (report in this order)

Never use open free-gen alone as the primary claim.

| Level | Metric | What it tests |
|-------|--------|----------------|
| 1 | **Recon Dice / IoU** | Encoder preserves thin geometry |
| 2 | **Closed-set probe acc** | Features are linearly readable (no LLM gen) |
| 3 | **Teacher-forced LLM answer-token acc** | VLM path under teacher force (stable) |
| 4 | **Digit-constrained CER / exact** | VLM free path with closed alphabet |
| 5 | **Open free-gen exact** | Diagnostic only (collapse / format noise) |

## Phase status

| Phase | Intent | Status |
|-------|--------|--------|
| 0 | This narrative + KPI ladder | **Done** (this doc) |
| 1 | Recurrence relative gains (`recur_T`, `K`) without frozen-LM OCR noise | **Done** — measured **ceiling null** on recon Dice; **null/negative** on digit-probe K (see `recurrence_phase1_conclusion.md`) |
| 2 | TF answer-span + digit-constrained CER (frozen small LM) | **Done** — honest **stuck**: A/B/B_vlcot TF≈0.33 exact=0 CER≈0.92 (see `vlm_path_tf_cer_conclusion.md`) |
| 3 | LoRA trainable language side (vs frozen) | **Done** — LoRA lifts TF (B 0.322→0.356); exact still 0 (see `vlm_path_lora_conclusion.md`) |
| 4+ | Longer curriculum / product multi-font OCR | Later |

## Related published artifacts

- Line recon (fair): `results/published/line_recon_64_fair.json`
- Probe vs VQA diagnosis: `results/published/vlm_probe_vs_vqa_*`
- Recurrence sweeps (Phase 1): `results/published/recur_recon_*`, `results/published/vlcot_probe_K_*`
- Phase-1 conclusion: `results/published/recurrence_phase1_conclusion.md`
- Phase-2 TF+CER: `results/published/vlm_path_tf_cer_table.json`, `vlm_path_tf_cer_conclusion.md`
- Phase-3 LoRA: `results/published/vlm_path_lora_table.json`, `vlm_path_lora_conclusion.md`
- Final thesis (point field): `results/published/FINAL_BIDIR_POINT_FIELD.md`
- Native multimodal workspace \(U\): `results/published/NATIVE_MULTIMODAL_WORKSPACE.md`
- **Transolver3 multimodal naming:** `results/published/TRANSOLVER3_MULTIMODAL.md`
