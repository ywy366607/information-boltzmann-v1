# Final architecture thesis: bidirectional full-res point field

> Historical thesis/supporting evidence. The current architecture source of truth is [`../../docs/NORTH_STAR.md`](../../docs/NORTH_STAR.md).

## One sentence

**Keep a full-resolution visual field \(X\) alive, evolve it with text state \(H\) every step via slice read + deslice write; do not replace vision with a static bag of LLM tokens.**

## Why this is the final form

Transolver-style physics attention is valuable **because**:

1. Soft **mass-normalized read** can pick thin structure without area dilution.
2. **Deslice write** returns information to **every point** (or a sparse support),
   so the working memory is still a spatial field, not \(G\) abstract tokens.

If the system collapses early to “vision tokens only” and never writes back to
points, deslice is decorative and the method is just another token mixer.
If language “reasons first” and vision only supplies a frozen prefix, there is
no **dynamic** vision—only delayed text.

Hence the product core is the coupled system:

\[
X_{t+1} = F_{\mathrm{vision}}(X_t, H_t)
\qquad
H_{t+1} = F_{\mathrm{language}}(H_t, \mathrm{Read}(X_{t+1}))
\]

with \(X \in \mathbb{R}^{B\times N\times C}\), \(N = HW\) **held every step**.

## Roles

| Object | Role |
|--------|------|
| \(X\) | **Primary visual memory** (full-res points). Always updated. |
| Slices | Temporary **view / interact space** for global SDPA (budget \(G\)). |
| Deslice | Write path: judgment / interaction → points (Transolver edge). |
| \(H\) | Text / control state (continuous and/or LLM hidden). Reads \(X'\), writes into \(X\). |
| Projected tokens | Optional **LLM interface** after the loop — not a substitute for \(X\). |

## Narrative cycle (must not reverse)

```
X (full-res field)
  → Read (text queries slices of current field)
  → stage judgment in H
  → write judgment back via deslice into points
  → reorganize field (slice interact + deslice)
  → next step reads the new field
```

## Implementation

- `fine_grain/cross_modal_slice_loop.py` — `F_vision` / `Read` / `F_language`
- Point stem keeps \(N=H\cdot W\); mixer deslice updates the same \(N\)
- `CrossModalSliceFrontend._last_x` retains final field for probes/recon
- Tests force **\(X'\) depends on \(H\)** (bidirectional, not FE-only)

## Relation to earlier experiments

| Path | Status vs thesis |
|------|------------------|
| Line recon + topk deslice | Proved **write-to-points** wins for thin structure |
| A/B VLM token frontends | Useful baselines; token-only path is **not** the end state |
| Frozen LM free-gen | Interface/readout problem; does not redefine the vision core |
| LoRA on LM | Helps \(H\) / readout; still secondary to keeping \(X\) live |
| **Bidir \((X,H)\) loop** | **Final product architecture** |

## Native multimodal generalization (locked companion)

Vision+language above is the first instance of a broader rule:

**Symmetric ports into a temporary shared workspace \(U\):**

- `Read_visual` / `Write_visual` — field \(X_v \leftrightarrow U\)
- `Read_language` / `Write_language` — state \(H \leftrightarrow U\)
- (later) audio / point cloud with the same Read/Write pair

**Each modality keeps its own topology:**

| Modality | Topology |
|----------|----------|
| Image | 2D pixel / full-res point field |
| Audio | High-frequency temporal field |
| Point cloud | Geometric point states |
| Language | Causal sequence |

**Shared is information, not “force everything into text tokens.”**  
Adding a modality = add a field + Read/Write into \(U\), not only bolt another encoder in front of the LLM.

Full write-up: **`NATIVE_MULTIMODAL_WORKSPACE.md`**.

## Non-goals (for this thesis)

- Replacing \(X\) with ViT patch tokens as the only state
- “Caption first, attend later” as the main control loop
- Dropping deslice because “LLM tokens are enough”
- Forcing every modality’s working memory to be a language-token sequence
