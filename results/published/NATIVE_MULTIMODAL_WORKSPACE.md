# Native multimodal: shared workspace \(U\), not “everything → text tokens”

> **This is Transolver3 applied to multimodal systems.**  
> Naming note: `TRANSOLVER3_MULTIMODAL.md`.  
> Legacy “vision = permanent slice tokens → LLM” is **not** Transolver3 product.

## One sentence

**Modalities keep their own topology; they only exchange *information* through a temporary shared workspace \(U\) via symmetric Read/Write ports — they are not permanently translated into language tokens (nor into a frozen bag of vision slice tokens).**

## Symmetric communication interfaces

| Port | Direction | Meaning |
|------|-----------|---------|
| **Read_visual** | visual field \(X_v\) → shared workspace \(U\) | Lift geometry-aware content into \(U\) (e.g. soft slice/pool) |
| **Write_visual** | \(U\) → visual field \(X_v\) | Write back into **points/pixels** (deslice / residual field update) |
| **Read_language** | language state \(H\) → \(U\) | Lift sequential / causal text state into \(U\) |
| **Write_language** | \(U\) → language state \(H\) | Update sequence / continuous \(H\) from \(U\) |

These four are **symmetric comms ports**, not a one-way encoder stack in front of an LLM.

Optional future ports (same pattern):

| Port | Direction |
|------|-----------|
| Read_audio / Write_audio | \(X_a \leftrightarrow U\) |
| Read_points / Write_points | \(X_p \leftrightarrow U\) |

If state widths are unified and \(U\) lives in a **shared canonical space** (common channel dim / norm), many of the per-modality maps can shrink further — fewer bespoke projectors, same Read/Write contract.

## What is shared vs what is not

| Shared | **Not** shared / not forced |
|--------|------------------------------|
| **Information** in temporary workspace \(U\) | Permanent collapse of every modality into **text tokens** |
| Communication protocol (Read_*/Write_*) | One topology (causal sequence) for all senses |
| Optional common channel width / norm on \(U\) | Stacking new encoders forever in front of the LLM only |

## Each modality keeps its own topology

```
  visual field   X_v     (2D pixel / point field)
  audio field    X_a     (high-rate temporal field)
  point/spatial  X_p     (geometric point states)
  language       H       (causal sequence / text state)
          │
          ▼
  temporary shared workspace U
          │
          ▼
  write back into each modality’s own state
```

| Modality | State | Topology kept |
|----------|--------|----------------|
| Image / video frames | \(X_v\) | **2D pixel (or full-res point) field** — \(N=H\cdot W\), deslice returns here |
| Audio | \(X_a\) | **High-frequency temporal field** — not forced into sentence tokens mid-loop |
| Point cloud / 3D | \(X_p\) | **Geometric point states** (neighbors / coords), not rasterized text |
| Language | \(H\) | **Causal sequence** (and/or continuous control state) |

**Shared is information, not an obligation that every modality become text tokens.**

## Why this is “truly native” multimodal

1. **No permanent vision→token translation as the working memory.**  
   Tokens (if any) are an **interface** to an external LM, not the sole visual store.

2. **New modalities do not require “another encoder bolted in front of the LLM” only.**  
   Add \(X_\star\) + `Read_★` / `Write_★` into the same \(U\) loop; keep that modality’s topology.

3. **Transolver / deslice stays meaningful.**  
   Write_visual lands on the **visual field**, not only on abstract token slots.

4. **Language is one peer, not the only world model.**  
   \(H\) reads/writes \(U\); it does not own all other states as frozen prefixes.

## Relation to current vision–language core

Today’s implementation (`cross_modal_slice_loop.py`) is the **vision + language** instance of this pattern:

| Native port | Approx. in code today |
|-------------|------------------------|
| Read_visual | soft mass pool / slices of \(X\) (and slice interact view) |
| Write_visual | H-conditioned deslice residual + Block deslice reorg on points |
| Read_language | \(H \to\) query / contribution into the interact path |
| Write_language | `F_language` residual update of \(H\) from read of reorganized field |
| Workspace \(U\) | temporary slice / interact space (budget \(G\)); not a replacement for \(X\) |

Full-res point-field thesis: `FINAL_BIDIR_POINT_FIELD.md`.

## Equations (multi-modality sketch)

For each modality \(m\) with state \(X^{(m)}\) (language may use \(H\)):

\[
U \leftarrow \mathrm{Mix}\big(\{\mathrm{Read}_m(X^{(m)})\}_m\big)
\]
\[
X^{(m)} \leftarrow \mathrm{Write}_m\big(X^{(m)}, U\big)
\]

Order and Mix details are implementation choices; the **invariant** is: topology of \(X^{(m)}\) is preserved, and \(U\) is ephemeral communication, not the only long-term store for every sense.

## Non-goals

- “Everything is a string of LLM tokens” as the only state representation  
- Adding modality \(k+1\) solely by stacking another frozen encoder in front of the same LM  
- Dropping field write-back because “the shared tokens already mixed”
