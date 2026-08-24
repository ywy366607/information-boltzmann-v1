# Transolver3 in multimodal (product naming)

> Supporting naming note. The current architecture source of truth is [`../../docs/NORTH_STAR.md`](../../docs/NORTH_STAR.md).

## Claim

**Transolver3’s natural multimodal form** is not “emit G slice tokens and feed
the LLM.” It is:

1. Each modality keeps a **native field** (own topology).
2. Soft **Read** into a temporary shared workspace **U** (slice / interact view).
3. **Write** back into that field (deslice / residual) so the field stays live.
4. Language is a peer with Read/Write on \(U\), not the only world model.

That is exactly the pattern in `NATIVE_MULTIMODAL_WORKSPACE.md` +
`FINAL_BIDIR_POINT_FIELD.md`.

## Abandoned as product vision

| Path | Status |
|------|--------|
| **B = permanent vision slice tokens → projector → LLM** (`SliceFrontend`) | **Dropped as product core** — historical A/B baseline only |
| Hybrid C token concat | Same: baseline / ablation |
| “Text reasons first, vision is frozen token prefix” | Rejected |

Why drop permanent slice tokens as vision?

- Once tokens replace the point field, **deslice has nowhere essential to land**.
- Transolver’s advantage (mass read + field write) is **discounted to a token mixer**.
- Multimodal extension becomes “more encoders in front of the LM,” not native fields.

## Product path (code)

| Name | Module |
|------|--------|
| Transolver3 V+L | `CrossModalSliceFrontend` (`meta.kind=B_xmodal`, `transolver=3_multimodal`) |
| build_frontend | kinds `B_xmodal` / `xmodal` / `B3` / `transolver3` |
| Legacy B tokens | `SliceFrontend` — keep for old tables; do not extend as product |

## Naming map

| Informal | Formal |
|----------|--------|
| Transolver++ / slice recon | Field + deslice wins on thin structure (encoder evidence) |
| **Transolver3 multimodal** | Shared **U** + per-modality **topology** + Read/Write |
| Old “VLM B frontend” | Slice **tokens** for LLM — **superseded** |
