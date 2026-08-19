# Bidirectional vision–language evolution

> **Product thesis:** `FINAL_BIDIR_POINT_FIELD.md` (full-res \(X\) + deslice) and
> `NATIVE_MULTIMODAL_WORKSPACE.md` (shared \(U\), Read/Write ports, keep topology).

## Equations

\[
X_{t+1} = F_{\mathrm{vision}}(X_t, H_t)
\]
\[
H_{t+1} = F_{\mathrm{language}}(H_t, \mathrm{Read}(X_{t+1}))
\]

Not a feed-forward vision encoder followed by a static LLM call: **both**
the visual field and the text state update each step, and language always
reads the **reorganized** field \(X_{t+1}\).

## Narrative ↔ maps

| Story | Map |
|-------|-----|
| Visual field \(X\) | state `X_t` |
| Text reads | `Read(X; H)` (after reorg: on \(X_{t+1}\)) |
| Stage judgment | inside `F_language` residual (and `H_t` as control for write) |
| Judgment writes back | part of `F_vision`: H → slice values → deslice into points |
| Field reorganizes | part of `F_vision`: Slice → SDPA → deslice |
| Next step re-reads | next layer’s `Read(X_{t+1})` |

## Code

| Symbol | Method |
|--------|--------|
| \(F_{\mathrm{vision}}\) | `CrossModalSliceLayer.F_vision` |
| \(\mathrm{Read}\) | `CrossModalSliceLayer.Read` |
| \(F_{\mathrm{language}}\) | `CrossModalSliceLayer.F_language` |
| one step | `CrossModalSliceLayer.forward` |
| L steps + tokens | `CrossModalSliceFrontend` |

Module: `fine_grain/cross_modal_slice_loop.py`  
Tests: `tests/test_cross_modal_slice_loop.py` (includes **X' depends on H**)

## Why the old 1→2→3→4 order was wrong

Previously: pure visual update **without** H, then read, then H update, then
writeback. That makes writeback only affect the *next* layer and breaks
\(X_{t+1}=F(X_t,H_t)\) inside the same step.

Now: **write + reorganize under H first**, then language reads the new field.
