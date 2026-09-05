# Architecture Documents

## Source of truth

- [`NORTH_STAR.md`](NORTH_STAR.md) — the locked Slice full-resolution vision–language co-evolution contract, active-inference claim boundary, unified ports, and generation gates.
- [`STATUS.md`](STATUS.md) — **current snapshot for handing to another model**. Checkpoints, Pythia T2I/edit facts, failed recipes, and the next allowed experiments. Not a new north star.
- [`PYTHIA_INTEGRATION_PLAN.md`](PYTHIA_INTEGRATION_PLAN.md) — implementation plan for using frozen Pythia as the token prior/likelihood inside the same Slice–MoT graph without creating a language or image bypass.
- [`REAL_DATA_PILOT.md`](REAL_DATA_PILOT.md) — real ShareGPT-4o T2I/IT2I/I2T/IT2T mapping, selective archive preparation, first causal metrics, and the next admission gate.
- [`../research_tree.json`](../research_tree.json) — current hypothesis, evidence, conflict, and experiment dependency tree.

Read `NORTH_STAR.md` and `STATUS.md` before changing `native_mot.py`, `omni_model.py`, unified port behavior, generation, or Pythia wiring.

## Supporting designs

- `native_slice_mot_vlm_experiment.md` — preregistered large-scale native VLM experiment.
- `slice_navit_dual_stream_mot.md` — optional patch observation stream; patch is not the product's persistent visual state.
- `VLM_OCR_FRONTEND.md` — historical frontend/OCR integration notes.
- `NATIVE_256_CAPACITY.md` — native 256px scratch training, four-color/nine-address bank, memory budget and exact-raster evaluation.
- `REAL_256_JOINT.md` — current real-data mainline: native 256px shared Pythia/Slice generation, editing, captioning, reconstruction, DAVIS segmentation and next-frame prediction.
- `MIXED_NATIVE_CHAMPION.md` — shared native 64px/256px training, dynamic square-grid handling, paired horizon controls and remaining unified-checkpoint requirements.
- `SEVEN_UNIFIED_SOLUTIONS.md` — seven distinct repair mechanisms and independent agent selection; S1 observation weighting first, S3 coordinate encoding as fallback; not yet implemented or validated.

## Historical evidence and specifications

Documents under `results/published/` record earlier theses, experiment conclusions, and metric snapshots. They remain useful evidence, but names such as `FINAL_*` no longer make them architectural sources of truth. Where they conflict with `NORTH_STAR.md`, follow the North Star and retain the conflict in `research_tree.json`.
