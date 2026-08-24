# Repository Guidelines

## Project Structure & Module Organization

`fine_grain/` is the Python package; preserve public imports from `fine_grain/__init__.py`. Put entry points in `scripts/`, tests in `tests/`, architecture documents in `docs/`, and showcase assets in `present/`. Commit only curated metrics under `results/published/`; datasets, checkpoints, and other result files are ignored.

## Architecture Guardrails

Read `docs/NORTH_STAR.md`, `docs/ACTIVE_INFERENCE_GDN2.md`, and `research_tree.json` before changing the multimodal graph. The deliverable is one checkpoint for text, perception, generation, editing, reconstruction, segmentation, and next-frame prediction. Preserve the full-resolution field, transient Slice workspace, joint language evolution, and Slice/Deslice write path. Express ports through modality precision, prediction horizon, and likelihood heads—not private backbones or output-controlled dynamics. Keep semantic F2 generation distinct from the physical-time GDN-2 prior while sharing the same graph. Add uncertainty loops only after the non-recurrent capability matrix passes. Update the research tree when a claim or invalidation changes.

## Build, Test, and Development Commands

- `python -m venv .venv` then `.venv\Scripts\Activate.ps1`: create a Windows environment.
- `pip install -r requirements.txt` then `pip install -e .`: install dependencies and the package.
- `pytest -q`: run the full CPU-oriented test suite.
- `python scripts/train_northstar_capabilities.py --active-gdn2 --active-gdn2-initial-trust 0.1 --no-future-language-evidence --temporal-only --steps 50`: reproduce the current causal Slice candidate.
- `pytest tests/test_native_mot.py -k test_native_mot_block_forward`: run one focused behavior.
- `python scripts/train_benchmark.py --task needle --arms patch4,slice --steps 50 --seeds 1`: run a short benchmark smoke test.
- `python scripts/train_benchmark.py --bench --device cuda`: run the optional CUDA benchmark.

## Coding Style & Naming Conventions

Target Python 3.10+ and follow PEP 8: four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Retain type hints and document non-obvious tensor semantics. Keep experiment CLIs explicit with `argparse`; avoid machine-specific paths. No formatter or linter is enforced, so match nearby code.

## Testing Guidelines

Use pytest and name tests `test_<behavior>`. Add deterministic tensor tests for model or routing changes; cover shapes, gradients, defaults, and invariants instead of long training convergence. Run the affected test file, `python tests/test_deslice_scatter_and_gate.py`, and the full suite when shared APIs change.

## Commit & Pull Request Guidelines

Recent history uses imperative subjects such as `Fix probe_token_acc causal alignment`. Keep commits focused. PRs should state the falsifiable claim, summarize code and metric changes, list validation commands, and link issues. Include figures for visual changes and compact artifacts under `results/published/`; exclude datasets, checkpoints, caches, and raw logs.
