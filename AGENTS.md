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

## Information Boltzmann experimental policy

- User directive (2026-09-10): capability research must use professional real-world datasets and sufficient training. Do not create or run synthetic/toy bit-memory, rule-switch, or cue-probe capability experiments, or make them prerequisites for dataset training.
- Before drawing positive or negative architecture/capacity/memory conclusions, record dataset revision and splits, parameter count, tokens/examples processed, actual optimizer updates, training curves, periodic independent validation, and the stopping criterion. Establish convergence or clearly label a budget-limited run inconclusive; a fixed short run is not evidence of convergence.
- Compare capacity and training budgets fairly. Online cumulative training CE is not final validation loss; identify tokenizer units and checkpoint provenance. Do not use the test split for tuning.
- Keep deterministic numerical and interface unit tests (conservation, analytic solutions, gradients, causality, checkpoint continuation). These verify implementation only and are not toy capability studies.

## Continuous-individual evaluation policy (2026-10-04)

- User directive: systems intended to persist and learn on infinite streams use active-learning evaluation as their primary protocol. Keep physical state, eligibility, weights, optimizer moments, pending gradients and elapsed time continuous. Use the actual learner; do not substitute an evaluation-only update algorithm.
- Cover three questions: first-pass prequential predictive quality, recovery after a real context change, and actual A-to-B-to-A revisit/relearning. Score predictions before the target can update their parameters. Report replay separately from first-pass traffic and preserve optimizer cadence across phase boundaries.
- Record real exposure intervals, fresh-stream cursors, all scored bridge transitions and actual optimizer updates. Never fill an intervening-duration field with unexecuted experience. Restore complete learning state on resume; no cold/reset evaluation of an ongoing individual.
- Historical frozen benchmarks retain their original provenance. Label new primary results as live/active-learning results rather than reinterpreting old scores. See docs/information_boltzmann/LIFELONG_EVALUATION.md.
- Fourth-pillar monitoring reports actual field/flux energy accounting, representation structure, temporal roughness, phase-separated risk drift and full physical-state conditional response. Keep predictive surprisal distinct from thermodynamic entropy flow; do not interpret low rank, smoother trajectories or small partial-gradient DNR as certified noise expulsion, criticality or optimal convergence. See docs/information_boltzmann/PERSISTENT_MEDIUM_HEALTH.md.
- Do not resurrect removed toy conclusions about decoder depth, memory failure, gamma superiority, or error-attribution root causes. The user has now authorized the first single-individual OWT run of 5000 optimizer updates, with learned local gamma and GPU/memory calibration. Preserve one continuous state and do not launch a population sweep as a substitute.

## Native fly efficiency policy (2026-10-19)

- Consider memory and compute during implementation, before adding a mechanism. Distinguish physical state, local eligibility, parameter sensitivities, optimizer state, autograd graphs, and CPU/GPU temporary workspace; record their dimensions and byte budgets.
- Prefer lossless index packing, shared immutable geometry, bounded workspaces and independent-RHS/edge blocking. Preserve all physical nodes, connections, real elapsed time and complete learning state; do not substitute pooling, shorter world clocks or reset histories for an efficient implementation.
- Ongoing native plasticity need not reproduce BPTT gradients. Fast local BPTT is allowed within a measured resource budget; do not require a dense global historical Jacobian to declare a biological pathway connected.
- Validate physical outputs and applicable derivatives after execution/storage changes. Report full-restore, forward, backward and optimizer memory separately; frozen replay alone is not training admission. Keep main training paused during repair unless the user resumes it.
- Custom autograd operators must put retained tensor data through `save_for_backward` so checkpoint hooks can release it. Flatten physical-state tensors into nested checkpoint inputs rather than capturing state objects in closures. Bound diagnostic temporaries such as finiteness checks on CPU and GPU, including strided large language-head gradients.

## Checkpoint retention during mechanism repair (2026-10-19)

- Full mechanism snapshots currently cost roughly 1.4–2.5 GB each even with zero new training. Budget complete snapshots and atomic replacement space before running a migration; do not place these archives on the system drive.
- For the ongoing mechanism-repair chain, normally retain the latest complete candidate and one immediate complete predecessor. After the new archive is verified and reports/configuration/provenance are preserved, remove obsolete reported smoke/interface-test payloads under the user's cleanup authorization. Write a deletion manifest and `CHECKPOINT_PRUNED.json`; historical result evidence is not proof that its old checkpoint file still exists.
- Keep the paused main individual, explicitly named key recovery points and separately requested historical baselines. Never infer that a reported mechanism test is a champion or keep every intermediate full snapshot indefinitely. Earlier smoke states with different brief exposures are obsolete test states, not necessarily byte-identical duplicates; describe deletion honestly.
- Preserve anatomy caches, source code, metrics, dialogue and failure evidence. Check actual free space after cleanup or relocation; moving data between drives does not reduce total storage. Do not change the system pagefile or interrupt unrelated jobs to free space.
