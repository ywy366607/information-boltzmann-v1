# Archived Exploratory Scripts

This directory houses experimental, diagnostic, and predecessor scripts that were developed during early phases of the Information Boltzmann research and have since been consolidated into the canonical continuous-individual pipelines.

## Subdirectories

1. **`diagnostics/`**:
   - Diagnostic tools used during debugging of learning rates, gradient norms, surrogate transitions, and pipeline clock behaviors.
   - Includes one-off inspection scripts for individual evaluation events and loss trajectories.
2. **`benchmarks/`**:
   - Measurement and profiling scripts for testing spatial basis diversity, phase-locking, delay compensation, and vocabulary GEMM kernels.
3. **`exploratory_training/`**:
   - Earlier prototype training pipelines (e.g. streaming e-prop, delayed reservoir, and earlier continuous pipelines) that were superseded by the unified truncated BPTT with surrogate gradients (`train_fly_bptt_stream.py`).
4. **`smoke_tests/`**:
   - Intermediate boundary checks, rate-distortion MCR2 dual-engine smoke tests, and interface tests developed during experimental iterations.
