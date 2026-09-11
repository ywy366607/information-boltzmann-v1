> Superseded for the single-individual OWT run: user explicitly authorized 5000 optimizer updates and GPU calibration. See INFORMATION_BOLTZMANN_INDIVIDUAL_RUN.md. The earlier hold below is historical.

# Training hold — 2026-09-10

User requested stopping training. The identified N=128 training process (PID 25224) was terminated. No new training is launched. Existing saved files are retained; unsaved in-memory progress is not claimed recovered.

All current CE/PPL and particle scaling results on the four-document smoke corpus are debugging measurements only. They do not establish generalization, useful continual learning, criticality or a scaling law. Non-collapse and finite moments over the observed interval are finite-time evidence, not an infinite-time guarantee.

The separate data preparation completed without training: pinned OpenWebText revision 433fe0f44ed7894fea29c08b3202aa348ccc6369, 10,000 source documents examined, exact-document hash splitting and deduplication. Prepared train/validation/test lengths: 48,145,263 / 468,139 / 420,184 byte tokens. Location: data/ib_owt_10000. This remains a subset of OpenWebText, not the full corpus. No model has been trained on this prepared corpus by this task.

Implemented repairs: gamma=0 thermal diffusion now agrees with the gamma→0 fluctuation-dissipation limit; incorrect adaptive controller is disabled; full-position-and-velocity response diagnostic and analytical oscillator test added; old criticality boolean classifications withdrawn; weight loading restores physical gamma/kappa; default trainer rejects tiny OWT fixtures and accidental overwrite of an existing run. 21 focused tests passed. The energy report no longer calls its endpoint estimate proof of NESS; substep power accounting remains outstanding. Full critical calibration and independent-stream evaluation remain pending; they were not replaced by more small-corpus optimization.

Training and deployment can have separate phase/clock/RNG/optimizer states with explicit parameter transfer. Coalgebraic behavioral equivalence should test causal prefixes, chunk invariance and checkpoint continuation; it cannot supply a stability or no-forgetting theorem.
