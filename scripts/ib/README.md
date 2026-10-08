# Information Boltzmann: Fruit Fly Spiking Connectome & Active Streaming Entry Points

This directory contains the runnable entry points for the continuous-individual streaming learning research. Shared model architecture, biophysical dynamics, and credit assignment algorithms live in `information_boltzmann/core/`, and runtime active lifelong evaluation protocols live in `information_boltzmann/runtime/`.

The 3D-medium causal-evolution candidate and its numerical acceptance are described in
[MEDIUM_CAUSAL_EVOLUTION_REPAIR_20261008.md](../../docs/information_boltzmann/MEDIUM_CAUSAL_EVOLUTION_REPAIR_20261008.md).
Its training entry point is `train_medium_active_stream.py`; resolved physical intervals,
finite aperture budgets, runtime material bandwidth and structural-posterior options are
explicit CLI branches. The previous continuous individual remains paused. Adaptive interval
execution currently uses eager scheduling with fused medium kernels; complete-window resource
calibration precedes any new capability training.

---

## 1. Champion Model: Drosophila Connectome Spiking LM (`q8_fly_ctm_settle14_100k`)

The current winning continuous-individual model is built on the real **Drosophila MaleCNS connectome** (~13.8k neurons, ~1.4M synapses across 8 axonal conduction delay tiers).

### Biophysical Architecture (`information_boltzmann/core/fly_reservoir.py`)
- **8-Tier Axonal Conduction Ring Buffers**: Pure ring-buffer delay queues `p_i(t - d)` ($d \in \{1, \dots, 8\}$), eliminating legacy IIR low-pass filter broadening and magnitude distortion.
- **Conductance-Based Synaptic Integration (COBA)**: Dynamic excitatory and inhibitory conductances ($g_E, g_I$) with biophysical reversal potentials ($E_{\text{exc}}=0\text{ mV}, E_{\text{inh}}=-75\text{ mV}$).
- **Adaptive Spiking Dynamics (ALIF + STP)**: Adaptive membrane threshold $b_i(t)$ with slow decay time constant, combined with Tsodyks-Markram short-term synaptic depression and facilitation ($x_i, u_i$).
- **Topographic Sensory Injection & Dynamic Admission**: Sensory writing into topographic antennal lobe input clusters; gated by homeostatic `DynamicAdmissionAgent` which monitors neural flux and settles between 3 and 14 ticks (mean ~6.2 ticks), accelerating streaming throughput by >2× without loss of predictive accuracy.
- **Continuous Thought Machine (CTM) Observation**: Multi-timescale readouts across physical relaxation ticks ($t_0 \dots t_S$).

### Continuous Streaming Learner (`information_boltzmann/core/fly_bptt_learning.py`)
- **Truncated BPTT with Surrogate Gradients**: 32-token observation windows with threshold / ATan surrogate derivatives for non-differentiable spike events.
- **Continuous Individual Protocol**: Membrane potentials $h$, axonal pulse rings, synaptic conductances, ALIF thresholds, STP resources, and **all AdamW optimizer moments** persist seamlessly across window boundaries and evaluation phases. Zero cold resets.

---

## 2. Canonical Commands

### Resume Continuous Training
To resume the champion fruit fly individual from its paused checkpoint:

```powershell
D:\conda_envs\vox\python.exe scripts/ib/train_fly_bptt_stream.py `
  --resume E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt `
  --output results/q8_fly_ctm_settle14_100k `
  --checkpoint-dir E:/ib_checkpoints/q8_fly_ctm_settle14_100k `
  --adaptive-admission --min-settle-ticks 3 --flux-baseline 0.048 --settle-ticks 14 `
  --window 32 --detach-reset --surrogate-mode threshold --transmission-mode incoming `
  --lambda-mcr2 0.0 --no-graph
```

### Soft Pause (Graceful Interruption)
To pause training without corrupting physical states or losing AdamW optimizer history:
```powershell
New-Item -Path "results/q8_fly_ctm_settle14_100k/STOP" -ItemType File -Force
```
The process will finish its current 32-token window, atomically serialize `last.pt`, update `progress.json` with `"status": "paused"`, and cleanly exit. Delete `STOP` before resuming.

### Active Lifelong Evaluation Suite
To execute the four-pillar continuous evaluation independently:
```powershell
D:\conda_envs\vox\python.exe scripts/ib/evaluate_life_form_suite.py `
  --checkpoint E:/ib_checkpoints/q8_fly_ctm_settle14_100k/last.pt `
  --output results/q8_fly_ctm_settle14_100k
```

---

## 3. Four-Pillar Active Evaluation Protocol

Standardized in `information_boltzmann/runtime/lifelong_evaluation.py` following continuous-individual evaluation rules:

1. **Pillar 1: Prequential Stream Predictive Quality**
   - Every target is scored *before* its corresponding optimizer update.
   - Scored against a frozen training-only Unigram reference. Current rolling gain: **+0.5959 nats**.
2. **Pillar 2: Context Change Recovery & Adaptive Generalization ($\mathrm{AG}$)**
   - When entering a fresh, unobserved stream $B$ (256 tokens), the individual undergoes an environmental transition.
   - Evaluated via the unified **Adaptive Generalization ($\mathrm{AG}$)** metric:
     $$\mathrm{AG} = \frac{2}{2 + \frac{\tau}{\tau_{\text{ref}}} + \exp(L_\infty - L_{\text{ref}})}$$
   - $\tau$: Confirmed sustained stabilization time (requires $\ge 2$ consecutive 16-token blocks below baseline without relapse).
   - $L_\infty$: Model steady-state plateau NLL on the trailing tail window (tokens 192~256).
   - $L_{\text{ref}}$: Reference Unigram baseline on the **exact same trailing tail window**.
   - Current 120k milestone result: **$\mathrm{AG} = 0.6016$** ($L_\infty = 5.9189$ vs $L_{\text{ref}} = 6.7184$, $\Delta L = -0.7995$, $Q = 0.6899$, $S = 0.5333$).
3. **Pillar 3: Revisit & Relearning Acceleration (A-to-B-to-A Revisit)**
   - Scored on the same text $A$ after 256 intervening events in stream $B$.
   - Current 120k milestone: $A_1 = 6.3225 \to A_2 = \mathbf{5.8990}$ nats (**$+0.4235$ nats saving / $+6.70\%$ gain**; Relearning acceleration: **$1.40\times$**).
4. **Pillar 4: Physical Medium Health & Representation Structure**
   - Centered effective rank of hidden neural activations ($\mathbf{6.85}$, preventing manifold dimensional collapse).
   - Firing rate stability (~$2.51\%$) and field energy balance (~$0.59$).

---

## 4. Directory Organization & Archive

- **Active Scripts (`scripts/ib/`)**:
  - `train_fly_bptt_stream.py`: Canonical fruit fly continuous training runner.
  - `evaluate_life_form_suite.py`: Four-pillar lifelong capability suite.
  - `visualize_fly_brain_conduction.py`: Spatiotemporal signal conduction visualization across connectome.
  - `verify_clocks_and_binding.py`: Hierarchical clock and temporal binding verification tool.
  - `build_fly_reservoir_graph.py`, `build_coba_graph.py`, `build_fly_reservoir_biological_graph.py`, `build_fly_reservoir_delayed_graph.py`: Connectome graph builder utilities.
  - `benchmark_fly_event_timing.py`: Spiking network timing and execution benchmark.
  - `extend_owt_gpt2_stream.py`: OpenWebText stream expander.
  - `train_medium_active_stream.py`, `benchmark_medium_event_fusion.py`: Non-fly continuous tensor medium model runner.
- **Archived Scripts (`scripts/ib/archive/`)**:
  - `diagnostics/`: One-off diagnostics, audits, and inspector scripts from exploratory phases.
  - `benchmarks/`: Exploratory profiling, measurement, and probe scripts.
  - `exploratory_training/`: Superseded earlier training experiments (e.g. streaming e-prop, early pipeline prototypes).
  - `smoke_tests/`: Intermediate numerical and boundary check scripts.
- **Checkpoints Retention Policy**:
  - **Preserved in Full**: `E:/ib_checkpoints/q8_fly_ctm_settle14_100k/` (`last.pt`, `best.pt`, `segments/learning_mode_before_100000/best.pt`).
  - **Purged**: Obsolete smoke test `.pt` checkpoints (~8 GB freed); all configuration files (`config.json`), execution logs (`progress.json`), metrics (`metrics.jsonl`), and evaluation ledgers (`lifelong_evaluation.jsonl`) remain permanently archived for historical reference.
