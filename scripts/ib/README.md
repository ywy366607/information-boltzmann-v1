# Information Boltzmann entry points

New runnable entry points live here. Their shared library code belongs in
`information_boltzmann/`, and each run must have a JSON configuration in
`configs/information_boltzmann/`.

The destination line is **M8 complete fields with path-adaptive physical time
T at K=64**.  The current joint training stage is its required M=1 port-law
closure: W4 predictive writing, a field-only kinetic read agent, and one
persistent posterior precision state.  It uses the same K=64 interior flow;
M8 may begin only after this shared boundary law has a registered language
result. Do not add one-off mechanism sweeps or task-private solvers here.
Historical scripts remain under `scripts/ib_local/` for reproducibility.

## Language capability score

Use `evaluate_continuous_owt.py` for every language headline number. For a kinetic checkpoint it derives an energy-aligned NESS initializer from
the saved terminal field, randomizes only its Fourier phase, assimilates 256
observed local tokens, then scores the immediate 128-token continuation. The
GDN-2 reference uses its saved mature recurrence state and the same local text
window; each report names the model-specific state regime.

```powershell
D:\conda_envs\vox\python.exe scripts\ib\evaluate_continuous_owt.py `
  --checkpoint results\<run>\best.pt `
  --output results\published\<run>_warm_local_sites.json
```

## Predictive-port joint training

```powershell
D:\conda_envs\vox\python.exe scripts\ib\train_q8_port_agents.py `
  --output results\q8_predictive_ports_k64_3000
```

The entry point captures an 8-token BPTT chunk in CUDA Graph and carries both
the field and its channelwise posterior precision across all chunks.  It saves
only `BBest.pt` and `last.pt`; both include the posterior precision required
by warm-local evaluation.
