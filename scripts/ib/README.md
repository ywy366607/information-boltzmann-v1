# Information Boltzmann entry points

New runnable entry points live here. Their shared library code belongs in
`information_boltzmann/`, and each run must have a JSON configuration in
`configs/information_boltzmann/`.

The active line is **M8 complete fields with path-adaptive physical time T at
K=64**. Do not add one-off mechanism sweeps or task-private solvers here.
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
