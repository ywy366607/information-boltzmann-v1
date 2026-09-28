# Information Boltzmann layout

| Location | Role |
| --- | --- |
| `information_boltzmann/` | Canonical shared field operators and task-neutral APIs. |
| `scripts/ib/` | Supported language training and evaluation entry points. |
| `configs/information_boltzmann/` | Versioned run contracts. |
| `tests/test_ib_*.py` and `tests/test_mt_ponder.py` | Deterministic operator and interface tests. |
| `results/published/` | Curated reports and the evidence registry. |
| `scripts/ib_local/` | Historical implementations and reproduction material. New research code does not go here. |

Every new experiment needs one configuration, one entry point, one result
directory, and one evaluation record.  Exploratory scripts and intermediate
checkpoints belong outside these locations until they earn a registered claim.
