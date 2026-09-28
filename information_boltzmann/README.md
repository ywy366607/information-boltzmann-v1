# Information Boltzmann

This package contains the canonical implementation of the shared kinetic
medium. It is independent of task-specific data formatting.

## Active contract

`M=8` complete periodic continuous-velocity fields evolve for learned pathwise physical
duration `T`; `K=64` is the fixed numerical integration resolution. The field
operators are W2 boundary write, spectral transport, invariant collision,
explicit bath, and query readout.

The same core serves ordered GPT-2 BPE language events and generic
observation/query ports used by Sudoku. Continuous no-reset OpenWebText NLL is
the primary capability metric. Sudoku blank-cell NLL and accuracy test whether
the same internal paths improve generic reasoning.

## API

```python
from information_boltzmann import CBIMTorus3D, CBIMUniversalPorts3D
```

Use `CBIMTorus3D` for sequential-token field evolution and
`CBIMUniversalPorts3D` for generic port-shaped observations and queries.
New code must import from this package. Historical implementations remain in
`scripts/legacy/ib_local/` and are available only through the compatibility
namespace `scripts.ib_local`.

See `docs/IB_NORTH_STAR_AND_NEXT_EXPERIMENT.md` for the experimental contract
and `docs/information_boltzmann/LAYOUT.md` for repository navigation.
