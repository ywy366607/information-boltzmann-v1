# CBIM-∞ 3D finite truncation

Status: implemented and training on OpenWebText as
`results/cbim_operator3d_3000`.

The persistent signed feature field is

\[
h_N(t)\in\mathbb R^{N_x\times N_y\times N_z\times d}.
\]

The current run uses `(4,4,4,128)`, equal to the 64 spatial sites and 128
channels of CBIM v2. Learned tensors contain no grid-sized parameters. The same
state dictionary loads strictly into another even spatial resolution; the grid
only supplies continuous torus coordinates and sampled Fourier wave numbers.

Each token applies state-aware local source/outflow, a 3D Fourier Cayley
transport, six shared conservative scattering layers (even/odd pairs on each
axis), and a state-only multi-query readout. Cayley transport preserves the L2
norm. Scattering preserves the spatial sum in every feature channel and total
quadratic energy. Source/outflow bounds every site's feature norm, and therefore
the volume-normalized energy, independently of the number of grid sites.

Strict cross-resolution loading proves parameterization independence. It does
not by itself prove numerical convergence to one continuum semigroup: the
nearest-neighbor context and finite scattering sweep still define a particular
discretization. A later refinement study must compare the projected trajectories
at `4^3`, `8^3`, and finer grids using identical weights. This does not block the
capacity-matched 3D training run.

Validation: seven 3D tests and the existing CUDA graph equivalence test pass.
The full FP32 `(B=1,T=128,4^3,d=128)` graph update runs at about 0.363 seconds on
the GTX 1650, with 365 MiB peak allocated in the isolated benchmark. The live
run enforces a 1.5-second ceiling after warmup and saves validation checkpoints
every 250 updates.
