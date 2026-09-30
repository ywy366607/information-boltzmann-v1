# Q8 bath-scale closure

The periodic Q8 field separates four roles: the predictive port supplies an
innovation, transport moves it, collision scatters it locally, and the bath
exports energy.  A spatial Laplacian in the bath has a fifth role: it erases
spatial variation.  On its own that can be a valid diffusion model, but it is
not a neutral way to stabilize a persistent spatial memory.

For the implemented spectral layer,

\[
\partial_t \widehat F_k=-\nu |k|^2\widehat F_k,
\qquad
\widehat F_k(T)=e^{-T\nu|k|^2}\widehat F_k(0).
\]

The unit three-torus has a DC mode with \(|k|^2=0\) and a first non-DC mode
with \(|k|^2=(2\pi)^2\).  With the historic unified-bath initialization
\(\nu=0.02\) and the registered duration \(T=64\), the first non-DC
amplitude is multiplied by

\[
e^{-64\cdot0.02\cdot(2\pi)^2}\simeq1.1\times10^{-22},
\]

or \(1.3\times10^{-44}\) in energy.  The DC state is the unique long-lived
spatial state of that operator.  This directly explains why both W2 and W4
unified-bath arms measured 100% DC spectrum.

`QuadraticTorusBath` is the current language main line because its local
energy outflow does not insert a fixed \(|k|^2\) preference.  It allows
transport, collision, and localized writing to retain their separate spatial
roles.  A future diffusion arm needs to specify a dimensionless retained-mode
factor, such as \(T\nu(2\pi/L)^2\), before it may be compared with this
main line.

The deterministic test
`test_unified_spectral_viscosity_annihilates_unit_torus_fundamental_at_t64`
checks this calculation against the actual implementation.  The associated
language claim still requires the registered 3000-update W4-quadratic K=64
run and four-site warm-local evaluation.
