# MaleCNS Graph-Field CBIM

## Purpose

This branch replaces the artificial periodic cubic lattice with a coarse field
whose geometry and adjacency come from the official MaleCNS v1.0 connectome.
It tests a specific claim: a biologically evolved sparse topology may support
more distributed and persistent internal organization than a uniform lattice,
while the Information-Boltzmann update remains the learned dynamical law.

MaleCNS is a static connectome, not a measured dynamical brain.  The model does
not reproduce the source fly or import a LIF simulator.  It uses the connectome
as a spatial/topological prior for an independently trained language system.

## Data reduction

`build_malecns_graph.py` consumes the CC-BY MaleCNS v1.0 flat-connectome files.
The first artifact uses 140,024 traced neurons with soma coordinates.  A
deterministic 3D MiniBatchKMeans partition creates 256 parcels.  Streaming over
151,856,684 aggregate connection rows then produces a 256 by 256 directed
weighted adjacency without materializing the raw table in RAM.

The checked run uses:

- 256 anatomical parcels;
- eight strongest undirected neighbors per parcel for local context;
- 32 nonconstant normalized graph-Laplacian modes;
- four disjoint high-weight edge matchings for nonlinear collision;
- parcel coordinates, degree, cell count and neurotransmitter mixture as fixed
  node features.

The ignored `.npz` artifact and its JSON metadata contain checksums of all three
official inputs.  Checkpoints also record the graph artifact checksum.

## Dynamics

For field state `h` with shape `[batch, parcel, channel]`, each token applies:

1. state-aware spatially localized source and local outflow;
2. orthogonal rotation of paired graph-Laplacian coefficients;
3. state- and edge-aware pair collision on disjoint connectome matchings;
4. state-only multi-query readout.

If `Phi` is the orthonormal truncated graph basis, transport projects
`c = Phi^T h`, rotates pairs of modal coefficients, and reconstructs
`h + Phi(c' - c)`.  The operation is exactly norm preserving on the retained
subspace and leaves the orthogonal complement unchanged.

For every collision edge `(i,j)`, the update preserves `h_i + h_j` and
`||h_i||^2 + ||h_j||^2`.  Its learned angle reads both endpoint states and edge
weight, anatomical direction and connection asymmetry.  Source/outflow is the
only open-system exchange term.

## Initial run

The first OWT run uses 128 GPT-2 BPE tokens per optimizer update, 64 feature
channels and 3,000 updates.  It is a budget-matched training study, not a claim
of convergence.  The speed gate on a GTX 1650 is 1.5 seconds per update and the
memory cap is 2 GiB.  The measured smoke performance was 0.52--0.56 seconds per
update with 526 MiB reserved by PyTorch; sampled GPU compute utilization was
100 percent.

The necessary controls after training are collision bypass, transport bypass,
read/write routing entropy, parcel lesions, graph-mode occupancy, and a
degree-preserving randomized topology.  The randomized graph control is needed
to attribute an effect specifically to MaleCNS topology rather than generic
sparse connectivity.
