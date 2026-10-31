# Native antennal-lobe relay (2026-10-18)

The connectome already contains local antennal-lobe neurons and projection neurons, including inhibitory AL-to-PN contacts. This addition gives those contacts a persistent native role: spikes from annotated ALIN/ALON/ALLN/LN and related local neurons form a slow trace, which is delivered through the original inhibitory AL-to-PN edge list and changes PN excitability. Contact topology and weights remain fixed.

The local trace decay and PN response are learnable. The response starts at zero, so attaching the mechanism has a frozen-equivalence control. The trace and normalized PN exposure are saved in `FlyPhysicalState`, and the two parameters are registered in normal optimizer migration and native policy credit.

The real continuation validates graph reachability, persistent state, and finite gradients only. It does not yet establish odor discrimination, sparse PN coding, or a causal language benefit. Those require a matched sensory replay with an AL-off/sham branch and the same clock.
