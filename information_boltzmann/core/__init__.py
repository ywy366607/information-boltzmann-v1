"""Canonical shared kinetic operators for Information Boltzmann."""

from .torus3d import (
    CBIMTorus3D,
    FullRankTorusWrite,
    KineticBeliefState,
    PredictiveImpedanceWriteAgent,
)
from .readout_probes import PredictivePhysicalReadAgent
from .universal_ports import CBIMUniversalPorts3D
from .mt_ponder import CBIMActivePonder3D, MTPonderOutput
from .mcr2_rate_distortion import (
    FlyAdaptiveAdmissionGatekeeper,
    MCR2Loss,
    compute_gaussian_coding_rate,
)

from .fly_bptt_learning import advance_fly_token_adaptive
from .variational_rolling_stream import (
    VariationalGaussianBelief,
    VariationalBeliefModulator,
    RollingStreamTransformer,
    VariationalRollingStreamLearner,
)

__all__ = [
    "CBIMTorus3D",
    "FullRankTorusWrite",
    "KineticBeliefState",
    "PredictiveImpedanceWriteAgent",
    "PredictivePhysicalReadAgent",
    "CBIMUniversalPorts3D",
    "CBIMActivePonder3D",
    "MTPonderOutput",
    "FlyAdaptiveAdmissionGatekeeper",
    "MCR2Loss",
    "compute_gaussian_coding_rate",
    "advance_fly_token_adaptive",
    "VariationalGaussianBelief",
    "VariationalBeliefModulator",
    "RollingStreamTransformer",
    "VariationalRollingStreamLearner",
]


