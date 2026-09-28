"""Information Boltzmann canonical package.

New experiments belong here.  ``scripts.ib_local`` is retained only for
historical checkpoint and test compatibility.
"""

from .core.torus3d import CBIMTorus3D
from .core.universal_ports import CBIMUniversalPorts3D

__all__ = ["CBIMTorus3D", "CBIMUniversalPorts3D"]
