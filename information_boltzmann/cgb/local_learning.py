"""Local learning affinities of the existing finite material reactions.

This is a constitutive interface, not a second update rule. It exposes the
actual electrochemical condition for a forward write/active event, including
the finite charge quantum, shared substrate and reciprocal support energy.
No moving firing-rate target, history buffer or external score is introduced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def depletion_shift(charged: int, depleted: int) -> float | None:
    """Chemical-affinity increase after one P -> D at fixed other state.

    None means the next forward event is forbidden because its P is gone.
    The reverse transition is still allowed; this is not an infinite energy
    value inserted into the physical free energy.
    """
    if (not isinstance(charged, int) or not isinstance(depleted, int)
            or charged < 1 or depleted < 0):
        raise ValueError('Available integer forward substrate required')
    if charged == 1:
        return None
    return math.log((depleted + 2) / (depleted + 1)) + math.log(charged / (charged - 1))


@dataclass(frozen=True)
class LocalLearningAffinity:
    kind: str
    resource_owner: int
    electrical_source: int
    electrical_target: int
    charged: int
    depleted: int
    support_occupied: int
    voltage_drop: float
    finite_quantum_energy: float
    drive_energy: float
    storage_energy: float
    composition_energy: float
    half_attempt_voltage_drop: float
    delta_u: float
    delta_g: float
    attempt_rate: float
    actual_rate: float
    next_consumption_threshold_shift: float | None

    def state_dict(self):
        return asdict(self)


def local_learning_affinity(model, event) -> LocalLearningAffinity:
    """Factor an admitted forward event in O(1), without altering state/RNG.

    half_attempt_voltage_drop is the drop at which this particular forward
    event has half its attempt rate. It is NOT the zero of net forward/reverse
    flux: the reverse lives in the adjacent state, with different composition.
    Current materials use kBT=1. Index >= n denotes an actual world capacitor.
    """
    if event.kind not in ('write', 'active') or event.sign != 1:
        raise ValueError('A forward write or active event is required')
    p = model.params
    if not p.charge_per_event > 0:
        raise ValueError('Positive physical charge quantum required')
    i = int(event.i)
    if not 0 <= i < model.n:
        raise ValueError('Known material resource owner required')
    if event.interface >= 0:
        a = int(event.interface)
        if (not hasattr(model, 'port_installed') or not 0 <= a < len(model.port_nodes)
                or not model.boundary_reactions_enabled or not model.port_installed[a]
                or i != int(model.port_nodes[a]) or int(event.j) != i):
            raise ValueError('Installed material-owned boundary contact required')
    else:
        e = int(event.edge)
        if (not 0 <= e < len(model.pre) or i != int(model.pre[e])
                or int(event.j) != int(model.post[e])):
            raise ValueError('Existing directed material bridge required')
    bound, trace, index = model._contact_arrays(event)
    if model.charged[i] < 1 or event.kind == 'write' and trace[index]:
        raise ValueError('Forward event lacks substrate or empty storage')
    changes = model._charge_changes(event)
    z = float(p.charge_per_event)
    if len(changes) != 2 or changes[0][1] != -z or changes[1][1] != z:
        raise ValueError('This factorization requires the declared charge-transfer law')
    source, target = int(changes[0][0]), int(changes[1][0])

    def electrical(node):
        if node < model.n:
            return float(model.charge[node]), float(model.capacitance[node])
        a = node - model.n
        return float(model.world_charge[a]), float(model.world_capacitance[a])

    qs, cs = electrical(source)
    qt, ct = electrical(target)
    drop = qs / cs - qt / ct
    quantum = z * z * .5 * (1 / cs + 1 / ct)
    fuel, spent = int(model.charged[i]), int(model.depleted[i])
    storage = float(p.trace_energy - p.capture_coupling * bound[index]) if event.kind == 'write' else 0.
    drive = -float(p.charged_energy)
    mixture = math.log(spent + 1) - math.log(fuel)
    du = -z * drop + quantum + drive + storage
    dg = du + mixture
    original_u, original_g = model.event_delta(event)
    if not (math.isclose(du, original_u, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(dg, original_g, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('Material charge/storage law changed; derive its factorization first')
    return LocalLearningAffinity(event.kind, i, source, target, fuel, spent,
        int(bound[index]), drop, quantum, drive, storage, mixture,
        (quantum + drive + storage + mixture) / z,
        float(original_u), float(original_g), float(model.attempt_rate(event)),
        float(model.event_rate(event)), None if fuel == 1 else depletion_shift(fuel, spent) / z)
