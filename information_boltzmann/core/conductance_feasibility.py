"""Constructive conductance checks; parameter witnesses, not task capability."""
import math

import torch

from .conductance_response import LocalConductanceResponse


def conductance_energy_bound(coefficients):
    """dH/dt <= -m*H + U2/(2*m) for gates in[0,1], fixed finite material.

    Existing conservative transport/collision contribute zero storage power.
    Additional bounded boundary power P adds P to this inequality.
    The ideal reversal batteries are counted in U, not declared free energy.
    """
    c = coefficients
    minimum = torch.minimum((c.leak / c.capacitance).amin(),
                             (c.resistance / c.inductance).amin())
    source_bound = (c.maximum * c.reversal.abs()).sum(-2) / c.capacitance.sqrt()
    source_squared = source_bound.square().sum(-1).mean()
    return {'minimum_release_rate': minimum, 'source_squared_bound': source_squared,
            'storage_bound_without_extra_boundary_power': source_squared / (2 * minimum.square())}


def local_ei_certificate():
    """One realizable local inhibition-stabilized response with damped oscillation.

    Desired Jacobian [[-2,1,-1],[15,-6,0],[20,0,-2]] is a freely selected
    existence witness, not initialization or a frequency imposed on the model.
    Includes actual two-state receptor derivatives through the learned actor.
    """
    rule = LocalConductanceResponse(2, 3, 4).double()
    with torch.no_grad():
        for parameter in rule.opening.parameters():
            parameter.zero_()
        rule.opening[0].weight[0, 0] = 1
        for index, closing, slope in ((0, 3.0, 15.0), (2, 1.0, 20.0)):
            bias = math.log(math.expm1(closing))
            rule.opening[-1].bias[index] = bias
            # SiLU'(0)=1/2, open fraction=1/2, softplus'=sigmoid(bias).
        for index, closing, slope in ((0, 3.0, 15.0), (2, 1.0, 20.0)):
            bias = rule.opening[-1].bias[index]
            rule.opening[-1].weight[index, 0] = 4 * slope / bias.sigmoid()
        rule.log_parameters.bias[6 * 2] = math.log(3.0)
    material = torch.zeros(1, 1, 1, 3, dtype=torch.float64)
    coefficients = rule.coefficients(material)

    def rhs(values):
        field = torch.stack((values[0], values[0] * 0)).reshape(1, 1, 1, 1, 2)
        flux = tuple(torch.zeros_like(field) for _ in range(3))
        gates = torch.stack((values[1], values[1] * 0 + 0.5,
                             values[2], values[2] * 0 + 0.5)).reshape(1, 1, 1, 1, 2, 2)
        df, _, ds = rule.rhs(field, flux, gates, coefficients)
        return torch.stack((df.flatten()[0], ds[..., 0, 0].flatten()[0], ds[..., 1, 0].flatten()[0]))

    equilibrium = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64)
    jacobian = torch.autograd.functional.jacobian(rhs, equilibrium)
    return {'fixed_point_residual': rhs(equilibrium).detach(), 'jacobian': jacobian,
            'closed_loop_eigenvalues': torch.linalg.eigvals(jacobian),
            'excitation_only_eigenvalues': torch.linalg.eigvals(jacobian[:2, :2])}
