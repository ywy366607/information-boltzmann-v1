"""Explicit conversion between real gains, wood forecasts and critic values.

Legacy saved bodies retain schema zero until an explicit event-boundary repair.
No running moments, weights, historical gains or delayed traces are rescaled.
"""
from scripts.ib_local.language_navigation_kanli_v4b import SCALE as CRITIC_GAIN_SCALE


WOOD_GAIN_SCALE = 10.


def td_gain_target(engine, previous_gain, current_value, compatible=True):
    """Return a TD target in raw gain units, before critic target conversion."""
    divisor = CRITIC_GAIN_SCALE if getattr(engine, 'gain_units_schema', 0) == 1 else 10.
    return previous_gain + (engine.gamma*current_value/divisor if compatible else 0.)


def wood_choice_logits(engine, forecast):
    """Express a 10*g wood forecast in the units used by critic precision."""
    conversion = CRITIC_GAIN_SCALE/WOOD_GAIN_SCALE if getattr(engine, 'gain_units_schema', 0) == 1 else 1.
    return engine.precision()*conversion*forecast


def configure_gain_units(engine):
    if getattr(engine, 'joint_world_cycle_schema', None) != 1:
        raise TypeError('unit repair requires the inherited whole-world cycle')
    engine.gain_units_schema = 1
