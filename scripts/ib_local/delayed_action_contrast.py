"""First-choice contrast receipt. No optimizer, reward or validation-data writes."""
from math import fsum, isfinite


def arrived_contrast(production, without_offset, chosen, outcome, forecast=None):
    """One natural draw, same H/units for the arrived score and pre-action forecast.

    The two policies share their support. Pass only legal actions. The caller
    owns the boundary, sampling law and action-independent probe protocol.
    """
    p, blind = tuple(production), tuple(without_offset)
    m = (0.,)*len(p) if forecast is None else tuple(forecast)
    if not (len(p) == len(blind) == len(m) and 0 <= chosen < len(p)
            and all(isfinite(x) and x > 0 for x in p)
            and all(isfinite(x) and x >= 0 for x in blind)
            and abs(fsum(p)-1) < 1e-6 and abs(fsum(blind)-1) < 1e-6
            and all(isfinite(x) for x in (*m, outcome))):
        raise ValueError('invalid legal-action receipt')
    d = tuple(x-y for x,y in zip(p, blind))
    return fsum(x*y for x,y in zip(d, m)) + d[chosen]/p[chosen]*(outcome-m[chosen])
