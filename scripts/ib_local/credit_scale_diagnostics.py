"""Research-only gradient controls; never installed in the living optimizer."""
import math

import torch


def magnitude(values):
    return math.sqrt(sum(float(x.double().square().sum()) for x in values))


def match_layer_norms(reference, candidate, groups):
    """Keep each layer's old-credit norm while replacing only its direction."""
    result = [torch.zeros_like(x) for x in candidate]
    metadata = {}
    for layer, indices in groups.items():
        target = magnitude([reference[i] for i in indices])
        actual = magnitude([candidate[i] for i in indices])
        if target and not actual:
            raise ValueError('nonzero reference cannot be matched by a zero direction')
        scale = target/actual if actual else 0.
        for i in indices:
            result[i] = candidate[i]*scale
        metadata[str(layer)] = dict(reference_norm=target, raw_candidate_norm=actual, scale=scale)
    return result, metadata


def equal_layer_norms(current, old, groups):
    """Upper-dose diagnostic: same L2/RMS norm as current within each layer.

    This changes the native gradient; it is not an unbiased correction or a
    proposed production schedule. Zero credit remains exactly zero.
    """
    result = [torch.zeros_like(x) for x in old]
    scales = {}
    for layer, indices in groups.items():
        target = magnitude([current[i] for i in indices])
        actual = magnitude([old[i] for i in indices])
        scale = target/actual if actual else 0.
        for i in indices:
            result[i] = old[i]*scale
        scales[str(layer)] = scale
    return result, scales


def parameter_effect(reference, candidate, source, total_parameters):
    """Count actual native FP16 differences, not nonzero gradient coordinates."""
    count = sum(int(torch.count_nonzero(a-b)) for a,b in zip(reference, candidate))
    coords = sum(x.numel() for x in reference)
    delta = [b.float()-a.float() for a,b in zip(reference, candidate)]
    amount = magnitude(delta)
    native_step = [a.float()-s.float() for a,s in zip(reference, source)]
    step = magnitude(native_step)
    return dict(changed_coordinates=count, coordinates=coords,
        fraction_trainable_coordinates=count/coords,
        fraction_whole_model=count/total_parameters,
        difference_norm=amount,
        rms_over_changed_coordinates=amount/math.sqrt(count) if count else 0.,
        rms_over_all_coordinates=amount/math.sqrt(coords),
        difference_to_reference_step_norm=amount/step if step else None,
        relative_to_source_weight_norm=amount/magnitude(source) if magnitude(source) else None)
