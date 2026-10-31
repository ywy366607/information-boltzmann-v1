"""Compile whole native contact families into the shared execution program.

The first template replaces existing verified KC delivery operators exactly.
It introduces no new biological law, parameter, physical history or soma
fallback. Subsequent receptor templates use the same typed/local interfaces.
"""
import hashlib

from .fly_native_runtime import NativeExecutionProgram, NativeStateLayout, NativeInstruction


def compile_existing_kc_routes(model):
    required = ('kct_index', 'apl_cable_index', 'danl_index', 'kc_terminal_digest',
                'apl_cable_digest', 'dan_release_digest')
    if any(not hasattr(model, name) for name in required):
        raise ValueError('KC/APL/DAN actual local anatomy must be assembled first')
    bindings = {
        'old_kc_packets': {'kind': 'component_state', 'component': 'kc_terminal', 'slot': 3},
        'new_kc_packets': {'kind': 'frame', 'name': 'kc_values', 'slot': 3},
        'fast_weights': {'kind': 'tensor', 'name': 'edge_weight_e'},
        'feedback_weights': {'kind': 'tensor', 'name': 'kcd_weight'},
        'apl_cells': {'kind': 'buffer', 'name': 'apl_cable_index'},
        'dan_density': {'kind': 'buffer', 'name': 'danl_density'},
        'dan_feedback_on': {'kind': 'flag', 'name': 'dan_local_feedback_enabled'},
    }
    instructions = []
    families = [
        ('kc_fast', 'kct_fast', 'pre_membrane', 'old_kc_packets', 'fast_weights', model.n_neurons, False),
        ('kc_apl', 'kct_apl', 'pre_membrane', 'old_kc_packets', 'fast_weights', model.aplc_owner.numel(), False),
        ('kc_dan', 'danl_in', 'post_release', 'new_kc_packets', 'feedback_weights', model.danl_owner.numel(), True),
    ]
    for label, prefix, phase, source, weights, size, weight_first in families:
        output_refs = []
        for lag in range(1, 5):
            arguments = [('packets', 'external:' + source), ('weights', 'external:' + weights)]
            for column in ('site', 'edge', 'fraction', 'order', 'count', 'target'):
                name = f'{label}_{lag}_{column}'
                buffer_name = f'{prefix}_{lag}_{column}'
                if buffer_name not in dict(model.named_buffers()):
                    raise ValueError('Missing actual native contact routing operand: ' + buffer_name)
                bindings[name] = {'kind': 'buffer', 'name': buffer_name}
                arguments.append((column, 'external:' + name))
            out = f'{label}_lag{lag}'
            instructions.append(NativeInstruction(out, 'site_packet_delivery', tuple(arguments), (out,), phase,
                (('size', int(size)), ('lag', lag), ('weight_first', weight_first))))
            output_refs.append((f'lag{lag}', 'new:' + out))
        instructions.append(NativeInstruction(label + '_sum', 'ordered_sum', tuple(output_refs), (label + '_sum',), phase))
        if label == 'kc_fast':
            instructions.append(NativeInstruction('fast_receivers', 'zero_at',
                (('value', 'new:kc_fast_sum'), ('index', 'external:apl_cells')), ('fast_receivers',), phase))
        if label == 'kc_dan':
            instructions.append(NativeInstruction('dan_normalized', 'multiply',
                (('value', 'new:kc_dan_sum'), ('factor', 'external:dan_density')), ('dan_normalized',), phase))
            instructions.append(NativeInstruction('dan_receivers', 'multiply',
                (('value', 'new:dan_normalized'), ('factor', 'external:dan_feedback_on')), ('dan_receivers',), phase))
    digest = hashlib.sha256(b''.join(getattr(model, name).detach().cpu().numpy().tobytes()
                        for name in ('kc_terminal_digest', 'apl_cable_digest', 'dan_release_digest'))).hexdigest()
    programme = NativeExecutionProgram(NativeStateLayout([]), instructions, tuple(bindings),
        anatomy_digest=digest,
        evidence={'scope': 'Lossless lowering of already implemented native contact delivery',
                  'kinetics': 'Existing full local states, resources and learning laws remain owned by their original components',
                  'not_new_biological_mechanisms': True},
        effectors=(('pre_membrane', 'extra_ge', 'fast_receivers'),
                   ('pre_membrane', 'apl_ge', 'kc_apl_sum'),
                   ('post_release', 'dan_receptor_input', 'dan_receivers')))
    identities = {name: getattr(model, name).detach().cpu().numpy().tobytes().hex()
                  for name in ('kc_terminal_digest', 'apl_cable_digest', 'dan_release_digest')}
    return {'program': programme.state_dict(), 'materials': {}, 'bindings': bindings,
            'identity_bindings': identities,
            'owned_routes': ['kc_fast', 'kc_apl', 'kc_dan'], 'state_device': 'brain'}
