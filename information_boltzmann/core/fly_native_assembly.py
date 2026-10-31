"""Batch anatomy/material assembly from the single executable registry."""
from importlib import import_module
from pathlib import Path

from .fly_native_runtime import attachment_order


SPATIAL_COMPONENTS = attachment_order()


def assemble_spatial_components(model, args, *, saved=None, previous_config=None):
    previous_config = previous_config or {}
    graph = Path(args.graph)
    completed = set()
    records = []
    for spec in SPATIAL_COMPONENTS:
        binding = spec.attachment
        module = import_module('.' + spec.module, __package__)
        parameters = spec.names('parameters')
        identities = spec.names('identities')
        keys = parameters + identities
        archived = saved is not None and any(key in saved for key in keys)
        enabled = bool(getattr(args, binding.config_flag, False))
        if not enabled and not archived:
            continue
        if set(binding.dependencies) - completed:
            raise ValueError(f'{spec.name}: missing anatomical dependencies {set(binding.dependencies) - completed}')
        eager = any(name == 'apl_geometry' or name.endswith('_geometry') for name in binding.arguments)
        if eager and not bool(getattr(args, 'no_graph', False)):
            raise ValueError('Native spatial mechanisms require eager/checkpointed execution (--no-graph)')
        if previous_config.get(binding.config_flag) and saved is not None and any(key not in saved for key in keys):
            raise ValueError('Incomplete continuing native component: ' + spec.name)
        # Earlier newborn phases can have initialized the same material law.
        # Reuse it so neither parameter order nor RNG nor state is replaced.
        existing = bool(keys) and all(hasattr(model, key) for key in keys)
        if not existing:
            builtins = {'graph': graph, 'annotations': graph.parent / 'body-annotations.feather',
                        'dendritic_shares': graph.parent / 'dendritic_share_Kenyon_Cell.feather',
                        'apl_geometry': graph.parent / 'spatial/apl_cable_native_v1.npz'}
            positional = [model]
            for name in binding.arguments:
                value = builtins[name] if name in builtins else getattr(args, name, None)
                if value is None:
                    raise ValueError('Missing actual anatomical geometry for ' + spec.name)
                positional.append(value)
            keywords = {'saved': saved if archived else None} if binding.accepts_saved else {}
            getattr(module, binding.function)(*positional, **keywords)
        elif saved is not None:
            raise ValueError('Continuing native component was attached twice: ' + spec.name)
        if binding.gate:
            setattr(model, binding.gate, enabled)
        for target, config_name in binding.additional_flags:
            setattr(model, target, bool(getattr(args, config_name, False)))
        completed.add(spec.name)
        records.append({'component': spec.name, 'enabled': enabled, 'archived': archived,
                        'reused_newborn_component': existing,
                        'identities': list(identities), 'parameters': list(parameters)})
    return records
