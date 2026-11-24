"""Connect relation retention evidence to new typed unknowns by fact identity."""


def configure(engine, mode):
    if mode not in ('off', 'relation'):
        raise ValueError('unknown retention scope prior')
    if getattr(engine, 'relation_retention_use_schema', None) != 2:
        raise ValueError('scope prior requires the existing finite-U/v5 retention law')
    engine.retention_scope_prior_mode = engine.model.retention_scope_prior_mode = mode
