"""Persist validation time in organism cycles, independently of worker restarts."""


def configure_validation_schedule(config, start_cycle, every, bytes_each):
    """Bind one fixed-data clock to the stage origin without changing learning."""
    if every == 0:
        return None
    if every < 1 or bytes_each < 512:
        raise ValueError('invalid independent validation budget')
    prior = config.get('validation_schedule')
    if prior is not None:
        if (prior.get('schema') != 1 or prior['every'] != every
                or prior['bytes_each'] != bytes_each):
            raise ValueError('inherited validation schedule requires explicit migration')
        if not 0 <= prior['origin_cycle'] <= start_cycle:
            raise ValueError('invalid inherited validation origin')
        return prior
    migration = config.get('world_source_migration', {})
    origin = migration.get('at_cycle', start_cycle) if config.get('world_source') == 'text' else start_cycle
    if not 0 <= origin <= start_cycle:
        raise ValueError('invalid validation origin')
    schedule = dict(schema=1, origin_cycle=origin, every=every, bytes_each=bytes_each,
        scope='fixed independent sidecars; cadence survives process restart')
    config['validation_schedule'] = schedule
    return schedule


def validation_due(cycle, schedule, *, final=False):
    return (final or (cycle > schedule['origin_cycle']
            and (cycle-schedule['origin_cycle']) % schedule['every'] == 0))
