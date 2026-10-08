"""Checkpoint configuration identity survives execution metadata and JSON."""
import json

import pytest

from scripts.ib.train_fly_rtc_dagger import (
    execution_metadata, parse_args, validate_resume_configuration,
)


@pytest.mark.parametrize('backend', ['student', 'exact'])
@pytest.mark.parametrize('horizon', [0, 14])
def test_backend_and_response_budget_survive_checkpoint_roundtrip(backend, horizon):
    args = parse_args(['--backend', backend, '--read-horizon', str(horizon), '--calibrate'])
    config = {k: str(v) if hasattr(v, 'resolve') else v for k, v in vars(args).items()}
    config.update(execution_metadata(args))
    restored = json.loads(json.dumps(config))
    assert restored['backend'] == backend
    validate_resume_configuration(restored, args)
    changed = parse_args(['--backend', backend, '--read-horizon', str(14-horizon), '--calibrate'])
    with pytest.raises(ValueError, match='read_horizon'):
        validate_resume_configuration(restored, changed)


def test_formal_budget_guard_and_causal_horizon_guard():
    with pytest.raises(SystemExit):
        parse_args(['--backend', 'exact', '--additional-tokens', '32'])
    with pytest.raises(SystemExit):
        parse_args(['--backend', 'exact', '--read-horizon', '15', '--calibrate'])
