"""Causal exposure/clock contracts; these are not capability experiments."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


_PATH=Path(__file__).resolve().parents[1]/'scripts/ib/generative/e2_sensitivity.py'
_SPEC=importlib.util.spec_from_file_location('e2_sensitivity',_PATH)
audit=importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def test_lag_changes_only_past_inputs_and_keeps_target_alignment():
    stream=np.arange(200)
    normal,targets=audit.exposure_tokens(stream,80,40)
    delayed,delayed_targets=audit.exposure_tokens(stream,80,40,lag=32)
    np.testing.assert_array_equal(targets,delayed_targets)
    np.testing.assert_array_equal(targets-normal,1)
    np.testing.assert_array_equal(targets-delayed,33)


@pytest.mark.parametrize('offset,words,lag',[(0,3,0),(10,3,32),(10,0,0),(10,3,-1),(198,3,0)])
def test_exposure_rejects_missing_context_or_out_of_range(offset,words,lag):
    with pytest.raises(ValueError):
        audit.exposure_tokens(np.arange(200),offset,words,lag)


def test_shared_clock_is_reproducible_positive_and_prefix_stable():
    short=audit.schedule(79,32,2,16,20261021)
    long=audit.schedule(128,32,2,16,20261021)
    assert len(short)==79 and np.all(short>=1)
    np.testing.assert_array_equal(short,long[:79])
    np.testing.assert_array_equal(short,audit.schedule(79,32,2,16,20261021))
