"""The replay guard checks serialized facts exactly, across tuple/list IO."""
import json
import pickle
from types import SimpleNamespace

import pytest

from scripts.run_native_preview_restore_comparison import assert_replayed_event
from scripts.audit_native_transition_ensemble import checkpoint_without_observer


def test_retention_moments_survive_json_sequence_representation():
    live = dict(choice='silence', metal_g=0.0125,
                shared_unknown=dict(identity='unknown:1733',
                    revision=dict(moments=[('episode:104', -0.006, 0.0001, 1)])))
    recorded = json.loads(json.dumps(live))
    assert live['shared_unknown'] != recorded['shared_unknown']
    assert_replayed_event(live, recorded, 1)


@pytest.mark.parametrize('key,value', [('metal_g',0.0126), ('choice','generate')])
def test_replay_still_rejects_changed_behavior_or_payoff(key,value):
    live = dict(choice='silence',metal_g=0.0125)
    recorded = json.loads(json.dumps(live))
    live[key]=value
    with pytest.raises(AssertionError,match=key):
        assert_replayed_event(live,recorded,1)


def test_checkpoint_omits_local_observer_and_keeps_body_fields():
    def observer():
        return 'measurement'
    observer._native_ensemble_observer=True
    engine=SimpleNamespace(_utter=observer,cycle=2497,pending=b'body')
    with checkpoint_without_observer(engine):
        recovered=pickle.loads(pickle.dumps(vars(engine)))
        assert recovered==dict(cycle=2497,pending=b'body')
    assert engine._utter is observer


def test_observer_reattached_after_save_error():
    def observer():
        pass
    observer._native_ensemble_observer=True
    engine=SimpleNamespace(_utter=observer)
    with pytest.raises(RuntimeError):
        with checkpoint_without_observer(engine):
            assert '_utter' not in vars(engine)
            raise RuntimeError('save failed')
    assert engine._utter is observer
