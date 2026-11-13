"""Same-code/source-age attribution checks; no optimizer or training."""
import copy

import numpy as np

from scripts.ib_local.age_resolved_native_s import AgeResolvedNativeSEligibility
from scripts.ib_local.native_s_eligibility import NativeSEligibility


SPECS = {l: dict(heads=1, key_dim=2, value_dim=2, width=2) for l in (1,12)}


def fill(producer, start):
    rng = np.random.default_rng(start+8)
    states = {l: rng.normal(size=(1,2,2)).astype(np.float32) for l in SPECS}
    producer.open_window(start, start//128, states)
    for layer in SPECS:
        p = dict(x=rng.normal(size=(128,2)).astype(np.float32),
                 k=rng.normal(size=(128,1,2)).astype(np.float32),
                 v=rng.normal(size=(128,1,2)).astype(np.float32),
                 g=-rng.uniform(.001,.04,size=(128,1)).astype(np.float32),
                 beta=rng.uniform(.1,.9,size=(128,1)).astype(np.float32),
                 a_slope=np.full((128,1),.5,np.float32),
                 Aexp=np.array([.7],np.float32), branch='recurrent')
        producer.add_chunk(layer, 0, p)


def test_age_metadata_never_changes_the_producer_or_raw_gradient():
    plain = NativeSEligibility(SPECS, block_size=3)
    aged = AgeResolvedNativeSEligibility(SPECS, block_size=3)
    teaching = {l: np.ones((1,2,2), np.float32) for l in SPECS}
    for start in (0,128,256):
        fill(plain, start)
        fill(aged, start)
        a, b = plain.raw_credit(teaching), aged.raw_credit(teaching)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        if start < 256:
            assert all(not v.any() for v in aged.strict_old_gradients.values())
        plain.advance_window()
        aged.advance_window()
        for name in ('U','V','fields','bills'):
            np.testing.assert_array_equal(getattr(plain.code,name), getattr(aged.code,name))
    fill(aged, 384)
    aged.raw_credit(teaching)
    assert aged.last_credit['strict_old_source_exclusive_stop'] == 256
    assert aged.last_credit['source_age_resolved']
    # Representation is intentionally approximate; old contribution may have
    # been dropped. A zero result is never evidence to fabricate old sources.
    assert all(np.isfinite(v).all() for v in aged.strict_old_gradients.values())


def test_pending_roundtrip_and_legacy_migration_do_not_backfill_ages():
    original = NativeSEligibility(SPECS, block_size=3)
    fill(original,0)
    original.advance_window()
    aged = AgeResolvedNativeSEligibility(SPECS, block_size=3)
    aged.load_state_dict(original.state_dict())
    teaching = {l: np.ones((1,2,2), np.float32) for l in SPECS}
    aged.raw_credit(teaching)
    assert all(not v.any() for v in aged.strict_old_gradients.values())
    migrated = AgeResolvedNativeSEligibility(SPECS, block_size=3)
    migrated.load_state_dict(aged.state_dict())
    fill(aged,128)
    restored = AgeResolvedNativeSEligibility(SPECS, block_size=3)
    restored.load_state_dict(copy.deepcopy(aged.state_dict()))
    aged.advance_window()
    restored.advance_window()
    for name in ('U','V','fields','bills'):
        np.testing.assert_array_equal(getattr(aged.code,name), getattr(restored.code,name))
    for name in ('older','previous','current'):
        np.testing.assert_array_equal(getattr(aged.source_ages,name), getattr(restored.source_ages,name))
    assert restored.nbytes - original.nbytes >= restored.source_ages.nbytes


def test_old_only_controls_leave_other_sources_common_and_mismatch_only_old():
    import pytest
    from scripts.ib_local.native_s_learner import select_native_credit, mismatch_credit
    old = {'layer1.a_proj.weight': np.array([[1.,2.,3.]],np.float32)}
    common = {'layer1.a_proj.weight': np.array([[.25,.5,.75]],np.float32)}
    raw = {n: common[n]+old[n] for n in old}
    cut = select_native_credit(raw,'cut_old',old)
    wrong = select_native_credit(raw,'mismatch_old',old)
    for name in old:
        np.testing.assert_array_equal(cut[name],common[name])
        np.testing.assert_array_equal(wrong[name],common[name]+mismatch_credit(old)[name])
    with pytest.raises(ValueError):
        select_native_credit(raw,'cut_old')
    with pytest.raises(ValueError):
        select_native_credit(raw,'cut_old',{n:v[0] for n,v in old.items()})
