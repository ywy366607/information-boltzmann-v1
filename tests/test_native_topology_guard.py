"""Only supervisor resource policy; no Torch or biological dynamics."""
from scripts.guard_native_topology import ResourcePressure, atomic_save_reserve_mib


def reading(commit=400, disk=4000):
    return dict(available_commit_mib=commit, system_disk_free_mib=disk)


def test_managed_pagefile_transient_does_not_stop_actual_event():
    pressure = ResourcePressure()
    assert pressure.check(reading(), 0) is None
    assert pressure.check(reading(200), 15) is None
    assert pressure.check(reading(1700), 20) is None
    assert pressure.check(reading(300), 31) is None
    assert pressure.check(reading(300), 60) is None
    assert pressure.check(reading(300), 61) is not None


def test_emergency_commit_and_atomic_save_space_remain_protected():
    assert ResourcePressure().check(reading(95), 0) is not None
    assert ResourcePressure().check(reading(2000, 1535), 0) is not None
    assert ResourcePressure().check(reading(512, 1536), 0) is None


def test_atomic_disk_reserve_uses_new_snapshot_and_database(tmp_path):
    with (tmp_path/'last.pt.gz').open('wb') as f:
        f.truncate(470*1048576)
    with (tmp_path/'causal_archive_current.sqlite').open('wb') as f:
        f.truncate(49*1048576)
    reserve = atomic_save_reserve_mib(tmp_path)
    assert reserve == 775
    assert ResourcePressure(reserve).check(reading(2000, 900), 0) is None
    assert ResourcePressure(reserve).check(reading(2000, 774), 0) is not None
