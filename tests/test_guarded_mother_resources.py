"""Resource gates must cover the drive receiving the complete body snapshot."""
from types import SimpleNamespace

from scripts.run_guarded_mother_continuation import ready_to_start, stop_reason


def test_current_process_memory_is_positive_on_windows():
    import os
    from scripts.run_guarded_mother_continuation import process_memory
    reading = process_memory(os.getpid())
    if os.name == 'nt':
        assert reading['working_set_mib'] > 0
        assert reading['private_commit_mib'] > 0
        assert reading['peak_working_set_mib'] >= reading['working_set_mib']
    else:
        assert reading is None


def test_supervisor_publisher_never_imports_torch_or_ib_package():
    import subprocess
    import sys
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    code="""
import sys, tempfile
from pathlib import Path
class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('scripts.ib_local'):
            raise AssertionError('supervisor loaded tensor/package runtime: '+fullname)
sys.meta_path.insert(0,NoTorch())
from scripts.run_guarded_mother_continuation import checkpoint_publisher
publish=checkpoint_publisher(Path.cwd())
with tempfile.TemporaryDirectory() as d:
    p=Path(d); (p/'status.tmp').write_text('ready')
    publish(p/'status.tmp',p/'status.json')
    assert (p/'status.json').read_text()=='ready'
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable,'-c',code],cwd=root,check=True,capture_output=True,text=True)


def test_full_system_disk_does_not_hide_full_output_disk():
    args = SimpleNamespace(minimum_start_commit_mib=3328,
                           minimum_system_disk_mib=512,
                           minimum_output_disk_mib=512)
    readings = dict(available_commit_mib=4000, system_disk_free_mib=900,
                    output_disk_free_mib=12)
    assert not ready_to_start(readings, args)
    readings['output_disk_free_mib'] = 600
    assert ready_to_start(readings, args)


def test_stop_leaves_output_room_for_atomic_save():
    readings = dict(available_commit_mib=1000, output_disk_free_mib=127,
                    system_disk_free_mib=900)
    assert stop_reason(readings) == 'Output disk below 128 MiB'
    readings['output_disk_free_mib'] = 512
    assert stop_reason(readings) is None
    readings['available_commit_mib'] = 50
    assert stop_reason(readings) == 'Host commit below 96 MiB'
    readings['available_commit_mib'] = 1000
    readings['system_disk_free_mib'] = 383
    assert stop_reason(readings) == 'System disk below 384 MiB'
