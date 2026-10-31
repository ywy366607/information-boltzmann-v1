"""Resource admission contracts; no NVIDIA driver/process manipulation."""
from types import SimpleNamespace

import pytest

from scripts.ib.fly_interface import native_causal_audit as audit


@pytest.mark.parametrize('memory,utilization,allowed,passes', [
    (67, 0, (11, 22), True),
    (67, 0, (11,), False),
    (129, 0, (11, 22), False),
    (67, 1, (11, 22), False),
    (67, 0, (), False),
])
def test_desktop_exception_requires_exact_pids_and_actual_idle_load(monkeypatch, memory, utilization, allowed, passes):
    def run(args, **kwargs):
        if '--query-compute-apps=pid,process_name,used_memory' in args:
            return SimpleNamespace(stdout='11, desktop.exe, [N/A]\n22, other.exe, [N/A]\n')
        return SimpleNamespace(stdout=f'{memory}, {utilization}\n')
    monkeypatch.setattr(audit.subprocess, 'run', run)
    if passes:
        assert audit.ensure_gpu_idle(allowed)['idle_desktop_pids'] == [11, 22]
    else:
        with pytest.raises(RuntimeError, match='another compute'):
            audit.ensure_gpu_idle(allowed)
