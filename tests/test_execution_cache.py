from pathlib import Path

from information_boltzmann.runtime.execution_cache import configure_execution_cache


def test_compiler_cache_uses_declared_workspace_and_honors_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv('TORCHINDUCTOR_CACHE_DIR', raising=False)
    monkeypatch.delenv('TRITON_CACHE_DIR', raising=False)
    chosen = configure_execution_cache(tmp_path / 'local')
    assert Path(chosen['TORCHINDUCTOR_CACHE_DIR']) == tmp_path / 'local' / 'torchinductor'
    assert Path(chosen['TRITON_CACHE_DIR']).is_dir()
    explicit = tmp_path / 'explicit'
    monkeypatch.setenv('TRITON_CACHE_DIR', str(explicit))
    assert configure_execution_cache(tmp_path / 'other')['TRITON_CACHE_DIR'] == str(explicit)
    assert explicit.is_dir()
