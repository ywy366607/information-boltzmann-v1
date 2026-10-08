"""Keep disposable CUDA compiler artifacts beside execution outputs by default."""
from pathlib import Path
import os


def configure_execution_cache(directory: str | Path) -> dict[str, str]:
    """Honor explicit environment overrides; avoid filling the Windows temp disk."""
    root = Path(directory).resolve()
    paths = {'TORCHINDUCTOR_CACHE_DIR': root / 'torchinductor',
             'TRITON_CACHE_DIR': root / 'torchinductor' / 'triton' / '0'}
    for key, default in paths.items():
        location = Path(os.environ.setdefault(key, str(default)))
        location.mkdir(parents=True, exist_ok=True)
    return {key: os.environ[key] for key in paths}
