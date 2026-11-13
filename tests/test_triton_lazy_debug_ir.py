"""No GPU: preserve binary/metadata while deferring debug-only text loads."""
from pathlib import Path

from scripts.ib_local.triton_lazy_debug_ir import patch_compiled_kernel


def test_binary_and_metadata_unchanged_debug_text_is_loaded_only_on_access(tmp_path):
    class Asm(dict):
        def __missing__(self, key):
            raise KeyError(key)
    seen = []
    class Kernel:
        def __init__(self, src, metadata_group, hash):
            seen.append(dict(metadata_group))
            self.asm = {Path(path).suffix[1:]: Path(path).read_bytes() if name.endswith('.cubin') else Path(path).read_text()
                        for name, path in metadata_group.items() if not name.endswith('.json')}
            self.kernel = self.asm['cubin']
            self.metadata = Path(metadata_group['kernel.json']).read_text()
    paths = {}
    for suffix, value in [('json', b'{"shared": 0}'), ('cubin', b'unchanged-binary'), ('llir', b'debug-ir')]:
        path = tmp_path / f'kernel.{suffix}'
        path.write_bytes(value)
        paths[path.name] = str(path)
    patch_compiled_kernel(Kernel, Asm)
    patched = Kernel(None, paths, 'held-hash')
    assert patched.kernel == b'unchanged-binary' and patched.metadata == '{"shared": 0}'
    assert set(seen[0]) == {'kernel.json', 'kernel.cubin'}
    assert 'llir' not in patched.asm
    assert patched.asm['llir'] == 'debug-ir'
    patch_compiled_kernel(Kernel, Asm)  # idempotent, no nested wrappers
    again = Kernel(None, paths, 'held-hash')
    assert again.kernel == patched.kernel


def test_unknown_backend_keeps_the_original_loading_behavior():
    class Kernel:
        def __init__(self, src, metadata_group, hash):
            self.received = metadata_group
    original = {'kernel.hsaco': 'not-read-by-fixture', 'kernel.llir': 'also-not-read'}
    patch_compiled_kernel(Kernel, dict)
    assert Kernel(None, original, 'x').received is original
