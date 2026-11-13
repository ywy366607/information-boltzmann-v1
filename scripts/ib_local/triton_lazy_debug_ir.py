"""Worker-local lazy loading of Triton's DEBUG IR text, never kernel binaries.

Triton 3.2 CompiledKernel eagerly reads ttir/ttgir/llir/ptx for every autotune
candidate. On this Windows host those debug strings exhaust commit memory.
The original constructor still reads unchanged metadata and cubin and performs
all original resource/launch checks. No installed library file is modified.
"""
from pathlib import Path

TEXT_IR = {'.ttir', '.ttgir', '.llir', '.ptx', '.amdgcn'}


def patch_compiled_kernel(kernel_class, asm_class):
    original = kernel_class.__init__
    if getattr(original, '_lazy_debug_ir', False):
        return

    class LazyAsm(asm_class):
        def __missing__(self, key):
            if key in self._deferred:
                value = self._deferred[key].read_text()
                self[key] = value
                return value
            return super().__missing__(key)

    def initialize(self, src, metadata_group, hash):
        # CUDA path only; unsupported binary kinds retain original behavior.
        if not any(str(name).endswith('.cubin') for name in metadata_group):
            return original(self, src, metadata_group, hash)
        deferred = {Path(path).suffix[1:]: Path(path) for name, path in metadata_group.items()
                    if Path(name).suffix in TEXT_IR}
        retained = {name: path for name, path in metadata_group.items() if Path(name).suffix not in TEXT_IR}
        original(self, src, retained, hash)
        asm = LazyAsm(self.asm)
        asm._deferred = deferred
        self.asm = asm
    initialize._lazy_debug_ir = True
    kernel_class.__init__ = initialize


def enable_lazy_debug_ir():
    from triton.compiler.compiler import CompiledKernel, AsmDict
    patch_compiled_kernel(CompiledKernel, AsmDict)
    return dict(debug_ir='lazy_on_explicit_access', cubin='unchanged_original_constructor',
                library_files_modified=False, resource_checks='original')
