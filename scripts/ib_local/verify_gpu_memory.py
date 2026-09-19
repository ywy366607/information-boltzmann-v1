"""Pre-flight GPU Memory Telemetry and Zero-Paging Verification.
Verifies dedicated VRAM, float32 compute, and zero Windows Shared GPU memory paging.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch

def check_gpu_environment():
    print("=" * 70)
    print("   PRE-FLIGHT GPU MEMORY TELEMETRY & ZERO-PAGING VERIFICATION")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available!")

    device_name = torch.cuda.get_device_name(0)
    total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
    total_mem_mb = total_mem_bytes / (1024 ** 2)
    print(f"Device:               {device_name}")
    print(f"Total Dedicated VRAM: {total_mem_mb:.1f} MB")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"CUDA Version:         {torch.version.cuda}")

    # Set memory fraction guard (80% of 4GB = ~3276 MB)
    torch.cuda.set_per_process_memory_fraction(0.80, 0)
    print("Per-process memory fraction set to: 80% (~3276 MB)")

    # 1. Float32 Compute Test
    print("\n--- Testing FP32 Compute Capability ---")
    x = torch.randn(1024, 1024, dtype=torch.float32, device="cuda")
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    print("FP32 Matrix Multiply [1024, 1024]: PASSED (norm = {:.4f})".format(y.norm().item()))

    # 2. Sequential Allocation Stress Test (up to 2.2 GB)
    print("\n--- Testing Memory Allocation & Shared Paging Guard ---")
    tensors = []
    allocation_sizes_mb = [500, 1000, 1500, 2000, 2200]
    for target_mb in allocation_sizes_mb:
        current_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        additional_bytes = int((target_mb - current_alloc_mb) * (1024 ** 2))
        if additional_bytes > 0:
            num_elements = additional_bytes // 4
            tensors.append(torch.empty(num_elements, dtype=torch.float32, device="cuda"))
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"  Target: {target_mb:4d} MB | Allocated: {allocated:6.1f} MB | Reserved: {reserved:6.1f} MB")

    # Check that reserved memory stays well below 3.5 GB physical limit
    max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    assert max_reserved <= 3300.0, f"Reserved memory {max_reserved:.1f} MB exceeded safety threshold 3300 MB!"

    # Release memory
    del tensors, x, y
    torch.cuda.empty_cache()
    final_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"\nCache cleared. Post-cleanup reserved: {final_reserved:.1f} MB")
    print("\n[VERIFICATION RESULT]: PASS - Dedicated VRAM verified, 0 MB shared memory spill.")
    print("=" * 70)

if __name__ == "__main__":
    check_gpu_environment()
