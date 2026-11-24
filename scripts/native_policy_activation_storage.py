"""Exact-layout bounded GPU/host activation caches with temporary disk spill."""
from contextlib import contextmanager, ExitStack, nullcontext
from pathlib import Path
import mmap
import shutil
import tempfile
import weakref

import torch


MIB = 1024**2


@contextmanager
def activation_storage(parameters, directory, *, threshold_bytes=65536,
                       mode='tiered', gpu_budget_bytes=512*MIB,
                       host_budget_bytes=128*MIB, disk_reserve_bytes=256*MIB):
    """Keep hot saved storage on GPU, then RAM, then disk, through backward.

    Budgets bound live eligible saved storage, rather than total device/process
    memory. Small tensors and parameter aliases stay native. Enable these hooks
    only around ordinary autograd forwards, outside torch.func transforms.
    ``disk`` retains the old spill policy; ``device`` is the native reference.
    Copies are synchronous so unpack cannot race an unfinished CUDA transfer.
    """
    if mode not in ('tiered', 'disk', 'device'):
        raise ValueError(f'unknown activation storage mode: {mode}')
    if min(threshold_bytes, gpu_budget_bytes, host_budget_bytes, disk_reserve_bytes) < 0:
        raise ValueError('activation storage budgets must be nonnegative')
    directory = Path(directory)
    protected = {(p.device, p.untyped_storage().data_ptr()) for p in parameters if p.is_cuda}
    # Leave space for gradients, optimizer temporaries and the backward scratch.
    # Each event re-evaluates this after the body's current allocations exist.
    requested_gpu_budget = gpu_budget_bytes
    devices = {device for device, _ in protected}
    if mode == 'tiered' and devices:
        free_gpu = min(torch.cuda.mem_get_info(device)[0] for device in devices)
        gpu_budget_bytes = min(gpu_budget_bytes, max(0, free_gpu-768*MIB))
    receipt = dict(mode=mode, packed_bytes=0, unpacked_bytes=0, packed_tensors=0,
                   threshold_bytes=threshold_bytes, gpu_budget_bytes=gpu_budget_bytes,
                   requested_gpu_budget_bytes=requested_gpu_budget,
                   host_budget_bytes=host_budget_bytes, gpu_live_bytes=0, host_live_bytes=0,
                   gpu_peak_bytes=0, host_peak_bytes=0, file_high_water_bytes=0,
                   disk_written_bytes=0, disk_read_bytes=0, host_packed_bytes=0,
                   deduplicated_saves=0)
    free, gpu_storages, copies = {}, {}, weakref.WeakValueDictionary()

    class Payload:
        def __init__(self, tensor, size, tier, *, host=None, offset=None, key=None):
            self.shape, self.stride = tuple(tensor.shape), tuple(tensor.stride())
            self.dtype, self.device = tensor.dtype, tensor.device
            self.size, self.tier = size, tier
            self.elements = size//tensor.element_size()
            self.host, self.offset, self.key = host, offset, key

        def __del__(self):
            if self.tier == 'host':
                receipt['host_live_bytes'] -= self.size
            elif self.tier == 'device':
                count, size = gpu_storages[self.key]
                if count == 1:
                    del gpu_storages[self.key]
                    receipt['gpu_live_bytes'] -= size
                else:
                    gpu_storages[self.key] = (count-1, size)
            else:
                free.setdefault(self.size, []).append(self.offset)

    with ExitStack() as stack:
        stream = None
        admission_refreshed = False

        def pack(tensor):
            nonlocal stream, admission_refreshed, gpu_budget_bytes
            size = tensor.numel()*tensor.element_size()
            storage = tensor.untyped_storage()
            key = (tensor.device, storage.data_ptr())
            if (mode == 'device' or not tensor.is_cuda or size < threshold_bytes or
                    key in protected or tensor.numel() == 0):
                return tensor.detach()
            if mode == 'tiered' and devices and not admission_refreshed:
                # Planning can leave free blocks in CUDA's caching allocator
                # between scope entry and the first autograd save. Decide
                # residency at this actual save boundary, after releasing
                # unused cached blocks; live tensors/graph pools stay intact.
                # Otherwise mem_get_info underestimates reusable capacity and
                # needlessly spills a fitting 30M body through RAM/SSD.
                torch.cuda.empty_cache()
                free_gpu = min(torch.cuda.mem_get_info(device)[0] for device in devices)
                gpu_budget_bytes = min(requested_gpu_budget, max(0, free_gpu-768*MIB))
                receipt['gpu_budget_bytes'] = gpu_budget_bytes
                receipt['admission_refreshed'] = True
                admission_refreshed = True
            source = tensor.detach().resolve_conj().resolve_neg()
            # A view keeps its whole allocation alive: account for the base.
            storage_size = storage.nbytes()
            if mode == 'tiered' and (key in gpu_storages or
                    receipt['gpu_live_bytes']+storage_size <= gpu_budget_bytes):
                count, _ = gpu_storages.get(key, (0, storage_size))
                gpu_storages[key] = (count+1, storage_size)
                if count == 0:
                    receipt['gpu_live_bytes'] += storage_size
                    receipt['gpu_peak_bytes'] = max(receipt['gpu_peak_bytes'], receipt['gpu_live_bytes'])
                return Payload(source, size, 'device', host=source, key=key)
            span = 1+sum((n-1)*s for n, s in zip(source.shape, source.stride()))
            size = span*source.element_size()
            copy_key = (key, tensor._version, tensor.storage_offset(), tensor.shape,
                        tensor.stride(), tensor.dtype, tensor.is_conj(), tensor.is_neg())
            previous = copies.get(copy_key)
            # The allocator can reuse an address after an offloaded tensor dies.
            # A live weak identity check prevents treating that new tensor as
            # an old saved value, even if shape and version happen to match.
            if previous is not None and previous.source_ref() is tensor:
                receipt['deduplicated_saves'] += 1
                return previous
            tier = ('host' if mode == 'tiered' and
                    receipt['host_live_bytes']+size <= host_budget_bytes else 'disk')
            offset = None
            if tier == 'disk':
                if stream is None:
                    directory.mkdir(parents=True, exist_ok=True)
                    stream = stack.enter_context(tempfile.TemporaryFile(
                        dir=directory, prefix='policy_activations_', suffix='.bin'))
                bucket = free.setdefault(size, [])
                if bucket:
                    offset = bucket.pop()
                else:
                    stream.seek(0, 2)
                    offset = stream.tell()
                    if shutil.disk_usage(directory).free < size+disk_reserve_bytes:
                        raise OSError(f'activation spill needs {size} bytes plus reserve in {directory}')
                    receipt['file_high_water_bytes'] = offset+size
            # Preserve values and strides; layout can affect GEMM accumulation.
            host = source.as_strided((span,), (1,), source.storage_offset()).cpu()
            receipt['packed_bytes'] += size
            receipt['packed_tensors'] += 1
            if tier == 'host':
                receipt['host_live_bytes'] += size
                receipt['host_peak_bytes'] = max(receipt['host_peak_bytes'], receipt['host_live_bytes'])
                receipt['host_packed_bytes'] += size
                payload = Payload(source, size, tier, host=host)
            else:
                stream.seek(offset)
                stream.write(memoryview(host.view(torch.uint8).numpy()))
                receipt['disk_written_bytes'] += size
                payload = Payload(source, size, tier, offset=offset)
            payload.source_ref = weakref.ref(tensor)
            copies[copy_key] = payload
            return payload

        def unpack(value):
            if torch.is_tensor(value):
                return value
            if value.tier == 'device':
                return value.host
            receipt['unpacked_bytes'] += value.size
            if value.tier == 'host':
                return value.host.to(value.device).as_strided(value.shape, value.stride)
            stream.flush()
            aligned = value.offset//mmap.ALLOCATIONGRANULARITY*mmap.ALLOCATIONGRANULARITY
            displacement = value.offset-aligned
            receipt['disk_read_bytes'] += value.size
            with mmap.mmap(stream.fileno(), value.size+displacement,
                           access=mmap.ACCESS_COPY, offset=aligned) as mapped:
                host = torch.frombuffer(mapped, dtype=value.dtype,
                                        count=value.elements, offset=displacement)
                restored = host.to(value.device).as_strided(value.shape, value.stride)
                del host
            return restored

        yield receipt, lambda: torch.autograd.graph.saved_tensors_hooks(pack, unpack)


@contextmanager
def policy_storage(parameters, directory, **options):
    with activation_storage(parameters, directory, **options) as (receipt, hooks):
        with hooks():
            yield receipt


def configure_runtime_storage(engine, directory, **options):
    """Bind storage to the existing objective/policy ports, never dynamics JVPs.

    The event owner calls event_storage through backward. These wrappers only
    enable hooks at the two ordinary-autograd ports, matching the audit runner.
    Storage configuration is runtime-only; it does not migrate a checkpoint.
    """
    if getattr(engine, '_activation_storage_config', None) is not None:
        raise ValueError('activation storage is already configured')
    engine._activation_storage_config = (Path(directory), options)
    from scripts.ib_local import run_navigation_residual_wood as legacy
    legacy._DERIVED.update({'_activation_storage_config', '_activation_hooks',
                            '_policy_activation_receipt', '_world_activation_receipt', '_policy_grad'})
    engine._activation_hooks = None
    objective, policy = engine.model.objective, engine._policy_grad

    def stored_objective(*args, **kwargs):
        hooks = engine._activation_hooks
        with (nullcontext() if hooks is None else hooks()):
            return objective(*args, **kwargs)

    def stored_policy(*args, **kwargs):
        if engine._activation_hooks is None:
            return policy(*args, **kwargs)
        # Independent lifetime: release policy payloads before live text graphs.
        with policy_storage(engine.params, Path(directory)/'policy', **options) as receipt:
            result = policy(*args, **kwargs)
        engine._policy_activation_receipt = receipt
        return result

    engine.model.objective = stored_objective
    engine._policy_grad = stored_policy


@contextmanager
def event_storage(engine):
    config = getattr(engine, '_activation_storage_config', None)
    if config is None:
        yield
        return
    directory, options = config
    if engine._activation_hooks is not None:
        raise RuntimeError('nested activation event scopes')
    with activation_storage(engine.params, directory/'world', **options) as (receipt, hooks):
        engine._activation_hooks = hooks
        try:
            yield
        finally:
            engine._activation_hooks = None
            engine._world_activation_receipt = receipt


def add_storage_arguments(parser):
    parser.add_argument('--activation-mode', choices=('tiered', 'disk', 'device'), default='tiered')
    parser.add_argument('--activation-gpu-mib', type=int, default=512)
    parser.add_argument('--activation-host-mib', type=int, default=128)


def storage_options(args):
    if min(args.activation_gpu_mib, args.activation_host_mib) < 0:
        raise ValueError('activation cache capacities must be nonnegative')
    return dict(mode=args.activation_mode, gpu_budget_bytes=args.activation_gpu_mib*MIB,
                host_budget_bytes=args.activation_host_mib*MIB)
