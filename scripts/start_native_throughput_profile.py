"""Run one serial, source-frozen profile on the existing complete native body."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import ctypes
import argparse


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def available_memory():
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                    ('physical', ctypes.c_ulonglong), ('available', ctypes.c_ulonglong),
                    ('page', ctypes.c_ulonglong), ('page_available', ctypes.c_ulonglong),
                    ('virtual', ctypes.c_ulonglong), ('virtual_available', ctypes.c_ulonglong),
                    ('extended', ctypes.c_ulonglong)]
    status = Memory()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('Cannot read physical-memory availability')
    return status.available


def gpu_processes():
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           check=True, text=True, capture_output=True)
    busy = {int(line.strip()) for line in query.stdout.splitlines() if line.strip().isdigit()}
    # A peer can still be restoring on CPU before CUDA becomes visible.
    if os.name == 'nt':
        query = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
            "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
            "Where-Object { $_.CommandLine -match 'scripts\\.(run_inquiry_navigation_trial|run_native_supply_life|run_wood_policy_use_worker|run_native_throughput_worker|run_native_preview_restore_comparison)' } | "
            "Select-Object -ExpandProperty ProcessId"], check=True, text=True, capture_output=True)
        busy.update(int(line.strip()) for line in query.stdout.splitlines() if line.strip().isdigit())
    return sorted(busy)


def reclaim_complete_table(destination):
    root = Path('D:/mengli-throughput-20261123').resolve()
    destination = Path(destination).resolve()
    if not destination.is_relative_to(root) or not destination.name.replace('_', '').isalnum():
        raise ValueError('unexpected benchmark output')
    body = destination/'body/last.pt.gz'
    metadata = json.loads(body.with_name('last.metadata.json').read_text(encoding='utf8'))
    with body.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
        if digest is None:
            hasher = hashlib.sha256()
            for block in iter(lambda: stream.read(8*1024**2), b''):
                hasher.update(block)
            digest = hasher.hexdigest()
    if digest != metadata['checkpoint_sha256']:
        raise ValueError('complete checkpoint checksum mismatch')
    temporary = (root/'temporary'/destination.name/'table').resolve()
    removed = []
    for path in temporary.rglob('rows.npy'):
        resolved = path.resolve()
        if not resolved.is_relative_to(temporary):
            raise ValueError('temporary table escaped its benchmark root')
        removed.append(dict(path=str(resolved), bytes=resolved.stat().st_size))
        resolved.unlink()
    return dict(checkpoint=str(body), sha256=digest, reclaimed=removed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='baseline')
    parser.add_argument('--python-profile', action='store_true')
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--activation-gpu-mib', type=int, default=512)
    parser.add_argument('--activation-host-mib', type=int, default=512)
    parser.add_argument('--activation-mode', choices=('tiered', 'disk', 'device'), default='tiered')
    parser.add_argument('--direct-typed-question', action='store_true')
    parser.add_argument('--factorized-endpoints', action='store_true')
    parser.add_argument('--shared-utter-preview', action='store_true')
    parser.add_argument('--fused-decode-step', action='store_true')
    parser.add_argument('--native-replay-batch-size', type=int, choices=(1,2,4,8))
    parser.add_argument('--policy-proposal-batch-size', type=int, choices=(1,2,4))
    parser.add_argument('--reuse-objective-logits', action='store_true')
    parser.add_argument('--warm-runtime', action='store_true')
    parser.add_argument('--checkpoint-compressor', choices=('gzip', 'isal'))
    parser.add_argument('--checkpoint-compression-workers', type=int, choices=(1,2,4))
    parser.add_argument('--internal-step-review', action='store_true')
    parser.add_argument('--phase-coupled-rhythm', action='store_true')
    parser.add_argument('--wiener-gate', action='store_true')
    parser.add_argument('--split-hamiltonian-theta', action='store_true')
    parser.add_argument('--learnable-precision-scales', action='store_true')
    parser.add_argument('--ridge-lr', type=float, default=None)
    parser.add_argument('--minimum-free-host-mib', type=int, default=3072,
                        help='explicit restore admission; low-memory loads retain gzip streaming')
    parser.add_argument('--shared-gpu-pid', type=int, action='append', default=[],
                        help='record an explicitly identified desktop process; still wait for GPU memory')
    parser.add_argument('--after-status', type=Path)
    parser.add_argument('--reuse-runtime', type=Path,
                        help='use a previously hashed frozen tree for a warm matched comparison')
    parser.add_argument('--checkpoint-spool', type=Path,
                        help='explicit local disk for bounded one-pass low-memory checkpoint loading')
    opts = parser.parse_args()
    if (not opts.name.replace('_', '').isalnum() or opts.cycles < 1 or
            opts.minimum_free_host_mib < 2048):
        parser.error('name must be alphanumeric; cycles must be positive; host admission must be at least 2048 MiB')
    project = Path.cwd()
    work = project / 'scratch/cgb_throughput_20261123'
    work.mkdir(exist_ok=True)
    if opts.reuse_runtime is not None:
        frozen = opts.reuse_runtime.resolve()
        if not frozen.is_relative_to(work.resolve()) or not frozen.name.endswith('_runtime'):
            parser.error('reuse-runtime must identify a frozen tree in this campaign')
        hashes = json.loads((work/(frozen.name.removesuffix('_runtime')+'_runtime_hashes.json')).read_text(encoding='utf8'))
        for relative, expected in hashes.items():
            if hashlib.sha256((frozen/relative).read_bytes()).hexdigest() != expected:
                raise ValueError('frozen runtime hash mismatch: '+relative)
    else:
        frozen = work / (opts.name + '_runtime')
        frozen.mkdir(exist_ok=False)
        hashes = {}
        for path in (project / 'scripts').rglob('*.py'):
            relative = path.relative_to(project)
            destination = frozen / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            hashes[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    write(work / (opts.name + '_runtime_hashes.json'), hashes)
    source = Path('E:/mengli-native-supply-completed-20261122/original/last.pt.gz')
    metadata = json.loads(source.with_name('last.metadata.json').read_text(encoding='utf8'))
    destination = Path('D:/mengli-throughput-20261123') / opts.name
    assert not destination.exists()
    protocol = dict(source=str(source), source_sha256=metadata['checkpoint_sha256'],
        source_cycle=metadata['cycle'], cycles=opts.cycles, world_calls='natural text only',
        runtime=str(frozen), output=str(destination), state='waiting_memory',
        shared_gpu_pids=opts.shared_gpu_pid)
    status_path = work / (opts.name + '_status.json')
    write(status_path, protocol)
    if opts.after_status:
        while True:
            before = json.loads(opts.after_status.read_text(encoding='utf8'))
            if before['state'] == 'failed':
                raise RuntimeError('preceding benchmark failed; inspect it before continuing')
            if before['state'] == 'completed':
                protocol['preceding_table_reclaim'] = reclaim_complete_table(before['output'])
                break
            protocol.update(state='waiting_predecessor', predecessor=str(opts.after_status))
            write(status_path, protocol)
            time.sleep(5)
    # Timing and 4 GB residency require exclusive GPU ownership. Another
    # maintainer may have a complete transaction in flight; let it finish.
    quiet_since = None
    while True:
        busy = sorted(set(gpu_processes()) - set(opts.shared_gpu_pid))
        memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
        free_gpu_mib = min(int(line.strip()) for line in memory.stdout.splitlines())
        protocol['free_gpu_mib'] = free_gpu_mib
        if free_gpu_mib < 3072:
            quiet_since = None
            protocol.update(state='waiting_gpu_memory', busy_pids=busy)
            write(status_path, protocol)
            time.sleep(5)
            continue
        if busy:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.monotonic()
        elif time.monotonic()-quiet_since >= 30:
            break
        protocol.update(state='waiting_gpu', busy_pids=busy)
        write(status_path, protocol)
        time.sleep(5)
    protocol.pop('busy_pids', None)
    protocol['state'] = 'waiting_memory'
    write(status_path, protocol)
    # Serialize heavy restores against the other maintainer's current audit.
    protocol['minimum_free_host_mib'] = opts.minimum_free_host_mib
    write(status_path, protocol)
    while available_memory() < opts.minimum_free_host_mib * 1024**2:
        time.sleep(5)
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    if opts.checkpoint_spool is not None:
        opts.checkpoint_spool.mkdir(parents=True, exist_ok=True)
        environment['CGB_CHECKPOINT_SPOOL'] = str(opts.checkpoint_spool.resolve())
        protocol['checkpoint_spool'] = environment['CGB_CHECKPOINT_SPOOL']
    environment['PYTHONPATH'] = str(frozen) + os.pathsep + str(project)
    code = 'import sys;sys.path.insert(0,' + repr(str(frozen)) + ');from scripts.run_native_throughput_worker import main;main()'
    command = [sys.executable, '-c', code, '--profile-core',
        '--checkpoint', str(source), '--work', str(destination), '--cycles', str(opts.cycles),
        '--native-decision', '--native-word-execution', '--successor-distribution',
        '--joint-successor', '--batch-native-words', '--batch-preview-scores',
        '--amortized-decision', '--latent-action-planner', '--persistent-rhythm', 'coupled',
        '--root-native-correction', 'observe', '--retention-scope-prior', 'off',
        '--profile-runtime', '--evaluate-every', str(opts.cycles),
        '--activation-mode', opts.activation_mode,
        '--activation-gpu-mib', str(opts.activation_gpu_mib),
        '--activation-host-mib', str(opts.activation_host_mib),
        '--table-storage', str(destination.parent/'temporary'/opts.name/'table'),
        '--world-activations', str(destination.parent/'temporary'/opts.name/'world'),
        '--policy-activations', str(destination.parent/'temporary'/opts.name/'policy')]
    if opts.python_profile:
        command.append('--python-profile')
    if opts.direct_typed_question:
        command.append('--direct-typed-question')
    if opts.factorized_endpoints:
        command.append('--factorized-endpoints')
    if opts.shared_utter_preview:
        command.append('--shared-utter-preview')
    if opts.fused_decode_step:
        command.append('--fused-decode-step')
    if opts.native_replay_batch_size is not None:
        command.extend(('--native-replay-batch-size', str(opts.native_replay_batch_size)))
    if opts.policy_proposal_batch_size is not None:
        command.extend(('--policy-proposal-batch-size', str(opts.policy_proposal_batch_size)))
    if opts.reuse_objective_logits:
        command.append('--reuse-objective-logits')
    if opts.warm_runtime:
        command.append('--warm-runtime')
    if opts.checkpoint_compressor is not None:
        command.extend(('--checkpoint-compressor', opts.checkpoint_compressor))
    if opts.checkpoint_compression_workers is not None:
        command.extend(('--checkpoint-compression-workers', str(opts.checkpoint_compression_workers)))
    if opts.internal_step_review:
        command.append('--internal-step-review')
    if opts.phase_coupled_rhythm:
        command.append('--phase-coupled-rhythm')
    if opts.wiener_gate:
        command.append('--wiener-gate')
    if opts.split_hamiltonian_theta:
        command.append('--split-hamiltonian-theta')
    if opts.learnable_precision_scales:
        command.append('--learnable-precision-scales')
    if opts.ridge_lr is not None:
        command.extend(('--ridge-lr', str(opts.ridge_lr)))
    protocol['state'] = 'running'
    with (work / (opts.name + '.log')).open('w', encoding='utf8') as output:
        process = subprocess.Popen(command, cwd=project, env=environment,
                                   stdout=output, stderr=subprocess.STDOUT)
        protocol['pid'] = process.pid
        write(status_path, protocol)
        code = process.wait()
    protocol.update(state='completed' if code == 0 else 'failed', exit_code=code)
    if code == 0:
        protocol['table_reclaim'] = reclaim_complete_table(destination)
    write(status_path, protocol)
    if code:
        raise RuntimeError('Native profile failed; see baseline.log')


if __name__ == '__main__':
    main()
