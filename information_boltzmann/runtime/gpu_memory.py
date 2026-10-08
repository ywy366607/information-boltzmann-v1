"""Read-only dedicated/shared memory counters for GPU execution measurements."""
import json
import os
import subprocess


def windows_gpu_memory():
    if os.name != 'nt':
        return None
    command = (
        '$rows=Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory '
        "| Where-Object {$_.Name -like 'pid_" + str(os.getpid()) + "_*'}; "
        '@($rows | Select-Object Name,DedicatedUsage,SharedUsage) | ConvertTo-Json -Compress')
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command],
                            capture_output=True, text=True, timeout=45, check=True)
    rows = json.loads(result.stdout or '[]')
    rows = [rows] if isinstance(rows, dict) else rows
    if not rows:
        return None
    return {'dedicated_bytes': sum(x['DedicatedUsage'] for x in rows),
            'shared_bytes': sum(x['SharedUsage'] for x in rows), 'rows': rows}


def total_gpu_memory(device_index=0):
    """Total dedicated usage includes other jobs sharing this device."""
    result = subprocess.run(
        ['nvidia-smi', f'--id={device_index}',
         '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu',
         '--format=csv,noheader,nounits'], capture_output=True, text=True,
        check=True, timeout=15)
    total, used, free, utilization = map(int, result.stdout.strip().split(','))
    return {'total_bytes': total * 2**20, 'used_bytes': used * 2**20,
            'free_bytes': free * 2**20, 'utilization_percent': utilization}
