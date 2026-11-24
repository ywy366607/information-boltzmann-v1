"""Run the approved complete comparison on NVMe after the G-drive stop.

Restore the complete 2437 boundary and check the interrupted G-drive event
prefix, then continue each private arm to 32 external rounds in saved chunks.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path('E:/cgb_preview_nvme_comparison_20261118')
COLD = Path('G:/cgb_preview_restore_comparison_20261118/legacy')
SOURCE = Path('C:/Users/85750/.codex/cgb_value_contract_repair_20261118_claude/body/last.pt.gz')
WORLD = Path('D:/information-boltzmann-v1/scratch/native_preview_world_20261118')
REPO = Path(__file__).resolve().parents[1]
SOURCE_SHA = '51380bcc2671eeb867414374fbc2af1afd47f4c147c85e6b954fcfd09e5e80c8'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def run(arm, *, segment):
    work = ROOT/arm
    command = [sys.executable, '-m', 'scripts.run_native_preview_restore_comparison',
               '--checkpoint', str(SOURCE), '--work', str(work), '--cycles', '8',
               '--world-activations', str(WORLD/arm)]
    if segment:
        command.append('--resume-work')
    if arm == 'legacy':
        command.append('--legacy-restore')
        if not segment:
            command += ['--replay-prefix', str(COLD/'events.jsonl')]
    subprocess.run(command, cwd=REPO, check=True)
    result = read(work/'result.json')
    assert result['status'] == 'completed'
    assert result['origin_cycle'] == 2437
    assert result['initial_source_sha256'] == SOURCE_SHA
    assert result['cycles'] == (segment+1)*8
    assert result['updates'] == result['cycles']*4
    assert result['external_bytes'] == result['cycles']*960
    assert result['cycle'] == 2437+result['cycles']
    return result


def main():
    metadata = read(SOURCE.parent/'last.metadata.json')
    assert metadata['cycle'] == 2437 and metadata['checkpoint_sha256'] == SOURCE_SHA
    assert read(COLD/'status.json')['status'] == 'stopped_by_user'
    ROOT.mkdir(exist_ok=False)
    status_path = ROOT/'pipeline_status.json'
    write(ROOT/'restart.json', dict(source=str(SOURCE), cycle=2437,
        source_sha256=SOURCE_SHA, interrupted_prefix=str(COLD/'events.jsonl'),
        prefix_policy='replay and check, not additional exposure',
        cold_G_records_retained=True, policy_table_sql_storage=str(ROOT),
        world_activation_storage=str(WORLD), external_cycles_per_arm=32))
    try:
        for arm in ('legacy', 'fixed'):
            for segment in range(4):
                write(status_path, dict(status='running', arm=arm, segment=segment+1,
                      additional_cycles=8, work=str(ROOT/arm)))
                result = run(arm, segment=segment)
                print(json.dumps(dict(arm=arm, cycles=result['cycles'],
                                      cycle=result['cycle'])), flush=True)
        output = REPO/'results/published/cgb_preview_restore_comparison_20261118.json'
        subprocess.run([sys.executable, '-m', 'scripts.summarize_native_preview_restore_comparison',
                        '--legacy', str(ROOT/'legacy'), '--fixed', str(ROOT/'fixed'),
                        '--output', str(output)], cwd=REPO, check=True)
        write(status_path, dict(status='completed', legacy_cycles=32, fixed_cycles=32,
              comparison=str(output), cold_G_records_retained=True))
    except BaseException as error:
        write(status_path, dict(status='failed', error=str(error), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
