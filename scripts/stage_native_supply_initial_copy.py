"""Stage this experiment's recoverable initial duplicate on another local drive.

Wait until the candidate has completed a real transaction (its initial restore
has closed), verify the initial rolling copy is still at the declared source
cycle, then move that one file and preserve its metadata. The final candidate
checkpoint is written normally by the original runner.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

from scripts.run_native_supply_comparison import digest, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--source-cycle', type=int, required=True)
    a = p.parse_args()
    root, destination = a.work.resolve(), a.destination.resolve()
    if root != Path('F:/mengli-native-supply-life-20261122/supply').resolve():
        raise ValueError('only the explicitly named private candidate directory is admitted')
    if destination != Path('E:/mengli-native-supply-initial-copy-20261122').resolve():
        raise ValueError('only the explicitly named local staging directory is admitted')
    source = root/'run/body/last.pt.gz'
    for _ in range(1200):
        status = root/'run/status.json'
        if status.exists():
            try:
                row = json.loads(status.read_text(encoding='utf8'))
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if row['status'] == 'running' and row.get('invocation_cycles',0) >= 1:
                metadata = source.with_name('last.metadata.json')
                info = json.loads(metadata.read_text(encoding='utf8'))
                assert info['cycle'] == a.source_cycle
                sha = digest(source)
                assert sha == info['checkpoint_sha256']
                assert not destination.exists()
                assert shutil.disk_usage(destination.parent).free > source.stat().st_size+50*1024**2
                destination.mkdir()
                shutil.copyfile(metadata,destination/metadata.name)
                shutil.move(str(source),str(destination/source.name))
                assert digest(destination/source.name) == sha
                receipt = dict(source=str(source), destination=str(destination/source.name),
                    cycle=info['cycle'], sha256=sha, observed_completed_transactions=row['invocation_cycles'],
                    purpose='local-drive staging of the closed initial duplicate; source 2805 body remains intact')
                write(destination/'move_receipt.json',receipt)
                print(json.dumps(receipt))
                return
            if row['status'] not in ('loading','running'):
                raise RuntimeError('candidate left expected restore/learning stages before staging')
        time.sleep(1)
    raise TimeoutError('candidate did not reach the declared staging boundary')


if __name__=='__main__':
    main()
