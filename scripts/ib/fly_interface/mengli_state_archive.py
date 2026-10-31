"""Full individual recovery points; never overwrite an earlier learning history."""
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
SIDECARS = ('config.json', 'progress.json', 'chunk_plan.json', 'caregiver_presence.json',
            'mother_state.json', 'mother.jsonl', 'caregiver_reviews.jsonl', 'learning_segments.jsonl')
CARE_FILES = ('data/harness/raising.json', 'data/harness/world.json',
              'docs/information_boltzmann/MENGLI_EDUCATION.json')


def save_point(run, checkpoint, trained, reason, root=ROOT):
    """Call only with a saved, quiescent learner. Compress without loading Torch."""
    run, checkpoint, root = Path(run), Path(checkpoint), Path(root)
    tag = re.sub(r'[^a-zA-Z0-9_-]', '_', reason)
    point = checkpoint.parent / 'recovery' / f'{trained:012d}_{tag}'
    # A segment-end point is also the next segment's start. Reuse those exact
    # bytes rather than duplicating a gigabyte under a second reason label.
    stat = checkpoint.stat()
    for previous in (checkpoint.parent / 'recovery').glob(f'{trained:012d}_*/manifest.json'):
        old = json.loads(previous.read_text(encoding='utf8'))
        if (old['source_size'], old['source_mtime_ns']) == (stat.st_size, stat.st_mtime_ns):
            return previous.parent
    if (point / 'manifest.json').exists():
        manifest = json.loads((point / 'manifest.json').read_text(encoding='utf8'))
        stat = checkpoint.stat()
        if (manifest['source_size'], manifest['source_mtime_ns']) != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError('Recovery point already exists for a different saved state')
        return point
    before = checkpoint.stat()
    # Reserve a complete next atomic last.pt write as well as a worst-case archive.
    if shutil.disk_usage(checkpoint.parent).free < 2 * before.st_size + 64 * 1024**2:
        raise RuntimeError('Insufficient space for recovery plus the next atomic checkpoint; keep saved state paused')
    point.mkdir(parents=True, exist_ok=True)
    target = point / 'checkpoint.pt.gz'
    partial = point / 'checkpoint.pt.gz.partial'
    digest = hashlib.sha256()
    with checkpoint.open('rb') as source, gzip.open(partial, 'wb', compresslevel=1) as archive:
        while block := source.read(4 * 1024**2):
            digest.update(block)
            archive.write(block)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        partial.unlink()
        raise RuntimeError('Checkpoint changed during archival; pause and save before retrying')
    partial.replace(target)
    for name in SIDECARS:
        if (run / name).exists():
            shutil.copy2(run / name, point / name)
    code_hashes = {}
    with zipfile.ZipFile(point / 'runtime.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
        sources = list((root / 'information_boltzmann').rglob('*.py')) + list((root / 'scripts/ib').rglob('*.py'))
        sources += [root / p for p in CARE_FILES]
        sources += list((root / 'data/harness/reading').glob('*.json')) + list((root / 'data/harness/reading').glob('*.txt'))
        for source in sorted(set(sources)):
            if source.is_file():
                relative = source.relative_to(root).as_posix()
                bundle.write(source, relative)
                code_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
    manifest = {'version': 1, 'trained': int(trained), 'reason': reason,
        'created_at': datetime.now(timezone.utc).isoformat(), 'run': str(run.resolve()),
        'source_size': before.st_size, 'source_mtime_ns': before.st_mtime_ns,
        'checkpoint_sha256': digest.hexdigest(), 'compressed_bytes': target.stat().st_size,
        'git_head': git.stdout.strip(), 'runtime_hashes': code_hashes,
        'review_status': 'unreviewed_candidate',
        'contains': 'complete learner checkpoint including optimizer, brain, RNG, consequence belief, pending credit and world',
        'restore_policy': 'new branch only; retain later history; code and caregiver plan require Ling review'}
    (point / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    return point


def restore_point(point, new_run, new_checkpoint_dir):
    """Restore all checkpoint bytes to a new branch; never auto-start or overwrite."""
    point, new_run, new_checkpoint_dir = map(Path, (point, new_run, new_checkpoint_dir))
    manifest = json.loads((point / 'manifest.json').read_text(encoding='utf8'))
    if new_run.resolve() == new_checkpoint_dir.resolve():
        raise ValueError('Use distinct empty run and checkpoint directories')
    for directory in (new_run, new_checkpoint_dir):
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('Recovery requires empty branch directories; original history is preserved')
    if shutil.disk_usage(new_checkpoint_dir.parent).free < manifest['source_size'] + 64 * 1024**2:
        raise RuntimeError('Insufficient space to restore the full individual')
    new_run.mkdir(parents=True, exist_ok=True)
    new_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    partial = new_checkpoint_dir / 'last.pt.partial'
    digest = hashlib.sha256()
    with gzip.open(point / 'checkpoint.pt.gz', 'rb') as source, partial.open('wb') as out:
        while block := source.read(4 * 1024**2):
            digest.update(block)
            out.write(block)
    if digest.hexdigest() != manifest['checkpoint_sha256']:
        partial.unlink()
        raise RuntimeError('Recovery checkpoint checksum mismatch')
    partial.replace(new_checkpoint_dir / 'last.pt')
    for name in ('config.json', 'mother_state.json', 'mother.jsonl', 'caregiver_reviews.jsonl', 'learning_segments.jsonl'):
        if (point / name).exists():
            shutil.copy2(point / name, new_run / name)
    # Never reuse a consumed caregiver approval, stale progress or an old lesson plan.
    progress = {'status': 'paused', 'bptt_train_tokens': manifest['trained'], 'needs_caregiver_review': True}
    (new_run / 'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf8')
    (new_run / 'STOP').write_text('Recovered branch awaits Ling code and caregiver review', encoding='utf8')
    (new_run / 'rollback_provenance.json').write_text(json.dumps({
        'source': str(point.resolve()), 'original_run': manifest['run'], 'trained': manifest['trained'],
        'later_history_preserved': True, 'new_run': str(new_run.resolve()),
        'code_and_plan_bundle': str((point / 'runtime.zip').resolve()), 'requires_review': True}, indent=2), encoding='utf8')
    return new_run


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    save = sub.add_parser('save')
    save.add_argument('--run', type=Path, required=True)
    save.add_argument('--checkpoint', type=Path, required=True)
    save.add_argument('--trained', type=int, required=True)
    save.add_argument('--reason', required=True)
    restore = sub.add_parser('restore')
    restore.add_argument('--point', type=Path, required=True)
    restore.add_argument('--new-run', type=Path, required=True)
    restore.add_argument('--new-checkpoint-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'save':
        print(save_point(args.run, args.checkpoint, args.trained, args.reason))
    else:
        print(restore_point(args.point, args.new_run, args.new_checkpoint_dir))


if __name__ == '__main__':
    main()
