"""Paired single-variable equal-wall-clock comparison: Original English vs Clean English Prose.

Protocol (strictly single-variable isolation from identical checkpoint & code):
  - Common source: D:/mengli-umwelt-10h-v12c-20261125/body/last.pt.gz
  - Only difference: `--english-corpus data/cgb_english_clean_prose_20261125/train.bin` on `clean` arm
  - Identical: Chinese mix (0.45), dialogue demo ratio (0.12), accompany/solo rhythm (35m/15m),
    piece_bytes (640), gpu_act_mb (512), context_tokens (524288), reward, LR, Mother policy,
    OtherWorldModel OFF (default), self_cons observe-only (default).
  - Pre-registered adjacent Mother dialogue windows (no cherry-picking):
    [0-600s], [1200-1800s], and [3000-3600s] saved verbatim for both arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

DEFAULT_SOURCE = Path(r'D:\mengli-umwelt-10h-v12c-20261125\body\last.pt.gz')
DEFAULT_CLEAN_EN = ROOT / 'data/cgb_english_clean_prose_20261125/train.bin'
DEFAULT_CN_TRAIN = ROOT / 'data/cgb_chinese_stories_20261124/train.bin'
DEFAULT_CN_VAL = ROOT / 'data/cgb_chinese_stories_20261124/validation.bin'
DEFAULT_DIALOGUES = ROOT / 'data/cgb_bilingual_dialogue_20261124/dialogues.jsonl'
DEFAULT_REPORT = ROOT / 'results/published/cgb_english_corpus_paired_1h_20261125.json'

FIXED_WINDOWS = [
    ('early_0_600s', 0.0, 600.0),
    ('mid_1200_1800s', 1200.0, 1800.0),
    ('late_3000_3600s', 3000.0, 3600.0),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    except Exception:
        return 'unknown'


def _summarize_arm(arm_name: str, work_dir: Path, report_path: Path) -> dict:
    rep = json.loads(report_path.read_text(encoding='utf-8'))
    m_path = work_dir / 'body' / 'metrics.jsonl'
    rows = [json.loads(l) for l in m_path.read_text(encoding='utf-8').splitlines() if l.strip()] if m_path.exists() else []

    total_updates = sum(int(r.get('updates') or 0) for r in rows)
    nan_count = sum(
        1 for r in rows for v in r.values()
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v))
    )
    self_cons_modes = sorted({
        str((r.get('self_evolution') or {}).get('self_cons_credit_mode'))
        for r in rows if isinstance(r.get('self_evolution'), dict)
    })
    self_evo_nonzero_g = sum(
        1 for r in rows
        if r.get('observe_target_id') == 'none' and abs(float(r.get('metal_g') or 0.0)) > 1e-12
    )
    other_world_active = sum(
        1 for r in rows
        if isinstance(r.get('gaze_credit'), dict) and r['gaze_credit'].get('other_model') is not None
    )

    # Actual bytes read & repeat exposure from world info on executive metrics
    exec_resp_seen = 0
    exec_hold_seen = 0
    exec_ext_bytes = 0
    en_book_turns = 0
    cn_book_turns = 0
    clean_corpus_turns = 0
    for r in rows:
        w = r.get('world') or {}
        exec_ext_bytes += int(r.get('external_bytes') or 0)
        exec_resp_seen += int(w.get('response_previously_seen_bytes') or 0)
        exec_hold_seen += int(w.get('holdout_previously_seen_bytes') or 0)
        src = str(w.get('source') or '')
        if src == 'chinese_storybook':
            cn_book_turns += 1
        elif src in ('text', 'matched', 'topic'):
            en_book_turns += 1
        if str(w.get('sidecar_corpus') or '') == DEFAULT_CLEAN_EN.name and src == 'text':
            clean_corpus_turns += 1

    # Extract pre-registered fixed-window verbatim adjacent Mother dialogue transcripts
    mother_turns = []
    for r in rows:
        w = r.get('world') or {}
        if w.get('channel') == 'dsh_mother' or w.get('mother_reply'):
            hx = r.get('utterance_hex') or ''
            utt = bytes.fromhex(hx).decode('utf-8', errors='replace') if hx else ''
            mother_turns.append(dict(
                cycle=int(r.get('cycle') or 0),
                elapsed_seconds=float(r.get('elapsed_seconds') or 0.0),
                interaction_mode=r.get('interaction_mode'),
                send_target_id=r.get('send_target_id'),
                observe_target_id=r.get('observe_target_id'),
                perspective_mode=w.get('perspective_mode'),
                stumble=(w.get('mirror_excerpt') or '')[:100],
                mengli=utt,
                mother_reply=w.get('mother_reply') or '',
                mother_response_nll=r.get('mother_response_nll'),
                self_speech_nll=r.get('self_speech_nll'),
                metal_g=r.get('metal_g'),
            ))

    windowed_dialogues = {}
    for wname, t0, t1 in FIXED_WINDOWS:
        windowed_dialogues[wname] = [
            t for t in mother_turns if t0 <= t['elapsed_seconds'] <= t1
        ]

    self_nlls = [float(r['self_speech_nll']) for r in rows if isinstance(r.get('self_speech_nll'), (int, float))]
    mother_nlls = [float(r['mother_response_nll']) for r in rows if isinstance(r.get('mother_response_nll'), (int, float)) and float(r['mother_response_nll']) > 0]
    raw_l2s = [float(r['draft_raw_action_l2']) for r in rows if isinstance(r.get('draft_raw_action_l2'), (int, float))]

    return dict(
        arm=arm_name,
        work_dir=str(work_dir),
        report_path=str(report_path),
        output=rep.get('output'),
        timing=rep.get('timing'),
        restore_exact=rep.get('restore_exact'),
        evaluation=rep.get('evaluation'),
        flux_trajectory=rep.get('flux_trajectory'),
        compliance=dict(
            nan_count=nan_count,
            self_cons_modes=self_cons_modes,
            self_evo_nonzero_metal_g=self_evo_nonzero_g,
            other_world_active_turns=other_world_active,
            max_draft_raw_action_l2=(max(raw_l2s) if raw_l2s else None),
        ),
        counts=dict(
            executive_cycles=len(rows),
            total_exec_updates=total_updates,
            mother_turns_in_run=len(mother_turns),
            en_book_exec_turns=en_book_turns,
            cn_book_exec_turns=cn_book_turns,
            clean_en_exec_turns=clean_corpus_turns,
            exec_external_bytes=exec_ext_bytes,
            exec_repeat_response_bytes=exec_resp_seen,
            exec_repeat_holdout_bytes=exec_hold_seen,
            mean_self_speech_nll=(round(sum(self_nlls) / len(self_nlls), 4) if self_nlls else None),
            mean_mother_response_nll=(round(sum(mother_nlls) / len(mother_nlls), 4) if mother_nlls else None),
        ),
        unified_port=rep.get('unified_port'),
        windowed_adjacent_dialogues=windowed_dialogues,
    )


def _run_arm(
    *,
    arm_name: str,
    source: Path,
    work_dir: Path,
    temp_dir: Path,
    report_path: Path,
    branch_id: str,
    budget_seconds: float,
    eval_every_seconds: float,
    save_every_seconds: float,
    english_corpus: Path | None,
    bind_dashboard: bool,
) -> dict:
    cmd = [
        PYTHON, str(ROOT / 'scripts/run_unified_mengli_life.py'),
        '--source', str(source),
        '--work', str(work_dir),
        '--temp-root', str(temp_dir),
        '--report', str(report_path),
        '--branch-id', branch_id,
        '--context-tokens', '524288',
        '--piece-bytes', '640',
        '--gpu-act-mb', '512',
        '--save-every-seconds', str(save_every_seconds),
        '--accompany-minutes', '35',
        '--solo-minutes', '15',
        '--budget-seconds', str(budget_seconds),
        '--eval-every-seconds', str(eval_every_seconds),
        '--chinese-corpus', str(DEFAULT_CN_TRAIN),
        '--chinese-val-file', str(DEFAULT_CN_VAL),
        '--chinese-mix-ratio', '0.45',
        '--dialogue-demo-corpus', str(DEFAULT_DIALOGUES),
        '--dialogue-demo-ratio', '0.12',
        '--reflect-every-turns', '16',
        '--mother-notes', str(work_dir / 'mother_notes.md'),
        '--inbox-file', str(work_dir / 'inbox.jsonl'),
    ]
    if english_corpus is not None:
        cmd.extend(['--english-corpus', str(english_corpus)])
    if bind_dashboard:
        cmd.append('--bind-dashboard')

    print(f'\n========== STARTING ARM [{arm_name}] (budget={budget_seconds}s) ==========', flush=True)
    print('  cmd:', ' '.join(cmd), flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), check=False)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f'Arm {arm_name} failed with exit code {proc.returncode}')
    print(f'========== FINISHED ARM [{arm_name}] in {wall_s:.1f}s ==========', flush=True)
    return _summarize_arm(arm_name, work_dir, report_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--budget-seconds', type=float, default=3600.0)
    p.add_argument('--eval-every-seconds', type=float, default=900.0)
    p.add_argument('--save-every-seconds', type=float, default=600.0)
    p.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    p.add_argument('--bind-dashboard', action='store_true')
    args = p.parse_args()

    source = args.source.resolve()
    src_sha_before = _sha256(source)
    git_head = _git_head()
    print(f'[PAIRED] source={source} sha256={src_sha_before[:16]} git={git_head[:12]}', flush=True)

    ref_work = Path(r'D:\mengli-en-ref-1h-20261125')
    ref_tmp = Path(r'E:\mengli-en-ref-1h-tmp-20261125')
    ref_rep = ROOT / 'results/published/cgb_en_ref_1h_20261125.json'

    clean_work = Path(r'D:\mengli-en-clean-1h-20261125')
    clean_tmp = Path(r'E:\mengli-en-clean-1h-tmp-20261125')
    clean_rep = ROOT / 'results/published/cgb_en_clean_1h_20261125.json'

    ref_summary = _run_arm(
        arm_name='reference_original_english',
        source=source,
        work_dir=ref_work,
        temp_dir=ref_tmp,
        report_path=ref_rep,
        branch_id='mengli-en-ref-1h-20261125',
        budget_seconds=args.budget_seconds,
        eval_every_seconds=args.eval_every_seconds,
        save_every_seconds=args.save_every_seconds,
        english_corpus=None,
        bind_dashboard=args.bind_dashboard,
    )

    src_sha_mid = _sha256(source)
    if src_sha_mid != src_sha_before:
        raise RuntimeError('Source checkpoint hash changed between paired arms!')

    clean_summary = _run_arm(
        arm_name='candidate_clean_english_prose',
        source=source,
        work_dir=clean_work,
        temp_dir=clean_tmp,
        report_path=clean_rep,
        branch_id='mengli-en-clean-1h-20261125',
        budget_seconds=args.budget_seconds,
        eval_every_seconds=args.eval_every_seconds,
        save_every_seconds=args.save_every_seconds,
        english_corpus=DEFAULT_CLEAN_EN,
        bind_dashboard=args.bind_dashboard,
    )

    payload = dict(
        schema=1,
        protocol='single_variable_english_corpus_paired_comparison',
        git_commit=git_head,
        source=dict(path=str(source), sha256=src_sha_before),
        clean_english_manifest=json.loads((DEFAULT_CLEAN_EN.parent / 'manifest.json').read_text(encoding='utf-8')),
        invariants=dict(
            chinese_mix_ratio=0.45,
            dialogue_demo_ratio=0.12,
            accompany_minutes=35,
            solo_minutes=15,
            piece_bytes=640,
            other_world_model='OFF',
            self_cons_credit_mode='observe',
            budget_seconds_per_arm=args.budget_seconds,
            eval_every_seconds=args.eval_every_seconds,
        ),
        arms=dict(
            reference=ref_summary,
            clean_english=clean_summary,
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\n[PAIRED] Complete! Saved paired report -> {args.report}', flush=True)


if __name__ == '__main__':
    main()
