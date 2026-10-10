"""Launch physical junctions with strict complete continuation.

Importing performs no training, argv mutation or filesystem write. The total
fresh-group budget is inherited; no source or budget exception is supplied.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path('results/medium_d768_physical_junction_gen3')
GEN2_CHECKPOINT = Path('results/medium_d768_capacity_growth_gen2/last.pt')
JUNCTION_PROTOCOL = 'capacity_paid_persistent_flux_junction_v2'

# The trainer checks checkpoint, data, source, clock, optimizer and budget.
PRODUCTION_FLAGS = (
    '--hopf-recomposition',
    '--structure-config', 'results/published/medium_gen2_structure_config_20261009.json',
    '--channels', '768', '--shape', '8', '8', '8',
    '--tokens', '32', '--chunk-tokens', '32',
    '--event-duration', '0.03327237442135811', '--execution', 'eager',
    '--activation-checkpointing', '--checkpoint-granularity', 'event',
    '--optimizer-state-offload', '--medium-segment-graph',
    '--read-key-execution', 'support', '--anisotropic-transport', '--temporal-read',
    '--material-bandwidth', 'runtime', '--material-initialization', 'spectral-xavier',
    '--material-field-std', '1.0', '--structure-dual-lr', '0.0625',
    '--pretrained-embedding', 'data/gpt2_model.safetensors',
    '--temporal-time-reference', '0.899332940578461',
    '--temporal-physical-reference', '0.899332940578461',
    '--lr', '0.0002', '--min-lr', '1e-06', '--warmup', '100', '--decay', '500',
    '--seed', '449', '--validate-every-tokens', '5000',
    '--eval-tokens', '256', '--eval-block-tokens', '16', '--replay-tokens', '128',
    '--shared-credit-execution', '--fused-optimizer', '--vram-limit-mib', '3900',
    '--intrinsic-time-reference', '0.03327237442135811',
    '--solver-max-step', '0.008318093605339527',
    '--observer-max-step', '0.03327237442135811',
    '--write-aperture-budget', '0.125', '--read-aperture-budget', '0.25',
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    result.add_argument('--resume', type=Path,
                        help='Default: output last.pt, then Gen2 last.pt')
    result.add_argument('--steps', type=int, default=5000,
                        help='Total fresh groups, not additional groups; preserve inherited budget')
    result.add_argument('--check-resume-only', action='store_true',
                        help='Read-only complete-continuation checking; consume no targets')
    return result


def resolve_path(path: Path, root: Path) -> Path:
    return (path if path.is_absolute() else root / path).resolve()


def read_config(directory: Path) -> dict | None:
    path = directory / 'config.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a configuration object: {path}')
    return value


def physical_junction_config(config: dict | None) -> bool:
    metadata = None if config is None else config.get('hopf_recomposition')
    return isinstance(metadata, dict) and metadata.get('protocol') == JUNCTION_PROTOCOL


def build_trainer_arguments(args: argparse.Namespace, *, root: Path = ROOT) -> list[str]:
    """Select a complete checkpoint without editing existing configurations."""
    if args.steps < 600:
        raise ValueError('Budget must accommodate the fixed 100 warmup + 500 decay groups')
    output = resolve_path(args.output, root)
    local_checkpoint = output / 'last.pt'
    output_config = read_config(output)
    if output_config is not None and not local_checkpoint.exists():
        if not physical_junction_config(output_config):
            raise ValueError('Output has an older or unrecognized protocol configuration but no '
                             'complete checkpoint; use a new directory and preserve the old record')
    if args.resume is None:
        source = (local_checkpoint if local_checkpoint.exists()
                  else resolve_path(GEN2_CHECKPOINT, root))
    else:
        source = resolve_path(args.resume, root)
        if local_checkpoint.exists() and source != local_checkpoint:
            raise ValueError('Output already owns another complete individual; use a distinct output')
    if not source.is_file():
        raise ValueError(f'Complete resume checkpoint does not exist: {source}')
    source_config = read_config(source.parent)
    if source_config is not None:
        if (source_config.get('constructor', {}).get('hopf_recomposition', False)
                and not physical_junction_config(source_config)):
            raise ValueError('Source uses the old readout branch protocol; use Gen2 or a '
                             'current physical-junction checkpoint')
        inherited_steps = source_config.get('training_groups')
        if inherited_steps is not None and inherited_steps != args.steps:
            raise ValueError(f'Total steps must match inherited budget {inherited_steps}; '
                             'budget extension has no continuation implementation')
    adoption = not (source == local_checkpoint or physical_junction_config(source_config))
    if adoption and source.parent == output:
        raise ValueError('Junction adoption must retain its source in a distinct directory')
    result = ['--output', str(output), '--resume', str(source), '--steps', str(args.steps)]
    if adoption:
        result.append('--adopt-hopf-branch')
    if args.check_resume_only:
        result.append('--check-resume-only')
    result.extend(PRODUCTION_FLAGS)
    return result


def main(argv: list[str] | None = None) -> None:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        arguments = build_trainer_arguments(args, root=ROOT)
    except (ValueError, json.JSONDecodeError) as error:
        cli.error(str(error))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.ib import train_medium_active_stream as trainer
    old_argv, old_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(ROOT)
        sys.argv = ['scripts/ib/train_medium_active_stream.py', *arguments]
        trainer.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == '__main__':
    main()
