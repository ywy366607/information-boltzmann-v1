import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set up exact arguments
checkpoint = ROOT / 'results/medium_d768_capacity_growth_gen2/last.pt'
if checkpoint.exists():
    resume_target = str(checkpoint)
    adopt_flags = []
else:
    resume_target = 'results/medium_d768_streaming_pathway_8x8x8_160k/last.pt'
    adopt_flags = ['--adopt-capacity-growth']

sys.argv = [
    'scripts/ib/train_medium_active_stream.py',
    '--output', 'results/medium_d768_capacity_growth_gen2',
    '--resume', resume_target,
    *adopt_flags,
    '--allow-entrypoint-change',
    '--structure-config', 'results/published/medium_gen2_structure_config_20261009.json',
    '--steps', '5000',
    '--channels', '768',
    '--shape', '8', '8', '8',
    '--tokens', '32',
    '--chunk-tokens', '32',
    '--event-duration', '0.03327237442135811',
    '--execution', 'eager',
    '--activation-checkpointing',
    '--checkpoint-granularity', 'event',
    '--optimizer-state-offload',
    '--medium-segment-graph',
    '--read-key-execution', 'support',
    '--anisotropic-transport',
    '--temporal-read',
    '--material-bandwidth', 'runtime',
    '--material-initialization', 'spectral-xavier',
    '--material-field-std', '1.0',
    '--structure-dual-lr', '0.0625',
    '--pretrained-embedding', 'data/gpt2_model.safetensors',
    '--temporal-time-reference', '0.899332940578461',
    '--temporal-physical-reference', '0.899332940578461',
    '--lr', '0.0002',
    '--min-lr', '1e-06',
    '--warmup', '100',
    '--decay', '500',
    '--seed', '449',
    '--validate-every-tokens', '5000',
    '--eval-tokens', '256',
    '--eval-block-tokens', '16',
    '--replay-tokens', '128',
    '--shared-credit-execution',
    '--fused-optimizer',
    '--vram-limit-mib', '3900',
    '--intrinsic-time-reference', '0.03327237442135811',
    '--solver-max-step', '0.008318093605339527',
    '--observer-max-step', '0.03327237442135811',
    '--write-aperture-budget', '0.125',
    '--read-aperture-budget', '0.25'
]

import scripts.ib.train_medium_active_stream as trainer

if __name__ == '__main__':
    trainer.main()
