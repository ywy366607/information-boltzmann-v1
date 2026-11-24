"""Continue the original measured life with its reconstructed live P-query cache."""
import sys
from pathlib import Path


def main():
    import torch
    from scripts.audit_wood_policy_restore import inherited_projection
    from scripts import run_wood_policy_use_worker as worker
    projection=Path(sys.argv[1])
    sys.argv=[sys.argv[0],*sys.argv[2:]]
    weight=torch.load(projection,map_location='cuda',weights_only=False)['weight']
    # This historical comparison must replay the *unversioned* writes of off2585.
    from unittest.mock import patch
    from information_boltzmann import budgeted_adamw_fused as kernel
    with patch.object(kernel, 'fused_step', kernel._launch), inherited_projection(weight):
        worker.main()


if __name__=='__main__':main()
