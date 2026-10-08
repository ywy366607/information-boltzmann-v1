import traceback
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import scripts.ib.diagnose_fly_surrogate_vs_hard as d
    d.main()
except Exception as e:
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
