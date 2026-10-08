import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

log_file = Path("D:/debug.log").open("w", encoding="utf-8")
def log(msg):
    log_file.write(str(msg) + "\n")
    log_file.flush()
    print(msg, flush=True)

try:
    log("Importing...")
    import scripts.ib.diagnose_fly_surrogate_vs_hard as diag
    log("Running main...")
    # Override print inside diag or run step by step
    diag.main()
    log("Finished successfully!")
except Exception as e:
    log("EXCEPTION CAUGHT:")
    traceback.print_exc(file=log_file)
    traceback.print_exc(file=sys.stdout)
    log_file.flush()
finally:
    log_file.close()
