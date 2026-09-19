"""Wait for the accepted v3 run, validate v4, then launch its OWT run."""
import sys

# Avoid shadowing the standard-library ``types`` module with ib_local/types.py
# when this file is executed directly by path.
if sys.path:
    sys.path.pop(0)

import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "results/cbim_malecns_v3_3000/progress.json"
V4 = ROOT / "results/cbim_malecns_v4_3000"
STATUS = V4 / "launcher_status.json"


def write(status, **values):
    V4.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps({"status": status, **values}), encoding="utf-8")
    temporary.replace(STATUS)
    print(json.dumps({"status": status, **values}), flush=True)


def main():
    write("waiting_for_v3")
    while True:
        try:
            progress = json.loads(V3.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(10)
            continue
        if progress.get("status") == "failed":
            write("blocked", reason="v3_failed", detail=progress.get("error"))
            return 2
        if progress.get("status") == "complete":
            break
        time.sleep(10)

    write("testing_v4")
    test = subprocess.run([
        sys.executable, "-m", "pytest", "-q",
        "tests/test_cbim_malecns_v4.py"], cwd=ROOT)
    if test.returncode:
        write("blocked", reason="v4_tests_failed", exit_code=test.returncode)
        return test.returncode

    write("training_v4")
    train = subprocess.run([
        sys.executable, "scripts/ib_local/train_cbim_malecns_v4.py",
        "--output", "results/cbim_malecns_v4_3000",
        "--steps", "3000", "--tokens", "128",
        "--validate-every", "250", "--validation-tokens", "4096"], cwd=ROOT)
    final = "complete" if train.returncode == 0 else "blocked"
    write(final, phase="v4_training", exit_code=train.returncode)
    return train.returncode


if __name__ == "__main__":
    raise SystemExit(main())
