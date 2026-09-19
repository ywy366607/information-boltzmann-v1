"""Launch the Phase 1 full-size d=128 MaleCNS training with live HTTP monitor."""
import sys
if sys.path:
    sys.path.pop(0)

import os
import time
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/cbim_malecns_d128_splitclip_fullbptt_3000"
DASHBOARD = ROOT / "present/cbim_malecns_internal_time_live.html"
PORT = 8085

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DASHBOARD, OUTPUT / "index.html")

    # Start HTTP monitor server on port 8085
    server_stdout = open(OUTPUT / "server.stdout.log", "w", encoding="utf-8")
    server_stderr = open(OUTPUT / "server.stderr.log", "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT), "--directory", str(OUTPUT)],
        stdout=server_stdout, stderr=server_stderr, cwd=str(ROOT)
    )
    (OUTPUT / "server_pid.txt").write_text(str(server_proc.pid), encoding="utf-8")
    print(f"HTTP monitor server running on http://127.0.0.1:{PORT}/ (PID {server_proc.pid})")

    # Launch training
    train_cmd = [
        sys.executable, "scripts/ib_local/train_cbim_malecns_internal_time.py",
        "--output", str(OUTPUT),
        "--content-dim", "16",
        "--velocities", "8",
        "--micro-steps", "1",
        "--bptt-chunk-tokens", "128",
        "--steps", "3000",
        "--validate-every", "250",
        "--validation-tokens", "4096"
    ]
    train_stdout = open(OUTPUT / "train.stdout.log", "w", encoding="utf-8")
    train_stderr = open(OUTPUT / "train.stderr.log", "w", encoding="utf-8")
    print("Starting training process...")
    train_proc = subprocess.Popen(
        train_cmd, stdout=train_stdout, stderr=train_stderr, cwd=str(ROOT)
    )
    (OUTPUT / "train_pid.txt").write_text(str(train_proc.pid), encoding="utf-8")
    print(f"Training started (PID {train_proc.pid}), streaming logs to {OUTPUT / 'train.stdout.log'}")

if __name__ == "__main__":
    main()
