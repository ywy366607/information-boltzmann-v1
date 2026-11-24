"""Compatibility entry point for the single Node mother-world service.

This module does not poll training logs, create sessions, or run a Python server.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Sequence


SERVICE_PATH = Path(__file__).resolve().with_name("mengli_service.mjs")


def service_command(args: Sequence[str] = ()) -> list[str]:
    return ["node", str(SERVICE_PATH), *args]


def run_server(port: int = 8900) -> int:
    """Preserve the old callable while delegating to the same Node service."""
    return subprocess.call(service_command(["--port", str(port)]))


def main(args: Sequence[str] | None = None) -> int:
    return subprocess.call(service_command(sys.argv[1:] if args is None else args))


if __name__ == "__main__":
    raise SystemExit(main())
