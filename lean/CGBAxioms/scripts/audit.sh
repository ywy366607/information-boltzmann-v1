#!/usr/bin/env bash
# Cross-platform full inventory; the manual Audit.lean list remains a historical smoke.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/audit.py "$@"
