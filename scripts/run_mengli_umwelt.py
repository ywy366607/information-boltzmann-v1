"""Canonical entry point for MengLi Umwelt (梦漓环世界) continuous life runs.

Connects MengLi's single UTF-8 Membrane port (`MembranePort`) to its living
Umwelt (shared English/Chinese storybooks, existing DSH Mother with adaptive
care plans & observable Horizon biofeedback, live human parent/guest inbox,
and pluggable world-side tools).
"""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from scripts.run_unified_mengli_life import main


if __name__ == '__main__':
    main()
