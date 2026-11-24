"""MengLi Umwelt (梦漓环世界): living environment, Membrane port, and observable Horizon.

Conceptual architecture (replacing the legacy "harness" terminology):
  - **Umwelt (环世界)**: MengLi's surrounding living environment — shared
    storybooks (English + Chinese), Mother caregiver with automatic memory
    compaction, live human parent/guests, and world-side tools.
  - **Membrane (界膜, ``MembranePort``)**: The single UTF-8 byte boundary
    connecting MengLi's body to its Umwelt. Zero extra neural heads or
    gradient ports are added to the body when new participants or tools join.
  - **Horizon (视界, ``HorizonSnapshot``)**: The shared observable window
    (current storybook page + read-only bodily biofeedback cues) with explicit
    perspective boundaries (``mirror_visible`` vs ``book_only``).
"""
from __future__ import annotations

from scripts.ib_local.body_mirror import (
    AdaptiveCarePlanTracker,
    BodyMirrorSnapshot as HorizonSnapshot,
    CARE_PLANS,
    STAGE_EMERGING,
    STAGE_INFANT,
    STAGE_ORDER,
    STAGE_REFLECTIVE,
    capture_body_mirror as capture_horizon_snapshot,
    classify_utterance,
    clean_utf8_excerpt,
    format_caregiver_turn,
)
from scripts.ib_local.mengli_harness.family_provider import FamilyCaregiver as UmweltCaregiver
from scripts.ib_local.mengli_harness.mengli_world_client import (
    DEFAULT_ENDPOINT as DEFAULT_UMWELT_BRIDGE_URL,
    DshMother as UmweltBridge,
    branch_for_output,
)
from scripts.ib_local.unified_world_port import (
    SidecarCorpusStream,
    UnifiedWorldPort as MembranePort,
)

__all__ = [
    'MembranePort',
    'HorizonSnapshot',
    'capture_horizon_snapshot',
    'format_caregiver_turn',
    'classify_utterance',
    'clean_utf8_excerpt',
    'AdaptiveCarePlanTracker',
    'CARE_PLANS',
    'STAGE_INFANT',
    'STAGE_EMERGING',
    'STAGE_REFLECTIVE',
    'STAGE_ORDER',
    'SidecarCorpusStream',
    'UmweltBridge',
    'UmweltCaregiver',
    'DEFAULT_UMWELT_BRIDGE_URL',
    'branch_for_output',
]
