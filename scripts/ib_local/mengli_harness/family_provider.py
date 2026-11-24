"""Route real caregiver inputs through the unchanged mother-world port.

Automatic DSH state stays at its legacy top-level keys. Family metadata is an
additional checkpoint field, so old DSH snapshots can be resumed unchanged.
"""
from __future__ import annotations

import copy


def delivery_receipt(text, capacity):
    """Exactly the prefix MotherReplyWorld emits, including its added newline."""
    candidate = (text.strip() + '\n').encode('utf-8')
    delivered = candidate[:capacity]
    try:
        decoded = delivered.decode('utf-8')
        complete = True
    except UnicodeDecodeError:
        decoded = delivered.decode('utf-8', 'replace')
        complete = False
    return dict(byte_start=0, byte_end=len(delivered), hex=delivered.hex(),
                text=decoded, utf8_complete=complete, capacity_bytes=capacity,
                candidate_bytes=len(candidate), truncated=len(delivered) < len(candidate))


class FamilyCaregiver:
    """One routed provider, no waiting or model substitution in world.take()."""

    def __init__(self, automatic):
        from scripts.ib_local.mengli_harness.mengli_world_client import DshMother
        if not isinstance(automatic, DshMother):
            raise ValueError('family caregiver requires the existing DSH provider')
        self.automatic = automatic
        self.plan = None
        self.synced_plan_version = None
        self.observations = []
        self.last_event = None
        self.parent_turns = 0
        self._turn = None
        self._reply = None
        self._telemetry = None
        self._restored_family = False

    def inherit_world_history(self, transcript):
        # A routed checkpoint already contains the exact automatic prefix.
        # Re-importing its world transcript would turn parent input into fake
        # mother assistant history when no automatic reply has happened yet.
        if not self._restored_family:
            self.automatic.inherit_world_history(transcript)

    def __getattr__(self, name):
        return getattr(self.automatic, name)

    @property
    def last_telemetry(self):
        return self._telemetry if self._telemetry is not None else self.automatic.last_telemetry

    def begin_turn(self, *, caregiver, text=None, plan=None, cycle, response_bytes):
        if self._turn is not None:
            raise RuntimeError('previous family turn has not committed')
        if caregiver not in ('mother', 'parent'):
            raise ValueError('caregiver must be mother or parent')
        if caregiver == 'parent' and (not isinstance(text, str) or not text.strip()):
            raise ValueError('parent input must contain text before starting a step')
        if plan is not None:
            if (not isinstance(plan, dict) or type(plan.get('version')) is not int
                    or plan['version'] < 0 or not isinstance(plan.get('text'), str)):
                raise ValueError('plan requires a nonnegative integer version and text')
            if self.plan is not None:
                if plan['version'] < self.plan['version']:
                    raise ValueError('plan version cannot move backwards')
                if plan['version'] == self.plan['version'] and plan != self.plan:
                    raise ValueError('same plan version has different text')
            self.plan = copy.deepcopy(plan)
        self._turn = dict(caregiver=caregiver, caregiver_text=text, cycle=cycle,
                          response_bytes=response_bytes,
                          plan_version=None if self.plan is None else self.plan['version'])
        self._telemetry = None

    def _sync_context(self):
        changed = self.plan is not None and self.plan['version'] != self.synced_plan_version
        if not changed and not self.observations:
            return
        self.automatic._restore()
        payload = dict(branch_id=self.branch_id,
                       checkpoint_ref=copy.deepcopy(self.checkpoint_ref),
                       observations=copy.deepcopy(self.observations))
        if changed:
            payload['plan'] = copy.deepcopy(self.plan)
        result = self.automatic._call('context', payload)
        self.automatic.checkpoint_ref = copy.deepcopy(result['checkpoint_ref'])
        if self.observations:
            last = self.observations[-1]
            # World bookkeeping still calls this boundary 'mother'. This is
            # matching metadata only; Node appends parent data as observations,
            # never as a fabricated assistant message.
            self.automatic.last_pair = [
                ['child', last['child_utterance']],
                ['mother', last['caregiver_text'].strip()]]
        self.observations = []
        if changed:
            self.synced_plan_version = self.plan['version']

    def reply(self, transcript):
        caregiver = 'mother' if self._turn is None else self._turn['caregiver']
        if self._reply is not None:
            raise RuntimeError('family input already delivered; do not retry a partial step')
        if caregiver == 'mother':
            self._sync_context()
            reply = self.automatic.reply(transcript)
            raw = self.automatic.pending['raw']
        else:
            raw = self._turn['caregiver_text']
            reply = raw.strip()
        if self._turn is not None:
            self._reply = dict(child_utterance=str(transcript[-1][1]),
                               caregiver_text=raw,
                               delivered=delivery_receipt(reply, self._turn['response_bytes']))
        return reply

    def commit_turn(self):
        caregiver = 'mother' if self._turn is None else self._turn['caregiver']
        if caregiver == 'mother':
            self.automatic.commit_turn()
        else:
            self.parent_turns += 1
            self._telemetry = dict(source='parent', api_usage=None, parent_turn=self.parent_turns)
        if self._turn is not None:
            if self._reply is None:
                raise RuntimeError('no actual family input to commit')
            event = dict(self._reply, caregiver=caregiver, cycle=self._turn['cycle'],
                         plan_version=self._turn['plan_version'],
                         event_id=f'{self.branch_id}:cycle:{self._turn["cycle"]}')
            if caregiver == 'parent':
                self.observations.append(copy.deepcopy(event))
                # Like automatic.commit_turn(), this external append follows
                # real world delivery and precedes the complete body save.
                # Raw history lives in the service JSONL, not an ever-growing
                # body checkpoint array. This call never generates a reply.
                self._sync_context()
            self.last_event = event
        self._turn = self._reply = None

    def state_dict(self):
        if self._turn is not None or self._reply is not None:
            raise RuntimeError('cannot checkpoint an incomplete family input')
        return dict(self.automatic.state_dict(), family_caregiver=dict(
            schema=1, plan=copy.deepcopy(self.plan),
            synced_plan_version=self.synced_plan_version,
            observations=copy.deepcopy(self.observations),
            last_event=copy.deepcopy(self.last_event), parent_turns=self.parent_turns))

    def load_state_dict(self, state):
        self.automatic.load_state_dict(state)
        saved = state.get('family_caregiver') or {}
        self._restored_family = bool(saved)
        if saved and saved.get('schema') != 1:
            raise ValueError('incompatible family caregiver state')
        self.plan = copy.deepcopy(saved.get('plan'))
        self.synced_plan_version = saved.get('synced_plan_version')
        self.observations = copy.deepcopy(saved.get('observations', []))
        self.last_event = copy.deepcopy(saved.get('last_event'))
        self.parent_turns = int(saved.get('parent_turns', 0))
        self._turn = self._reply = self._telemetry = None
