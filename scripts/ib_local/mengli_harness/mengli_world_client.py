"""Mother-world provider backed by the official, tool-free DSH service.

Only the newest child turn crosses the bridge.  The service owns the mother's
session; the existing world still decides when the real reply was delivered.
"""
from __future__ import annotations

import copy
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request

from scripts.ib_local.mother_reply_world import (
    RESPONSIVE_SYSTEM, RESPONSIVE_SYSTEM_V2, RESPONSIVE_SYSTEM_V3,
    RESPONSIVE_TAIL_V3, UMWELT_MOTHER_SYSTEM_V1,
)


DEFAULT_ENDPOINT = 'http://127.0.0.1:8900'
DEFAULT_SETTINGS = dict(model='deepseek-flash', temperature=.7, max_tokens=120,
                        context_tokens=8192, headroom_tokens=1024,
                        retain_tokens=2048, summary_tokens=512)
SETTINGS_32K = dict(model='deepseek-flash', temperature=.7, max_tokens=120,
                    context_tokens=32768, headroom_tokens=2048,
                    retain_tokens=6144, summary_tokens=1024)
SETTINGS_64K = dict(model='deepseek-flash', temperature=.7, max_tokens=120,
                    context_tokens=65536, headroom_tokens=2048,
                    retain_tokens=None, summary_tokens=1024)
SETTINGS_128K = dict(model='deepseek-flash', temperature=.7, max_tokens=120,
                     context_tokens=131072, headroom_tokens=4096,
                     retain_tokens=None, summary_tokens=1536)
SETTINGS_256K = dict(model='deepseek-flash', temperature=.7, max_tokens=120,
                     context_tokens=262144, headroom_tokens=4096,
                     retain_tokens=None, summary_tokens=2048)
SETTINGS_512K = dict(model='deepseek-flash', temperature=.7, max_tokens=120,
                     context_tokens=524288, headroom_tokens=4096,
                     retain_tokens=None, summary_tokens=2048)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def branch_for_output(output):
    from pathlib import Path
    return 'mengli-' + fingerprint(str(Path(output).resolve()))[:24]


def policy_prompt(policy):
    prompts = dict(responsive=RESPONSIVE_SYSTEM, responsive_v2=RESPONSIVE_SYSTEM_V2,
                   responsive_v3=RESPONSIVE_SYSTEM_V3 + '\n\n' + RESPONSIVE_TAIL_V3,
                   umwelt_mother_v1=UMWELT_MOTHER_SYSTEM_V1)
    if policy not in prompts:
        raise ValueError('DSH migration needs an explicit responsive caregiver policy')
    return prompts[policy]


def migration_turns(context=None, transcript=()):
    """Import only the actual checkpoint's committed pairs, never metrics tails."""
    if context:
        messages = context
        provenance = 'checkpoint_raw_context'
    else:
        messages = [dict(role='assistant' if who == 'mother' else 'user', content=text,
                         delivered=text) for who, text in transcript]
        provenance = 'checkpoint_delivered_only'
    turns = []
    for i in range(0, len(messages), 2):
        pair = messages[i:i + 2]
        if len(pair) != 2 or pair[0]['role'] != 'user' or pair[1]['role'] != 'assistant':
            raise ValueError('checkpoint caregiver history must contain complete pairs')
        turns.append(dict(input=pair[0]['content'], raw=pair[1]['content'],
                          delivered=pair[1].get('delivered', pair[1]['content']),
                          provenance=provenance))
    return turns


class DshMother:
    def __init__(self, *, endpoint_url=DEFAULT_ENDPOINT, branch_id,
                 policy='responsive_v3', settings=None, provenance=None, transport=None,
                 max_retries=0):
        endpoint = urllib.parse.urlsplit(endpoint_url)
        if (endpoint.scheme != 'http' or endpoint.hostname not in ('127.0.0.1', 'localhost')
                or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment
                or endpoint.path not in ('', '/')):
            raise ValueError('the caregiver bridge must be a loopback HTTP endpoint')
        self.endpoint_url = endpoint_url.rstrip('/')
        self.branch_id = str(branch_id)
        if not self.branch_id:
            raise ValueError('an independent caregiver branch identity is required')
        self.policy = policy
        self.settings = dict(DEFAULT_SETTINGS, **(settings or {}))
        self.settings['temperature'] = .7 if policy in ('responsive_v3', 'umwelt_mother_v1') else 0.
        self.provenance = copy.deepcopy(provenance or {})
        self.checkpoint_ref = None
        self.turn_count = 0
        self.last_pair = None
        self.last_telemetry = {}
        self.usage_totals = None
        self.pending = None
        self._migration = []
        self._ready = False
        self._transport = transport
        self._max_retries = max(0, int(max_retries))

    def _call(self, operation, payload):
        if self._transport is not None:
            return self._transport(operation, copy.deepcopy(payload))
        import time
        last_err = None
        for attempt in range(self._max_retries + 1):
            request = urllib.request.Request(
                f'{self.endpoint_url}/api/mengli/{operation}',
                data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.loads(response.read().decode('utf-8'))
                if not isinstance(result, dict) or result.get('ok') is False:
                    raise RuntimeError('caregiver harness rejected the event; delivery not confirmed')
                return result
            except urllib.error.HTTPError as error:
                last_err = RuntimeError(f'caregiver harness HTTP {error.code}; delivery not confirmed')
                if error.code not in (429, 500, 502, 503, 504) or attempt >= self._max_retries:
                    raise last_err from None
            except urllib.error.URLError:
                last_err = RuntimeError('caregiver harness unavailable; delivery not confirmed')
                if attempt >= self._max_retries:
                    raise last_err from None
            time.sleep(min(20.0, 3.0 * (2 ** attempt)))
        raise last_err

    def _restore(self):
        if self._ready:
            return
        migration = None
        if self.checkpoint_ref is None:
            migration = dict(turns=copy.deepcopy(self._migration),
                             system_prompt=policy_prompt(self.policy),
                             provenance=copy.deepcopy(self.provenance))
        result = self._call('restore', dict(branch_id=self.branch_id,
            checkpoint_ref=copy.deepcopy(self.checkpoint_ref),
            settings=copy.deepcopy(self.settings), migration=migration,
            usage_totals=copy.deepcopy(self.usage_totals)))
        self.checkpoint_ref = copy.deepcopy(result['checkpoint_ref'])
        self.usage_totals = copy.deepcopy(result.get('usage_totals', self.usage_totals))
        self._ready = True

    def inherit_world_history(self, transcript):
        if self.checkpoint_ref is None and not self._migration:
            self._migration = migration_turns(transcript=transcript)
        if self._migration and self.last_pair is None:
            last = self._migration[-1]
            self.last_pair = [['child', last['input']], ['mother', last['delivered']]]

    def set_policy(self, policy):
        policy_prompt(policy)
        if policy == self.policy:
            return
        if self.checkpoint_ref is not None:
            raise ValueError('changing a committed DSH persona requires a new explicit migration')
        self.policy = policy
        self.settings['temperature'] = .7 if policy in ('responsive_v3', 'umwelt_mother_v1') else 0.

    def reply(self, transcript):
        turns = [[str(who), str(text)] for who, text in transcript]
        if not turns or turns[-1][0] != 'child':
            raise ValueError('a caregiver request must end in the actual child turn')
        key = fingerprint(turns)
        if self.pending is not None:
            if self.pending['transcript_key'] != key:
                raise RuntimeError('previous caregiver event has not committed')
            return self.pending['delivered']
        if self.last_pair is not None and turns[-3:-1] != self.last_pair:
            raise ValueError('caregiver context does not match the actual continuing world')
        self._restore()
        step_id = fingerprint(dict(branch_id=self.branch_id,
                                   checkpoint_ref=self.checkpoint_ref, transcript=turns))
        result = self._call('prepare', dict(branch_id=self.branch_id,
            checkpoint_ref=copy.deepcopy(self.checkpoint_ref), step_id=step_id,
            transcript=turns[-1][1]))
        raw, delivered = result['raw'], result['delivered']
        if not isinstance(raw, str) or not raw.strip() or delivered != raw.strip():
            raise RuntimeError('caregiver harness returned an invalid real reply')
        self.pending = dict(step_id=step_id, transcript_key=key,
                            child=turns[-1][1], raw=raw, delivered=delivered)
        self.last_telemetry = copy.deepcopy(result.get('telemetry', {}))
        return delivered

    def commit_turn(self):
        if self.pending is None:
            return
        result = self._call('commit', dict(branch_id=self.branch_id,
                                          step_id=self.pending['step_id']))
        self.checkpoint_ref = copy.deepcopy(result['checkpoint_ref'])
        self.last_pair = [['child', self.pending['child']], ['mother', self.pending['delivered']]]
        self.last_telemetry = copy.deepcopy(result.get('telemetry', self.last_telemetry))
        self.usage_totals = copy.deepcopy(result.get('usage_totals', self.usage_totals))
        if self.usage_totals is not None:
            self.last_telemetry['summary_calls'] = self.last_telemetry.get('summary_usage')
            self.last_telemetry['reply_usage'] = copy.deepcopy(self.usage_totals.get('reply_usage'))
            self.last_telemetry['summary_usage'] = copy.deepcopy(self.usage_totals.get('summary_usage'))
            usage = self.usage_totals.get('reply_usage') or {}
            hit, miss = usage.get('cache_hit_tokens'), usage.get('cache_miss_tokens')
            self.last_telemetry['provider_hit_rate'] = (
                hit / (hit + miss) if hit is not None and miss is not None and hit + miss else None)
        self.turn_count += 1
        self.pending = None

    def reflect(self, dossier: str) -> dict:
        """Run a read-only KV-cache-hitting reflection side-query on the committed session."""
        if self.pending is not None:
            raise RuntimeError('commit pending caregiver turn before reflecting')
        self._restore()
        result = self._call('reflect', dict(
            branch_id=self.branch_id,
            checkpoint_ref=copy.deepcopy(self.checkpoint_ref),
            dossier=str(dossier),
        ))
        if result.get('usage_totals') is not None:
            self.usage_totals = copy.deepcopy(result['usage_totals'])
        return result

    def state_dict(self):
        if self.pending is not None:
            raise RuntimeError('cannot checkpoint an undelivered caregiver event')
        return dict(policy=self.policy, dsh_harness=dict(schema=1,
            branch_id=self.branch_id, settings=copy.deepcopy(self.settings),
            checkpoint_ref=copy.deepcopy(self.checkpoint_ref), turn_count=self.turn_count,
            last_pair=copy.deepcopy(self.last_pair),
            last_telemetry=copy.deepcopy(self.last_telemetry),
            usage_totals=copy.deepcopy(self.usage_totals),
            migration=copy.deepcopy(self._migration) if self.checkpoint_ref is None else [],
            provenance=copy.deepcopy(self.provenance)))

    def load_state_dict(self, state):
        saved = state.get('dsh_harness')
        if saved is None:
            old = state.get('context_harness', {})
            self.checkpoint_ref = None
            self._migration = migration_turns(old.get('context'))
            self.turn_count = int(old.get('turn_count', 0))
            self.last_pair = None
            self.last_telemetry = {}
            self.usage_totals = None
            if old.get('meter') is not None:
                meter = old['meter']
                self.usage_totals = dict(
                    reply_usage={key: meter.get(key) for key in (
                        'prompt_tokens', 'completion_tokens', 'cache_hit_tokens', 'cache_miss_tokens')},
                    summary_usage=None, source_counts=dict(api=meter.get('api_successes', 0),
                        local=meter.get('local_cache_hits', 0), shared=meter.get('shared_cache_hits', 0)),
                    api_calls=meter.get('api_calls', 0), failures=meter.get('api_failures', 0))
                self.usage_totals['reply_usage']['total_tokens'] = (
                    meter['prompt_tokens'] + meter['completion_tokens']
                    if 'prompt_tokens' in meter and 'completion_tokens' in meter else None)
            if self._migration:
                last = self._migration[-1]
                self.last_pair = [['child', last['input']], ['mother', last['delivered']]]
        else:
            if saved.get('schema') != 1:
                raise ValueError('incompatible saved DSH caregiver settings')
            saved_s = saved.get('settings')
            window_upgrade_only = (
                isinstance(saved_s, dict)
                and int(self.settings.get('context_tokens', 0)) > int(saved_s.get('context_tokens', 0))
                and {k: v for k, v in saved_s.items() if k != 'context_tokens'}
                    == {k: v for k, v in self.settings.items() if k != 'context_tokens'}
            )
            if saved_s != self.settings and not window_upgrade_only:
                raise ValueError('incompatible saved DSH caregiver settings')
            if state.get('policy') != self.policy:
                raise ValueError('caregiver policy differs from saved DSH world')
            self.checkpoint_ref = copy.deepcopy(saved['checkpoint_ref'])
            self._migration = copy.deepcopy(saved.get('migration', []))
            self.turn_count = int(saved['turn_count'])
            self.last_pair = copy.deepcopy(saved['last_pair'])
            self.last_telemetry = copy.deepcopy(saved['last_telemetry'])
            self.usage_totals = copy.deepcopy(saved.get('usage_totals'))
            self.provenance = copy.deepcopy(saved['provenance'])
        self.pending = None
        self._ready = False


# The old name now refers to the actual world-provider protocol, not a demo API.
MengLiHarnessClient = DshMother
