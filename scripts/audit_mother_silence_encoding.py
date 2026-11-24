"""Compare silence encodings on forks of committed real mother-world histories.

Never commits the resulting replies to the living body's branch. Run after the
active training segment has stopped so the shared mother service is not delayed.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path
from uuid import uuid4


def post(endpoint, operation, payload):
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request(
        endpoint.rstrip('/') + '/api/mengli/' + operation,
        data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', required=True,
                        help='Saved run config carrying a committed mother origin')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit('Refusing to overwrite existing audit output')
    result = {'scope': 'separate branches, no body update and no reply committed',
              'conditions': {'marker': '<quiet_turn/>',
                             'description': '[The child does not speak this turn.]'},
              'records': []}
    output.parent.mkdir(parents=True, exist_ok=True)
    for config_path in args.config:
        config = json.loads(Path(config_path).read_text(encoding='utf-8'))
        origin = config['mother_harness_origin_ref']
        harness = config['mother_context_harness']
        for condition, transcript in result['conditions'].items():
            branch = 'silence-audit-' + uuid4().hex
            restored = post(harness['endpoint_url'], 'restore', {
                'branch_id': branch, 'checkpoint_ref': origin,
                'settings': harness['settings']})
            response = post(harness['endpoint_url'], 'prepare', {
                'branch_id': branch, 'checkpoint_ref': restored['checkpoint_ref'],
                'step_id': 'silence-encoding', 'transcript': transcript})
            result['records'].append({'config': str(config_path), 'origin_ref': origin,
                'condition': condition, 'branch_id': branch,
                'reply': response['delivered'], 'source': response.get('source'),
                'usage': response.get('usage'), 'at_unix': time.time()})
            tmp = output.with_suffix(output.suffix + '.tmp')
            tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            tmp.replace(output)


if __name__ == '__main__':
    main()
