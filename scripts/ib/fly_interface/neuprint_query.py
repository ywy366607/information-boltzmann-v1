"""Minimal neuPrint HTTPS client: token from NEUPRINT_TOKEN, Cypher on the server, prints the bytes received."""
import os, sys, json, requests
URL = 'https://neuprint.janelia.org'
H = {'Authorization': 'Bearer ' + os.environ['NEUPRINT_TOKEN'], 'Content-Type': 'application/json'}
def cypher(q, dataset='male-cns:v1.0'):
    r = requests.post(URL + '/api/custom/custom', headers=H, data=json.dumps({'cypher': q, 'dataset': dataset}), timeout=600)
    print(f'[{r.status_code}, {len(r.content)} bytes]', file=sys.stderr)
    r.raise_for_status()
    return r.json()
if __name__ == '__main__':
    r = requests.get(URL + '/api/dbmeta/datasets', headers=H, timeout=60)
    print(r.status_code, len(r.content), 'bytes')
    if r.ok:
        print([k for k in r.json()][:20])
