"""Fixed-trajectory, time-ordered replay of the proposed gain clipping."""
import json
import statistics

from scripts.ib_local.audit_shadow_gain_trace import BASE, RUNS


def main():
    rows = []
    for name in RUNS:
        rows.extend(json.loads(s) for s in (BASE/name/'metrics.jsonl').read_text(
            encoding='utf-8').splitlines() if s.strip())
    rows.sort(key=lambda r:r['cycle'])
    if [r['cycle'] for r in rows] != list(range(2945, 2994)):
        raise ValueError('incomplete history')
    means = [float(x[3]) for x in rows[0]['wood_all_bases']]
    past, scale, results = [], None, []
    for row in rows:
        option = {'silence':0, 'generate':1, 'reuse':2}[row['choice']]
        target = float(row['wood_target'][3]); err = target-means[option]
        if scale is None and len(past) >= 8:
            scale = max(1e-6, statistics.median(past))
        limited = err if scale is None else max(-3*scale, min(3*scale, err))
        results.append(dict(cycle=row['cycle'], target=target,
            original=row['wood_base'][3], candidate=means[option],
            innovation=err, clipped_innovation=limited, scale=scale))
        means = [v+.05*(1 if i==option else .4)*limited for i,v in enumerate(means)]
        if scale is not None:
            scale = .95*scale+.05*abs(limited)
        past.append(abs(err))
    phases=[]
    for lo,hi in ((2945,2958),(2960,2968),(2969,2976),(2977,2984),(2985,2993),(2960,2993)):
        part=[r for r in results if lo <= r['cycle'] <= hi]
        phases.append(dict(cycles=[lo,hi], n=len(part),
            original_mae=statistics.mean(abs(r['target']-r['original']) for r in part),
            candidate_mae=statistics.mean(abs(r['target']-r['candidate']) for r in part)))
    print(json.dumps(dict(scope='fixed realized actions and outcomes; no body update',
                          parameters=dict(warmup=8,threshold_multiple=3,scale_decay=.95),
                          phases=phases,rows=results),indent=2))


if __name__ == '__main__':
    main()
