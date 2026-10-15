"""Does the brain's own three-factor memory learn?  An optogenetic-style pairing (review 2026-10-15, step 1): the
learner is tested apart from the question of what dopamine carries.  Fixed weights; only the online rules run
(the compartment memory phi on Kenyon -> MBON, as in life).  From the checkpoint's state:
  trial = REST silent ticks (true silence), then a word held WORD ticks; dopamine = the compartment's DANs driven at
  LEVEL x rheobase for PULSE ticks: 'forward' starts at tick FWD_AT of the word (after its Kenyon cells fire),
  'backward' ends BWD_LEAD ticks before the word starts.
Protocol per arm (forward / backward / none): pre-test X and Z (no dopamine); 10 pairings of X (Z interleaved without
dopamine); post-test X and Z; 256 words of the lineage's conversation; retest X and Z.
Reported: the MBON's spikes during the second half of the word (its response to X and Z) at each test, and phi on the
compartment edges onto that MBON from the Kenyon cells the word fired.
Usage: python pairing_probe.py <checkpoint dir> <run dir> [MBON type=MBON11] [DAN type=PPL101] [out.json]"""
import sys, json, os, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork, RUN = Path(sys.argv[1]), Path(sys.argv[2])
MBON_T = sys.argv[3] if len(sys.argv) > 3 else 'MBON11'
DAN_T = sys.argv[4] if len(sys.argv) > 4 else 'PPL101'
OUT = Path(sys.argv[5]) if len(sys.argv) > 5 else None
REST, WORD, PULSE, LEVEL = 32, 32, 8, float(os.environ.get('PAIR_LEVEL', '2.0'))
FWD_AT, BWD_LEAD = 16, 4
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
s0 = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
if getattr(model, 'dopamine_sensitivity', None) is not None:
    opts['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, s0.h)
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values, ann['class'].astype(str).values
mbon = torch.as_tensor(np.flatnonzero(typ == MBON_T), device='cuda')
dan = torch.as_tensor(np.flatnonzero(typ == DAN_T), device='cuda')
kc = torch.as_tensor(np.flatnonzero(cls == 'Kenyon_Cell'), device='cuda')
on_mbon = torch.isin(model.comp_post.long(), mbon)
with torch.no_grad():
    rates = model.get_decay_rates()[0]
    rheo = (model.get_thresholds().expand_as(rates) * -torch.log(rates.clamp(1e-5, 1 - 1e-7)))[0]
pulse_drive = torch.zeros(1, model.n_neurons, device='cuda')
pulse_drive[0, dan] = LEVEL * rheo[dan]
import tiktoken
tok = tiktoken.get_encoding('gpt2')
X, Z = tok.encode(' flower')[0], tok.encode(' water')[0]
lines = [json.loads(l) for l in (RUN / 'mother.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
filler = []
for r in lines:
    t = r['text'].strip()
    if t and t != '...' and not t.startswith('Mengli'):
        filler += tok.encode(' ' + t)
    if len(filler) >= 256:
        break


def tick(s, w, on=True, drive=None):
    model.external_drive = drive
    s = fl.advance_fly_token_stream(model, s, torch.tensor([int(w)], device='cuda'), [1], ticks=1, hold_input=True,
                                    fresh=False, sensory_on=on, **opts)[0]
    model.external_drive = None
    return s


def trial(s, w, mode):
    for t in range(REST):
        s = tick(s, w, on=False, drive=pulse_drive if mode == 'backward' and REST - BWD_LEAD - PULSE <= t < REST - BWD_LEAD else None)
    resp, fired = 0.0, torch.zeros(kc.numel(), dtype=torch.bool, device='cuda')
    th = model.get_thresholds().reshape(-1)
    for t in range(WORD):
        s = tick(s, w, drive=pulse_drive if mode == 'forward' and FWD_AT <= t < FWD_AT + PULSE else None)
        spk = s.ring[0][0] > 0
        fired |= spk[kc]
        if t >= WORD // 2:
            # the MBON's depolarisation (v / theta, its spikes reset it, so a mean is a finer readout than spike counts)
            # plus its spikes, per tick
            vt = s.h[0, mbon] / (th.index_select(0, mbon) if th.numel() > 1 else th)
            resp += float(vt.mean() + spk[mbon].float().mean()) / (WORD // 2)
    s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
    return s, resp, fired


def phi_on(s, fired_kc):
    fired_all = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda')
    fired_all[kc[fired_kc]] = True
    sel = on_mbon & fired_all[model.comp_pre.long()]
    return float(s.comp_memory[0][sel].mean()) if bool(sel.any()) else None


rep = {'mbon': MBON_T, 'dan': DAN_T, 'level': LEVEL, 'arms': {}}
with torch.no_grad():
    for arm in ('none', 'forward', 'backward'):
        s, rows = s0, {}
        for name in ('pre',):
            s, rx, fx = trial(s, X, 'none'); s, rz, fz = trial(s, Z, 'none')
            rows[name] = {'X': rx, 'Z': rz, 'phiX': phi_on(s, fx), 'phiZ': phi_on(s, fz)}
        for k in range(10):
            s, _, _ = trial(s, X, arm)
            s, _, _ = trial(s, Z, 'none')
        s, rx, fx = trial(s, X, 'none'); s, rz, fz = trial(s, Z, 'none')
        rows['post'] = {'X': rx, 'Z': rz, 'phiX': phi_on(s, fx), 'phiZ': phi_on(s, fz)}
        for w in filler:
            for t in range(WORD):
                s = tick(s, w)
            s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        s, rx, fx = trial(s, X, 'none'); s, rz, fz = trial(s, Z, 'none')
        rows['retest'] = {'X': rx, 'Z': rz, 'phiX': phi_on(s, fx), 'phiZ': phi_on(s, fz)}
        rep['arms'][arm] = rows
        ch = lambda a, b: (b - a) / a if a else None
        print(arm, json.dumps({k: {kk: (round(vv, 4) if vv is not None else None) for kk, vv in v.items()} for k, v in rows.items()}),
              '| MBON change X %s Z %s' % (ch(rows['pre']['X'], rows['post']['X']), ch(rows['pre']['Z'], rows['post']['Z'])), flush=True)
if OUT:
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
