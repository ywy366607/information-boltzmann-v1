"""M2/M3 probe (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md): eyes and output in a
world-paced stream.  Fixed weights, the checkpoint's own physical state and act integrator, the same validation text for
every condition (256 words burn-in, then 1,024 scored).  The rules are the training rules: the eye decides before each
word is written (binary eyes: a look sampled from the policy with a fixed seed; else a gain); the main speaker asks a
question with every word ("what comes next?"), answered by its next word; background (noise) words neither ask nor
close; a commit answers the oldest open question (value -CE of the read then against that question's answer);
questions close at the first main onset where the brain looks (or after `ai_depth` + 1 main onsets), unanswered =
-log V.  For continuous-eye checkpoints (M1/M2) the eyes look iff the gain >= 0.5 (open in practice).

Conditions: geo mu (each word's interval K ~ Geom(1/mu), same seed per mu), fixed16 (every word 16 ticks: the latency
the brain chooses when the world does not interrupt), and noisy geo mu (in every 4th stretch of 32 words, words 8..15
replaced by draws from the training noise alphabet).  Eye modes: own (the head decides) / open (forced open).

Reported per (condition, mode): answered fraction, latency [min | median | mean | max] among answered (ticks since the
question opened), answer value, commits per word, Spearman(latency, difficulty) among answered, the commit hazard by
tick for the easiest and hardest quartile (censoring-aware), the eye at main onsets with / without an open question
(and by that question's difficulty), questions extended by closed eyes, arrival and survival CE; on the noisy
condition the open fraction on noise vs text onsets.  Per-word arrays are kept for paired comparisons (ai_report.py).
Usage: python act_probe.py <checkpoint dir> [out .jsonl]   env ACT_MUS (16,8,4,2), ACT_FIXED (16), ACT_NOISY (4,2),
ACT_MODES (own,open)
"""
import sys, os, json, math, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core.fly_active_inference import ai_eye, ai_step, eye_act, motor_rest, precision_of
BURN, SCORE = int(os.environ.get('ACT_BURN', BURN)), int(os.environ.get('ACT_SCORE', SCORE))
ints = lambda key, default: tuple(int(x) for x in os.environ.get(key, default).split(',') if x)
MUS, FIXED, NOISY = ints('ACT_MUS', '16,8,4,2'), ints('ACT_FIXED', '16'), ints('ACT_NOISY', '4,2')

fork = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else fork / 'act_probe.jsonl')
model, cfg, L = load(fork)
assert getattr(model, 'ai_w', None) is not None, 'no act head in this checkpoint'
BINARY, DEPTH = bool(cfg.get('eye_binary', False)), int(cfg.get('ai_depth') or 0)
HOLD, SMAX = bool(cfg.get('hold_input', False)), float(cfg.get('sense_noise') or 0.0)
if os.environ.get('ACT_SMAX'):
    # zero-shot: put a checkpoint into the sustained noisy world
    HOLD, SMAX = True, float(os.environ['ACT_SMAX'])
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons
writer = model.topographic_writer
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region = 60_000
clean = np.array(val[region - 1:region + BURN + SCORE + 1], dtype=np.int64)       # input i = seq[i], target = seq[i+1]
alphabet = np.random.default_rng(int(cfg.get('noise_seed') or 7)).choice(50257, int(cfg.get('noise_alphabet') or 16), replace=False)
noisy = clean.copy()
noise_mask = np.zeros(len(clean), bool)
nrng = np.random.default_rng(11)
for i in range(len(clean)):
    if (i // 32) % 4 == 1 and 8 <= i % 32 < 16:
        noisy[i] = nrng.choice(alphabet); noise_mask[i] = True
g0 = L.get('ai_g')
MOTOR = getattr(model, 'ai_mode', None) == 'motor'
if g0 is None:
    g0 = motor_rest(model, torch.zeros(1, 1)) if MOTOR else torch.tensor([[math.log(0.95 / 0.05), 0.0]])
LOGV = math.log(50257)
# the value of not answering: the listener's prior (unigram CE of the answer) if the checkpoint was trained so
_counts = np.bincount(np.asarray(np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')[:20_000_000], dtype=np.int64),
                      minlength=50257).astype(np.float64) + 0.5
UNIGRAM = -np.log(_counts / _counts.sum())
PRIOR_MISS = bool(cfg.get('miss_prior', False))
d = lambda x: [float(np.nanmin(x)), float(np.nanmedian(x)), float(np.nanmean(x)), float(np.nanmax(x))] if np.isfinite(x).any() else None


def run(name, mode, mu=None, fixed=None, use_noise=False):
    seq = noisy if use_noise else clean
    mask = noise_mask if use_noise else np.zeros(len(seq), bool)
    inputs, targets = seq[:-1], seq[1:]
    n_words = len(inputs)
    main = ~mask[:-1]                                          # input i is a main word
    answer_of = np.zeros(n_words, np.int64)                    # the main speaker's next word after input i
    nxt = int(seq[-1])
    for i in range(n_words - 1, -1, -1):
        if not mask[i + 1]:
            nxt = int(seq[i + 1])
        answer_of[i] = nxt
    rng = np.random.default_rng(1000 + (mu or 0))
    torch.manual_seed(1234 + (mu or 0))
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    g = g0.clone().float().cuda()
    q_answer = (UNIGRAM[answer_of].copy() if PRIOR_MISS else np.full(n_words, LOGV)); q_latency = np.full(n_words, -1.0); q_ext = np.zeros(n_words)
    q_open_tick = np.zeros(n_words)
    eye_on = np.zeros(n_words); K_of = np.zeros(n_words); ce1 = np.zeros(n_words); ent1 = np.zeros(n_words)
    had_open = np.zeros(n_words, bool); open_hard = np.full(n_words, np.nan)
    n_commits = np.zeros(n_words); arrival = np.zeros(n_words); survival = np.zeros(n_words)
    oracle_ce = np.zeros(n_words); oracle_tick = np.zeros(n_words); tick_curve = []
    sigma_of = np.zeros(n_words); eye_word = np.zeros(n_words)
    exp_value = np.zeros(n_words); exp_latency = np.zeros(n_words); latency_entropy = np.zeros(n_words); hz_first = np.zeros(n_words)
    gamma_word = np.full(n_words, np.nan)
    open_q, clock = [], 0
    srng = np.random.default_rng(2000 + (mu or 0))
    passed = np.zeros(n_words)
    with torch.no_grad():
        for i in range(n_words):
            token = torch.tensor([int(inputs[i])], device='cuda')
            K = fixed if fixed else int(rng.geometric(1.0 / mu))
            K_of[i] = K
            ces, q_ces, q_hz = [], [], []
            noise = None
            if HOLD and SMAX > 0:
                # sustained noisy input: the word is heard every tick as a noisy sample; clarity drawn per word
                sigma_of[i] = SMAX * srng.random()
                emb = model.embedding(token)
                noise = torch.as_tensor(srng.standard_normal((K, emb.shape[-1])), dtype=emb.dtype, device='cuda') \
                    * (sigma_of[i] * float(emb.norm()) / math.sqrt(emb.shape[-1]))
            if main[i]:
                had_open[i] = bool(open_q)
                if open_q:
                    open_hard[i] = ce1[open_q[0]]
                # a question stays open for at most DEPTH further main words
                for q in open_q:
                    passed[q] += 1
                open_q = [q for q in open_q if passed[q] <= DEPTH]
            eye_sum = 0.0
            for t in range(1, K + 1):
                clock += 1
                if t == 1 or HOLD:
                    # 'glance' (diagnostic): look at a word's first tick only, then close to settle
                    forced = 1.0 if mode == 'open' else ((1.0 if t == 1 else 0.0) if mode == 'glance' else None)
                    if BINARY:
                        e, look = eye_act(model, g, forced, sample=(mode == 'own'))
                    else:
                        e = ai_eye(model, g, forced); look = float(e) >= 0.5
                    eye_sum += float(e)
                    if t == 1:
                        eye_on[i] = float(e)
                    if main[i]:
                        # receiving the main speaker's word (eyes open while it is written) closes every older open
                        # question (unanswered = miss); closed eyes keep them open
                        if t == 1 and not look:
                            for q in open_q:
                                q_ext[q] += 1
                        if look:
                            open_q = [q for q in open_q if q >= i]
                        if t == 1:
                            open_q.append(i); q_open_tick[i] = clock
                    sample = None if noise is None else model.embedding(token) + noise[t - 1:t]
                    drive, baseline = fl.write_input(model, s, token, sample)
                    quiet_baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                    baseline = e * baseline + (1.0 - e) * quiet_baseline
                    s = fl.step_fly_physical_tick(model, s, token, drive * e, baseline, opts, base_rates=opts['base_rates'])
                else:
                    baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                    s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
                hz, g, committed = ai_step(model, s, g)
                q_hz.append(float(hz))
                lat = fl.read_fly_motor(model, s)
                logp = torch.log_softmax(model.decoder(model.read_norm(lat)).float(), -1)[0]
                ces.append(float(-logp[int(targets[i])]))
                own = float(-logp[int(answer_of[i])])                 # the read against the question opened at onset i
                q_ces.append(own)
                if t == 1:
                    ce1[i] = own; ent1[i] = float(-(logp.exp() * logp).sum())
                if committed:
                    n_commits[i] += 1
                    if open_q:
                        q = open_q.pop(0)                              # the oldest open question
                        q_answer[q] = float(-logp[int(answer_of[q])]); q_latency[q] = clock - q_open_tick[q] + 1
            s = fl.expect_next(model, s, lat)
            eye_word[i] = eye_sum / (K if HOLD else 1)
            gw = precision_of(model, g)
            if gw is not None:
                gamma_word[i] = gw                              # the dopamine-encoded precision at the word's end
            arrival[i] = ces[-1]; survival[i] = sum(ces) / (mu or K)
            # hindsight oracle within the word's own interval: the best tick to have answered (an upper bound on what
            # waiting could gain; biased up by taking a min over noisy values)
            oracle_ce[i] = min(q_ces); oracle_tick[i] = int(np.argmin(q_ces)) + 1
            # exact expectations under this word's hazards within its own interval (motor acts: sampled commits)
            hh = np.clip(np.array(q_hz), 1e-6, 1 - 1e-6); Sb = np.concatenate([[1.0], np.cumprod(1 - hh)])
            pt = Sb[:-1] * hh; pmiss = Sb[-1]
            miss_i = UNIGRAM[answer_of[i]] if PRIOR_MISS else LOGV
            exp_value[i] = float((pt * -np.array(q_ces)).sum() + pmiss * -miss_i)
            exp_latency[i] = float((pt * np.arange(1, len(hh) + 1)).sum() / max(pt.sum(), 1e-9))
            p_all = np.concatenate([pt, [pmiss]]); latency_entropy[i] = float(-(p_all * np.log(np.clip(p_all, 1e-12, 1))).sum())
            hz_first[i] = hh[0]
            if fixed:
                tick_curve.append(q_ces[:16])
    sl = slice(BURN, BURN + SCORE)
    A = {'answer_ce': q_answer[sl], 'latency': q_latency[sl], 'ext': q_ext[sl], 'eye': eye_on[sl], 'K': K_of[sl],
         'ce1': ce1[sl], 'ent1': ent1[sl], 'n_commits': n_commits[sl], 'arrival': arrival[sl], 'survival': survival[sl],
         'had_open': had_open[sl].astype(float), 'open_hard': open_hard[sl], 'main': main[sl].astype(float),
         'noise': mask[:-1][sl].astype(float), 'oracle_ce': oracle_ce[sl], 'oracle_tick': oracle_tick[sl],
         'sigma': sigma_of[sl], 'eye_word': eye_word[sl], 'exp_value': exp_value[sl], 'exp_latency': exp_latency[sl],
         'latency_entropy': latency_entropy[sl], 'hz_first': hz_first[sl], 'prior_ce': UNIGRAM[answer_of[sl]], 'gamma': gamma_word[sl]}
    asked = A['main'] > 0
    answered = asked & (A['latency'] > 0)
    for k in ('answer_ce', 'latency', 'ext', 'ce1', 'ent1'):
        A[k] = np.where(asked, A[k], np.nan)
    rep = {'checkpoint': str(fork), 'condition': name, 'mode': mode, 'mu': mu, 'fixed': fixed, 'binary_eyes': BINARY,
           'depth': DEPTH, 'answered_frac': float(answered[asked].mean()), 'latency_min_median_mean_max': d(A['latency'][answered]),
           'answer_value_mean': float(-np.nanmean(A['answer_ce'])), 'commits_per_word': float(A['n_commits'].mean()),
           'questions_extended_frac': float((A['ext'][asked] > 0).mean()),
           'arrival_ce_min_median_mean_max': d(A['arrival']), 'survival_ce_min_median_mean_max': d(A['survival']),
           'eye_min_median_mean_max': d(A['eye']), 'eye_open_frac': float((A['eye'] >= 0.5).mean())}
    # diversity and adaptivity of the output, exact under the hazards (fixed16: no interruptions)
    rep['exp_value_mean'] = float(np.nanmean(np.where(asked, A['exp_value'], np.nan)))
    rep['exp_latency_min_median_mean_max'] = d(A['exp_latency'][asked])
    rep['latency_entropy_mean'] = float(np.mean(A['latency_entropy'][asked]))
    if np.isfinite(A['gamma']).any():
        gm = A['gamma'][asked]
        rep['gamma_min_median_mean_max'] = d(gm)
        rep['gamma_cv'] = float(np.std(gm) / max(np.mean(gm), 1e-12))
        rep['rho_exp_latency_gamma'] = float(spearmanr(A['exp_latency'][asked], gm)[0])
        rep['rho_latency_entropy_gamma'] = float(spearmanr(A['latency_entropy'][asked], gm)[0])
    rep['rho_exp_latency_ce1'] = float(spearmanr(A['exp_latency'][asked], A['ce1'][asked])[0])
    rep['rho_exp_latency_oracle_tick'] = float(spearmanr(A['exp_latency'][asked], A['oracle_tick'][asked])[0])
    hq_ = asked & (A['ce1'] >= np.nanquantile(A['ce1'], 0.75)); eq_ = asked & (A['ce1'] <= np.nanquantile(A['ce1'], 0.25))
    rep['exp_latency_easy_hard'] = [float(A['exp_latency'][eq_].mean()), float(A['exp_latency'][hq_].mean())]
    rep['answer_vs_prior_gain'] = float(np.nanmean(np.where(asked, A['prior_ce'] - A['answer_ce'], np.nan)))
    # what waiting could buy at all: the hindsight-best tick per word vs tick 1, and whether it tracks difficulty
    rep['oracle_gain_over_tick1'] = float(np.nanmean(A['ce1'] - A['oracle_ce']))
    rep['oracle_tick_min_median_mean_max'] = d(A['oracle_tick'][asked])
    rep['rho_oracle_tick_ce1'] = float(spearmanr(A['oracle_tick'][asked], A['ce1'][asked])[0])
    if fixed and tick_curve:
        curve = np.array(tick_curve)[sl]
        hard_q = A['ce1'] >= np.nanquantile(A['ce1'], 0.75); easy_q = A['ce1'] <= np.nanquantile(A['ce1'], 0.25)
        for label, sel in (('all', asked), ('easy', asked & easy_q), ('hard', asked & hard_q)):
            m = curve[sel].mean(0)
            rep[f'ce_by_tick_{label}'] = [round(float(x), 3) for x in m]
            rep[f'mean_curve_best_gain_{label}'] = float(m[0] - m.min())       # no hindsight bias: the mean curve
    if A['sigma'].std() > 0:
        # sustained noisy input: perceptual difficulty is the word's noise; answers are about what follows it
        rep['rho_oracle_tick_sigma'] = float(spearmanr(A['oracle_tick'][asked], A['sigma'][asked])[0])
        if answered.sum() > 10:
            rep['rho_latency_sigma'] = float(spearmanr(A['latency'][answered], A['sigma'][answered])[0])
            noisy_half = A['sigma'] >= np.median(A['sigma'])
            rep['latency_mean_clear_half'] = float(A['latency'][answered & ~noisy_half].mean())
            rep['latency_mean_noisy_half'] = float(A['latency'][answered & noisy_half].mean())
        if fixed and tick_curve:
            curve = np.array(tick_curve)[sl]
            for label, sel in (('clear', asked & (A['sigma'] <= np.quantile(A['sigma'], 0.25))),
                               ('noisy', asked & (A['sigma'] >= np.quantile(A['sigma'], 0.75)))):
                m = curve[sel].mean(0)
                rep[f'ce_by_tick_{label}'] = [round(float(x), 3) for x in m]
                rep[f'mean_curve_best_gain_{label}'] = float(m[0] - m.min())
        rep['eye_word_mean'] = float(A['eye_word'].mean())
    if answered.sum() > 10:
        rep['rho_latency_ce1'] = float(spearmanr(A['latency'][answered], A['ce1'][answered])[0])
        rep['rho_latency_entropy1'] = float(spearmanr(A['latency'][answered], A['ent1'][answered])[0])
    q1, q3 = np.nanquantile(A['ce1'], [0.25, 0.75])
    for label, sel in (('easy', asked & (A['ce1'] <= q1)), ('hard', asked & (A['ce1'] >= q3))):
        haz = {}
        for t in range(1, 17):
            at_risk = sel & (A['K'] >= t) & ((A['latency'] < 0) | (A['latency'] >= t))
            if at_risk.sum() >= 10:
                haz[t] = round(float((A['latency'][at_risk] == t).mean()), 4)
        rep[f'hazard_{label}'] = haz
        rep[f'latency_mean_{label}'] = float(A['latency'][sel & answered].mean()) if (sel & answered).any() else None
    if mode == 'own':
        # the eye at main onsets against the questions that onset would close: any open, and how hard
        closed = A['eye'] < 0.5
        has_open = asked & (A['had_open'] > 0)
        rep['closed_frac'] = float(closed[asked].mean())
        rep['closed_frac_when_open_question'] = float(closed[has_open].mean()) if has_open.any() else None
        rep['closed_frac_when_none_open'] = float(closed[asked & ~has_open].mean()) if (asked & ~has_open).any() else None
        if has_open.sum() >= 8:
            hard = A['open_hard'][has_open]
            hq = hard >= np.median(hard)
            rep['closed_frac_open_hard_half'] = float(closed[has_open][hq].mean())
            rep['closed_frac_open_easy_half'] = float(closed[has_open][~hq].mean())
    if use_noise:
        nz = A['noise'] > 0
        rep['eye_open_frac_noise'] = float((A['eye'][nz] >= 0.5).mean()); rep['eye_open_frac_text'] = float((A['eye'][~nz] >= 0.5).mean())
        late = nz & np.concatenate([[False, False], nz[:-2] & nz[1:-1]])         # third noise word onward
        rep['eye_open_frac_noise_late'] = float((A['eye'][late] >= 0.5).mean()) if late.any() else None
        resume = (~nz) & np.concatenate([[False], nz[:-1]])                     # the first text word after noise
        rep['arrival_ce_text'] = float(A['arrival'][~nz].mean())
        rep['eye_open_frac_resume'] = float((A['eye'][resume] >= 0.5).mean()) if resume.any() else None
    rep.update({f'per_word_{k}': [None if not np.isfinite(x) else float(x) for x in v] for k, v in A.items()})
    print(json.dumps({k: v for k, v in rep.items() if k != 'checkpoint' and not k.startswith('per_word')}), flush=True)
    with open(out, 'a') as f:
        f.write(json.dumps(rep) + '\n')


for mode in tuple(os.environ.get('ACT_MODES', 'own,open').split(',')):
    for fx in FIXED:
        run(f'fixed{fx}', mode, fixed=fx)
    for mu in MUS:
        run(f'geo{mu}', mode, mu=mu)
    for mu in NOISY:
        run(f'noisy_geo{mu}', mode, mu=mu, use_noise=True)
