"""Acceptance of the event-driven execution (contract v2 in CGB_UNIFIED_REFERENCE_20261026,
"事件驱动执行与按事件计费"). Writes results/published/cgb_event_driven_20261027.json."""
from collections import Counter, deque
import importlib.util
import json
import math
import random
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.stats import chi2

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from validate_cgb_unified_reference import ROOT, life  # noqa: E402

sys.path.insert(0, str(ROOT / "information_boltzmann/core"))
spec = importlib.util.spec_from_file_location("cgb_local_medium", ROOT / "information_boltzmann/core/cgb_local_medium.py")
local = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = local
spec.loader.exec_module(local)
import cgb_local_medium_ct as CT  # noqa: E402


def pack(law, states):
    word = 0
    for edge, (p, q, w) in enumerate(states):
        word |= ((p * (law.charge_cap + 1) + q) * (law.slot_cap + 1) + w) << (edge * law.bits)
    return word.to_bytes(law.state_bytes, "little")


def make(law, states, reservoir, stock):
    body = local.LocalMedium(law, pack(law, list(states)))
    o = local.OpenMedium(body, reservoir)
    o.stock = stock
    return o


def key(o):
    return (tuple(o.body.get(e) for e in range(o.body.law.edges)), o.stock)


def energy_total(o):
    """E_base + c * content + mu0 * S (uniform mu): pi ~ 2**(-2 * this)."""
    f, b = o.body.law, o.body
    st = [b.get(e) for e in range(f.edges)]
    e = sum(b.unit_energy(s) for s in st)
    e += sum(f.potential[sum(st[r * f.labels + c][0] for c in range(f.labels))] for r in range(f.rows))
    e += sum(f.potential[sum(st[r * f.labels + c][1] for r in range(f.rows))] for c in range(f.labels))
    content = sum(sum(s) for s in st)
    return e + o.reservoir.content_cost * content + o.reservoir.chemical_base * o.stock


def reachable(law, start, reservoir, stock0, label, kappa):
    """BFS over states reachable by enabled events; returns states, index, generator rows."""
    seen = {}
    order = []
    rows = {}
    dq = deque([(tuple(start), stock0)])
    while dq:
        st, S = dq.popleft()
        if (st, S) in seen:
            continue
        seen[(st, S)] = len(order)
        order.append((st, S))
        o = make(law, st, reservoir, S)
        c = CT.ContinuousOpenMedium(o, tau=1.0, max_events=0, family_conductance=kappa, seed=0)
        r = c.rates(label)
        out = []
        for i in np.nonzero(r)[0]:
            o2 = make(law, st, reservoir, S)
            edits, ex, _b, sign = o2.prepare(int(i), label)
            o2._commit(edits, sign)
            k2 = key(o2)
            out.append((int(i), k2, float(r[i]), ex))
            if k2 not in seen:
                dq.append(k2)
        rows[(st, S)] = out
    return order, seen, rows


LAWS = [
    ("2x2_Q1W1", local.LocalLaw(2, 2, 1, 1, (0, 1, 3), epsilon=1, coupling=1)),
    ("2x3_Q2W1", local.LocalLaw(2, 3, 2, 1, (0, 1, -1, 2, 0, 1, 4), epsilon=-1, coupling=1)),
    ("3x2_Q1W2", local.LocalLaw(3, 2, 1, 2, (0, 2, 1, 3), epsilon=1, coupling=-1)),
]


RATE_LAWS = LAWS + [
    ("1x4_Q2W1", local.LocalLaw(1, 4, 2, 1, (0, 1, 0, 2, 1, 3, 0, 1, 2), epsilon=1, coupling=1)),
    ("3x1_Q1W1", local.LocalLaw(3, 1, 1, 1, (0, -1, 2, 1), epsilon=-1, coupling=-1)),
]


def random_state(law, rng):
    return [(rng.randint(0, law.charge_cap), rng.randint(0, law.charge_cap), rng.randint(0, law.slot_cap))
            for _ in range(law.edges)]


def check_rates():
    """1. vectorized rates == kappa * 2**(prepare exponent), every event, many states."""
    rng = random.Random(7)
    n = worst = 0
    for name, law in RATE_LAWS:
        for trial in range(60):
            res = local.ReservoirLaw(4, content_cost=rng.randint(-2, 2), chemical_base=rng.randint(-2, 2),
                                     stimulus=rng.randint(0, 3))
            o = make(law, random_state(law, rng), res, rng.choice([0, 1, 3]))
            kappa = tuple(rng.choice([0.0, 0.25, 1.0, 3.0]) for _ in range(5))
            c = CT.ContinuousOpenMedium(o, tau=1.0, max_events=0, family_conductance=kappa, seed=0)
            label = rng.randrange(law.labels)
            a, b = c.rates(label), c.rates_reference(label)
            worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(1e-300, np.maximum(np.abs(a), np.abs(b))))))
            n += 1
    return dict(states=n, laws=[x for x, _ in RATE_LAWS], max_relative_error=worst, passed=worst == 0.0)


def check_balance():
    """2. exact integer ratio check and pi Q = 0 on reachable sets (uniform mu)."""
    rows_out = []
    ok = True
    for name, law, start, stock in [
        ("2x2_Q1W1", LAWS[0][1], [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)], 2),
        ("2x3_Q2W1", LAWS[1][1], [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)], 1),
        ("3x2_Q1W2", LAWS[2][1], [(1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0), (0, 0, 1), (0, 0, 0)], 1),
    ]:
        res = local.ReservoirLaw(4, content_cost=1, chemical_base=-1, stimulus=0)
        kappa = (0.5, 2.0, 1.0, 0.25, 1.5)
        order, index, rows = reachable(law, start, res, stock, 0, kappa)
        bad = 0
        Eof = {k: energy_total(make(law, k[0], res, k[1])) for k in order}
        for x, out in rows.items():
            for i, y, r, ex in out:
                o_y = make(law, y[0], res, y[1])
                j = o_y.reverse(i)
                _e, ex_back, _b, _s = o_y.prepare(j, 0)
                if ex_back is None or ex - ex_back != -2 * (Eof[y] - Eof[x]):
                    bad += 1
        n = len(order)
        Qm = np.zeros((n, n))
        for x, out in rows.items():
            for _i, y, r, _ex in out:
                Qm[index[x], index[y]] += r
        Qm -= np.diag(Qm.sum(1))
        e = np.array([Eof[k] for k in order], dtype=float)
        pi = np.exp2(-2 * (e - e.min()))
        pi /= pi.sum()
        resid = float(np.abs(pi @ Qm).max() / np.abs(Qm).max())
        rows_out.append(dict(law=name, states=n, transitions=sum(len(v) for v in rows.values()),
                             ratio_mismatches=bad, pi_Q_max_relative=resid))
        ok &= bad == 0 and resid < 1e-12
    # negative control: give every cross-end move a factor (1 + N) read from the clique IT reads
    # (forward reads one clique, its reverse the other); the exact ratio check must then fail
    law = LAWS[1][1]
    res = local.ReservoirLaw(4, content_cost=1, chemical_base=-1, stimulus=0)
    start_ = [(1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
    order, index, rows = reachable(law, start_, res, 1, 0, (1, 1, 1, 1, 1))
    Eof = {k: energy_total(make(law, k[0], res, k[1])) for k in order}

    def own_read(o, i):
        _kind, edge, _other, _sign, ch = o.body.decode(i)
        r_, c_ = divmod(edge, law.labels)
        return o.body.total(ch, r_ if ch == 0 else c_)
    neg = checked = 0
    for x, out in rows.items():
        o_x = make(law, x[0], res, x[1])
        for i, y, r, ex in out:
            if i >= 4 * law.edges:
                continue
            o_y = make(law, y[0], res, y[1])
            j = o_y.reverse(i)
            _e, ex_back, _b, _s = o_y.prepare(j, 0)
            lhs = 2.0 ** ex * (1 + own_read(o_x, i)) / (2.0 ** ex_back * (1 + own_read(o_y, j)))
            checked += 1
            neg += int(abs(lhs - 2.0 ** (-2 * (Eof[y] - Eof[x]))) > 1e-9 * lhs)
    return dict(bodies=rows_out, negative_control=dict(cross_end_transitions=checked, ratio_failures=neg),
                passed=bool(ok and neg > 0))


def exact_dist(law, start, res, stock, label, kappa, tau):
    order, index, rows = reachable(law, start, res, stock, label, kappa)
    n = len(order)
    Qm = np.zeros((n, n))
    for x, out in rows.items():
        for _i, y, r, _ex in out:
            Qm[index[x], index[y]] += r
    Qm -= np.diag(Qm.sum(1))
    return order, index, rows, expm(Qm * tau)[index[(tuple(start), stock)]]


def chisq(counts, probs, N):
    exp_ = probs * N
    big = exp_ >= 5
    obs = np.append(counts[big], counts[~big].sum())
    ex = np.append(exp_[big], exp_[~big].sum())
    keep = ex > 0
    stat = float(((obs[keep] - ex[keep]) ** 2 / ex[keep]).sum())
    dof = int(keep.sum()) - 1
    return stat, dof, float(chi2.sf(stat, dof))


def check_ct_vs_expm():
    """3. event-driven end-state law == expm(Q tau); cap case == its exact truncated law."""
    law = LAWS[0][1]
    start = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)]
    res = local.ReservoirLaw(4, content_cost=1, chemical_base=0, stimulus=1)
    kappa, tau, N, label = (0.5, 1.0, 1.0, 0.5, 1.0), 0.7, 20000, 1
    order, index, rows, p = exact_dist(law, start, res, 2, label, kappa, tau)
    counts = np.zeros(len(order))
    capc = np.zeros(len(order))
    for k in range(N):
        o = make(law, start, res, 2)
        c = CT.ContinuousOpenMedium(o, tau=tau, max_events=1 << 20, family_conductance=kappa, seed=k)
        c.evolve(label, tau)
        counts[index[key(o)]] += 1
        o = make(law, start, res, 2)
        c = CT.ContinuousOpenMedium(o, tau=tau, max_events=1, family_conductance=kappa, seed=10 ** 6 + k)
        c.evolve(label, tau)
        capc[index[key(o)]] += 1
    out0 = rows[(tuple(start), 2)]
    R0 = sum(r for _i, _y, r, _ex in out0)
    pc = np.zeros(len(order))
    pc[index[(tuple(start), 2)]] = math.exp(-R0 * tau)
    for _i, y, r, _ex in out0:
        pc[index[y]] += (1 - math.exp(-R0 * tau)) * r / R0
    s1, d1, p1 = chisq(counts, p, N)
    s2, d2, p2 = chisq(capc, pc, N)
    return dict(states=len(order), runs=N, tau=tau, chi2=s1, dof=d1, p_value=p1,
                tv=float(0.5 * np.abs(counts / N - p).sum()),
                cap1_chi2=s2, cap1_dof=d2, cap1_p_value=p2, passed=bool(p1 > 1e-3 and p2 > 1e-3))


def check_first_reaction():
    """3b. P1b: independent local clocks (first reaction) give the same law as expm(Q tau)."""
    law = LAWS[0][1]
    start = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)]
    res = local.ReservoirLaw(4, content_cost=1, chemical_base=0, stimulus=1)
    kappa, tau, N, label = (0.5, 1.0, 1.0, 0.5, 1.0), 0.7, 10000, 1
    order, index, rows, p = exact_dist(law, start, res, 2, label, kappa, tau)
    counts = np.zeros(len(order))
    for k in range(N):
        o = make(law, start, res, 2)
        CT.ContinuousOpenMedium(o, tau=tau, max_events=1 << 20, family_conductance=kappa,
                                seed=7 * 10 ** 6 + k).evolve_first_reaction(label, tau)
        counts[index[key(o)]] += 1
    st, d, pv = chisq(counts, p, N)
    return dict(states=len(order), runs=N, chi2=st, dof=d, p_value=pv, passed=bool(pv > 1e-3))


def check_nfold_vs_attempts():
    """4. matched-kappa rejection-free chain == the attempt chain (sample_proposal + dyadic tickets)."""
    law = LAWS[0][1]
    start = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)]
    res = local.ReservoirLaw(4, content_cost=1, chemical_base=0, stimulus=1)
    A, N, label = 12, 20000, 1
    kappa = CT.matched_conductance(make(law, start, res, 2))
    order, index, rows = reachable(law, start, res, 2, label, kappa)
    n = len(order)
    P = np.zeros((n, n))
    for x, out in rows.items():
        for _i, y, r, _ex in out:
            P[index[x], index[y]] += r
    P += np.diag(1 - P.sum(1))
    assert P.min() >= -1e-12
    p = np.linalg.matrix_power(P, A)[index[(tuple(start), 2)]]
    nf = np.zeros(n)
    old = np.zeros(n)
    for k in range(N):
        o = make(law, start, res, 2)
        CT.ContinuousOpenMedium(o, tau=float(A), max_events=1 << 20, family_conductance=kappa,
                                seed=k).evolve_attempts(label, A)
        nf[index[key(o)]] += 1
        o = make(law, start, res, 2)
        rng = random.Random(5 * 10 ** 6 + k)
        for _ in range(A):            # the attempt loop of OpenMedium.tick_window
            i = o.sample_proposal(rng)
            bound = o.proposal_bound(i)
            ticket = rng.getrandbits(2 * bound)
            edits, ex, bound, sign = o.prepare(i, label)
            if ex is not None and ticket >> (bound + ex) == 0:
                o._commit(edits, sign)
        old[index[key(o)]] += 1
    s1, d1, p1 = chisq(nf, p, N)
    s2, d2, p2 = chisq(old, p, N)
    return dict(states=n, attempts=A, runs=N, kappa=list(kappa), nfold_chi2=s1, nfold_dof=d1, nfold_p=p1,
                attempt_loop_chi2=s2, attempt_loop_dof=d2, attempt_loop_p=p2,
                passed=bool(p1 > 1e-3 and p2 > 1e-3))


def fresh_open(source, stimulus=2, supply=65):
    f = local.LocalLaw(1, 256, 1, 1, (0,) * 257)
    body = local.LocalMedium(f, pack(f, [(1, 1, 0)] * f.edges))
    opened = local.OpenMedium(body, local.ReservoirLaw(8, content_cost=1, stimulus=stimulus))
    world = life.PrefixReplayWorld(source, source_id="open-material-interface", reserve=200,
                                  capacity=200, supply_per_tick=supply)
    return opened, world


def owt_source():
    from cgb_observation_stream import load_owt_sources
    data = ROOT / "data/ib_owt_10000"
    _, validation, _, provenance = load_owt_sources(data / "train.npy", data / "validation.npy", "byte")
    return bytes(int(v) for v in validation[:256]), provenance


def summary(xs):
    xs = np.asarray(xs, dtype=float)
    return dict(min=float(xs.min()), median=float(np.median(xs)), mean=float(xs.mean()), max=float(xs.max()))


def target_full(opened, label):
    """Before publication: no edge of the target Q clique can take one more charge."""
    f = opened.body.law
    return all(opened.body.get(r * f.labels + label)[1] >= f.charge_cap for r in range(f.rows))


def conserved(opened):
    return (sum(sum(opened.body.get(e)) for e in range(opened.body.law.edges)) + opened.stock
            == opened.initial_content + opened.reservoir.source_initial)


def run_tick_window(source, seed):
    """GPT's kernel as in validate_cgb_local_medium.validate_window_real_bytes (A=64)."""
    opened, world = fresh_open(source)
    opened._opportunities_per_tick = 64
    full = accepted = spent = bits = 0
    t0 = time.process_time()
    for tick in range(len(source)):
        rng = random.Random(seed * 10 ** 6 + 20261027 + tick)
        plan = tuple((i, rng.getrandbits(2 * opened.proposal_bound(i)))
                     for i in (opened.sample_proposal(rng) for _ in range(64)))
        full += int(target_full(opened, source[tick]))
        r = opened.tick_window(plan, world)
        accepted += r["accepted_reactions"]
        spent += r["spent"]
        bits += r["planned_random_bits"]
        assert conserved(opened)
    return dict(events=accepted, fees=spent, intake=opened.intake, release=opened.released,
                net_slots=sum(opened.body.get(e)[2] for e in range(opened.body.law.edges)),
                target_full_fraction=full / len(source), planned_ticket_bits=bits,
                cpu_seconds=time.process_time() - t0)


def run_ct(source, seed, *, mode, kappa_scale, stimulus):
    opened, world = fresh_open(source, stimulus=stimulus)
    base = CT.matched_conductance(fresh_open(source)[0])      # matched clocks of the h=2 law, all arms
    kappa = tuple(k * s_ for k, s_ in zip(base, kappa_scale))
    c = CT.ContinuousOpenMedium(opened, tau=64.0, max_events=4096, family_conductance=kappa, seed=seed)
    full = events = fees = capped = refused = 0
    t0 = time.process_time()
    for tick in range(len(source)):
        full += int(target_full(opened, source[tick]))
        if mode == "events":
            r = c.window(world)
            events += r["events"]
            fees += r["spent"]
            capped += int(r["capped"])
            refused += int(r["refused"])
        else:                                   # attempt tariff: pay each attempt, the first refusal ends
            opened.masses()
            b = world.begin()
            obs = world.publish()
            paid = 0
            while paid < 64 and world.pay_write(opened.body.law.attempt_fee):
                paid += 1
            refused += int(paid < 64)
            n, _m = c.evolve_attempts(obs.payload, paid) if b.active else (0, 0)
            events += n
            fees += paid
        assert conserved(opened)
    assert c.check_sync()
    return dict(events=events, fees=fees, intake=opened.intake, release=opened.released,
                intake_on_published_byte=c.intake_on_target, net_slots=int(c.w.sum()),
                capped_windows=capped, refused_windows=refused, target_full_fraction=full / len(source),
                random_draws=c.draws, family_events=list(c.family_events),
                cpu_seconds=time.process_time() - t0)


SEEDS = (1, 2, 3, 4, 5, 6)


def check_real_bytes():
    """5. real OWT, 6 seeds per arm, min/median/mean/max; interface and cost facts only."""
    source, provenance = owt_source()
    arms = {"tick_window_A64_attempt_tariff": [run_tick_window(source, sd) for sd in SEEDS]}
    arms["nfold_A64_attempt_tariff_matched"] = [run_ct(source, sd, mode="attempts", kappa_scale=(1,) * 5,
                                                       stimulus=2) for sd in SEEDS]
    for h in (2, 6):
        for name, scale in (("matched", (1, 1, 1, 1, 1)), ("quiet_internal_1_16", (1 / 16,) * 4 + (1,))):
            arms[f"event_tariff_{name}_h{h}"] = [run_ct(source, sd, mode="events", kappa_scale=scale,
                                                        stimulus=h) for sd in SEEDS]
    keys = ("events", "fees", "intake", "release", "intake_on_published_byte", "net_slots",
            "target_full_fraction", "cpu_seconds", "capped_windows", "refused_windows")
    table = {a: {k: summary([r[k] for r in runs]) for k in keys if k in runs[0]} for a, runs in arms.items()}
    return dict(source_provenance=provenance, fresh_bytes=len(source), seeds=list(SEEDS), arms=table, runs=arms,
                exchange_conductance_matched_at_h2_in_every_arm=True, conservation_every_window=True,
                arrays_in_sync=True, capability_claim=False, passed=True)


def check_restore():
    """6. snapshot between windows (body, stock, world, RNG) -> identical future."""
    source, _ = owt_source()
    base = CT.matched_conductance(fresh_open(source)[0])
    opened, world = fresh_open(source)
    c = CT.ContinuousOpenMedium(opened, tau=64.0, max_events=4096, family_conductance=base, seed=3)
    for _ in range(20):
        c.window(world)
    snap = json.loads(json.dumps(c.state_dict()))
    wsnap = json.loads(json.dumps(world.state_dict()))
    first = [c.window(world) for _ in range(20)]
    end = (c.state_dict(), world.state_dict())
    o2, w2 = fresh_open(source)
    c2 = CT.ContinuousOpenMedium(o2, tau=64.0, max_events=4096, family_conductance=base, seed=999)
    c2.load_state_dict(snap)
    w2.load_state_dict(wsnap)
    second = [c2.window(w2) for _ in range(20)]
    same = first == second and end == (c2.state_dict(), w2.state_dict())
    return dict(windows_before=20, windows_after=20, identical_future=bool(same), passed=bool(same))


def main():
    out = {}
    for name, fn in (("rates", check_rates), ("balance", check_balance), ("ct_vs_expm", check_ct_vs_expm),
                     ("first_reaction", check_first_reaction),
                     ("nfold_vs_attempts", check_nfold_vs_attempts), ("restore", check_restore),
                     ("real_bytes", check_real_bytes)):
        t0 = time.time()
        out[name] = fn()
        out[name]["seconds"] = round(time.time() - t0, 1)
        print(name, json.dumps(out[name])[:600], flush=True)
    out["passed"] = all(v.get("passed", True) for v in out.values() if isinstance(v, dict))
    out["contract"] = "docs/information_boltzmann/CGB_UNIFIED_REFERENCE_20261026.md (event-driven execution, v2)"
    out["scope"] = ("Execution and tariff facts for the existing generator; no language capability, no optimal "
                    "family clocks, event tariff argued physically (no erasure in a rejected attempt), not a Lean theorem.")
    (ROOT / "results/published/cgb_event_driven_20261027.json").write_text(json.dumps(out, indent=1))
    print("passed", out["passed"])


if __name__ == "__main__":
    main()
