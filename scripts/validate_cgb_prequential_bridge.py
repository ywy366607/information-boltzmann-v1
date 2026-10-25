"""Numerical bridge/joint-state contracts; not a language capability study."""

import importlib.util
from fractions import Fraction
import json
import math
from pathlib import Path
from random import Random
import sys


def load_modules():
    root = Path(__file__).resolve().parents[1] / "information_boltzmann/core"
    loaded = {}
    for name in ("cgb_low_resource_life", "cgb_information_ledger", "cgb_language_kernel", "cgb_prequential_bridge"):
        spec = importlib.util.spec_from_file_location(name, root / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded[name] = module
        spec.loader.exec_module(module)
    return loaded


def state_key(port):
    state = port.state_dict() if hasattr(port, "state_dict") else (vars(port.ledger), port.last_tick, list(port.learner.calls))
    return json.dumps(state, sort_keys=True, default=lambda value: value.tolist())


def reject(operation, port=None):
    before = state_key(port) if port else None
    try:
        operation()
    except (ValueError, TypeError):
        assert port is None or state_key(port) == before
        return
    raise AssertionError("invalid operation admitted")


def joint_rule(code, hidden, bit, context, depth, prefix):
    return (code + 9 if hidden else 1) * (3 if hidden == bit else 1) * (2 if hidden == ((context[0] >> depth) & 1) else 1)


def advance(context, token, posterior):
    return (token,)

def oracle_probability(codes, context, vocab, token):
    probability, prefix = Fraction(1), 0
    for depth, code in enumerate(codes):
        remaining, a = len(codes) - depth - 1, code + 9
        allowed = [b for b in (0, 1) if (2 * prefix + b) * 2**remaining < vocab]
        bit = (token >> remaining) & 1
        masses = (6 + a, 2 + 3 * a) if not ((context[0] >> depth) & 1) else (3 + 2 * a, 1 + 6 * a)
        probability *= Fraction(masses[bit], sum(masses[b] for b in allowed))
        prefix = 2 * prefix + bit
    return probability


def validate_tree_port(module):
    def fresh(vocab=5, learning=True, seed=71, legacy=False):
        boot = min(3, 2**((vocab - 1).bit_length()) - 1)
        return module.PackedTreePort(vocab, joint_rule, advance if legacy else None, context=(boot,),
                                    seed=seed, learning=learning, rule_id="joint-fixture-v2")
    vocabularies = (1, 2, 3, 5, 8, 9, 17, 256, 257)
    for vocab in vocabularies:
        port = fresh(vocab)
        before, frame = state_key(port), port.predict()
        assert sum((frame.probability(t) for t in range(vocab)), Fraction(0)) == 1
        assert state_key(port) == before and port.predict() == frame and port.state_dict()["version"] == 2
        for target in range(vocab):
            assert frame.probability(target) == oracle_probability(frame.codes, frame.context, vocab, target)
            assert math.isclose(frame.nll(target), -math.log(float(frame.probability(target))))
        assert all(0 <= port.sample() < vocab for _ in range(32))
        reject(lambda: frame.probability(vocab))
    port = fresh()
    for target in (4, 1, 3, 0, 2, 4, 2, 0):
        frame, old_updates = port.predict(), port.updates
        old = tuple(frame.probability(t) for t in range(5))
        rng, mask = Random(), 0
        rng.setstate(port.rng.getstate())
        for depth, code in enumerate(frame.codes):
            bit, memory = (target >> (len(frame.codes) - depth - 1)) & 1, (frame.context[0] >> depth) & 1
            m0 = (3 if bit == 0 else 1) * (2 if memory == 0 else 1)
            m1 = (code + 9) * (3 if bit == 1 else 1) * (2 if memory == 1 else 1)
            mask |= int(rng.randrange(m0 + m1) >= m0) << depth
        assert math.isclose(port.observe(target), -math.log(float(old[target])))
        assert port.context == (mask,) and port.rng.getstate() == rng.getstate()
        assert tuple(frame.probability(t) for t in range(5)) == old
        changed = module.PackedTreeFrame(5, port.predict().codes, frame.context, frame.tick, frame.rule)
        assert (changed.probability(target) > old[target]) == (port.updates > old_updates)
    checkpoint, restored = port.state_dict(), fresh()
    restored.load_state_dict(json.loads(json.dumps(checkpoint)))
    assert state_key(restored) == state_key(port)
    for target in (1, 4, 0, 3):
        assert restored.sample() == port.sample() and restored.observe(target) == port.observe(target)
        assert state_key(restored) == state_key(port)
    a, b = fresh(), fresh()
    for _ in range(7):
        a.sample()
    for target in (4, 0, 2):
        assert a.observe(target) == b.observe(target)
        assert a.context == b.context and a.packed.tolist() == b.packed.tolist()
    for operation in (lambda: port.observe(2, external=False), lambda: port.observe(-1)):
        reject(operation, port)
    for changes in (dict(rule_id="wrong"), dict(generation_rng_state=()), dict(context=()),
                    dict(context=(8,)), dict(transition="legacy"), dict(version=1)):
        reject(lambda: port.load_state_dict(dict(checkpoint, **changes)), port)
    reject(lambda: fresh(vocab=9).load_state_dict(checkpoint))
    reject(lambda: module.PackedTreePort(5, joint_rule, context=(-1,), rule_id="invalid"))
    legacy = fresh(legacy=True)
    old = legacy.state_dict()
    assert old["version"] == 1 and old["transition"] == "legacy"
    del old["transition"]
    legacy.load_state_dict(old)
    legacy.observe(2)
    assert legacy.context == (2,)
    reject(lambda: port.load_state_dict(old), port)
    reject(lambda: legacy.load_state_dict(checkpoint), legacy)
    for changes in (dict(version=2), dict(transition="joint")):
        reject(lambda: legacy.load_state_dict(dict(old, **changes)), legacy)
    frozen = fresh(learning=False)
    packed = frozen.packed.tolist()
    frozen.observe(1)
    assert frozen.packed.tolist() == packed and frozen.updates == 0
    invalid = module.PackedTreePort(5, joint_rule, lambda *_: (True,), context=(3,), rule_id="bad-advance")
    reject(lambda: invalid.observe(4), invalid)
    bad_rule = lambda c, h, b, s, d, p: 0 if d else joint_rule(c, h, b, s, d, p)
    invalid = module.PackedTreePort(5, bad_rule, context=(3,), rule_id="bad-prefix")
    reject(invalid.sample, invalid)
    tickets = 0
    for code in range(-8, 8):
        for memory in (0, 1):
            for bit in (0, 1):
                frame = module.PackedTreeFrame(2, (code,), (memory,), 0, joint_rule)
                rule, count, context = frame._law(0, 0, bit)
                masses = [(code + 9 if h else 1) * (3 if h == bit else 1) * (2 if h == memory else 1) for h in (0, 1)]
                counts = [0, 0]
                for ticket in range(sum(masses)):
                    hidden, label = module.kernel.local_sample(code, context, count, rule, ticket)
                    counts[hidden] += 1
                    assert label == bit
                assert counts == masses
                tickets += sum(masses)
    frames = [module.PackedTreeFrame(2, (code,), (0,), 0, joint_rule) for code in (0, 1)]
    assert frames[0].probability(1) != frames[1].probability(1)
    posterior = []
    for frame in frames:
        rule, count, context = frame._law(0, 0, 1)
        masses = [rule(frame.codes[0], h, context)[0] for h in (0, 1)]
        posterior.append(Fraction(masses[1], sum(masses)))
    assert posterior == [Fraction(27, 29), Fraction(30, 32)] and posterior[0] != posterior[1]
    dependent = [module.PackedTreeFrame(2, (0,), (m,), 0, joint_rule).probability(1) for m in (0, 1)]
    cut = lambda c, h, b, s, d, p: (c + 9 if h else 1) * (3 if h == b else 1)
    independent = [module.PackedTreeFrame(2, (0,), (m,), 0, cut).probability(1) for m in (0, 1)]
    assert dependent[0] != dependent[1] and independent[0] == independent[1]
    return dict(normalization_vocabularies=vocabularies, exact_leaf_probabilities=558,
                joint_posterior_tickets=tickets, joint_state_continuation=True, resume_and_rng_isolation=True)

def main():
    modules = load_modules()
    class RecordingLearner:
        def __init__(self): self.calls = []
        def predict(self): self.calls.append("predict")
        def observe(self, token, *, external=True): self.calls.append((token, external)); return 0.0
    learner = RecordingLearner()
    ledger = modules["cgb_information_ledger"].PredictiveInformationLedger(reference_name="fixed")
    bridge = modules["cgb_prequential_bridge"].CGBPrequentialBridge(learner, ledger, lambda _: 0.0)
    Observation = modules["cgb_low_resource_life"].Observation
    result = bridge.step(Observation(0, 17), write_units=1, dissipation_units=2)
    assert result.target == 17 and result.ledger_row.useful_bits == 0.0
    assert learner.calls == ["predict", (17, True)]
    reject(lambda: bridge.step(Observation(2, 18), write_units=0, dissipation_units=0), bridge)
    for value in (-1, math.nan, math.inf):
        bridge.reference_nll_nats = lambda _, value=value: value
        reject(lambda: bridge.step(Observation(1, 18), write_units=0, dissipation_units=0), bridge)
        bridge.reference_nll_nats = lambda _: 0.0
        reject(lambda: bridge.step(Observation(1, 18), write_units=0, dissipation_units=0, calibrated_energy=value), bridge)
    print("cgb_prequential_bridge_ok", validate_tree_port(modules["cgb_prequential_bridge"]))


if __name__ == "__main__":
    main()
