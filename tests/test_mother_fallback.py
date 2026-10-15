"""FallbackMother: the reserve mouth answers only a turn the primary leaves silent."""
from information_boltzmann.core.fly_mother_world import FallbackMother


class _Mouth:
    def __init__(self, name, answers):
        self.name, self.answers, self.calls, self.plan = name, list(answers), 0, None

    def reply(self, transcript):
        self.calls += 1
        return self.answers.pop(0)

    def update_plan(self, plan):
        self.plan = plan


def test_fallback_only_on_silence():
    a, b = _Mouth('A', ['hi', '', 'yes']), _Mouth('B', ['reserve'])
    m = FallbackMother(a, b)
    out = [m.reply([('fly', 'x')]) for _ in range(3)]
    assert out == ['hi', 'reserve', 'yes'] and b.calls == 1 and m.fallback_turns == 1
    m.update_plan({'k': 1})
    assert a.plan == b.plan == {'k': 1} and m.name == 'A (fallback B)'
