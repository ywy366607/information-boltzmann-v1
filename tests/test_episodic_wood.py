"""Numerical/interface invariants, not synthetic capability experiments."""
import torch
from scripts.ib_local.navigation_episodic_wood import candidate_read, within_option_shuffle


def read(options, bank_options, values, *, exclude=None, topk=8):
    source = torch.eye(2)
    return candidate_read(source[:len(options)], source[:len(options)], options,
                          source[:len(bank_options)], source[:len(bank_options)],
                          bank_options, values, torch.arange(len(bank_options)),
                          exclude=exclude, topk=topk)


def test_same_option_and_empty_read_are_exact():
    values = torch.tensor([[1., 2.], [5., 8.]])
    out, packet = read(torch.tensor([0, 2]), torch.tensor([0, 1]), values)
    torch.testing.assert_close(out, torch.tensor([[1., 2.], [0., 0.]]), rtol=0, atol=0)
    torch.testing.assert_close(packet['weights'].sum(-1), torch.tensor([1., 0.]), rtol=0, atol=0)


def test_replay_cannot_read_its_own_target():
    out, packet = read(torch.tensor([0, 0]), torch.tensor([0, 0]),
                       torch.tensor([[1., 2.], [5., 8.]]), exclude=torch.tensor([0, 1]))
    torch.testing.assert_close(out, torch.tensor([[5., 8.], [1., 2.]]), rtol=0, atol=0)


def test_convex_read_cannot_amplify_stored_innovation_range():
    values = torch.tensor([[-3., 2.], [5., 8.]])
    out, _ = read(torch.tensor([0, 0]), torch.tensor([0, 0]), values)
    assert torch.all(out >= values.amin(0)) and torch.all(out <= values.amax(0))


def test_current_reencoding_has_gradients_without_historical_graph():
    encoder = torch.nn.Linear(3, 2)
    source = torch.tensor([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.]])
    query = torch.tensor([[.2, .9, 1.]])
    out, _ = candidate_read(encoder(query), encoder(query+.1), torch.tensor([0]),
        encoder(source), encoder(source+.2), torch.zeros(3, dtype=torch.long),
        torch.arange(3.)[:, None], torch.arange(3))
    out.sum().backward()
    assert encoder.weight.grad.abs().sum() > 0
    assert not source.requires_grad


def test_shuffle_retains_option_marginals_without_global_rng_change():
    values = torch.arange(20.).reshape(5, 4)
    option = torch.tensor([0, 1, 0, 1, 2])
    rng = torch.get_rng_state().clone()
    shuffled = within_option_shuffle(values, option)
    assert torch.equal(rng, torch.get_rng_state())
    for i in range(3):
        assert torch.equal(values[option == i].sum(0), shuffled[option == i].sum(0))
    assert not torch.equal(values[:4], shuffled[:4])


def test_no_valid_rows_is_finite_and_zero():
    query = torch.ones(3, 2)
    out, packet = candidate_read(query, query, torch.arange(3), query[:0], query[:0],
        torch.empty(0, dtype=torch.long), torch.empty(0, 4), torch.empty(0, dtype=torch.long))
    assert torch.equal(out, torch.zeros(3, 4))
    assert packet['weights'].shape == (3, 0)


def test_final_gain_training_reuses_pre_action_feeling_not_settled_result():
    from scripts.ib_local.navigation_episodic_wood import EpisodicWoodEngine
    engine = object.__new__(EpisodicWoodEngine)
    engine.device = torch.device('cpu')
    engine._pending_record = dict(choice=1, feeling=torch.tensor([1., 2., 3.]))
    engine._wood_source = torch.ones(2)
    engine._wood_action = torch.zeros(2)
    engine._wood_replay = None
    engine._forecast = dict(base=torch.zeros(3, 4))
    engine.intero = lambda: torch.tensor([[100., 200., 300.]])
    engine._training_batch(torch.zeros(4))
    assert torch.equal(engine._episodic_query_feeling, torch.tensor([[1., 2., 3.]]))
