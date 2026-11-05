"""Temporal wiring tests only; no claim about language capability."""
import numpy as np
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.language_navigation_stream_v3 import StreamingIndividual
from scripts.ib_local.language_navigation_world_v3 import StreamingLanguageWorld
from scripts.ib_local.latent_navigation_v3 import LatentNavigationV3
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.train_gdn_fast_table import TableGDN


def test_streaming_step_consumes_only_real_observed_bytes():
    torch.manual_seed(9)
    base = GatedDeltaNetLM(256, 16, 2, 2, 'decay_then_delta', 32, 4, 32., decay_init='log_timescale')
    model = LatentNavigationV3(TableGDN(base, True), 8, 8, 1, 4, 2,
                               max_chunk=4, history_size=4)
    # Deterministic unit test for the silent branch; not an action-rate policy.
    with torch.no_grad():
        model.action_decisions.weight.zero_()
        model.action_decisions.bias.fill_(-100)
    data = np.frombuffer(b'a real finite language stream with ordinary bytes', np.uint8)
    world = StreamingLanguageWorld(data, buckets=64, seed=4)
    table = RetrievedCreditTable(orders=(0, 1), heads=1, slots_log2=3)
    person = StreamingIndividual(model, table, world, model.initial_state(1), ord(' '))
    for _ in range(5):
        assert person.step() == b''
    assert person.external_bytes == 5 and person.self_bytes == 0
    assert world.cursor == 5 and int(world.seen.sum()) == 5
    assert int(model.memory.clock) == 5 and not bool(model.memory.occupied.any())


def test_new_actions_are_recorded_as_joint_history_not_sole_cause():
    torch.manual_seed(12)
    base = GatedDeltaNetLM(256, 16, 2, 2, 'decay_then_delta', 32, 4, 32., decay_init='log_timescale')
    model = LatentNavigationV3(TableGDN(base, True), 8, 8, 1, 4, 2,
                               max_chunk=4, history_size=4)
    # Force an action in this numerical wiring test; no behavioral claim.
    with torch.no_grad():
        model.action_decisions.weight.zero_()
        model.action_decisions.bias.copy_(torch.tensor([100., -100.]))
    data = np.frombuffer(b'continuously observed language content ' * 8, np.uint8)
    world = StreamingLanguageWorld(data, buckets=64, seed=4)
    table = RetrievedCreditTable(orders=(0, 1), heads=1, slots_log2=3)
    person = StreamingIndividual(model, table, world, model.initial_state(1), ord(' '), max_utterance=1)
    person.step()
    assert person.progress_count == 0  # future probe has not arrived
    for _ in range(2):
        person.step()
    assert int(model.memory.episodes) == 0
    person.step()
    assert int(model.memory.episodes) == 1
    assert int(model.memory.clock) == person.events == 8
    assert person.overlapping_action_events == 6
    assert int(model.memory.occupied.sum()) == 1
    assert int(model.memory.intervention_count[model.memory.occupied][0]) == 3
    assert bool(model.memory.progress_valid.any())
    assert person.progress_count > 0
