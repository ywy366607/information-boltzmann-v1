"""Private, read-only trajectories of ONE frozen body with distinct use masks.

The batch axis is independent replays, not additional living individuals.
Each trajectory keeps its own original language/workspace recurrent state and
versioned exclusion. No result is committed by this module.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReplayPanel:
    model: object
    exclusions: torch.Tensor
    valid: torch.Tensor
    length: int


_panel = ContextVar('native_readonly_replay_panel', default=None)


def configure(engine, config, *, batch_size=None, reuse_actual=None):
    settings = dict(config.get('native_replay_runtime', {}))
    width = settings.get('batch_size', 1) if batch_size is None else batch_size
    if width not in (1,2,4,8):
        raise ValueError('private replay width must be 1, 2, 4 or 8')
    reuse = settings.get('reuse_objective_logits', False) if reuse_actual is None else reuse_actual
    engine.native_replay_batch_size = width
    engine.native_replay_reuse_actual = bool(reuse)
    config['native_replay_runtime'] = dict(schema=1, batch_size=width,
        reuse_objective_logits=bool(reuse), scope='fixed-parameter arrived-language use measurement')


def for_model(model):
    panel = _panel.get()
    return panel if panel is not None and panel.model is model else None


def for_field(field):
    panel = _panel.get()
    return panel if panel is not None and panel.model.body.coevolution is field else None


def exclude_reads(memory, slots, valid):
    panel = _panel.get()
    if panel is None or panel.model.memory is not memory:
        return valid
    if len(slots) != len(panel.exclusions)*panel.length:
        raise ValueError('deep read rows must preserve private batch/time order')
    excluded = panel.exclusions.repeat_interleave(panel.length, dim=0)
    return valid & ~((slots == excluded[:, :1]) &
                     (memory.read_credit_version[slots] == excluded[:, 1:]))


@contextmanager
def replay_scope(model, exclusions, length):
    if torch.is_grad_enabled() or _panel.get() is not None:
        raise ValueError('private replay panels require independent no-grad scope')
    from scripts.ib_local.native_word_batch import request
    if request.get() is not None:
        raise ValueError('full replay and endpoint requests have separate contracts')
    field = model.body.coevolution
    exclusions = torch.as_tensor(exclusions, dtype=torch.long, device=field.valid.device)
    if exclusions.ndim != 2 or exclusions.shape[1] != 2 or length < 1:
        raise ValueError('one slot/version exclusion per nonempty private trajectory')
    valid = field.valid[None] & ~((field.slots_id[None] == exclusions[:, :1]) &
                                 (field.versions[None] == exclusions[:, 1:]))
    batch = len(exclusions)
    previous = {}
    # Original masks mark byte positions. Expand only this private B axis;
    # actual observation/utterance scopes retain their original tensor owners.
    for name in ('_question_input_mask', '_question_output_mask', '_expression_remaining'):
        value = getattr(model, name, None)
        if torch.is_tensor(value) and value.ndim == 2 and value.shape == (1, length):
            previous[name] = value
            setattr(model, name, value.expand(batch, -1))
    token = _panel.set(ReplayPanel(model, exclusions, valid, length))
    try:
        yield
    finally:
        _panel.reset(token)
        for name, value in previous.items():
            setattr(model, name, value)


@torch.no_grad()
def logits(engine, ids, start, logp, exclusions, *, batch_size=4):
    """One original logits tensor per withdrawal, with bounded replay width."""
    if batch_size < 1 or ids.shape[0] != 1:
        raise ValueError('bounded replicas of one real input required')
    for begin in range(0, len(exclusions), batch_size):
        group = exclusions[begin:begin+batch_size]
        b = len(group)
        with replay_scope(engine.model, group, ids.shape[1]):
            result = engine.model.predict(ids.expand(b, -1),
                start.expand(-1, b, -1).contiguous(), logp.expand(b, -1, -1))[0]
        yield from result.split(1, dim=0)
