"""Frozen continuous-language evaluation with the training boundary protocol.

Carry the native state, pending byte, lexical history and observed memories.
Only observations advance the stream. Independent-request resets are an
explicit diagnostic, never the default of this evaluator. No optimizer step.
"""
from contextlib import nullcontext
from hashlib import sha256

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.latent_navigation_v2 import TransitionRead
from scripts.ib_local.native_state_policy import apply_native_boundary


class CarriedStream:
    """Local continuation of a complete engine, inside a reversible branch.

    As in training, ``state`` precedes ``pending``. An arriving block supplies
    real next-byte targets; the block's last byte remains pending. Thus no
    reset, duplicate boundary byte or fabricated final target is introduced.
    """
    def __init__(self, engine, table, *, reset_native=False, window=128):
        if not 1 <= window <= 128:
            raise ValueError('bounded128-byte observation window required')
        self.engine, self.table, self.window = engine, table, window
        self.state = (engine.model.initial_state(1, device=engine.device) if reset_native
                      else engine.recurrent.detach().clone())
        self.pending = int(engine.pending)

    @torch.no_grad()
    def observe(self, observed, *, training_forward=False):
        """Score/assimilate real arrivals; freeze all parameter updates.

        training_forward is an audit reference: use the real objective's
        forward and its original read receipt, then the same observed commit.
        Auxiliary losses are computed but never differentiated or optimized.
        """
        observed = np.asarray(observed, dtype=np.uint8)
        if observed.ndim != 1 or not len(observed):
            raise ValueError('nonempty one-dimensional real byte stream required')
        engine, model = self.engine, self.engine.model
        bank = getattr(model, 'local_replay', None)
        all_logits, all_targets, records = [], [], []
        raw = np.concatenate((np.array([self.pending], dtype=np.uint8), observed))
        saved = {n:(n in model.__dict__, getattr(model, n, None))
                 for n in ('_last_sigreg', '_near_outcome')}
        try:
            for start in range(0, len(observed), self.window):
                end = min(start+self.window, len(observed))
                part = raw[start:end+1]
                _, probability, _ = self.table.run_credit(part, max_window=self.window)
                ids, target, logp = engine._inputs(part, probability)
                carried = self.state
                self.state = apply_native_boundary(engine, self.state, self.window)
                initial = self.state.clone()
                scope = (bank.observation(dict(lr=0., betas=(.9,.999), eps=1e-8), end-start)
                         if bank is not None else nullcontext())
                with scope:
                    if training_forward:
                        _, state, features, diag = model.objective(ids, target, self.state, logp,
                            torch.randn(64, 32, device=engine.device))
                        logits = diag['logits']
                        receipt = TransitionRead(None, None, diag['read_slots'], diag['read_weights'])
                    else:
                        if hasattr(model.memory, 'begin_read_credit'):
                            model.memory.begin_read_credit()
                        logits, state, _, features, receipt = model.predict_observed(ids, self.state, logp)
                    if hasattr(model.memory, 'settle_read_credit'):
                        model.memory.settle_read_credit()
                    model.commit_observed(features, receipt)
                    if bank is not None:
                        bank.settle_and_commit()
                self.state, self.pending = state.detach(), int(raw[end])
                all_logits.append(logits.detach().float().cpu())
                all_targets.append(target.detach().cpu())
                records.append(dict(arrivals=end-start, first_input=int(part[0]),
                    last_pending=self.pending,
                    carried_state_sha256=sha256(carried.cpu().contiguous().numpy().tobytes()).hexdigest(),
                    incoming_state_sha256=sha256(initial.cpu().contiguous().numpy().tobytes()).hexdigest(),
                    outgoing_state_sha256=sha256(self.state.cpu().contiguous().numpy().tobytes()).hexdigest()))
        finally:
            for name, (owned, value) in saved.items():
                if owned:
                    setattr(model, name, value)
                elif name in model.__dict__:
                    delattr(model, name)
        logits, targets = torch.cat(all_logits, 1), torch.cat(all_targets, 1)
        losses = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='none')
        return dict(logits=logits, targets=targets, byte_nll=losses.tolist(),
            nll=float(losses.mean()), observations=len(observed), windows=records,
            final_state=self.state.cpu().clone(), final_pending=self.pending)
