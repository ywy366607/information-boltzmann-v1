"""Current-body witness scoring, with equal opportunity for every reply.

The original 32-slot reservoir is reused. Only the surprise gate before its
admission is removed. Equal reply length makes one uniformly sampled window
per reply an explicit sampling contract, not a claim about unseen futures.
"""
import numpy as np
import torch

from scripts.ib_local.language_navigation_kanli_v4c import WINDOW
from scripts.ib_local.language_navigation_kanli_v4r import CurrentWitnessEngine
from scripts.ib_local.language_navigation_v2 import FLOOR


class RepresentativeWitnessEngine(CurrentWitnessEngine):
    @torch.no_grad()
    def _admit(self, response, response_probs, token_ce, own):
        if len(response) < WINDOW + 1:
            return False
        start = int(self.w_rng.integers(0, len(response)-WINDOW))
        heard = float(token_ce[own+start+1:own+start+WINDOW+1].mean())
        ids = torch.as_tensor(response[start:start+WINDOW].astype(np.int64), device=self.device)[None]
        target = torch.as_tensor(response[start+1:start+WINDOW+1].astype(np.int64), device=self.device)[None]
        logp = torch.as_tensor(np.log(np.maximum(response_probs[start+1:start+WINDOW+1], FLOOR)),
                                 device=self.device)[None]
        self.w_admitted += 1
        if len(self.w_ids) < self.capacity:
            slot = len(self.w_ids)
            self.w_ids.append(ids); self.w_tgt.append(target)
            self.w_logp.append(logp); self.w_hear.append(heard)
        else:
            slot = int(self.w_rng.integers(0, self.w_admitted))
            if slot < self.capacity:
                self.w_ids[slot], self.w_tgt[slot] = ids, target
                self.w_logp[slot], self.w_hear[slot] = logp, heard
        return True
