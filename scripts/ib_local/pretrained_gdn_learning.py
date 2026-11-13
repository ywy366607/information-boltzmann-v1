"""Continuous pretrained-GDN learning with bounded local experience replays."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager

import torch
from torch.nn import functional as F

from scripts.ib_local.language_navigation_kanli_v4l import live_sigreg
from scripts.ib_local.pretrained_gdn import clone_cache, new_recurrent_cache, cache_nbytes
from scripts.ib_local.pretrained_gdn_experience import LayerExperienceRecall, copy_tree


def language_loss(logits, target, loss_mask=None):
    """Preserve token losses, optionally supervise only explicitly marked answers."""
    token_loss = F.cross_entropy(logits.float().flatten(0, 1), target.flatten(), reduction='none')
    if loss_mask is None:
        return token_loss, token_loss.mean()
    if loss_mask.shape != target.shape or loss_mask.dtype != torch.bool or not bool(loss_mask.any()):
        raise ValueError('nonempty boolean loss mask must match target shape')
    return token_loss, token_loss[loss_mask.flatten().to(token_loss.device)].mean()


def with_null_optimizer_group(stored,current,null_parameter_count):
    """Preserve all old Adam identities when appending the new null group."""
    result = copy_tree(stored)
    if len(result['param_groups']) == 2:
        group = dict(current['param_groups'][2])
        used = [i for g in result['param_groups'] for i in g['params']]
        start = max(used,default=-1)+1
        group['params'] = list(range(start,start+null_parameter_count))
        result['param_groups'].append(group)
    if len(result['param_groups']) != 3:
        raise ValueError('incompatible null-read optimizer groups')
    return result


class PretrainedExperienceLearner:
    def __init__(self, model, *, layers=(1, 12), capacity=512, detach_replay=False,
                 consolidation=True, consolidation_rule='retention', read_interval=1):
        if len(set(layers)) != len(layers) or capacity % len(layers):
            raise ValueError('unique layers and equally divided fixed capacity required')
        self.model, self.layers, self.capacity = model, tuple(layers), capacity
        model.requires_grad_(False)
        self.memories = {i: LayerExperienceRecall(model.config.hidden_size,
            capacity=capacity//len(layers), device=next(model.parameters()).device,
            consolidation=consolidation, consolidation_rule=consolidation_rule,
            read_interval=read_interval) for i in layers}
        self.native = []
        for layer in layers:
            for name, parameter in model.model.layers[layer].attn.named_parameters():
                parameter.requires_grad_(True)
                self.native.append((f'layer{layer}.{name}', parameter))
        self.masters = [torch.nn.Parameter(p.detach().float().clone()) for _, p in self.native]
        self.null_parameters = [p for m in self.memories.values() for p in (m.null_key,m.null_bias)]
        null_ids = {id(p) for p in self.null_parameters}
        self.extras = [p for m in self.memories.values() for p in m.parameters() if id(p) not in null_ids]
        self.optimizer = torch.optim.AdamW([
            dict(params=self.masters, lr=1e-4), dict(params=self.extras, lr=3e-4),
            dict(params=self.null_parameters,lr=3e-4)],
            eps=1e-8, weight_decay=0, foreach=False)
        self.directions = torch.randn(64, 16)
        self.cache = new_recurrent_cache()
        self.updates, self.tokens = 0, 0
        self.joint_capture_enabled = False
        self.set_replay_gradient(detach_replay)

    def set_replay_gradient(self, detach):
        for memory in self.memories.values():
            memory.detach_replay = detach

    def set_long_credit_cut(self, older_than=None):
        """Cut only old native derivatives, preserving values and encoder learning."""
        if older_than is not None and (type(older_than) is not int or older_than < 0):
            raise ValueError('nonnegative age threshold or None required')
        for memory in self.memories.values():
            memory.detach_replay = False
            memory.detach_replay_older_than = older_than

    def enable_full_window_participation(self):
        """Preserve all128 actual local inputs in future capsules, not just65.

        This storage migration leaves current retrieval/derivatives unchanged.
        A later contextual read must explicitly use the recorded continuation;
        merely saving it is not credit or a new forward memory capability.
        """
        for wrapper in self.memories.values():
            if wrapper.current is not None or wrapper.memory._read_credit_active:
                raise RuntimeError('finish the current observation before extending participation')
        for wrapper in self.memories.values():
            wrapper.capture_full_window = True

    def enable_joint_participation(self):
        """Archive coherent boundaries in the same two tables; no forward change."""
        self.enable_full_window_participation()
        self.joint_capture_enabled=True

    def enable_complete_reactivation(self):
        """Explicit migration of the local read program, not a state reset.

        Replay all already saved inputs, using the current query to read their
        per-position responses. Original65 capsules remain65; future captures
        keep all128. No parameters, Adam moments or native states are replaced.
        """
        self.enable_full_window_participation()
        for wrapper in self.memories.values():
            wrapper.reactivation_mode = 'complete_attention'

    @property
    def experience_storage_nbytes(self):
        seen=set()
        return sum(wrapper.storage_nbytes_shared(seen) for wrapper in self.memories.values())

    def archive_joint_participation(self,x,boundary,slots,parameter_version):
        """One shared archive of vectors+native boundary for the two local entries."""
        from scripts.ib_local.pretrained_gdn_experience import JointExperience
        if x.shape[0]!=1 or x.shape[1]!=128:
            raise ValueError('joint archive requires one actual128-input window')
        with torch.no_grad():
            features=copy_tree(self.model.get_input_embeddings()(x))
        held=dict(states=copy_tree(boundary.states),seen_tokens=boundary._seen_tokens)
        joint=JointExperience(features,held,parameter_version)
        for layer,slot in slots.items():
            self.memories[layer].capsules[slot].joint=joint
        return joint

    def enable_null_read(self):
        """Add a learned zero-valued choice; preserve all existing experience."""
        for wrapper in self.memories.values():
            if wrapper.current is not None or wrapper.memory._read_credit_active:
                raise RuntimeError('finish the observation before enabling null read')
        for wrapper in self.memories.values():
            wrapper.enable_null_read()

    def set_recall_placement(self, placement):
        """Explicit candidate change; old checkpoints retain input placement."""
        if placement not in ('input','output'):
            raise ValueError('unknown recall placement')
        for wrapper in self.memories.values():
            if wrapper.current is not None or wrapper.memory._read_credit_active:
                raise RuntimeError('finish the live observation before changing placement')
        for wrapper in self.memories.values():
            wrapper.recall_placement = placement

    def set_read_interval(self, interval):
        """Explicitly migrate cadence without changing slots, weights or clocks.

        1 queries every position; N holds each read until the next N-token
        refresh; None preserves the historical single-read-per-observation path.
        Old checkpoints restore None, irrespective of constructor defaults.
        """
        LayerExperienceRecall.validate_read_interval(interval)
        for wrapper in self.memories.values():
            if wrapper.current is not None or wrapper.memory._read_credit_active:
                raise RuntimeError('finish the live observation before changing read cadence')
        for wrapper in self.memories.values():
            wrapper.read_interval = interval

    def enable_credit_retention(self):
        """Explicitly migrate a restored individual without resetting its life.

        Loading old checkpoints first preserves their recorded policy. This
        upgrade consumes the existing read moments and changes future eviction
        only; no parameters, moments, slots, cache, clocks or RNG are reset.
        """
        for wrapper in self.memories.values():
            if wrapper.current is not None or wrapper.memory._read_credit_active:
                raise RuntimeError('finish the current observation before policy migration')
            if (wrapper.memory.consolidation_enabled and
                    wrapper.memory.consolidation_rule != 'retention'):
                raise ValueError('joint read-policy checkpoint needs an explicit read-policy decision')
        for wrapper in self.memories.values():
            wrapper.memory.consolidation_enabled = True
            wrapper.memory.consolidation_rule = 'retention'

    def enable_utility_retention(self):
        """Explicit candidate migration: retain by observed removal benefit.

        Existing read parameters and moments continue to control read dose.
        New finite utility evidence starts neutral, not relabelled gradients.
        """
        for wrapper in self.memories.values():
            if wrapper.current is not None or wrapper.memory._read_credit_active:
                raise RuntimeError('finish live credit before retention migration')
            if not wrapper.null_read_enabled:
                raise ValueError('enable the existing null choice before utility retention')
        for wrapper in self.memories.values():
            wrapper.memory.consolidation_enabled = True
            wrapper.memory.consolidation_rule = 'utility'

    def _regularizer(self):
        values = []
        for wrapper in self.memories.values():
            m = wrapper.memory
            current = wrapper.encoder(wrapper.current['source'][0].float().cpu())
            history = wrapper.encoder(m.history)
            values.append(live_sigreg(current, history, m.history_valid, self.directions))
        return torch.stack(values).mean()

    @torch.no_grad()
    def _removal_objective(self, x, target, cache, identity, loss_mask, read_position):
        with self.observation(suppress_record=identity), ExitStack() as stack:
            for layer, wrapper in self.memories.items():
                stack.enter_context(wrapper.attach(self.model,layer=layer,read_position=read_position))
            output = self.model(x,past_key_values=clone_cache(cache),use_cache=True)
            _, ce = language_loss(output.logits,target,loss_mask)
            reg = self._regularizer()
            return float(10*ce+.001*reg.to(ce.device))

    @contextmanager
    def observation(self, *, suppress_older_than=None, suppress_record=None):
        """Diagnostics cannot overwrite a training window's receipts/graphs."""
        names = ('current', 'replayed', 'replayed_by_slot', 'last_read', 'causal_receipt',
                 'suppress_older_than', 'suppressed_record', 'query_clock_offsets')
        saved = {i: [getattr(m, name) for name in names] for i, m in self.memories.items()}
        try:
            for layer,m in self.memories.items():
                m.current, m.replayed, m.last_read = None, [], None
                m.replayed_by_slot = {}
                m.suppress_older_than = suppress_older_than
                m.suppressed_record = (suppress_record[1:] if suppress_record is not None
                                       and suppress_record[0] == layer else None)
            yield
        finally:
            for i, m in self.memories.items():
                for name, value in zip(names, saved[i]):
                    setattr(m, name, value)

    @torch.no_grad()
    def predict(self, x, cache=None, *, suppress_older_than=None, read_position=0,
                clock_offset=0, on_prediction=None):
        with self.observation(suppress_older_than=suppress_older_than), ExitStack() as stack:
            for layer, memory in self.memories.items():
                stack.enter_context(memory.attach(self.model, layer=layer, read_position=read_position,
                                                  capture=False, clock_offset=clock_offset))
            output = self.model(x, past_key_values=clone_cache(self.cache if cache is None else cache),
                                use_cache=True)
            if on_prediction is not None:
                on_prediction()
            return output.logits, output.past_key_values

    @torch.no_grad()
    def evaluate(self, ids, *, window=128):
        """Frozen sidecar from the complete live state; no write/update/reset."""
        cache = clone_cache(self.cache)
        total, count = 0., 0
        for start in range(0, len(ids)-window, window):
            x = ids[start:start+window][None].to(next(self.model.parameters()).device)
            target = ids[start+1:start+window+1][None].to(x.device)
            logits, cache = self.predict(x, cache)
            total += float(F.cross_entropy(logits.float().flatten(0, 1), target.flatten(), reduction='sum'))
            count += target.numel()
        if not count:
            raise ValueError('empty sidecar')
        return dict(nll_sum=total, tokens=count, nats_per_token=total/count)

    @torch.no_grad()
    def observe(self, x, *, return_logits=False, read_position=0,
                cache_override=None, preserve_live_cache=False):
        """Consume prompt text and write observed episodes, without a weight update."""
        with ExitStack() as stack:
            for layer, memory in self.memories.items():
                stack.enter_context(memory.attach(self.model, layer=layer, read_position=read_position))
            cache = self.cache if cache_override is None else clone_cache(cache_override)
            output = self.model(x, past_key_values=cache, use_cache=True)
        if not preserve_live_cache:
            self.cache = clone_cache(output.past_key_values)
        slots = {layer: memory.commit() for layer, memory in self.memories.items()}
        self.tokens += x.numel()
        return (slots, output.logits) if return_logits else slots

    def step(self, x, target, *, diagnose=False, revisit_mask=None, loss_mask=None,
             read_position=0, diagnose_retention=False, cache_override=None,
             preserve_live_cache=False):
        self.optimizer.zero_grad(set_to_none=True)
        for _, param in self.native:
            param.grad = None
        for memory in self.memories.values():
            memory.begin(self.optimizer.param_groups[1], query_limit=x.shape[1])
        utility_enabled = any(w.memory.consolidation_enabled and w.memory.consolidation_rule == 'utility'
                              for w in self.memories.values())
        measure_utility = utility_enabled or diagnose_retention
        execution_cache = self.cache if cache_override is None else clone_cache(cache_override)
        before = clone_cache(execution_cache) if diagnose or measure_utility else None
        prior_retention = {}
        if measure_utility:
            from scripts.ib_local.experience_consolidation import bounded_preference
            prior_retention = {layer:bounded_preference(w.memory).clone()
                               for layer,w in self.memories.items()}
        with ExitStack() as stack:
            for layer, memory in self.memories.items():
                stack.enter_context(memory.attach(self.model, layer=layer, read_position=read_position))
            output = self.model(x, past_key_values=execution_cache, use_cache=True)
        token_loss, ce = language_loss(output.logits, target, loss_mask)
        reads = {}
        for layer, memory in self.memories.items():
            m, read = memory.memory, memory.last_read
            valid = m.occupied[read.slots]
            positions = memory.current['query_positions']
            position_tensor = torch.tensor(positions,device=read.slots.device)[:,None]
            ages = (m.clock+position_tensor-m.birth[read.slots])[valid]
            weights = read.weights.detach()[valid]
            reads[str(layer)] = dict(ages_tokens=ages.tolist(), weights=weights.tolist(),
                                     slots=read.slots.detach()[valid].tolist(),
                                     query_position=read_position,
                                     query_positions=positions, query_count=len(positions),
                                     read_interval=memory.read_interval,
                                     unique_replays=len(memory.replayed_by_slot))
            if len(positions) > 1:
                reads[str(layer)]['per_query'] = dict(
                    slots=read.slots.detach().tolist(),weights=read.weights.detach().tolist())
            if memory.null_read_enabled:
                reads[str(layer)]['null_weight'] = float((1-read.weights.sum(-1)).detach().mean())
        reg = self._regularizer()
        loss = 10*ce + .001*reg.to(ce.device)
        utility = {}
        if measure_utility:
            factual = float(loss.detach())
            for layer,wrapper in self.memories.items():
                m, read = wrapper.memory, wrapper.last_read
                if not diagnose_retention and not (m.consolidation_enabled and m.consolidation_rule == 'utility'):
                    continue
                utility[layer] = []
                # Bound finite-removal diagnostics independently of cadence.
                # One removed identity is cut at every position where it reads.
                selected = list(dict.fromkeys(read.slots.flatten().tolist()))
                reads[str(layer)]['removal_candidates'] = len(selected)
                for slot in selected[:wrapper.memory.topk]:
                    if not bool(m.occupied[slot]):
                        continue
                    version = int(m.read_credit_version[slot])
                    without = self._removal_objective(x,target,before,(layer,slot,version),loss_mask,read_position)
                    utility[layer].append(dict(slot=slot,version=version,benefit=without-factual))
        diagnostics = {}
        if diagnose:
            with torch.no_grad():
                same, _ = self.predict(x, before, read_position=read_position)
                diagnostics['symmetric_logit_max'] = float((same-output.logits).abs().max())
                if diagnostics['symmetric_logit_max'] != 0:
                    raise AssertionError('same complete start produced different diagnostic predictions')
                cut, _ = self.predict(x, before, suppress_older_than=x.shape[1],
                                      read_position=read_position)
                _, cut_ce = language_loss(cut, target, loss_mask)
                diagnostics['old_read_cut_nll_delta'] = float(cut_ce-ce.detach())
            # Held read is the exact forward control: identical values, but no
            # derivative from old native computation into the original weights.
            from scripts.ib_local.reactivated_credit_audit import audit_reactivated_credit
            diagnostics['reactivated_old_credit'] = {
                str(layer): audit_reactivated_credit(
                    ce, memory, [(n, p) for n, p in self.native
                                 if n.startswith(f'layer{layer}.') and p.requires_grad],
                    min_age=x.shape[1])
                for layer, memory in self.memories.items()
            }
            if not next(iter(self.memories.values())).detach_replay:
                for layer, memory in self.memories.items():
                    responses = memory.replayed
                    if not responses:
                        continue
                    teaching = torch.autograd.grad(ce, responses, retain_graph=True, allow_unused=True)
                    active = [(r, g) for r, g in zip(responses, teaching) if g is not None]
                    if active:
                        selected = [(n, p) for n, p in self.native if n.startswith(f'layer{layer}.')]
                        gradients = torch.autograd.grad([r for r, _ in active], [p for _, p in selected],
                            grad_outputs=[g.detach() for _, g in active], retain_graph=True, allow_unused=True)
                        diagnostics[f'layer{layer}_old_native_grad_norms'] = {
                            n: None if g is None else float(g.float().norm()) for (n, _), g in zip(selected, gradients)}
        loss.backward()
        credit = {str(i): m.memory.settle_read_credit() for i, m in self.memories.items()}
        for layer, memory in self.memories.items():
            if memory.causal_relation is not None:
                credit[str(layer)]['causal_relation'] = memory.settle_causal_relation()
        if measure_utility:
            from scripts.ib_local.experience_consolidation import settle_retention_utility
            for layer,observations in utility.items():
                from scripts.ib_local.experience_consolidation import bounded_preference
                memory = self.memories[layer].memory
                prior = prior_retention[layer]
                apply_utility = memory.consolidation_enabled and memory.consolidation_rule == 'utility'
                settled = (settle_retention_utility(memory,observations) if apply_utility else
                           dict(records=0,stale=0,diagnostic_only=True))
                after = bounded_preference(memory)
                for item in observations:
                    item.update(preference_before=float(prior[item['slot']]),
                                preference_after=float(after[item['slot']]))
                credit[str(layer)]['retention_utility'] = dict(
                    settled,
                    observations=observations)
        for layer,memory in self.memories.items():
            if memory.null_read_enabled:
                gradient = memory.null_bias.grad
                credit[str(layer)]['null_bias_gradient'] = None if gradient is None else float(gradient)
                credit[str(layer)]['null_key_gradient_norm'] = (None if memory.null_key.grad is None else
                                                               float(memory.null_key.grad.norm()))
        self.apply_gradients()
        if not preserve_live_cache:
            self.cache = clone_cache(output.past_key_values)
        committed_slots={layer:memory.commit() for layer,memory in self.memories.items()}
        self.updates += 1
        self.tokens += x.numel()
        traffic = {}
        if revisit_mask is not None:
            mask = revisit_mask.flatten().to(token_loss.device)
            traffic = {name: dict(nll_sum=float(token_loss.detach()[valid].sum()), tokens=int(valid.sum()))
                       for name, valid in [('revisit', mask), ('fresh', ~mask)]}
        supervised_tokens = target.numel() if loss_mask is None else int(loss_mask.sum())
        return dict(nll_sum=float(token_loss.detach().sum()), target_tokens=target.numel(), traffic=traffic,
            supervised_tokens=supervised_tokens, supervised_nll_sum=float(ce.detach())*supervised_tokens,
            ce=float(ce.detach()), last_target_nll=float(token_loss[-1].detach()), sigreg=float(reg.detach()),
            reads=reads, diagnostics=diagnostics, credit=credit,
            updates=self.updates, tokens=self.tokens, recurrence_bytes=cache_nbytes(self.cache),
            experience_cpu_bytes=self.experience_storage_nbytes,committed_slots=committed_slots)

    def apply_gradients(self):
        """The same finite checks, native clipping and Adam for both settlements."""
        for master, (_, param) in zip(self.masters, self.native):
            if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                raise FloatingPointError('nonfinite native gradient')
            master.grad = None if param.grad is None else param.grad.detach().float().clone()
        torch.nn.utils.clip_grad_norm_(self.masters, 1.)
        for param in self.extras+self.null_parameters:
            if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                raise FloatingPointError('nonfinite recall gradient')
        self.optimizer.step()
        with torch.no_grad():
            for master, (_, param) in zip(self.masters, self.native):
                param.copy_(master.to(param.dtype))

    def state_dict(self):
        return dict(schema=1, layers=self.layers, capacity=self.capacity,
            native={n: p.detach().cpu().clone() for n, p in self.native},
            masters=[p.detach().cpu().clone() for p in self.masters],
            memories={i: copy_tree(m.state_dict()) for i, m in self.memories.items()},
            optimizer=copy_tree(self.optimizer.state_dict()), directions=self.directions.clone(),
            cache=dict(states=copy_tree(self.cache.states), seen_tokens=self.cache._seen_tokens),
            updates=self.updates, tokens=self.tokens,joint_capture_enabled=self.joint_capture_enabled,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())

    def load_state_dict(self, state, *, detach_replay=None):
        if state['schema'] != 1 or tuple(state['layers']) != self.layers or state['capacity'] != self.capacity:
            raise ValueError('incompatible complete individual')
        with torch.no_grad():
            for name, param in self.native:
                param.copy_(state['native'][name].to(param.device))
            for master, stored in zip(self.masters, state['masters']):
                master.copy_(stored.to(master.device))
        for layer, memory in self.memories.items():
            memory.load_state_dict(state['memories'][layer])
        # CPU optimizer tensors can otherwise alias a reusable source snapshot:
        # Adam's in-place moment updates would silently alter the next branch.
        self.optimizer.load_state_dict(with_null_optimizer_group(
            state['optimizer'],self.optimizer.state_dict(),len(self.null_parameters)))
        self.directions.copy_(state['directions'])
        self.cache = new_recurrent_cache(state['cache']['seen_tokens'])
        self.cache.states = copy_tree(state['cache']['states'], next(self.model.parameters()).device)
        self.updates, self.tokens = state['updates'], state['tokens']
        self.joint_capture_enabled=state.get('joint_capture_enabled',False)
        torch.set_rng_state(state['rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
        if detach_replay is not None:
            self.set_replay_gradient(detach_replay)
