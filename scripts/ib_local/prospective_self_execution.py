"""Native intention and known words, re-executed at the current physical anchor.

The world forecast is a pre-question boundary. This operator jointly predicts
the post-question decision source and the spoken action, then translates BOTH
from the same current anchor. At the actual anchor it is the original executor.
"""
import torch


def typed_filler(model, source, proposal):
    from scripts.ib_local.causal_relation_program import Port
    z=model.encoder(source)
    return (proposal['target']-z if proposal['role']==Port.MOTHER else
            model.encoder(proposal['action'])-z if proposal['role']==Port.ACTION else
            model.encoder(proposal['source'])-z)


@torch.no_grad()
def question_context(engine, source, proposal):
    from scripts.ib_local.navigation_relation_wood import local_relation
    from scripts.ib_local.navigation_mother_boundary import mother_question_context
    from scripts.ib_local.navigation_persistent_question import retained_receipt,relation_from_receipt
    from scripts.ib_local.preview_runtime import read, tensor_key
    model=engine.model
    def _base():
        context,_=local_relation(model,source,source,ridge=engine.relation_ridge,scale=engine.relation_scale)
        context,_=mother_question_context(engine,source,context)
        packet=retained_receipt(model,source)
        kind='mother' if getattr(engine,'mother_shadow_enabled',False) else 'decision'
        relation,_=relation_from_receipt(engine,source,source,packet,outcome_kind=kind)
        return context+model.question_reactivation(relation)
    base = read(model, 'pse_question_context_base', tensor_key(source), _base, anchors=(source,))
    return base+model.question_reactivation(typed_filler(model,source,proposal))


@torch.no_grad()
def execute(engine, future_source, proposal, slots):
    from scripts.ib_local.known_utterance_runtime import known_film
    from scripts.ib_local.table_branch import table_branch
    from scripts.ib_local.preview_runtime import read,tensor_key
    model=engine.model;m=model.memory
    context=question_context(engine,future_source,proposal)[0]
    old_context=getattr(model,'_question_context',None)
    old_state=engine.decision.state.detach().clone()
    old_receipt=getattr(engine,'_last_expression_receipt',None)
    has_cache=hasattr(engine,'_utterance_payload_cache')
    old_cache=getattr(engine,'_utterance_payload_cache',None)
    try:
        model._question_context=None
        anchor_read_key = (tensor_key(engine.recurrent), int(m.clock), int(engine.table.history), engine.pending)
        anchor = read(model, 'pse_base_anchor', anchor_read_key,
                      lambda: engine._read_source().detach().clone(), anchors=(engine.recurrent,))
        model._question_context=context
        post_anchor=engine._read_source().detach().clone()
        post=future_source+(post_anchor-anchor)
        reuse_slot = None
        if not getattr(model, '_expression_speech_only', False):
            current=m.read_encoded(model.encoder(post),model.encoder)
            valid=(m.word_count[current.slots[0]]>0)&(m.progress[current.slots[0]]>0)
            valid &= engine.reward_label_valid()[current.slots[0]]&m.occupied[current.slots[0]]
            weights=current.weights[0]*valid
            reuse_slot=int(current.slots[0,weights.argmax()]) if bool(weights.sum()>0) else None
        required=list(dict.fromkeys([*slots,*([] if reuse_slot is None else [reuse_slot])]))
        endpoints={}
        engine._utterance_payload_cache={}
        # Cache only the same physical/parameter/question anchor, never the
        # live self-hearing cache, so shadow words cannot become real receipts.
        anchor_key=(tuple(tensor_key(p) for p in model.parameters()),
                    tuple(anchor.flatten().tolist()),tuple(context.flatten().tolist()),int(m.clock),engine.pending)
        if getattr(engine, 'native_word_runtime', 'serial') == 'batch':
            from scripts.ib_local.native_word_batch import endpoints as word_endpoints
            from scripts.ib_local.preview_runtime import read_many
            payloads = [bytes(m.words[slot, :int(m.word_count[slot])].cpu().tolist()) for slot in required]
            keys = [(anchor_key, words) for words in payloads]
            def run_many(missing):
                values = word_endpoints(engine, [key[1] for key in missing], batch_size=8)
                return [values[key[1]] for key in missing]
            values = read_many(model, 'self_word_execution', keys, run_many, anchors=(anchor, context))
            for slot, endpoint in zip(required, values):
                endpoints[slot] = future_source+(endpoint-anchor)
        else:
            for slot in required:
                words=bytes(m.words[slot,:int(m.word_count[slot])].cpu().tolist())
                def run(words=words):
                    with table_branch(engine.table) as branch:known_film(engine,words,branch)
                    return engine._draft_actions.pop()[None].detach().clone()
                endpoint=read(model,'self_word_execution',(anchor_key,words),run,anchors=(anchor,context))
                endpoints[slot]=future_source+(endpoint-anchor)
        actions=torch.cat([endpoints[s] for s in slots]) if slots else future_source[:0]
        reuse=None if reuse_slot is None else endpoints[reuse_slot]
        return post,actions,reuse
    finally:
        model._question_context=old_context;engine.decision.state.copy_(old_state)
        engine._last_expression_receipt=old_receipt
        if has_cache:engine._utterance_payload_cache=old_cache
        else:del engine._utterance_payload_cache
