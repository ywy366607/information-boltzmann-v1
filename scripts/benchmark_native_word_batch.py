"""Actual frozen body: word execution equivalence, timing and response split.

No optimizer updates or world calls. Read-only comparison on real stored words
and a real past labelled event, all in the current body's SAME coordinates.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args(); a.work.mkdir(exist_ok=False, parents=True)
    import numpy as np
    import torch
    from dataclasses import replace
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.known_utterance_runtime import known_film
    from scripts.ib_local.native_word_batch import endpoints
    from scripts.ib_local.table_branch import table_branch
    from scripts.ib_local.preview_runtime import preview_scope
    from scripts.ib_local import epistemic_value_reference as ev, finite_use_value as finite
    from scripts.ib_local import joint_response
    from scripts.ib_local.causal_relation_program import mechanism, Port, log_score
    from scripts.ib_local.relation_evidence import mechanism_evidence
    from scripts.ib_local.shared_unknown_bindings import _read_evidence
    from scripts.ib_local.finite_use_bounds import payment_ceiling
    torch.set_num_threads(1)
    source_hash = sha256_file(a.checkpoint)
    opts = continuation_parser().parse_args(['--checkpoint', str(a.checkpoint), '--output', str(a.work/'body'),
        '--cycles','1','--world-source','text','--autonomous-events','--optimizer-policy','budgeted',
        '--table-storage',str(a.work/'table'),'--causal-storage',str(a.work/'causal')])
    loaded = prepare_continuation(opts)
    e = loaded.engine; model = e.model
    try:
        ev.configure(e, schema=5, word_execution='native_anchor')
        m = model.memory
        slots = torch.nonzero(m.occupied & (m.word_count > 0)).flatten()
        slots = slots[torch.argsort(m.last_write[slots], descending=True)]
        words = []
        for slot in slots.tolist():
            w = bytes(m.words[slot, :int(m.word_count[slot])].cpu().tolist())
            if w not in words:
                words.append(w)
            if len(words) == 4:
                break
        assert len(words) >= 2, 'require multiple actual remembered word programs'
        cpu = torch.get_rng_state().clone(); gpu = torch.cuda.get_rng_state().clone()
        recurrent = e.recurrent.clone(); decision = e.decision.state.clone()
        sql = model._causal_archive.db.total_changes
        versions = tuple(p._version for p in model.parameters())
        context = e.internal_pending.get('question_context')
        old_context = getattr(model, '_question_context', None)
        model._question_context = context
        old_receipt = getattr(e, '_last_expression_receipt', None)
        old_cache = getattr(e, '_utterance_payload_cache', None)
        had_cache = hasattr(e, '_utterance_payload_cache')
        def serial():
            out = {}
            for w in words:
                with table_branch(e.table) as branch:
                    known_film(e, w, branch)
                out[w] = e._draft_actions.pop()[None].detach().clone()
            e.decision.state.copy_(decision)
            return out
        def batch():
            return endpoints(e, words)
        timing = {'serial':[], 'batch':[]}; errors=[]
        with torch.no_grad():
            # Warm both real kernels before the paired alternating timing.
            with preview_scope(model): reference, actual = serial(), batch()
            for w in words:
                torch.testing.assert_close(actual[w], reference[w], atol=3e-5, rtol=3e-5)
            for name in ('serial','batch','batch','serial'):
                with preview_scope(model):
                    torch.cuda.synchronize(); started=time.perf_counter()
                    out = serial() if name=='serial' else batch()
                    torch.cuda.synchronize(); timing[name].append(time.perf_counter()-started)
                for w in words:
                    errors.append(float((out[w]-reference[w]).abs().max()))
                    torch.testing.assert_close(out[w], reference[w], atol=3e-5, rtol=3e-5)
        model._question_context = old_context
        e._last_expression_receipt = old_receipt
        if had_cache:e._utterance_payload_cache=old_cache
        elif hasattr(e,'_utterance_payload_cache'):del e._utterance_payload_cache
        word_checks = dict(cpu_rng=torch.equal(cpu,torch.get_rng_state()),
            gpu_rng=torch.equal(gpu,torch.cuda.get_rng_state()),
            recurrent=torch.equal(recurrent,e.recurrent),decision=torch.equal(decision,e.decision.state),
            no_sql=model._causal_archive.db.total_changes==sql,
            parameter_versions=versions==tuple(p._version for p in model.parameters()))
        assert all(word_checks.values()), word_checks
        report=dict(valid=True, cycle=loaded.cycle, source_sha256=source_hash,
            word_lengths=list(map(len,words)), experience_layer=getattr(model,'experience_layer',model.body.inject_layer),
            layers=len(model.base.layers), seconds=timing,
            median_seconds={k:statistics.median(v) for k,v in timing.items()},
            max_endpoint_error=max(errors), checks=word_checks,
            cuda_peak_allocated=torch.cuda.max_memory_allocated(), updates=0, world_calls=0)
        # Separate forward sampling from predictive training on an ACTUAL
        # archived event. Remove its own outcome from the fitted support.
        identity = e.internal_pending['causal_program']['episode_identity']
        fact = _read_evidence(model._causal_archive,int(identity.split(':')[1]))[0]
        st = e.internal_pending['causal_program']['state']
        st = replace(st, constraints=tuple(f for f in st.constraints if f.fact_id!=identity),
            extra=tuple(f for f in st.extra if f.fact_id!=identity), organization=None)
        facts = tuple(f for f in mechanism_evidence(model,st,Port.MOTHER) if f.fact_id!=identity)
        assert facts and all(f.fact_id!=identity for f in facts)
        with torch.no_grad():
            d = mechanism(model,st,fact.source,fact.action,kind=Port.MOTHER,
                          exclude_fact=identity,ridge=e.relation_ridge)
            target = model.encoder(fact.outcome)
            before = finite.score_scope(e,st,facts)
            values = {}
            for schema in (5,6):
                ev.configure(e,schema=schema,word_execution='native_anchor')
                with preview_scope(model):
                    values[str(schema)] = float(finite.expected(e,st,fact.source,fact.action,d,facts)[0])
            encoded = ev.from_body(e,st,facts)
            actual = ev.encode_fact(model,fact)
            revised = ev.revise(model,encoded,actual,ridge=e.relation_ridge)
            actual_use = float((ev.batched_scores(model,revised,ridge=e.relation_ridge)-before).mean())
            ceiling = float(payment_ceiling(before,model.relation_basis.noise(mother=True),len(st.valid)+1))
        dmean = d['mean'].detach().requires_grad_()
        frozen = dict(d,mean=dmean)
        old = log_score(frozen,target).mean()
        new = log_score(joint_response.distribution(frozen),target).mean()
        go = torch.autograd.grad(-old,dmean,retain_graph=True)[0]
        gn = torch.autograd.grad(-new,dmean)[0]
        report['same_boundary_response'] = dict(event=identity, evidence=[f.fact_id for f in facts],
            target_excluded=True, forward_expected_use=values, actual_same_U_use=actual_use,
            fixed_prior_payment_upper_bound=ceiling,
            predictive_score=dict(marginal=float(old.detach()),joint=float(new.detach())),
            mean_gradient=dict(marginal_norm=float(go.norm()),joint_norm=float(gn.norm()),
                               cosine=float(torch.nn.functional.cosine_similarity(go,gn)[0])),
            interpretation='same actual event/current encoder/weights/U; scores label different predictive laws; no parameter update')
        report['source_unchanged'] = sha256_file(a.checkpoint)==source_hash
        assert report['source_unchanged']
        (a.work/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:
        model._causal_archive.close()


if __name__=='__main__':main()
