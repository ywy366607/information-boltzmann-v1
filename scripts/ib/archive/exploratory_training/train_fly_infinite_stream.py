"""Persistent real-OWT learning with three-factor sensory and recurrent updates.

Validation is fresh active experience on the same individual. The actual online
learner, optimizer cadence and physical/eligibility state continue across A/B/A.
"""
from __future__ import annotations
import argparse
import json
import shutil
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_online_learning import FlyOnlineLearner
from information_boltzmann.runtime.lifelong_evaluation import recovery_summary,savings_summary
from information_boltzmann.runtime.forgetting_curve import FirstRevisitCurve


def atomic_json(path, data):
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data,allow_nan=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def atomic_save(path, data):
    tmp=path.with_suffix('.pt.tmp')
    try:
        torch.save(data,tmp)
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


def plastic_model_state(model):
    names={name for name,_ in model.named_parameters()}
    names.update(('edge_weight_e','edge_weight_i'))
    state = {name:value for name,value in model.state_dict().items() if name in names}
    state.update(edge_weight_e=model.edge_weight_e,edge_weight_i=model.edge_weight_i)
    return state


def count_unigrams(data, vocab):
    counts=np.ones(vocab,dtype=np.float64)
    for left in range(0,len(data),1000000):
        counts+=np.bincount(np.asarray(data[left:left+1000000],dtype=np.int64),minlength=vocab)
    return np.log(counts/counts.sum()).astype(np.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--graph',type=Path,default=Path('data/malecns_v1/fly_reservoir_coba.npz'))
    parser.add_argument('--output',type=Path,default=Path('results/q8_fly_infinite_stream_three_factor_100k'))
    parser.add_argument('--resume',type=Path,default=None,help='Complete continuing-individual checkpoint')
    parser.add_argument('--initialize-from',type=Path,default=None,help='Weights only; declared new individual')
    parser.add_argument('--pretrained-embedding',type=Path,default=Path('data/gpt2_model.safetensors'))
    parser.add_argument('--total-tokens',type=int,default=100000)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--lr-synapse',type=float,default=1e-5)
    parser.add_argument('--lr-sensory',type=float,default=1e-4)
    parser.add_argument('--grad-accum-tokens',type=int,default=4)
    parser.add_argument('--synapse-update-interval',type=int,default=2)
    parser.add_argument('--d-model',type=int,default=768)
    parser.add_argument('--read-norm-init',type=float,default=.1)
    parser.add_argument('--dopamine-k-on',type=float,default=.2)
    parser.add_argument('--dopamine-k-off',type=float,default=.05)
    parser.add_argument('--dopamine-q0',type=float,default=.05)
    parser.add_argument('--validate-every-tokens',type=int,default=5000)
    parser.add_argument('--log-every-tokens',type=int,default=100)
    parser.add_argument('--eval-shock-tokens',type=int,default=150)
    parser.add_argument('--eval-ebb-a',type=int,default=40)
    parser.add_argument('--eval-ebb-b',type=int,default=80)
    parser.add_argument('--eval-ftle-steps',type=int,default=50,help='Full physical-state conditional finite-difference steps on fresh B')
    parser.add_argument('--forgetting-lags',default='64,256,1024,4096,8192',help='Measured intervening-event intervals; not dynamical parameters')
    parser.add_argument('--forgetting-episode-tokens',type=int,default=128)
    parser.add_argument('--forgetting-cohort-every-tokens',type=int,default=5000)
    parser.add_argument('--initial-checkpoint-tokens',type=int,default=128)
    parser.add_argument('--decoder-bias',dest='decoder_bias',action='store_true',default=True)
    parser.add_argument('--no-decoder-bias',dest='decoder_bias',action='store_false')
    for component in ('decoder','synapses','sensory'):
        parser.add_argument('--unfreeze-'+component,dest='train_'+component,action='store_true',default=True)
        parser.add_argument('--freeze-'+component,dest='train_'+component,action='store_false')
    parser.add_argument('--checkpoint',action='store_true',default=True)
    parser.add_argument('--no-checkpoint',dest='checkpoint',action='store_false',help='Numerical/runtime calibration only')
    parser.add_argument('--vram-limit-mib',type=float,default=3900)
    args=parser.parse_args()
    if not torch.cuda.is_available(): parser.error('CUDA required')
    if min(args.total_tokens,args.log_every_tokens,args.validate_every_tokens,args.eval_shock_tokens,args.eval_ebb_a,args.eval_ebb_b)<1:
        parser.error('Token budgets and reporting intervals must be positive')
    if args.resume and args.initialize_from: parser.error('Choose resume or initialize-from')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(11)
    train=np.load(args.data/'train.npy',mmap_mode='r')
    val=np.load(args.data/'validation.npy',mmap_mode='r')
    if args.total_tokens+1>len(train): parser.error('Fresh train stream exhausted')
    print('Loading connectome and COBA/ALIF/STP model...',flush=True)
    m=FlyReservoirLM(args.graph,vocab_size=50257,d_model=args.d_model,injection='topographic',
                     read_surface='output',synapse_model='coba',use_alif=True,use_stp=True,
                     decoder_bias=args.decoder_bias).cuda()
    cursor=val_cursor=0
    best=math.inf
    init='GPT2 embedding/decoder and train-only unigram prior; new individual'
    saved=None
    with torch.no_grad():
        if args.resume or args.initialize_from:
            saved=torch.load(args.resume or args.initialize_from,map_location='cpu',weights_only=False)
            if args.resume and saved.get('format')!='fly-online-v2':
                raise ValueError('Resume needs complete lifecycle state; use initialize-from for weights-only assets')
            weights=dict(saved['model'])
            for name in ('edge_weight_e','edge_weight_i'):
                if name in weights: getattr(m,name).copy_(weights.pop(name))
            incompatible=m.load_state_dict(weights,strict=False)
            if incompatible.unexpected_keys: raise ValueError(incompatible.unexpected_keys)
            init='complete lifecycle resume' if args.resume else 'weights-only initialization of new individual'
        elif args.pretrained_embedding.exists() and args.d_model==768:
            from safetensors.torch import load_file
            pretrained=load_file(str(args.pretrained_embedding))
            m.embedding.weight.copy_(pretrained['wte.weight'])
            m.decoder.weight.copy_(pretrained['wte.weight'])
            del pretrained
            if m.decoder.bias is not None:
                prior=count_unigrams(train,50257)
                m.decoder.bias.copy_(torch.from_numpy(prior))
            m.read_norm.weight.fill_(args.read_norm_init)
        else:
            init='random weights; new individual'
    learner=FlyOnlineLearner(m,lr=args.lr,lr_synapse=args.lr_synapse,lr_sensory=args.lr_sensory,
         grad_accum_tokens=args.grad_accum_tokens,synapse_update_interval=args.synapse_update_interval,
         train_decoder=args.train_decoder,train_synapses=args.train_synapses,train_sensory=args.train_sensory,
         dopamine_k_on=args.dopamine_k_on,dopamine_k_off=args.dopamine_k_off,dopamine_q0=args.dopamine_q0)
    if args.resume:
        learner.load_state_dict(saved['learner'])
        cursor=saved['train_cursor']; val_cursor=saved['val_cursor']; best=saved['best_live_nll']
        torch.set_rng_state(saved['rng_cpu']); torch.cuda.set_rng_state(saved['rng_cuda'])
    forgetting=FirstRevisitCurve([int(x) for x in args.forgetting_lags.split(',')],
        episode_tokens=args.forgetting_episode_tokens,cohort_every=args.forgetting_cohort_every_tokens,train_cursor=cursor)
    if saved is not None and 'forgetting_measurement' in saved:
        forgetting.load_state_dict(saved['forgetting_measurement'])
        measurements=args.output/'first_revisit_episodes.jsonl'
        if measurements.exists():
            with measurements.open(encoding='utf-8') as source:
                forgetting.rebuild_summary((json.loads(line) for line in source),through_event=learner.events)
            atomic_json(args.output/'forgetting_curve.json',forgetting.curve())
    lifecycle_measurement=saved.get('lifecycle_measurement') if saved is not None else None
    del saved
    if 'weights' in locals(): del weights
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(initialization=init,protocol='one continuing active learner; prequential score before update; A/B/A with scored bridges',
                  recurrent_credit='factorized e-prop + low-rank DFA approximation',
                  sensory_credit='conditional local forward sensitivities incl. baseline/reset/ALIF and gates',
                  trained_physical_parameters=list(learner.physical_names),
                  fixed_physical_parameters=['logit_u0','log_tau_fac','log_tau_rec'],
                  dopamine_rates='engineering initial values in model-event units, not measured biological constants',
                  n_neurons=m.n_neurons,n_synapses=m.edge_weight_e.numel()+m.edge_weight_i.numel(),
                  parameter_count=sum(p.numel() for p in m.parameters()),
                  train_shape=list(train.shape),validation_shape=list(val.shape),pid=os.getpid())
    config.update(recurrent_credit='destination-conditioned COBA/reset/ALIF eligibility + low-rank DFA direction',
                  weight_updates='compensated float32 updates with persisted rounding residuals',
                  credit_execution='float32 Triton; mask coefficient gathers only for exactly zero histories and input; all pending updates retained',
                  credit_migration=learner.credit_migration,
                  forgetting_protocol='independent real-OWT episodes, one revisit per episode, active pre-target-update scoring')
    atomic_json(args.output/'config.json',config)
    start=time.perf_counter(); initial_cursor=cursor
    print(f'Online OWT learning: {args.total_tokens:,} train predictions; no resets, no BPTT.',flush=True)
    def log(row,name='metrics.jsonl'):
        with (args.output/name).open('a',encoding='utf-8') as out:
            out.write(json.dumps(row,allow_nan=False)+'\n')
    def observe(token):
        if learner.previous_token is None:
            learner.previous_token=int(token)
            return None
        inp=torch.tensor([learner.previous_token],device='cuda')
        target=torch.tensor([int(token)],device='cuda')
        score=learner.step(inp,target)
        learner.previous_token=int(token)
        return score
    def health():
        return {'field_energy':float(learner.h.square().mean()),
                'firing_rate':float((learner.ring[0]>0).float().mean()),
                'dopamine_mean_q':float(learner.dopamine.q.mean()),
                'dopamine_max_q':float(learner.dopamine.q.max()),
                'vram_allocated_mib':torch.cuda.memory_allocated()/2**20,
                'vram_reserved_mib':torch.cuda.memory_reserved()/2**20,
                'vram_peak_mib':torch.cuda.max_memory_allocated()/2**20,
                'read_norm_mean':float(m.read_norm.weight.mean()),
                'elapsed_events':learner.events,'optimizer_updates':learner.updates,
                'synapse_updates':learner.synapse_updates,'pending_gradients':learner.pending,
                'synapse_rounding_residual_rms':float(torch.sqrt(sum(s.rounding_residual.square().sum() for s in learner.edge_credit)/max(sum(s.rounding_residual.numel() for s in learner.edge_credit),1))) if learner.edge_credit else 0.,
                'sensory_rounding_residual_rms':float(torch.sqrt(sum(r.square().sum() for r in learner.projection_residuals)/max(sum(r.numel() for r in learner.projection_residuals),1))) if learner.projection_residuals else 0.,
                'conditional_local_log_gain':float(learner.local_log_gain/max(learner.events,1)),
                'energy_ledger_units':'squared model voltage, not ATP/Joule or Shannon entropy',
                'cumulative_input_work':float(learner.energy_totals[0]),
                'cumulative_recurrent_work':float(learner.energy_totals[1]),
                'cumulative_passive_attenuation':float(learner.energy_totals[2]),
                'cumulative_reset_outflow':float(learner.energy_totals[3]),
                'energy_balance_residual':float(learner.energy_totals[4])}
    def progress(status):
        row=dict(status=status,tokens_streamed=cursor,total_target_tokens=args.total_tokens,
                 ema_stream_loss=learner.ema,best_live_nll=best if math.isfinite(best) else None,
                 speed_tokens_per_sec=(cursor-initial_cursor)/max(time.perf_counter()-start,1e-6),**health())
        atomic_json(args.output/'progress.json',row)
        return row
    def representation_report():
        count=min(learner.events,128)
        if count<2: return {}
        Z=learner.latent_window[:count].detach().cpu()
        if learner.events>128: Z=torch.roll(Z,-learner.events%128,0)
        centered=Z-Z.mean(0)
        sv=torch.linalg.svdvals(centered).square()
        total=sv.sum()
        if total<=0: return {'centered_effective_rank':0.0,'temporal_roughness':0.0}
        p=sv/total
        entropy=-(p[p>0]*p[p>0].log()).sum()
        return {'centered_effective_rank':float(entropy.exp()),
                'centered_participation_ratio':float(total.square()/sv.square().sum()),
                'temporal_roughness':float((Z[1:]-Z[:-1]).square().sum(-1).mean()/2),
                'centered_feature_variance':float(centered.square().sum(-1).mean()),
                'measurement_events':count,'interpretation':'descriptive structure; no noise/criticality certification'}
    def checkpoint(is_best):
        if not args.checkpoint: return
        # Static graph coordinates/indices are reloaded from the declared graph.
        data={'format':'fly-online-v2','model':plastic_model_state(m),'learner':learner.state_dict(),
              'train_cursor':cursor,'val_cursor':val_cursor,'best_live_nll':best,'config':config,
              'forgetting_measurement':forgetting.state_dict(),
              'lifecycle_measurement':{'recent_a':recent_a,'recent_a_scores':recent_a_scores,'window':window},
              'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()}
        required=sum(t.numel()*t.element_size() for t in m.parameters())+sum(t.numel()*t.element_size() for t in (m.edge_weight_e,m.edge_weight_i))
        def tensor_bytes(obj):
            if isinstance(obj,torch.Tensor): return obj.numel()*obj.element_size()
            if isinstance(obj,dict): return sum(tensor_bytes(v) for v in obj.values())
            if isinstance(obj,(list,tuple)): return sum(tensor_bytes(v) for v in obj)
            return 0
        required+=tensor_bytes(data['learner'])
        if shutil.disk_usage(args.output).free<required+64*2**20:
            raise OSError(f'Checkpoint needs {required/2**30:.2f} GiB free for atomic save')
        atomic_save(args.output/'last.pt',data)
        if is_best:
            # Best weights are an inference/branch asset, distinctly not a life resume.
            atomic_save(args.output/'best.pt',{'model':plastic_model_state(m),'config':config,
                         'tokens_streamed':cursor,'best_live_nll':best,'format':'fly-best-weights-v2'})
    if learner.previous_token is None: observe(train[0])
    window=[]; recent_a=[]; recent_a_scores=[]
    if lifecycle_measurement:
        window=lifecycle_measurement['window']; recent_a=lifecycle_measurement['recent_a']; recent_a_scores=lifecycle_measurement['recent_a_scores']
    def first_revisits():
        while True:
            episode=forgetting.pop_due(learner.events)
            if episode is None: return
            event_start,updates_start=learner.events,learner.updates
            scores=[observe(token) for token in episode['tokens']]
            report=forgetting.record_revisit(episode,scores,event_start=event_start,event_end=learner.events,
                updates_start=updates_start,updates_end=learner.updates)
            report['train_tokens']=cursor
            log(report,'first_revisit_episodes.jsonl')
            atomic_json(args.output/'forgetting_curve.json',forgetting.curve())
            print(f"[FIRST REVISIT {episode['episode_id']}] interval={report['actual_intervening_events']} events; initial={report['initial_nll']:.4f}; revisit={report['first_revisit_nll']:.4f}; gap={report['forgetting_nll_gap']:+.4f}",flush=True)
    try:
        while cursor<args.total_tokens:
            # Target cursor+1; after any replay/validation the bridge is scored
            # from the actual last observed token, never an invented train prefix.
            target=int(train[cursor+1])
            score=observe(target); cursor+=1
            forgetting.fresh_observation(target,score,event=learner.events,train_cursor=cursor,updates=learner.updates)
            window.append(score); recent_a.append(target); recent_a_scores.append(score)
            if len(recent_a)>args.eval_ebb_a:
                recent_a.pop(0); recent_a_scores.pop(0)
            first_revisits()
            if cursor%args.log_every_tokens==0:
                row=progress('running'); row['window_prequential_nll']=float(np.mean(window)); window.clear()
                log(row)
                print(f"Token {cursor}/{args.total_tokens} | EMA NLL {learner.ema:.4f} | {row['speed_tokens_per_sec']:.1f} tok/s | peak {row['vram_peak_mib']:.0f} MiB",flush=True)
                if max(row['vram_reserved_mib'],row['vram_peak_mib'])>args.vram_limit_mib:
                    raise MemoryError('Dedicated CUDA allocation budget exceeded')
            if cursor==args.initial_checkpoint_tokens: checkpoint(False)
            if (args.output/'STOP').exists():
                checkpoint(False); progress('paused'); return
            if cursor%args.validate_every_tokens==0 or cursor==args.total_tokens:
                # A1 is freshly observed in the actual continuing stream.
                # Use its recorded scores, then fresh B, then replay A2.
                A=np.asarray(recent_a,dtype=np.int64)
                count=args.eval_shock_tokens+args.eval_ebb_b
                if val_cursor+count>len(val): raise RuntimeError('Fresh validation stream exhausted')
                event_start=learner.events
                tangent=None; gains=[]; epsilon=1e-4
                def physical_state():
                    return [learner.h,*learner.ring,*[learner.syn[k] for k in ('ge','gi','b','x','u')],m.topographic_writer.a_adapt]
                B=[]
                for event,t in enumerate(val[val_cursor:val_cursor+count]):
                    if event<args.eval_ftle_steps:
                        real=physical_state()
                        if tangent is None:
                            tangent=[torch.randn_like(tensor) for tensor in real]
                            norm=torch.sqrt(sum(direction.square().sum() for direction in tangent))
                            tangent=[direction*(epsilon/norm) for direction in tangent]
                        perturbed=[value+direction for value,direction in zip(real,tangent)]
                        old_baseline=m.topographic_writer.a_adapt.clone()
                        with torch.no_grad():
                            m.topographic_writer.a_adapt.copy_(perturbed[-1])
                            shadow=m.step(perturbed[0],torch.tensor([learner.previous_token],device='cuda'),
                                spike_ring=tuple(perturbed[1:5]),**dict(zip(('ge','gi','b','x','u'),perturbed[5:10])),
                                base_rates=learner.rates,thresholds=learner.threshold,
                                conductance_gains=learner.gains,alif_params=learner.alif,stp_params=learner.stp)
                            shadow_state=[shadow[0],*shadow[2],*shadow[3:],m.topographic_writer.a_adapt.clone()]
                            m.topographic_writer.a_adapt.copy_(old_baseline)
                        B.append(observe(t))
                        differences=[s-r for s,r in zip(shadow_state,physical_state())]
                        distance=torch.sqrt(sum(delta.square().sum() for delta in differences)).clamp_min(1e-30)
                        gains.append(float((distance/epsilon).log()))
                        tangent=[delta*(epsilon/distance) for delta in differences]
                    else:
                        B.append(observe(t))
                val_cursor+=count
                A2=[observe(t) for t in A]
                live=float(np.mean(B))
                is_best=live<best
                if is_best: best=live
                report={'train_tokens':cursor,'event_interval':[event_start,learner.events],
                        'fresh_validation_cursor':val_cursor,'live_prequential_nll':live,
                        'B_curve':B,'A1_curve':list(recent_a_scores),'A2_replay_curve':A2,'A_tokens':A.tolist(),
                        'A1_prequential_nll':float(np.mean(recent_a_scores)),
                        'A2_prequential_nll':float(np.mean(A2)),
                        'revisit_nll_change':float(np.mean(recent_a_scores)-np.mean(A2)),
                        'opening_B_nll':float(np.mean(B[:min(32,len(B))])),
                        'late_B_nll':float(np.mean(B[-min(32,len(B)):])),
                        'actual_intervening_events':count,'replay_events':len(A2),
                        'first_pass_events':count,'optimizer_updates':learner.updates,
                        'protocol':'same learner, scored train-B, B-A and A-train bridges',
                        'recovery':recovery_summary(B,block_tokens=16,hold_blocks=2),
                        'savings':savings_summary(recent_a_scores,A2,intervening_events=count,block_tokens=16,hold_blocks=2),
                        'representation':representation_report(),
                        'conditional_full_state_ftle':float(np.mean(gains)) if gains else None,
                        'ftle_steps':len(gains),'ftle_perturbation':epsilon,
                        'ftle_scope':'full physical state incl. writer adaptation; conditioned on inputs and current learned parameters; units per model event'}
                log(report,'lifelong_evaluation.jsonl'); log(report)
                print(f'[ACTIVE EVAL {cursor}] fresh B NLL={live:.4f}; replay A NLL={np.mean(A2):.4f}',flush=True)
                checkpoint(is_best); progress('running')
        atomic_json(args.output/'forgetting_curve.json',forgetting.curve())
        progress('completed')
    except BaseException as exc:
        row=progress('failed'); row['error']=repr(exc)
        atomic_json(args.output/'progress.json',row)
        raise

if __name__=='__main__': main()
