"""Frozen CBIM scattering intervention on held-out OpenWebText BPE tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.ib_local.cbim_field import CBIMFieldModel


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--validation-tokens', type=int, default=4096)
    p.add_argument('--history', type=int, default=256)
    p.add_argument('--horizon', type=int, default=128)
    p.add_argument('--sites', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    model = CBIMFieldModel().cuda().eval()
    model.load_state_dict(saved['model'])
    tokens = saved['config']['tokens']
    if a.validation_tokens % tokens:
        p.error('validation tokens must be divisible by checkpoint token window')
    data = np.load(a.data/'validation.npy', mmap_mode='r')

    def window(start, length, disabled):
        h = torch.zeros(1, model.L, model.d, device='cuda')
        weighted, count = 0., 0
        for offset in range(0, length, tokens):
            ids = torch.as_tensor(np.array(data[start+offset:start+offset+tokens]), dtype=torch.long, device='cuda')[None]
            targets = torch.as_tensor(np.array(data[start+offset+1:start+offset+tokens+1]), dtype=torch.long, device='cuda')[None]
            loss, h, _ = model(ids, targets, h, disable_scattering=disabled)
            weighted += float(loss) * tokens
            count += tokens
        return weighted/count

    # This exactly reproduces training validation: first window burn-in then
    # score 4096 held-out tokens with a separately initialized state.
    def full(disabled):
        h = torch.zeros(1, model.L, model.d, device='cuda')
        total = 0.
        for offset in range(0, a.validation_tokens + tokens, tokens):
            ids = torch.as_tensor(np.array(data[offset:offset+tokens]), dtype=torch.long, device='cuda')[None]
            targets = torch.as_tensor(np.array(data[offset+1:offset+tokens+1]), dtype=torch.long, device='cuda')[None]
            loss, h, _ = model(ids, targets, h, disable_scattering=disabled)
            if offset:
                total += float(loss)*tokens
        return total/a.validation_tokens

    full_keep, full_off = full(False), full(True)
    sites = []
    for site in range(a.sites):
        start = 8192 + site * 4096
        keep = window(start, a.history+a.horizon, False)
        off = window(start, a.history+a.horizon, True)
        # Remove the common history contribution by score-only replay.
        # h is deterministic, so create it once then score the final horizon.
        def scored(disabled):
            h = torch.zeros(1, model.L, model.d, device='cuda')
            for off0 in range(0, a.history, tokens):
                ids = torch.as_tensor(np.array(data[start+off0:start+off0+tokens]),dtype=torch.long,device='cuda')[None]
                targets = torch.as_tensor(np.array(data[start+off0+1:start+off0+tokens+1]),dtype=torch.long,device='cuda')[None]
                _,h,_=model(ids,targets,h,disable_scattering=disabled)
            total=0.
            for off0 in range(a.history,a.history+a.horizon,tokens):
                ids=torch.as_tensor(np.array(data[start+off0:start+off0+tokens]),dtype=torch.long,device='cuda')[None]
                targets=torch.as_tensor(np.array(data[start+off0+1:start+off0+tokens+1]),dtype=torch.long,device='cuda')[None]
                loss,h,_=model(ids,targets,h,disable_scattering=disabled);total+=float(loss)*tokens
            return total/a.horizon
        keep, off = scored(False), scored(True)
        sites.append({'site':site,'start':start,'keep_nll':keep,'no_scattering_nll':off,'delta_nll':off-keep})
    report = {'checkpoint_step':saved['step'],'events':saved['events'],
        'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'protocol':'Frozen BBest. Shared held-out GPT-2 BPE validation. no_scattering bypasses only ConservativeScattering; writer, Cayley transport and state-only readout remain active.',
        'full_validation_tokens':a.validation_tokens,
        'full_validation':{'keep_nll':full_keep,'no_scattering_nll':full_off,'delta_nll':full_off-full_keep},
        'sites':sites,
        'site_mean_delta_nll':float(np.mean([s['delta_nll'] for s in sites]))}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
