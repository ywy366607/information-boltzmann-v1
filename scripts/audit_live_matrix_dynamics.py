"""Read selected saved S/P storage without importing Torch or loading a body.

The passive pickle reader accepts only known data constructors. Matrix force
diagnostics use the production block equations on real saved states. This
does not execute a model, contact a world, or measure behavioral benefit.
"""
from __future__ import annotations

import argparse
import builtins
import codecs
import collections
import gzip
import importlib
import io
import json
import math
from pathlib import Path
import pickle
import zipfile

import numpy as np


class Passive:
    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class Storage:
    def __init__(self, key, dtype, count):
        self.key, self.dtype, self.count = key, dtype, count


class View:
    def __init__(self, storage, offset, shape, stride, dtype=None):
        self.storage, self.offset = storage, offset
        self.shape, self.stride = tuple(shape), tuple(stride)
        self.dtype = np.dtype(dtype if dtype is not None else storage.dtype)


def rebuild(storage, offset, shape, stride, *args):
    return View(storage, offset, shape, stride)


def rebuild_v3(storage, offset, shape, stride, requires_grad, hooks, dtype, *args):
    return View(storage, offset, shape, stride, dtype)


class DataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch' and name.endswith('Storage'):
            kinds = dict(Float='f4', Double='f8', Long='i8', Int='i4',
                         Byte='u1', Bool='?', Half='f2')
            return np.dtype(kinds[name[:-7]])
        if module == 'torch' and name in ('uint16', 'uint64'):
            return np.dtype(name)
        if (module, name) == ('torch.storage', 'UntypedStorage'):
            return np.dtype('u1')
        if module == 'torch._utils':
            return {'_rebuild_tensor_v2': rebuild, '_rebuild_tensor_v3': rebuild_v3}[name]
        if (module, name) == ('torch', 'device') or module == 'torch.return_types':
            return Passive
        if module.startswith('scripts.ib_local.'):
            allowed = {'Evidence', 'Port', 'RelationState', 'Variable', 'Running', 'ActionEvidence'}
            if name in allowed:
                return Passive
        if module == 'collections' and name == 'OrderedDict':
            return collections.OrderedDict
        if module in ('__builtin__', 'builtins') and name in ('bytes', 'set'):
            return getattr(builtins, name)
        if (module, name) == ('_codecs', 'encode'):
            return codecs.encode
        if module.startswith('numpy'):
            allowed = {'dtype', 'ndarray', '_reconstruct', 'PCG64', '__bit_generator_ctor',
                       '__generator_ctor', 'SeedSequence', '__pyx_unpickle_SeedSequence'}
            if name in allowed:
                return getattr(importlib.import_module(module), name)
        raise pickle.UnpicklingError(f'Unsupported data constructor: {module}.{name}')

    def persistent_load(self, identifier):
        kind, dtype, key, location, count, *other = identifier
        if kind != 'storage' or not isinstance(count, int) or count < 0:
            raise pickle.UnpicklingError('Invalid storage descriptor')
        return Storage(key, dtype, count)


def read_matrices(checkpoint):
    with Path(checkpoint).open('rb') as raw, gzip.GzipFile(fileobj=raw) as stream, zipfile.ZipFile(stream) as archive:
        info = next(x for x in archive.infolist() if x.filename.endswith('/data.pkl'))
        if info.file_size > 16*2**20:
            raise ValueError('Metadata budget exceeded')
        state = DataOnly(io.BytesIO(archive.read(info))).load()
        prefix = info.filename[:-len('data.pkl')]
        cache = {}

        def array(view):
            if not isinstance(view, View):
                raise TypeError('Expected tensor metadata')
            key = view.storage.key
            if key not in cache:
                entry = archive.getinfo(f'{prefix}data/{key}')
                if entry.file_size > 16*2**20:
                    raise ValueError('Requested storage exceeds S/P audit budget')
                cache[key] = archive.read(entry)
            data = cache[key]
            final = view.offset + sum((n-1)*s for n, s in zip(view.shape, view.stride))
            if any(s < 0 for s in view.stride) or final*view.dtype.itemsize >= len(data):
                raise ValueError('Invalid tensor view')
            return np.ndarray(view.shape, dtype=view.dtype, buffer=data,
                offset=view.offset*view.dtype.itemsize,
                strides=tuple(s*view.dtype.itemsize for s in view.stride)).copy()

        engine = state['engine']
        recurrent = array(engine['recurrent'])
        momenta = np.stack([array(v) for v in engine['bfss_momenta']])
        config = state.get('config', {})
        result = dict(cycle=state['cycle'], recurrent=recurrent, momenta=momenta,
            config=config, requested_storage_bytes=sum(map(len, cache.values())),
            engine_keys=list(engine))
        metadata = Path(checkpoint).with_name('last.metadata.json')
        if metadata.exists():
            receipt = json.loads(metadata.read_text(encoding='utf-8'))
            if receipt.get('cycle') == state['cycle']:
                result['checkpoint_sha256'] = receipt['checkpoint_sha256']
        return result


def force(s, coupling):
    si, sj = s[..., :, None, :, :], s[..., None, :, :, :]
    a = si @ sj.swapaxes(-1, -2) - sj @ si.swapaxes(-1, -2)
    b = si.swapaxes(-1, -2) @ sj - sj.swapaxes(-1, -2) @ si
    potential = coupling/16*(np.sum(a*a, axis=(-4, -3, -2, -1)) +
                             np.sum(b*b, axis=(-4, -3, -2, -1)))
    gradient = coupling/2*np.sum(a @ sj - sj @ b, axis=-3)
    return potential, gradient


def quiet(s, p, dt, omega, coupling):
    half = p - dt/2*force(s, coupling)[1]
    cosine, sine = math.cos(omega*dt), math.sin(omega*dt)
    next_s = cosine*s + sine/omega*half
    next_p = cosine*half - omega*sine*s
    return next_s, next_p-dt/2*force(next_s, coupling)[1]


def diagnose(saved):
    p = saved['momenta'].astype(np.float64)
    layers, batch, heads, dk, dv = p.shape
    s = saved['recurrent'][..., :heads*dk*dv].reshape(p.shape).astype(np.float64)
    spec = saved['config']['bfss_dynamics']
    dt, omega, coupling = (spec[k] for k in ('dt', 'frequency', 'coupling'))
    full_s, full_p = quiet(s, p, dt, omega, coupling)
    uncoupled_s, uncoupled_p = quiet(s, p, dt, omega, 0.)
    potential, gradient = force(s, coupling)
    direction = p/max(float(np.linalg.norm(p)), 1e-30)
    epsilon = 1e-5
    numerical = (force(s+epsilon*direction, coupling)[0] -
                 force(s-epsilon*direction, coupling)[0])/(2*epsilon)
    analytic = np.sum(gradient*direction, axis=(-3, -2, -1))
    rows = []
    for i in range(layers):
        norm = lambda x: float(np.linalg.norm(x))
        sn, pn, gn = norm(s[i]), norm(p[i]), norm(gradient[i])
        rows.append(dict(layer=i+1, state_norm=sn, momentum_norm=pn,
            momentum_minus_state_relative=norm(p[i]-s[i])/max(sn, 1e-30),
            momentum_state_cosine=float(np.sum(p[i]*s[i]))/max(sn*pn, 1e-30),
            largest_head_momentum_energy_fraction=max(norm(v)**2 for v in p[i, 0])/max(pn**2, 1e-30),
            momentum_head_norms=[norm(v) for v in p[i, 0]],
            coupling_potential=float(potential[i, 0]), coupling_force_norm=gn,
            harmonic_force_norm=omega**2*sn,
            coupling_over_harmonic=gn/(omega**2*sn) if sn else None,
            quiet_state_change=norm(full_s[i]-s[i]),
            coupling_only_state_delta=norm(full_s[i]-uncoupled_s[i]),
            coupling_only_momentum_delta=norm(full_p[i]-uncoupled_p[i]),
            coupling_fraction_of_state_motion=norm(full_s[i]-uncoupled_s[i])/max(norm(full_s[i]-s[i]),1e-30)))
    return dict(schema=1, cycle=saved['cycle'], dynamics=spec, layers=rows,
        checkpoint_sha256=saved.get('checkpoint_sha256'),
        dtype='float64 diagnostic of saved float32 matrices',
        actual_state_directional_derivative_max_error=float(np.max(np.abs(numerical-analytic))),
        total_quiet_state_change=float(np.linalg.norm(full_s-s)),
        total_coupling_state_delta=float(np.linalg.norm(full_s-uncoupled_s)),
        total_coupling_momentum_delta=float(np.linalg.norm(full_p-uncoupled_p)),
        whole_body_loaded=False, torch_imported=False,
        requested_storage_bytes=saved['requested_storage_bytes'], model_updates=0, world_calls=0,
        scope='Actual saved matrices; one next quiet step with/without coupling. No subsequent model or action rollout; no behavioral-benefit claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    saved = read_matrices(args.checkpoint)
    result = diagnose(saved)
    args.report.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(cycle=result['cycle'], requested_storage_bytes=result['requested_storage_bytes'],
        strongest=sorted(result['layers'], key=lambda r:r['coupling_force_norm'], reverse=True)[:3]), indent=2))


if __name__ == '__main__':
    main()
