"""Bounded, disposable receipts and carrier-fate observation.

These values never enter an event rate. All initial material state, future
RNG and the observer's small continuation state are archived separately.
"""
from __future__ import annotations

import copy

import numpy as np


class ExchangeObservation:
    def __init__(self,model,focus_ports=(),state=None):
        if state is not None:
            self.data=copy.deepcopy(state)
            self.validate(model)
            return
        keys=self.contact_keys(model)
        alive={key:index for index,key in enumerate(keys)}
        self.data=dict(start_ms=model.t,initial_drive=model.charged.tolist(),
            initial_config_free_energy=model.free_energy()-model.energy(),
            initial_reaction_heat=model.reaction_heat,initial_reaction_counts=dict(model.reaction_counts),
            initial_work=model.work_in,initial_heat=model.heat_out,initial_contact_count=len(keys),
            focus_ids=[alive['p:'+str(port)] for port in focus_ports if 'p:'+str(port) in alive],
            alive=alive,next_id=len(keys),deaths=0,events=0,counts={},by_kind={},
            drive_delta=[0]*model.n,electrical_energy_delta=0.,drive_energy_delta=0.,
            other_energy_delta=0.,delta_u=0.,delta_g=0.)

    @staticmethod
    def contact_keys(model):
        keys=['i:'+str(int(i))+':'+str(int(j)) for i,j in zip(model.pre,model.post)]
        if hasattr(model,'port_installed'):
            keys+=['p:'+str(port) for port,n in enumerate(model.port_installed) if n]
        return sorted(keys)

    @staticmethod
    def event_key(receipt):
        port=receipt.get('interface',-1)
        return 'p:'+str(port) if port>=0 else 'i:'+str(receipt['i'])+':'+str(receipt['j'])

    def observe(self,receipt):
        data=self.data
        kind=receipt['kind'];sign=receipt['sign']
        label=('port_' if receipt.get('interface',-1)>=0 else '')+kind
        counter=label+('+' if sign>0 else '-')
        data['counts'][counter]=data['counts'].get(counter,0)+1
        group=data['by_kind'].setdefault(label,dict(events=0,electrical_energy_delta=0.,
            drive_energy_delta=0.,other_energy_delta=0.,delta_u=0.,delta_g=0.))
        group['events']+=1;data['events']+=1
        for name in ('electrical_energy_delta','drive_energy_delta','other_energy_delta','delta_u','delta_g'):
            group[name]+=receipt[name];data[name]+=receipt[name]
        for name,node,change in receipt['composition_changes']:
            if name=='charged':data['drive_delta'][node]+=change
        if kind not in ('exchange','conversion','release'):
            key=self.event_key(receipt)
            if kind=='assembly' and sign>0:
                if key in data['alive']:raise ValueError('Carrier already installed')
                data['alive'][key]=data['next_id'];data['next_id']+=1
            if key not in data['alive']:raise ValueError('Unknown live storage carrier')
            receipt['observed_carrier_id']=data['alive'][key]
            if kind=='assembly' and sign<0:
                data['alive'].pop(key);data['deaths']+=1

    def validate(self,model):
        data=self.data
        if set(data['alive'])!=set(self.contact_keys(model)):
            raise ValueError('Observation inventory does not match the material')
        expected=np.array(data['initial_drive'])+np.array(data['drive_delta'])
        np.testing.assert_array_equal(expected,model.charged)
        expected_energy=model.params.charged_energy*sum(data['drive_delta'])
        if abs(expected_energy-data['drive_energy_delta'])>1e-8:
            raise ValueError('Drive receipt does not close')
        if abs(model.reaction_heat-data['initial_reaction_heat']+data['delta_u'])>1e-8:
            raise ValueError('Reaction heat receipt does not close')
        config=model.free_energy()-model.energy()-data['initial_config_free_energy']
        if abs(config-data['delta_g']+data['delta_u'])>1e-8:
            raise ValueError('Configuration entropy receipt does not close')
        if len(data['alive'])!=data['next_id']-data['deaths']:
            raise ValueError('Carrier births/deaths do not close')
        for key in set(model.reaction_counts)|set(data['initial_reaction_counts'])|set(data['counts']):
            if model.reaction_counts.get(key,0)-data['initial_reaction_counts'].get(key,0)!=data['counts'].get(key,0):
                raise ValueError('Reaction receipt count mismatch')

    def summary(self,model):
        self.validate(model)
        data=self.data;live=set(data['alive'].values())
        conversion=data['by_kind'].get('conversion',{})
        return dict(time_ms=model.t,events=data['events'],
            conversion_electrical_delivery=-conversion.get('electrical_energy_delta',0.),
            drive_energy_change=data['drive_energy_delta'],
            source_work=model.work_in-data['initial_work'],heat_out=model.heat_out-data['initial_heat'],
            initial_contacts=data['initial_contact_count'],live_contacts=len(live),
            initial_carriers_surviving=sum(identity<data['initial_contact_count'] for identity in live),
            focus_carriers_surviving=sum(identity in live for identity in data['focus_ids']),
            focus_carriers_initial=len(data['focus_ids']),
            births=data['next_id']-data['initial_contact_count'],deaths=data['deaths'],
            local_drive_change=list(data['drive_delta']),by_kind=copy.deepcopy(data['by_kind']))

    def state_dict(self):
        return copy.deepcopy(self.data)
