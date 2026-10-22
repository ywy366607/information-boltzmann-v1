"""Existing finite contact reactions at a material-owned world interface.

Electrical terminals and material owners are distinct physical roles. The
same reaction stoichiometry/energy/acceptance/apply path handles internal
bridges and finite additional boundary contacts; no new learning species.
"""
from __future__ import annotations

import copy

import numpy as np

from .identity_material import IdentityMaterialProgram, IdentityMobileExchange
from .mobile_exchange import MaterialEvent, MobileExchangeParams


class BoundaryIdentityMobileExchange(IdentityMobileExchange):
    _arrays = IdentityMobileExchange._arrays+('port_installed','port_bound','port_trace')

    def __init__(self, *args, port_installed=0, port_bound=0, port_trace=0,
                 boundary_reactions_enabled=True, boundary_conduction_enabled=True, **kwargs):
        shape = np.shape(kwargs.get('port_nodes', ()))
        for name, value in (('port_installed',port_installed),('port_bound',port_bound),('port_trace',port_trace)):
            array = np.broadcast_to(np.asarray(value),shape)
            if not np.isfinite(array).all() or not np.equal(array,np.floor(array)).all():
                raise ValueError('Integer finite boundary configuration required')
            setattr(self,name,array.astype(np.int64).copy())
        self.boundary_reactions_enabled = boundary_reactions_enabled
        self.boundary_conduction_enabled = boundary_conduction_enabled
        super().__init__(*args,**kwargs)
        self.initial_material += self.port_installed.sum()
        self.initial_support += self.port_bound.sum()

    def _validate(self):
        super()._validate()
        for name in ('boundary_reactions_enabled','boundary_conduction_enabled'):
            if not isinstance(getattr(self,name),bool):
                raise ValueError('Explicit Boolean diagnostic barrier required')
        for name in ('port_installed','port_bound','port_trace'):
            array = getattr(self,name)
            if array.shape != self.port_nodes.shape or array.dtype.kind not in 'iu' or np.any((array<0)|(array>1)):
                raise ValueError('Binary occupied boundary configuration required: '+name)
        if np.any(self.port_bound>self.port_installed) or np.any(self.port_trace>self.port_installed):
            raise ValueError('Uninstalled contact cannot store support or trace')

    def energy(self):
        p = self.params
        return float(super().energy()+p.structure_energy*self.port_installed.sum()
            +p.bound_energy*self.port_bound.sum()+p.trace_energy*self.port_trace.sum()
            -p.capture_coupling*np.dot(self.port_trace,self.port_bound))

    def _contact_arrays(self,event):
        if event.interface >= 0:
            return self.port_bound,self.port_trace,event.interface
        return super()._contact_arrays(event)

    def _charge_changes(self,event):
        if event.interface >= 0 and (event.kind in ('write','active')
                or event.kind == 'thermal' and self.params.polarized_storage):
            z = event.sign*self.params.charge_per_event
            return [(self.n+event.interface,-z),(int(self.port_nodes[event.interface]),z)]
        return super()._charge_changes(event)

    def _assembly_delta(self,event,degree):
        if event.interface >= 0:
            # One labeled slot; no hidden free-end or identity multiplicity.
            return event.sign*self.params.structure_energy,0.
        return super()._assembly_delta(event,degree)

    def _assembly_attempt(self,event,degree):
        if event.interface >= 0:
            return self.params.assembly_rate
        return super()._assembly_attempt(event,degree)

    def _apply_assembly(self,event):
        if event.interface >= 0:
            self.port_installed[event.interface] += event.sign
        else:
            super()._apply_assembly(event)

    def effective_port_conductance(self):
        if not self.boundary_conduction_enabled:
            return self.port_conductance
        p = self.params
        return self.port_conductance+self.unit_conductance*self.port_installed*(p.gate_low+(p.gate_high-p.gate_low)*self.port_trace)

    def contact_conductance_increment(self,event):
        if event.interface >= 0 and not self.boundary_conduction_enabled:
            return 0.
        return super().contact_conductance_increment(event)

    def events(self,pairs=None,degree=None):
        events = super().events(pairs,degree)
        if not self.boundary_reactions_enabled:
            return events
        for port,node in enumerate(self.port_nodes):
            node = int(node)
            if not self.port_installed[port]:
                if self.material[node] and self.charged[node]:
                    events.append(MaterialEvent('assembly',node,node,edge=port,interface=port))
                continue
            events.extend(self._installed_contact_events(node,node,port,interface=port))
        return [event for event in events if self.attempt_rate(event,degree)>0.]

    def balances(self):
        values = super().balances()
        values['material'] += self.port_installed.sum()
        values['support'] += self.port_bound.sum()
        values['boundary_slots'] = int(np.sum(1-self.port_installed)+self.port_installed.sum()-len(self.port_nodes))
        return values

    def state_dict(self):
        state = super().state_dict()
        state.update(kind='boundary_identity_mobile_exchange',schema=1,field_scheme=self.field_scheme,
            boundary_reactions_enabled=self.boundary_reactions_enabled,
            boundary_conduction_enabled=self.boundary_conduction_enabled)
        return state

    @classmethod
    def from_identity(cls,model):
        if type(model) is not IdentityMobileExchange:
            raise TypeError('Use an existing complete identity-material individual')
        state = copy.deepcopy(model.state_dict())
        shape = model.port_nodes.shape
        for name in ('port_installed','port_bound','port_trace'):
            state['arrays'][name] = np.zeros(shape,dtype=np.int64)
        state.update(kind='boundary_identity_mobile_exchange',schema=1,field_scheme=model.field_scheme,
                     boundary_reactions_enabled=True,boundary_conduction_enabled=True)
        return cls.from_state_dict(state)

    @classmethod
    def from_state_dict(cls,state):
        if state.get('kind')!='boundary_identity_mobile_exchange' or state.get('schema')!=1:
            raise ValueError('Unsupported boundary material archive')
        if state.get('field_scheme') not in ('sequential_ou','symmetric_ou'):
            raise ValueError('Boundary archive needs an explicit field execution mode')
        arrays,genome = state['arrays'],state['genome']
        program = IdentityMaterialProgram({name:arrays['gene_'+name] for name in IdentityMaterialProgram.array_names},
                                         genome['types'],genome['provenance'])
        if program.fingerprint()!=genome['sha256']:
            raise ValueError('Boundary material genome fingerprint changed')
        obj = cls(arrays['coordinates'],arrays['material'],arrays['free'],arrays['charged'],arrays['depleted'],
            end_capacity=arrays['end_capacity'],carried=arrays['carried'],charge=arrays['charge'],
            capacitance=arrays['capacitance'],pre=state['pre'],post=state['post'],bound=arrays['bound'],
            trace=arrays['trace'],unit_conductance=state['unit_conductance'],params=MobileExchangeParams(**state['params']),
            port_nodes=state['port_nodes'],world_charge=arrays['world_charge'],world_capacitance=arrays['world_capacitance'],
            port_conductance=arrays['port_conductance'],source_conductance=arrays['source_conductance'],program=program,
            cell_type=arrays['cell_type'],recognition_energy_fraction=state['recognition_energy_fraction'],
            recognition_enabled=state['recognition_enabled'],transport_enabled=state['transport_enabled'],
            port_installed=arrays['port_installed'],port_bound=arrays['port_bound'],port_trace=arrays['port_trace'],
            boundary_reactions_enabled=state['boundary_reactions_enabled'],
            boundary_conduction_enabled=state['boundary_conduction_enabled'])
        obj = cls._restore_archive(obj,state)
        for name in program.array_names:
            setattr(obj,'gene_'+name,program.arrays[name])
        obj.cell_type.setflags(write=False)
        return obj
