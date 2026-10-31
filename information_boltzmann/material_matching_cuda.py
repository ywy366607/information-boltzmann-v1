"""CUDA numerical executor for the registered cap2 material law.

Matching preparation is parallel; a reflected prefix enforces the SAME stock
order as serial matching execution. This is a new sweep clock, not the CPU
random-site clock. No old clock's lifetime certificate is inherited.

Numerical/interface use only. Candidate selection is unbiased (each block
size divides468). The GPU coin uses a31-bit grid and FP64 exponentials; rare
rates/rounding differ from the ideal unlimited-bit CPU sampler. No model
training or unqualified ideal-kernel refinement is admitted by this module.
"""
from functools import lru_cache
from fractions import Fraction
from copy import copy
import math

import torch

MATCHING_CLOCK = "readout_matching_sweep_v1"
FULL_LIFELONG_ADMITTED = False
COIN_LEVELS = 1 << 31


@lru_cache(maxsize=1)
def _initializer():
    """Nearest readout, uniform existing microstate at that readout level."""
    from numba import cuda
    fibers = tuple((p*3+q)*3+w for level in range(7)
                   for p in range(3) for q in range(3) for w in range(3)
                   if p+q+w == level)
    starts = (0,1,4,10,17,23,26,27)

    @cuda.jit(fastmath=False)
    def initialize(parameter,words,codes,quantum,offset):
        i=cuda.grid(1)
        if i<codes.size:
            level=max(0,min(6,int(math.floor((parameter[i]-offset)/quantum+0.5))))
            codes[i]=fibers[starts[level]+words[i]%(starts[level+1]-starts[level])]
    return initialize


@lru_cache(maxsize=1)
def _kernels():
    from numba import cuda, float64
    from numba.cuda import libdevice

    # Candidate order is exactly the existing CPU fiber order.
    fibers = tuple((p*3+q)*3+w for level in range(7)
                   for p in range(3) for q in range(3) for w in range(3)
                   if p+q+w == level)
    starts = (0, 1, 4, 10, 17, 23, 26, 27)
    minimum_codes = (0, 1, 2, 13, 14, 17, 23, 26)
    minimum_starts = (0, 1, 2, 3, 4, 5, 7, 8)

    @cuda.jit(device=True, inline=True)
    def location(lane, color, rows, cols):
        if rows <= cols:
            return lane, (lane+color) % cols
        return (lane+color) % rows, lane

    @cuda.jit(device=True, inline=True)
    def mass(code):
        return code//9+(code//3)%3+code%3

    @cuda.jit(device=True, inline=True)
    def energy(code, R, C, rows, cols, force):
        p, q, w = code//9, (code//3)%3, code%3
        return (float64(14*(p+q+w)-12*p*q*w)
                +float64(2)*float64(R+p)*float64(R+p)/float64(cols)
                +float64(2)*float64(C+q)*float64(C+q)/float64(rows)
                +force*float64(p+q+w))

    @cuda.jit(fastmath=False)
    def prepare(codes, row_p, col_q, gradient, tickets, proposals, delta,
                powers, color, rows, cols, force_scale, has_gradient):
        lane = cuda.grid(1)
        if lane >= proposals.size:
            return
        row, col = location(lane, color, rows, cols)
        index = row*cols+col
        old = int(codes[index]); level = mass(old)
        direction = 1 if tickets[lane,0] else -1
        proposals[lane], delta[lane], powers[lane,0], powers[lane,1] = old, 0, 0, 0
        if level+direction < 0 or level+direction > 6:
            return
        lower = min(level,level+direction)
        size = starts[lower+2]-starts[lower]
        candidate = fibers[starts[lower]+tickets[lane,1] % size]
        R, C = int(row_p[row])-old//9, int(col_q[col])-(old//3)%3
        force = float64(gradient[index])*force_scale if has_gradient else float64(0)
        floor = float64(math.inf)
        for k in range(minimum_starts[lower],minimum_starts[lower+2]):
            floor = min(floor,energy(minimum_codes[k],R,C,rows,cols,force))
        power = max(float64(0),float64(2)*(energy(candidate,R,C,rows,cols,force)-floor))
        powers[lane,0] = power
        # Integer ticket comparison; p=1 is accepted without an exponential.
        probability = float64(1) if power == 0 else libdevice.exp2(-power)
        cutoff = math.floor(probability*float64(2147483648))
        powers[lane,1] = cutoff
        if tickets[lane,2] < cutoff:
            proposals[lane] = candidate
            delta[lane] = mass(candidate)-level

    @cuda.jit(fastmath=False)
    def commit(parameter, codes, row_p, col_q, stock, tickets, proposals,
               cumulative, peaks, color, rows, cols, quantum, offset):
        lane = cuda.grid(1)
        if lane >= proposals.size:
            return
        prior = 0 if lane == 0 else cumulative[lane-1]
        peak = 0 if lane == 0 else max(0,peaks[lane-1])
        available = stock[0]-prior+max(0,peak-stock[0])
        # An upward block is wholly disabled at zero stock, including its
        # same-mass internal candidates. Prefix reflection alone changes only
        # the actual material delta; this guard also preserves that microstate.
        if tickets[lane,0] and available == 0:
            return
        row, col = location(lane,color,rows,cols)
        index = row*cols+col
        old, new = int(codes[index]), int(proposals[lane])
        codes[index] = new
        row_p[row] += new//9-old//9
        col_q[col] += (new//3)%3-(old//3)%3
        parameter[index] = offset+quantum*float64(mass(new))

    return prepare, commit


class CudaMatchingMaterial:
    """One contiguous FP32 parameter matrix, with device-owned material.

    One byte per cell currently stores its27-state code; this is lossless byte
    alignment, not5-bit packing. Shared row/column totals use int32. Stock is
    ONE int64 scalar. All transient proposal/prefix/ticket arrays have only
    min(rows,cols) entries, never one history vector per model parameter.

    A sweep visits every cell once through nonintersecting row/column colors.
    The declared physical tariff remains13 units per attempted cell. The API
    does not claim that these units are measured joules. No external-wear
    clock is silently added; callers may change the body only between colors.
    """

    clock = MATCHING_CLOCK

    def __init__(self, parameter, codes, *, stock, quantum, offset=Fraction(0),
                 energy_unit=Fraction(1), seed=0):
        if (parameter.device.type != "cuda" or parameter.dtype != torch.float32
                or parameter.ndim != 2 or not parameter.is_contiguous()):
            raise ValueError("requires a contiguous CUDA FP32 parameter matrix")
        if (codes.device != parameter.device or codes.dtype != torch.uint8
                or codes.shape != parameter.shape or not codes.is_contiguous()):
            raise ValueError("requires same-device uint8 material codes of matrix shape")
        self.rows, self.cols = parameter.shape
        if not min(self.rows,self.cols) or max(self.rows,self.cols) >= (1 << 29):
            raise ValueError("nonempty matrix and int32-safe pool sizes required")
        if type(stock) is not int or not 0 <= stock < (1 << 62):
            raise ValueError("invalid global stock")
        if any(isinstance(v,bool) or not isinstance(v,(int,Fraction))
               for v in (quantum,offset,energy_unit)) or quantum <= 0 or energy_unit <= 0:
            raise ValueError("explicit positive rational units required")
        self.quantum, self.offset, self.energy_unit = map(Fraction,(quantum,offset,energy_unit))
        if not 0 < float(self.quantum/self.energy_unit) <= 1e200:
            raise ValueError("nonfinite force scale")
        for n in range(7):
            value = self.offset+n*self.quantum
            f = torch.tensor(float(value),dtype=torch.float32).item()
            if not math.isfinite(f) or Fraction.from_float(f) != value:
                raise ValueError("readout alphabet must be exactly representable")
        if torch.any(codes > 26).item():
            raise ValueError("invalid material state")
        self.parameter = parameter
        self.codes = codes.detach().clone().view(-1)
        grid = self.codes.view(self.rows,self.cols)
        self.row_p = (grid//9).sum(1,dtype=torch.int32)
        self.col_q = ((grid//3)%3).sum(0,dtype=torch.int32)
        self.stock = torch.tensor([stock],dtype=torch.int64,device=parameter.device)
        self.supply = stock+int((grid//9+(grid//3)%3+grid%3).sum(dtype=torch.int64).item())
        self.generator = torch.Generator(device=parameter.device).manual_seed(seed)
        self.color = self.attempts = self.fee_units = 0
        self.lanes = min(self.rows,self.cols)
        device = parameter.device
        # Reused across EVERY color, including CUDA graph capture. Thus a
        # captured sweep does not retain one proposal workspace per parameter.
        self._workspace = dict(
            tickets=torch.empty((self.lanes,3),device=device,dtype=torch.int32),
            proposals=torch.empty(self.lanes,device=device,dtype=torch.int32),
            delta=torch.empty(self.lanes,device=device,dtype=torch.int32),
            powers=torch.empty((self.lanes,2),device=device,dtype=torch.float64),
            cumulative=torch.empty(self.lanes,device=device,dtype=torch.int32),
            peaks=torch.empty(self.lanes,device=device,dtype=torch.int32),
            indices=torch.empty(self.lanes,device=device,dtype=torch.int64),
            stock_a=torch.empty_like(self.stock),stock_b=torch.empty_like(self.stock))
        self._graph = self._graph_gradient = None
        # No autograd graph is retained for the decoded physical parameters.
        with torch.no_grad():
            p, q, w = self.codes//9, (self.codes//3)%3, self.codes%3
            parameter.copy_((float(self.offset)+float(self.quantum)*(p+q+w)).view_as(parameter))

    def _validate_gradient(self, gradient):
        if gradient is None:
            return
        if (gradient.device != self.parameter.device or gradient.dtype != torch.float32
                or gradient.shape != self.parameter.shape or not gradient.is_contiguous()):
            raise ValueError("gradient must match contiguous CUDA parameter matrix")
        first, last = self.parameter.data_ptr(), self.parameter.data_ptr()+4*self.parameter.numel()
        grad_first, grad_last = gradient.data_ptr(), gradient.data_ptr()+4*gradient.numel()
        if max(first,grad_first) < min(last,grad_last) or not torch.isfinite(gradient).all().item():
            raise ValueError("gradient must be finite and disjoint from the parameter")

    @torch.no_grad()
    def sweep(self, gradient):
        """Visit every site once; validate the frozen gradient only once."""
        self._validate_gradient(gradient)
        for _ in range(max(self.rows,self.cols)):
            self._execute_color(gradient)
        torch.autograd.graph.increment_version(self.parameter)

    @torch.no_grad()
    def capture_sweep(self, gradient):
        """Capture fixed storage, without consuming physical time or RNG.

        Kernel warmup and the complete state restoration precede capture.
        Replay reads the current contents of the SAME gradient storage; it
        never freezes its numerical values. A checkpoint load invalidates it.
        """
        self._validate_gradient(gradient)
        snapshot = self.state_dict()
        self.sweep(gradient)
        torch.cuda.synchronize(self.parameter.device)
        self.load_state_dict(snapshot)
        graph = torch.cuda.CUDAGraph()
        graph.register_generator_state(self.generator)
        before=(self.color,self.attempts,self.fee_units)
        with torch.cuda.graph(graph):
            for _ in range(max(self.rows,self.cols)):
                self._execute_color(gradient)
        self.color,self.attempts,self.fee_units=before
        self._graph, self._graph_gradient = graph, gradient
        self._graph_color = self.color
        self._graph_storage = self._storage_signature(gradient)

    def _storage_signature(self, gradient):
        return tuple(x.data_ptr() for x in (self.parameter,self.codes,self.row_p,self.col_q,self.stock)) + (
            0 if gradient is None else gradient.data_ptr(),)

    @torch.no_grad()
    def graph_sweep(self, gradient):
        """Replay one complete sweep with preserved RNG/body/stock semantics."""
        self._validate_gradient(gradient)
        if (self._graph is None or gradient is not self._graph_gradient
                or self.color != self._graph_color
                or self._storage_signature(gradient) != self._graph_storage):
            raise ValueError("capture this exact gradient storage before replay")
        self._graph.replay()
        count=self.rows*self.cols
        self.attempts+=count
        self.fee_units+=13*count
        torch.autograd.graph.increment_version(self.parameter)

    @torch.no_grad()
    def color_step(self, gradient, *, tickets=None, audit=False):
        """One matching, with GPU random words or caller-supplied test words.

        Tickets columns: direction in{0,1}, uniform candidate word in[0,468),
        acceptance word in[0,2^31). Diagnostics may synchronize the host only
        when requested. Gradient finiteness is validated before any mutation.
        """
        self._validate_gradient(gradient)
        result = self._execute_color(gradient,tickets=tickets,audit=audit)
        torch.autograd.graph.increment_version(self.parameter)
        return result

    def _execute_color(self, gradient, *, tickets=None, audit=False):
        from numba import cuda
        device = self.parameter.device
        if tickets is not None:
            if (tickets.device != self.parameter.device or tickets.dtype != torch.int32
                    or tickets.shape != (self.lanes,3) or not tickets.is_contiguous()):
                raise ValueError("invalid ticket layout")
            bounds = torch.tensor([2,468,COIN_LEVELS],device=tickets.device,dtype=torch.int64)
            if not ((tickets >= 0) & (tickets < bounds)).all().item():
                raise ValueError("ticket outside sampling range")
        else:
            tickets = self._workspace['tickets']
            for column,limit in enumerate((2,468,COIN_LEVELS)):
                torch.randint(limit,(self.lanes,),device=device,dtype=torch.int32,
                              generator=self.generator,out=tickets[:,column])
        proposals,delta,powers = (self._workspace[k] for k in ('proposals','delta','powers'))
        prepare, commit = _kernels()
        gradient_view = self.parameter.view(-1) if gradient is None else gradient.view(-1)
        with torch.cuda.device(self.parameter.device):
            stream = cuda.external_stream(torch.cuda.current_stream(self.parameter.device).cuda_stream)
            def view(x):
                return cuda.as_cuda_array(x.detach(),sync=False)
            prepare[(self.lanes+127)//128,128,stream](
                view(self.codes),view(self.row_p),view(self.col_q),view(gradient_view),
                view(tickets),view(proposals),view(delta),view(powers),self.color,
                self.rows,self.cols,float(self.quantum/self.energy_unit),gradient is not None)
            cumulative,peaks,indices = (self._workspace[k] for k in ('cumulative','peaks','indices'))
            torch.cumsum(delta,0,dtype=torch.int32,out=cumulative)
            torch.cummax(cumulative,0,out=(peaks,indices))
            commit[(self.lanes+127)//128,128,stream](
                view(self.parameter.view(-1)),view(self.codes),view(self.row_p),view(self.col_q),
                view(self.stock),view(tickets),view(proposals),view(cumulative),view(peaks),
                self.color,self.rows,self.cols,float(self.quantum),float(self.offset))
            a,b=self._workspace['stock_a'],self._workspace['stock_b']
            torch.sub(peaks[-1],self.stock,out=a)
            a.clamp_(min=0)
            torch.sub(self.stock,cumulative[-1],out=b)
            torch.add(a,b,out=self.stock)
        self.attempts += self.lanes
        self.fee_units += 13*self.lanes
        self.color = (self.color+1) % max(self.rows,self.cols)
        if audit:
            return dict(proposals=proposals.clone(),powers=powers[:,0].clone(),cutoffs=powers[:,1].clone(),
                        delta=delta.clone(),tickets=tickets.clone())

    def state_dict(self):
        """Complete physical and random continuation; no gradient is serialized."""
        return dict(version=1,clock=self.clock,shape=(self.rows,self.cols),supply=self.supply,
            units=tuple((v.numerator,v.denominator) for v in
                        (self.quantum,self.offset,self.energy_unit)),
            codes=self.codes.clone(),stock=self.stock.clone(),rng=self.generator.get_state().clone(),
            color=self.color,attempts=self.attempts,fee_units=self.fee_units)

    @torch.no_grad()
    def load_state_dict(self, state):
        """Validate completely before replacing the live body, cache or RNG."""
        prepared=self._validate_state_dict(state)
        self._restore_validated_state(state,prepared)

    def _validate_state_dict(self, state):
        """Prepare owned successor arrays without modifying live parameters."""
        keys = {'version','clock','shape','supply','units','codes','stock','rng','color','attempts','fee_units'}
        units = tuple((v.numerator,v.denominator) for v in (self.quantum,self.offset,self.energy_unit))
        if (set(state) != keys or state['version'] != 1 or state['clock'] != self.clock
                or tuple(state['shape']) != (self.rows,self.cols) or state['units'] != units
                or state['supply'] != self.supply):
            raise ValueError("incompatible matching checkpoint")
        codes, stock = state['codes'], state['stock']
        if (not isinstance(codes,torch.Tensor) or codes.device != self.parameter.device
                or codes.dtype != torch.uint8 or codes.shape != self.codes.shape
                or not codes.is_contiguous() or torch.any(codes > 26).item()
                or not isinstance(stock,torch.Tensor) or stock.device != self.parameter.device
                or stock.dtype != torch.int64 or stock.shape != (1,) or stock.item() < 0):
            raise ValueError("invalid physical checkpoint")
        mass = int((codes//9+(codes//3)%3+codes%3).sum(dtype=torch.int64).item())
        if mass+stock.item() != self.supply:
            raise ValueError("checkpoint violates total supply")
        if (any(type(state[k]) is not int or state[k] < 0 for k in ('color','attempts','fee_units'))
                or state['attempts'] % self.lanes
                or state['color'] != (state['attempts']//self.lanes) % max(self.rows,self.cols)
                or state['fee_units'] != 13*state['attempts']):
            raise ValueError("invalid matching clock or fee ledger")
        generator = torch.Generator(device=self.parameter.device)
        generator.set_state(state['rng'])
        new_codes = codes.clone()
        grid = new_codes.view(self.rows,self.cols)
        row_p = (grid//9).sum(1,dtype=torch.int32)
        col_q = ((grid//3)%3).sum(0,dtype=torch.int32)
        return new_codes,stock.clone(),row_p,col_q,generator

    @torch.no_grad()
    def _restore_validated_state(self,state,prepared):
        self.codes,self.stock,self.row_p,self.col_q,self.generator=prepared
        self._graph = self._graph_gradient = None
        self.color, self.attempts, self.fee_units = (state[k] for k in ('color','attempts','fee_units'))
        grid=self.codes.view(self.rows,self.cols)
        self.parameter.copy_((float(self.offset)+float(self.quantum)*(grid//9+(grid//3)%3+grid%3)).view_as(self.parameter))


class CudaMaterialOptimizer(torch.optim.Optimizer):
    """Standard optimizer interface over the registered CUDA compartments.

    Each parameter tensor owns one independent resource compartment. Matrices
    retain their shape; vectors use a near-square exact factorization without
    padding, extra cells or topology indices. Initial floating values are
    projected once to the nearest seven-level readout. Within that level the
    original material microstates are initialized uniformly, using a separate
    initialization RNG. This is an explicit initialization convention, not a
    pretrained-model losslessness claim.

    lr scales the supplied gradient force, while quantum sets actual write
    size. lr=0 and missing gradients retain autonomous physical motion. State
    contains no Adam moments/shadow weights. zero_grad defaults to preserving
    gradient storage for graph replay. Only numerical/interface execution is
    admitted here; a registered learning experiment needs its applicable gate.
    """

    def __init__(self,params,*,quantum,energy_unit=Fraction(1),offset=None,
                 stock_per_cell=1,lr=1.0,seed=0,use_graph=True):
        if offset is None:
            offset=-3*quantum
        super().__init__(params,dict(quantum=quantum,energy_unit=energy_unit,
                                    offset=offset,stock_per_cell=stock_per_cell,lr=lr))
        self.use_graph=bool(use_graph)
        self._entries=[]
        entries=[];ranges=[];devices=set()
        for group in self.param_groups:
            self._check_group(group)
            for parameter in group['params']:
                if (parameter.device.type!='cuda' or parameter.dtype!=torch.float32
                        or not parameter.is_contiguous() or parameter.numel()==0):
                    raise ValueError('nonempty contiguous CUDA FP32 parameters required')
                if not torch.isfinite(parameter).all().item():
                    raise ValueError('initial parameters must be finite')
                devices.add(parameter.device)
                ranges.append((parameter.data_ptr(),parameter.data_ptr()+4*parameter.numel()))
                if parameter.ndim>=2:
                    rows=parameter.shape[0]
                else:
                    rows=math.isqrt(parameter.numel())
                    while parameter.numel()%rows:rows-=1
                shape=(rows,parameter.numel()//rows)
                if (max(shape)>=(1<<29)
                        or group['stock_per_cell']*parameter.numel()>=(1<<62)):
                    raise ValueError('material shape or stock outside integer domain')
                entries.append((parameter,group,shape))
        ranges.sort()
        if len(devices)!=1 or any(b[0]<a[1] for a,b in zip(ranges,ranges[1:])):
            raise ValueError('parameters must share one CUDA device and disjoint storage')
        # All layout validation precedes any projection of live parameters.
        from numba import cuda
        for number,(parameter,group,shape) in enumerate(entries):
            q,o=Fraction(group['quantum']),Fraction(group['offset'])
            generator=torch.Generator(device=parameter.device).manual_seed((seed+number)^0x5DEECE66D)
            words=torch.randint(42,(parameter.numel(),),device=parameter.device,dtype=torch.uint8,generator=generator)
            codes=torch.empty_like(words)
            with torch.cuda.device(parameter.device):
                stream=cuda.external_stream(torch.cuda.current_stream(parameter.device).cuda_stream)
                _initializer()[(parameter.numel()+255)//256,256,stream](
                    cuda.as_cuda_array(parameter.detach().view(-1),sync=False),
                    cuda.as_cuda_array(words,sync=False),cuda.as_cuda_array(codes,sync=False),float(q),float(o))
            owner=CudaMatchingMaterial(parameter.view(shape),codes.view(shape),
                stock=group['stock_per_cell']*parameter.numel(),quantum=q,offset=o,
                energy_unit=self._effective_unit(group),seed=seed+number)
            self._entries.append(dict(parameter=parameter,group=group,owner=owner,
                                      gradient_source=None,gradient_view=None))

    @staticmethod
    def _effective_unit(group):
        return Fraction(group['energy_unit'])/(Fraction.from_float(float(group['lr'])) if group['lr'] else 1)

    @staticmethod
    def _check_group(group):
        if set(group)!={'params','quantum','energy_unit','offset','stock_per_cell','lr'}:
            raise ValueError('unsupported material parameter-group options')
        if (any(isinstance(group[k],bool) or not isinstance(group[k],(int,Fraction))
                for k in ('quantum','energy_unit','offset'))
                or group['quantum']<=0 or group['energy_unit']<=0
                or type(group['stock_per_cell']) is not int or group['stock_per_cell']<0
                or not math.isfinite(float(group['lr'])) or group['lr']<0):
            raise ValueError('invalid material force/readout/resource units')
        unit=CudaMaterialOptimizer._effective_unit(group)
        if not 0<float(Fraction(group['quantum'])/unit)<=1e200:
            raise ValueError('force units outside numerical domain')
        for n in range(7):
            value=Fraction(group['offset'])+n*Fraction(group['quantum'])
            fp=torch.tensor(float(value),dtype=torch.float32).item()
            if not math.isfinite(fp) or Fraction.from_float(fp)!=value:
                raise ValueError('material alphabet must be exactly representable')

    def add_param_group(self,group):
        if getattr(self,'_entries',None):
            raise ValueError('material parameter layout is fixed')
        return super().add_param_group(group)

    def zero_grad(self,set_to_none=False):
        return super().zero_grad(set_to_none=set_to_none)

    def _gradient_view(self,entry):
        parameter,owner=entry['parameter'],entry['owner']
        source=parameter.grad
        if source is None:return None
        if (source.device!=parameter.device or source.dtype!=parameter.dtype
                or source.shape!=parameter.shape or not source.is_contiguous()):
            raise ValueError('material gradients require matching contiguous FP32 layout')
        if (entry['gradient_source'] is not source or entry['gradient_view'] is None
                or entry['gradient_view'].data_ptr()!=source.data_ptr()):
            entry['gradient_source']=source
            entry['gradient_view']=source.view(owner.rows,owner.cols)
        return entry['gradient_view']

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():loss=closure()
        gradients=[];ranges=[]
        # Validate ALL compartments before any payment/physical change/RNG.
        for entry in self._entries:
            self._check_group(entry['group'])
            p,owner=entry['parameter'],entry['owner']
            if (p.data_ptr()!=owner.parameter.data_ptr() or not p.is_contiguous()
                    or p.numel()!=owner.rows*owner.cols
                    or group_units_changed(entry)):
                raise ValueError('parameter storage changed after material construction')
            gradient=self._gradient_view(entry)
            owner._validate_gradient(gradient)
            if gradient is not None:
                ranges.append((gradient.data_ptr(),gradient.data_ptr()+4*gradient.numel()))
            gradients.append(gradient if entry['group']['lr'] else None)
        parameters=[(e['parameter'].data_ptr(),e['parameter'].data_ptr()+4*e['parameter'].numel()) for e in self._entries]
        for a,b in ranges:
            if any(max(a,c)<min(b,d) for c,d in parameters):
                raise ValueError('gradient aliases material parameter storage')
        for entry,gradient in zip(self._entries,gradients):
            owner,group=entry['owner'],entry['group']
            unit=self._effective_unit(group)
            if unit!=owner.energy_unit:
                owner.energy_unit=unit
                owner._graph=owner._graph_gradient=None
            if self.use_graph:
                if (owner._graph is None or gradient is not owner._graph_gradient
                        or owner._storage_signature(gradient)!=owner._graph_storage):
                    owner.capture_sweep(gradient)
                owner.graph_sweep(gradient)
            else:owner.sweep(gradient)
        return loss

    def state_dict(self):
        result=super().state_dict()
        result['cuda_material']=dict(version=1,clock=MATCHING_CLOCK,
            compartments=[entry['owner'].state_dict() for entry in self._entries])
        return result

    def load_state_dict(self,state):
        if set(state)!={'state','param_groups','cuda_material'} or state['state']:
            raise ValueError('invalid material optimizer checkpoint')
        meta=state['cuda_material'];groups=state['param_groups']
        if (meta.get('version')!=1 or meta.get('clock')!=MATCHING_CLOCK
                or len(groups)!=len(self.param_groups)
                or len(meta.get('compartments',[]))!=len(self._entries)):
            raise ValueError('material checkpoint layout differs')
        loaded=[]
        for group,current in zip(groups,self.param_groups):
            self._check_group(group)
            if len(group['params'])!=len(current['params']):raise ValueError('parameter count differs')
            for _ in group['params']:loaded.append(group)
        prepared=[]
        for entry,body,group in zip(self._entries,meta['compartments'],loaded):
            owner=copy(entry['owner'])
            if (owner.quantum!=group['quantum'] or owner.offset!=group['offset']
                    or entry['group']['stock_per_cell']!=group['stock_per_cell']):
                raise ValueError('checkpoint physical units differ')
            owner.energy_unit=self._effective_unit(group)
            if type(body.get('supply')) is not int or not 0<=body['supply']<(1<<63):
                raise ValueError('invalid restored supply')
            owner.supply=body['supply']
            prepared.append((owner,owner._validate_state_dict(body)))
        super().load_state_dict(dict(state={},param_groups=groups))
        new_groups=[g for group in self.param_groups for g in [group]*len(group['params'])]
        for entry,body,group,(owner,data) in zip(self._entries,meta['compartments'],new_groups,prepared):
            owner._restore_validated_state(body,data)
            entry.update(owner=owner,group=group,gradient_source=None,gradient_view=None)


def group_units_changed(entry):
    """Readout units are physical; lr/energy units only change the task force."""
    return (Fraction(entry['group']['quantum'])!=entry['owner'].quantum
            or Fraction(entry['group']['offset'])!=entry['owner'].offset)
