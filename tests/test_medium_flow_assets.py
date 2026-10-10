"""Numerical checks for display interpolation, not a physics simulation."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_periodic_energy_vectors_and_unoriented_material_lines():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node needed for browser asset numerical checks')
    asset = Path(__file__).resolve().parents[1] / 'present' / 'medium_flow.js'
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context={window:{},Math,Date};vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const flow=context.window.MediumFlow;
const vectors=Array.from({length:8},()=>[1,0,0]);
for(const p of [[.2,.4,.8],[-.1,1.2,2.4],[0,0,0]]) {
 const v=flow.sample(vectors,[2,2,2],p);
 assert(Math.abs(v[0]-1)<1e-12&&v[1]===0&&v[2]===0);
}
const unoriented=vectors.map((v,i)=>[i%2?-1:1,0,0]);
assert(Math.abs(flow.sample(unoriented,[2,2,2],[.25,.25,.25],[1,0,0])[0]-1)<1e-12);
const snapshot={shape:[2,2,2],wall_time:Date.now()/1000,transport_energy_current:vectors};
const tracers=new flow.Tracers(1);tracers.set(snapshot);
const before=tracers.particles[0].p.slice();
const ctx={beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(){},fill(){}};
const project=p=>[p[0],p[1]];
tracers.draw(ctx,0,project,1,1,()=>true);
tracers.draw(ctx,50,project,1,1,()=>true);
assert(tracers.particles[0].p[0]>before[0]);
assert(tracers.particles[0].p[1]===before[1]);
const eig=flow.principal([[2,1,0],[1,2,0],[0,0,1]]);
assert(Math.abs(eig.speed-Math.sqrt(3))<1e-10);
assert(Math.abs(Math.abs(eig.axis[0])-Math.SQRT1_2)<1e-10);
assert(Math.abs(eig.gap-2/3)<1e-10);
const sh=[4,4,4],coordinates=Array.from({length:64},(_,i)=>[Math.floor(i/16)/4,Math.floor(i/4)%4/4,i%4/4]);
const b=Array.from({length:64},()=>[[3,0,0],[0,1,0],[0,0,1]]);
const lines=flow.streamlines({shape:sh,coordinates,effective_transport_factor:b},{count:8});
assert(lines.length===8);
let xLength=0;
for(const line of lines)for(const part of line.segments)for(let i=1;i<part.length;i++){
 const a=part[i-1],v=part[i];
 assert(Math.abs(v.speed-3)<1e-10);
 assert(Math.abs(v.p[0]-a.p[0])<.1); // no long chord across a display seam
 assert(Math.abs(v.p[1]-a.p[1])<1e-10&&Math.abs(v.p[2]-a.p[2])<1e-10);
 xLength+=Math.abs(v.p[0]-a.p[0]);
}
assert(xLength>5);
const iso=b.map(()=>[[1,0,0],[0,1,0],[0,0,1]]);
assert(flow.streamlines({shape:sh,coordinates,effective_transport_factor:iso}).length===0);
const opposite=flow.principal([[9,0,0],[0,1,0],[0,0,1]],[-1,0,0]);
assert(opposite.axis[0]<-.99);
const edgeValues=coordinates.map(p=>[p[0],0,0]);
assert(Math.abs(flow.sampleCurrent(edgeValues,sh,[.375,.2,.7])[0]-.25)<1e-12);
assert(Math.abs(flow.sampleCurrent(edgeValues,sh,[0,.2,.7])[0]-.375)<1e-12);
const current=coordinates.map(()=>[1,0,0]);
const aligned={shape:sh,coordinates,transport_energy_current:current,material_principal_axis:current};
assert(Math.abs(flow.currentAlignment(aligned)-1)<1e-12);
const across={...aligned,material_principal_axis:current.map(()=>[0,1,0])};
assert(flow.currentAlignment(across)===0);
const energyLines=flow.currentLines(aligned,{count:8});
assert(energyLines.length===8);
for(const line of energyLines)for(const part of line.segments)for(let i=1;i<part.length;i++){
 assert(Math.abs(part[i].p[0]-part[i-1].p[0])<.1);
 assert(Math.abs(part[i].p[1]-part[i-1].p[1])<1e-10);
}
assert(flow.junctionSummary(snapshot)===null);
assert(flow.junctionSummary({...snapshot,hopf_branch:{fraction:.5}})===null);
const j={protocol:'capacity_paid_persistent_flux_junction_v2',capacity_metric:'squared_coupling_norm',
 pairs:[[0,1],[0,2],[1,2]],resource_owner_rows:[0,1,2],route_mask:[1,1,1],
 physical_length:[.125,.125,.125],signed_fraction:[[0,0,0],[0,0,0]],
 rates:[[0,0,0],[0,0,0]],full_row_capacity:[[1,2,3],[1,2,3]],
 junction_row_capacity:[[0,0,0],[0,0,0]],capacity_balance_error:0,
 forest:[{label:'q0',children:[{label:'q1',children:[]}]}],scope:'instantaneous'};
const disabled=flow.junctionSummary({...snapshot,hopf_branch:j});
assert(disabled.paidShare===0&&disabled.maxAbsRate===0&&disabled.balance===0);
const active=flow.junctionSummary({...snapshot,hopf_branch:{...j,
 signed_fraction:[[.2,0,0],[.2,0,0]],rates:[[1,0,0],[-2,0,0]],
 junction_row_capacity:[[.2,0,0],[.2,0,0]]}});
assert(Math.abs(active.paidShare-.08/28)<1e-12);
assert(Math.abs(active.rows[0].paidShare-.04)<1e-12&&active.rows[0].meanRate===-.5);
assert(active.rows[0].meanAbsKappa===.2);
assert(active.maxAbsRate===2&&active.rows[0].physicalLength===.125);
"""
    subprocess.run([node, '-e', script, str(asset)], check=True, capture_output=True, text=True)
