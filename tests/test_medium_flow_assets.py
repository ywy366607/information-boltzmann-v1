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
"""
    subprocess.run([node, '-e', script, str(asset)], check=True, capture_output=True, text=True)
