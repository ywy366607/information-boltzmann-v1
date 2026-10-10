"""Display statistics follow the periodic physical grid and actual tensor."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_periodic_clusters_actual_speed_and_dashboard_syntax():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node needed for browser asset numerical checks')
    present = Path(__file__).resolve().parents[1] / 'present'
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context={window:{},Math};vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1]+'/medium_structure.js','utf8'),context);
const s=context.window.MediumStructure;
assert.strictEqual(s.analyze(Array(64).fill(1),[4,4,4]).count,0);
const v=Array(64).fill(0);v[0]=1;v[48]=1;
assert.strictEqual(s.analyze(v,[4,4,4]).sizes[0],2);
assert.strictEqual(s.analyze(Array.from({length:64},(_,i)=>i),[4,4,4],50).count,32);
const snap={material_tensor_eigenvalues:[[1,4,9]],speed:[[100,100,100]]};
assert.strictEqual(s.field('speed','',snap,null)[0],3);
assert.strictEqual(s.field('anisotropy','',snap,null)[0],3);
assert.strictEqual(s.compositionRGB([0,0,0])[0],150);
const html=fs.readFileSync(process.argv[1]+'/medium_cosmos_live.html','utf8');
for(const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g))new Function(match[1]);
"""
    subprocess.run([node, '-e', script, str(present)], check=True, capture_output=True, text=True)
