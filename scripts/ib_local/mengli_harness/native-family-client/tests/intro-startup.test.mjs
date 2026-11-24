import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Execute the production startup block with scene mounts observed. This catches
// transient mounts even when the final DOM would already have faded them away.
const source=fs.readFileSync(new URL('../identity-assets/life-core.js',import.meta.url),'utf8');
const start=source.indexOf('  function pageShow(event)');
const end=source.indexOf("  window.addEventListener('pageshow', pageShow);",start);
assert(start>=0&&end>start);
const boot=source.slice(start,end)+"  window.addEventListener('pageshow', pageShow);";
function startup(login,{blockedStorage=false}={}) {
  const calls=[],listeners=new Map();
  const context=vm.createContext({MENGLI_LOGIN_PAGE:login,entered:false,Event,
    replay:()=>calls.push('mount-login'),locate:()=>calls.push('locate'),
    sessionStorage:{removeItem(){if(blockedStorage)throw new Error('blocked');calls.push('consume-marker');}},
    window:{dispatchEvent:event=>calls.push(event.type),addEventListener:(name,listener)=>listeners.set(name,listener)}});
  vm.runInContext(boot,context);
  return {calls,context,restore:()=>listeners.get('pageshow')({persisted:true})};
}
test('authenticated startup and restored history never mount a password scene, including with storage blocked',()=>{
  for(const blockedStorage of [false,true]){
    const run=startup(false,{blockedStorage});
    assert.equal(run.context.entered,true);
    assert.equal(run.calls.filter(call=>call==='mengli:enter').length,1);
    run.restore();
    assert.equal(run.calls.includes('mount-login'),false);
    assert.equal(run.calls.filter(call=>call==='locate').length,2);
  }
});
test('anonymous documents retain their login scene on initial load and restored history',()=>{
  const run=startup(true);
  assert.deepEqual(run.calls,['mount-login']);
  assert.equal(run.context.entered,false);
  run.restore();
  assert.deepEqual(run.calls,['mount-login','mount-login']);
});
