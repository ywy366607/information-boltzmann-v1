import test from 'node:test';
import assert from 'node:assert/strict';
import { aggregateLayers, coreTarget, easeCore, isLearningCommit, telemetryFocus,
  originalPortraitBinding, originalPortraitPosition, telemetryLayout, originalPortraitGestures,
  bodyVisualDynamics, portraitArmPosition, portraitLegPart, portraitLegPieceBinding, telemetryPort } from '../life-core-state.mjs';

test('entrance leaders converge at the selected radius and clockwise head, with cycle fixed', () => {
  assert.deepEqual(telemetryPort({layer:1,head:1},32,8),[340,213]);
  assert.deepEqual(telemetryPort({layer:32,head:1},32,8),[340,102]);
  assert.deepEqual(telemetryPort({layer:32,head:3},32,8),[470,232]);
  const first = telemetryLayout(0,[340,213],true), next = telemetryLayout(1,[470,232],true);
  assert.equal(first[0].path,next[0].path);
  assert(next.slice(1).every(slot=>slot.path.endsWith('L 470.00 232.00')));
  const surfaces = coreTarget({latest:{momentum_norms:[0]},dynamics:{frequency:[[null,0]]}}).surfaces;
  assert.equal(telemetryFocus(surfaces).head,2);
});

test('leg binding leaves the upper body and hands outside its deformation support', () => {
  for (let frame=0;frame<=228;frame++) for(const [x,y] of [[340,350],[678,150],[90,1010],[380,750]]) {
    assert.equal(originalPortraitBinding(x,y).blend,0);
    const actual=originalPortraitPosition(x,y,frame/60);
    assert(Math.abs(actual[0]-x)<1e-10&&Math.abs(actual[1]-y)<1e-10);
  }
});

test('readouts retain fixed locations while their own leaders change shape and length', () => {
  let previous=null;
  for(let beat=0;beat<32;beat++){
    const slots=telemetryLayout(beat);
    assert.deepEqual(slots.map(slot=>slot.key),['cycle','depth','momentum','frequency']);
    for(const slot of slots){
      assert(slot.x>=-80&&slot.x+132<=780&&slot.y>=-24&&slot.y+54<=520);
      assert(slot.x+132<300||slot.x>450);
      assert(slot.path.startsWith(`M ${slot.side>0?slot.x:slot.x+132} ${slot.y+54} `));
      if(previous){assert.equal(slot.x,previous[slot.key].x);assert.equal(slot.y,previous[slot.key].y);assert.notEqual(slot.path,previous[slot.key].path);}
    }
    previous=Object.fromEntries(slots.map(slot=>[slot.key,slot]));
  }
});

test('visual rates follow actual angular frequency and momentum-derived amplitude', () => {
  assert.deepEqual(bodyVisualDynamics(null),{clockRate:0,transportRate:0});
  assert.deepEqual(bodyVisualDynamics({frequency:0,amplitude:.9}),{clockRate:0,transportRate:0});
  const surfaces=coreTarget({latest:{momentum_norms:[1,100]},dynamics:{frequency:[[.3],[39]]}}).surfaces;
  const low=bodyVisualDynamics(telemetryFocus(surfaces,0)),high=bodyVisualDynamics(telemetryFocus(surfaces,1));
  assert(high.transportRate>low.transportRate&&high.clockRate>low.clockRate);
  assert(bodyVisualDynamics({frequency:1,amplitude:.8}).transportRate>bodyVisualDynamics({frequency:1,amplitude:.2}).transportRate);
});

test('serial elbow and wrist joints bend the forearm while preserving hand size and support point', () => {
  const hand=[[680,150],[660,170],[650,210]];
  for(let i=0;i<306;i++){
    const t=i/60,points=hand.map(([x,y])=>portraitArmPosition(x,y,t));
    for(let k=1;k<hand.length;k++)assert(Math.abs(Math.hypot(points[k][0]-points[0][0],points[k][1]-points[0][1])-Math.hypot(hand[k][0]-hand[0][0],hand[k][1]-hand[0][1]))<1e-9);
    const support=portraitArmPosition(90,1010,t);assert(Math.hypot(support[0]-90,support[1]-1010)<1e-9);
  }
  const t=1.275,g=originalPortraitGestures(t),actual=portraitArmPosition(680,150,t);
  const shoulderOnly=[450+(680-450)*Math.cos(g.arm)-(150-530)*Math.sin(g.arm),530+(680-450)*Math.sin(g.arm)+(150-530)*Math.cos(g.arm)];
  assert(Math.hypot(actual[0]-shoulderOnly[0],actual[1]-shoulderOnly[1])>30);
});

test('idle gestures have distinct timing with brief blinks and bounded joint angles', () => {
  let blinkFrames=0;
  for(let i=0;i<456;i++){
    const g=originalPortraitGestures(i/60);
    assert(Math.abs(g.head)<=.018&&Math.abs(g.arm)<=.021);
    assert(Math.abs(g.leftEar)<=.075&&Math.abs(g.rightEar)<=.065);
    assert(g.blink>=0&&g.blink<=.92);
    if(g.blink>.1)blinkFrames++;
  }
  assert(blinkFrames>=8&&blinkFrames<=18);
  assert(originalPortraitGestures(1.25).leftEar>.05);
  assert(originalPortraitGestures(4.1).rightEar<-.04);
});

test('shoe interiors remain rigid and the neutral pose reconstructs source coordinates', () => {
  const groups=[[[665,1350],[720,1430],[790,1460]],[[860,1290],[900,1340],[950,1380]]];
  for (const points of groups) for(let frame=0;frame<=228;frame++) {
    const transformed=points.map(([x,y])=>originalPortraitPosition(x,y,frame/60));
    for(let i=1;i<points.length;i++) {
      const before=Math.hypot(points[i][0]-points[0][0],points[i][1]-points[0][1]);
      const after=Math.hypot(transformed[i][0]-transformed[0][0],transformed[i][1]-transformed[0][1]);
      assert(Math.abs(before-after)<1e-9);
    }
  }
  for(let y=0;y<=1536;y+=24)for(let x=0;x<=1024;x+=24){
    const p=originalPortraitPosition(x,y,0);assert(Math.abs(p[0]-x)<1e-10&&Math.abs(p[1]-y)<1e-10);
  }
});

test('shoe edges share one leg transform, including the former high-stretch seam', () => {
  for(let x=515;x<=1024;x+=8){
    assert.deepEqual(portraitLegPieceBinding(x,1080,0),portraitLegPieceBinding(x,1080,1));
  }
  // The old blended mesh stretched this far-shoe edge from 2px to 10.56px
  // at t=.95. Both edge pixels now belong to the same independent piece.
  for(const [x,y] of [[855,1400],[817,1350]])for(const t of [.95,2.85]){
    assert.equal(portraitLegPart(x-1,y),0);assert.equal(portraitLegPart(x+1,y),0);
    const a=originalPortraitPosition(x-1,y,t),b=originalPortraitPosition(x+1,y,t);
    assert(Math.abs(Math.hypot(b[0]-a[0],b[1]-a[1])-2)<1e-9);
  }
  const outlines=[[[630,1300],[775,1320],[800,1360],[812,1400],[848,1460]],
    [[776,1300],[805,1340],[840,1400],[1000,1400]]];
  for(const [part,points] of [[1,outlines[0]],[0,outlines[1]]]){
    for(const [x,y] of points){assert.equal(portraitLegPart(x,y),part);assert.equal(portraitLegPieceBinding(x,y,part).near,part);}
    for(let i=0;i<=228;i++){
      const transformed=points.map(([x,y])=>originalPortraitPosition(x,y,i/60));
      for(let k=1;k<points.length;k++){
        const length=Math.hypot(points[k][0]-points[0][0],points[k][1]-points[0][1]);
        assert(Math.abs(Math.hypot(transformed[k][0]-transformed[0][0],transformed[k][1]-transformed[0][1])-length)<1e-9);
      }
    }
  }
});

test('telemetry scan visits all real layer/head pairs with their own values', () => {
  const surfaces = coreTarget({latest:{momentum_norms:Array.from({length:32},(_,i)=>i+1)},
    dynamics:{frequency:Array.from({length:32},(_,i)=>Array.from({length:8},(_,j)=>i+j/10))}}).surfaces;
  const seen=new Set();
  for(let i=0;i<256;i++){
    const focus=telemetryFocus(surfaces,i);seen.add(`${focus.layer}/${focus.head}`);
    assert.equal(focus.momentum,focus.layer);
    assert.equal(focus.frequency,focus.layer-1+(focus.head-1)/10);
    assert(focus.period>=3.2&&focus.period<=4.8+1e-12);
  }
  assert.equal(seen.size,256);assert.deepEqual(telemetryFocus(surfaces,256),telemetryFocus(surfaces,0));
  assert.equal(telemetryFocus([],0),null);
});


test('all available layers participate; depth alone does not amplify the core', () => {
  for (const count of [1, 4, 8, 32]) {
    const result = aggregateLayers(Array(count).fill(12));
    assert.equal(result.count, count);
    assert.equal(result.mean, 12);
    assert.equal(result.dispersion, 0);
    assert.equal(coreTarget({ latest: { momentum_norms: Array(count).fill(12) } }).amplitude,
      coreTarget({ latest: { momentum_norms: [12] } }).amplitude);
  }
  assert.equal(aggregateLayers([0, null, undefined, NaN, -1, 6]).mean, 3);
  assert.equal(aggregateLayers([0]).mean, 0);
  assert.equal(aggregateLayers([null]).mean, null);
  assert.equal(aggregateLayers([1, 2, 3, 4, 5, 6, 7, 8]).mean, 4.5);
});

test('layer changes interpolate from the current visual state; frequency is a bounded time mapping', () => {
  const before = coreTarget({ latest: { momentum_norms: [1] } });
  const after = coreTarget({ latest: { momentum_norms: Array(8).fill(20) }, dynamics: { frequency: .8 } });
  const halfway = easeCore(before, after, 1);
  assert(halfway.amplitude > before.amplitude && halfway.amplitude < after.amplitude);
  const unchanged = easeCore(before, after, 0);
  for (const key of ['amplitude', 'variation', 'speed']) assert.equal(unchanged[key], before[key]);
  assert(after.speed < 1 && after.speed > .3);
  const frequencies = coreTarget({ dynamics: { frequencies: [.4, 1.2] } });
  assert.equal(frequencies.speed, after.speed);
});

test('32 layers use three contiguous summary bands without losing missing layer positions', () => {
  const values = Array.from({ length: 32 }, (_, i) => i);
  values[4] = null;
  const result = coreTarget({ latest: { momentum_norms: values } });
  assert.equal(result.depth, 32);
  assert.equal(result.count, 31);
  assert.deepEqual(result.bands.map(b => [b.start, b.end]), [[1, 10], [11, 21], [22, 32]]);
  assert.equal(result.bands.reduce((n, b) => n + b.count, 0), 31);
  assert.equal(result.bands[0].mean, (45 - 4) / 9);
  const changed = coreTarget({ latest: { momentum_norms: values.map((v, i) => i > 20 ? 100 : v) } });
  assert.equal(changed.bands[0].amplitude, result.bands[0].amplitude);
  assert.equal(changed.bands[1].amplitude, result.bands[1].amplitude);
  assert(changed.bands[2].amplitude > result.bands[2].amplitude);
  assert.equal(coreTarget(null).depth, 0);
  assert.deepEqual(coreTarget(null).bands, []);
});

test('learning light requires a new committed event for the same individual, not a read or replay', () => {
  const previous = { identity: { individual_id: 'fixture' }, branch_id: 'fixture', source: { cycle: 1 } };
  const next = { ...previous, source: { cycle: 2 }, latest: { updates: 4 } };
  assert.equal(isLearningCommit(previous, next), false);
  assert.equal(isLearningCommit(previous, next, true), true);
  assert.equal(isLearningCommit(next, next, true), false);
  assert.equal(isLearningCommit(previous, { ...next, latest: { updates: 0 } }, true), false);
  assert.equal(isLearningCommit(previous, { ...next, branch_id: 'other' }, true), false);
});

test('32 individual surfaces preserve gaps and map each layer frequency without magnifying tiny differences', () => {
  const values = Array.from({ length: 32 }, (_, i) => i + 1);
  values[4] = null;
  const frequency = Array.from({ length: 32 }, (_, i) => Array(8).fill(.8 + i * .00001));
  const target = coreTarget({ latest: { momentum_norms: values }, dynamics: { frequency } });
  assert.equal(target.surfaces.length, 32);
  assert.equal(target.surfaces[4].measured, false);
  assert.equal(target.surfaces[4].layer, 5);
  assert(Math.abs(target.surfaces[0].frequency - .8) < 1e-12);
  assert(Math.abs(target.surfaces[31].frequency - .80031) < 1e-12);
  assert(target.surfaces[31].speed - target.surfaces[0].speed < .001);
  assert(target.surfaces[31].amplitude > target.surfaces[0].amplitude);
  assert.equal(target.surfaces[0].headFrequencies.length, 8);
  assert.equal(target.surfaces[31].headFrequencies[7], frequency[31][7]);
  assert.notEqual(target.surfaces[0].colour, target.surfaces[31].colour);
  assert.equal(coreTarget({ latest: { momentum_norms: [0] } }).surfaces[0].amplitude, 0);
  assert.equal(coreTarget({ latest: { momentum_norms: [5] } }).surfaces[0].speed, 0);
  const eased = easeCore(coreTarget(null), target, .1);
  assert.equal(eased.surfaces.length, 32);
  assert(eased.surfaces.every(s => Number.isFinite(s.speed)));
});
