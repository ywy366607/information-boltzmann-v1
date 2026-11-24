// Display mapping only: these values never advance or alter the body.
export function aggregateLayers(values) {
  const valid = (Array.isArray(values) ? values : []).filter(value =>
    typeof value === 'number' && Number.isFinite(value) && value >= 0);
  if (!valid.length) return { count: 0, mean: null, dispersion: null };
  const mean = valid.reduce((sum, value) => sum + value / valid.length, 0);
  const dispersion = Math.sqrt(valid.reduce((sum, value) => sum + ((value - mean) / Math.max(1, mean)) ** 2 / valid.length, 0));
  return { count: valid.length, mean, dispersion };
}

export function coreTarget(card) {
  const values = Array.isArray(card?.latest?.momentum_norms) ? card.latest.momentum_norms : [];
  const layers = aggregateLayers(values);
  const frequency = card?.dynamics?.frequencies ?? card?.dynamics?.frequency;
  const frequencyValues = Array.isArray(frequency) ? frequency : [frequency];
  const frequencies = aggregateLayers(frequencyValues.flat());
  const visualSpeed = value => value === null ? 0 : .3 + .7 * value / (1 + value);
  const shape = readings => ({
    amplitude: readings.mean === null ? .32 : Math.log1p(readings.mean) / (3 + Math.log1p(readings.mean)),
    variation: readings.dispersion === null ? .15 : readings.dispersion / (1 + readings.dispersion),
  });
  // Retain summary statistics for compatibility; render one surface per layer.
  const groups = Math.min(3, values.length);
  const bands = Array.from({ length: groups }, (_, i) => {
    const start = Math.floor(i * values.length / groups), end = Math.floor((i + 1) * values.length / groups);
    const readings = aggregateLayers(values.slice(start, end));
    return { start: start + 1, end, ...readings, ...shape(readings) };
  });
  const surfaces = values.map((value, index) => {
    const readings = aggregateLayers([value]);
    const sample = Array.isArray(frequency) ? frequency[index] : frequency;
    const angular = aggregateLayers(Array.isArray(sample) ? sample : [sample]);
    return { layer: index + 1, measured: readings.count === 1, ...readings, ...shape(readings),
      frequency: angular.mean, speed: visualSpeed(angular.mean),
      headFrequencies: (Array.isArray(sample) ? sample : [sample]).map(value =>
        typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null),
      // Colour encodes depth, not health, reward, or an invented emotion.
      colour: `hsl(${182 + 104 * index / Math.max(1, values.length - 1)} 76% 66%)`,
    };
  });
  return { ...layers,
    depth: values.length, bands, surfaces, ...shape(layers),
    // Internal angular frequency is mapped to visual time, never labelled Hz.
    speed: visualSpeed(frequencies.mean),
  };
}

// The layout is a stable visual embedding, not the physical position of a layer.
// Motion reads the head spectrum without treating its mean as the whole rhythm.
export function nebulaLayerPose(surface, phases = []) {
  const index = surface.layer - 1;
  const fraction = value => value - Math.floor(value);
  const angle = index * 2.399963229728653;
  const radius = 20 + 69 * Math.sqrt(fraction((index + 1) * .61803398875));
  const amplitude = surface.measured ? surface.amplitude : 0;
  const mean = fn => phases.length ? phases.reduce((sum, phase, head) =>
    sum + fn(phase + head * Math.PI * 2 / phases.length) / phases.length, 0) : 0;
  const wave = mean(Math.sin), quadrature = mean(Math.cos);
  return {
    x: 416 + Math.cos(angle) * radius * .85 + wave * amplitude * 19,
    y: 192 + Math.sin(angle) * radius + quadrature * amplitude * 21,
    width: 175 + 74 * fraction(index * .754877666),
    height: 120 + 85 * fraction(index * .569840291),
    rotation: angle * .34 + wave * amplitude * .19,
    scale: 1 + quadrature * amplitude * .13,
    opacity: surface.measured ? .11 + amplitude * .25 : .025,
  };
}

export function easeCore(current, target, elapsedSeconds) {
  const amount = 1 - Math.exp(-Math.max(0, elapsedSeconds) / .7);
  const result = Object.fromEntries(['amplitude', 'variation', 'speed'].map(key =>
    [key, current[key] + (target[key] - current[key]) * amount]));
  if (target.bands) result.bands = target.bands.map((band, i) => {
    const previous = current.bands?.[i] ?? current;
    return { ...band, ...Object.fromEntries(['amplitude', 'variation'].map(key =>
      [key, previous[key] + (band[key] - previous[key]) * amount])) };
  });
  if (target.surfaces) result.surfaces = target.surfaces.map((surface, i) => {
    const previous = current.surfaces?.[i] ?? surface;
    return { ...surface, ...Object.fromEntries(['amplitude', 'speed'].map(key =>
      [key, previous[key] + (surface[key] - previous[key]) * amount])) };
  });
  return result;
}

export function isLearningCommit(previous, next, committed = false) {
  return Boolean(committed && previous && next &&
    previous.identity?.individual_id === next.identity?.individual_id &&
    previous.branch_id === next.branch_id &&
    next.source?.cycle > previous.source?.cycle && next.latest?.updates > 0);
}

// Scan real layer/head pairs. The stride visits the whole set before repeating;
// values retain their original layer/head identity rather than becoming ranges.
export function telemetryFocus(surfaces, step = 0) {
  const pairs = surfaces.flatMap(surface => surface.headFrequencies.flatMap((frequency, index) =>
    Number.isFinite(frequency) ? [{ layer: surface.layer, head: index + 1,
      momentum: surface.measured ? surface.mean : null, frequency,
      amplitude: surface.measured ? surface.amplitude : 0 }] : []));
  if (!pairs.length) return null;
  const gcd = (a,b) => b ? gcd(b,a % b) : a;
  let stride = Math.max(1,Math.floor(pairs.length * .28));
  while (gcd(stride,pairs.length) !== 1) stride++;
  const focus = pairs[((Math.max(0,Math.floor(step))*stride)%pairs.length)];
  return { ...focus, period: 3.2 + 1.6 / (1 + Math.log1p(focus.frequency)) };
}

// Authored on the unchanged 1024x1536 portrait, in source-pixel coordinates.
// The separating path follows the visible overlap between the two original legs.
export function originalPortraitBinding(x, y) {
  const path = [[1000,665],[1150,710],[1250,748],[1320,795],[1400,855],[1536,920]];
  let boundary = path[0][1];
  for (let i=1;i<path.length;i++) {
    const t=Math.max(0,Math.min(1,(y-path[i-1][0])/(path[i][0]-path[i-1][0])));
    boundary+=(path[i][1]-path[i-1][1])*t;
  }
  const smooth = (a,b,v) => {const t=Math.max(0,Math.min(1,(v-a)/(b-a)));return t*t*(3-2*t);};
  const near=1-smooth(boundary-8,boundary+8,x);
  const pivot=[709+(616-709)*near,960+(990-960)*near];
  const blend=smooth(pivot[1]-25,pivot[1]+95,y)*smooth(515,560,x);
  return { near, blend, pivot };
}

// The lower legs are independently rendered pieces. This contour follows the
// visible near-shoe silhouette where it occludes the far heel, then the gap.
export function portraitLegSeam() {
  return [[1080,690],[1200,736],[1260,748],[1300,772],[1320,785],[1340,795],
    [1360,803],[1380,812],[1400,825],[1420,843],[1440,870],[1536,920]];
}

export function portraitLegPart(x,y) {
  const seam=portraitLegSeam();
  let boundary=seam[0][1];
  for(let i=1;i<seam.length;i++){
    const t=Math.max(0,Math.min(1,(y-seam[i-1][0])/(seam[i][0]-seam[i-1][0])));
    boundary+=(seam[i][1]-seam[i-1][1])*t;
  }
  return y>=1080&&x>=515&&x<boundary?1:0;
}

export function portraitLegPieceBinding(x,y,part) {
  const b=originalPortraitBinding(x,y);
  const t=Math.max(0,Math.min(1,(y-1080)/80)),rigid=t*t*(3-2*t);
  const near=b.near+(part-b.near)*rigid;
  const pivot=b.pivot.map((v,i)=>v+((part?[616,990]:[709,960])[i]-v)*rigid);
  return {...b,near,pivot};
}

export function originalPortraitAngles(seconds) {
  const wave=Math.sin(Math.PI*2*seconds/3.8);
  return { near: .055*wave, far: -.045*wave };
}

// Independently timed idle gestures, in radians. The character is a visual
// companion; these gestures do not represent measured physical body signals.
export function originalPortraitGestures(seconds) {
  const turn = Math.PI * 2;
  const phase = seconds % 7.6;
  const twitch = (centre, width) => Math.exp(-(((phase-centre)/width)**2));
  const eyePhase = seconds % 6.7;
  const blink = eyePhase > 2 && eyePhase < 2.26 ? Math.sin((eyePhase-2)/.26*Math.PI)**2 : 0;
  return {
    head: .018*Math.sin(turn*seconds/7.6),
    arm: .021*Math.sin(turn*seconds/5.1),
    elbow: .14*Math.sin(turn*seconds/5.1),
    wrist: -.035*Math.sin(turn*seconds/5.1),
    leftEar: .075*(twitch(1.25,.14)-.5*twitch(1.48,.16)),
    rightEar: -.065*(twitch(4.1,.15)-.45*twitch(4.33,.17)),
    blink: .92*blink,
    gaze: [1.1*Math.sin(turn*seconds/8.9), .65*Math.sin(turn*seconds/6.1)],
  };
}

// In the entrance, layer increases radially and head advances clockwise from north.
export function telemetryPort(focus, layers, heads) {
  if (!focus || !layers || !heads) return null;
  const radius = 19 + 111 * (focus.layer - 1) / Math.max(1, layers - 1);
  const angle = 2 * Math.PI * (focus.head - 1) / heads;
  return [340 + radius * Math.sin(angle), 232 - radius * Math.cos(angle)];
}

// Labels stay in place; the entrance's three layer/head leaders share one port.
export function telemetryLayout(step = 0, port = null, polar = false) {
  const n=Math.max(0,Math.floor(step));
  const slots=[
    {x:-12,y:40,end:[245,156],side:1},
    {x:563,y:2,end:[452,133],side:-1},
    {x:548,y:344,end:[446,285],side:-1},
    {x:64,y:420,end:[244,329],side:1},
  ];
  return ['cycle','depth','momentum','frequency'].map((key,i)=>{
    const slot=slots[i],variant=(n+i)%4;
    const x=slot.x,y=slot.y;
    const startX=slot.side>0?x:x+132,startY=y+54;
    const endX=slot.end[0]+slot.side*[0,24,-15,10][variant],endY=slot.end[1]+[0,-18,16,8][variant];
    const bendX=startX+slot.side*(64+((n*7+i*13)%51));
    const leader=variant===1?`M ${startX} ${startY} L ${endX} ${endY}`:
      variant===3?`M ${startX} ${startY} H ${bendX} V ${endY} H ${endX}`:
      `M ${startX} ${startY} H ${bendX} L ${endX} ${endY}`;
    const fragments=[
      `M -76 66 H ${95+n%4*13} M 153 92 H 187`,
      `M 651 -19 V 28 M 615 39 H ${758+n%3*19}`,
      `M 707 319 V ${389+n%3*11} M 501 418 H 752`,
      `M -45 492 H ${201+n%3*24} M 213 478 H 236`,
    ];
    let path=leader+' '+fragments[i];
    if (polar) {
      const end = i === 0 ? [244,156] : port;
      const kink = [158,502,525,198][i];
      path = end ? `M ${startX} ${startY} H ${kink} L ${end[0].toFixed(2)} ${end[1].toFixed(2)}` : '';
    }
    return {key,x,y,side:slot.side,path};
  });
}

// Bounded display-time mapping from the selected actual head/layer readings.
// Zero angular frequency produces zero transport; missing data stays static.
export function bodyVisualDynamics(focus) {
  const frequency=focus?.frequency;
  if(!Number.isFinite(frequency)||frequency<0)return{clockRate:0,transportRate:0};
  const activity=Math.sqrt(frequency/(1+frequency));
  const amplitude=Number.isFinite(focus?.amplitude)?Math.max(0,Math.min(1,focus.amplitude)):0;
  return {
    clockRate:activity*(.85+.55*amplitude),
    transportRate:.24*activity*(.6+.55*amplitude),
  };
}

// The raised arm has serial shoulder -> elbow -> wrist joints. Pixels on the
// hand inherit all three; the elbow remains the forearm's actual pivot.
export function portraitUpperBinding(x,y) {
  const smooth=(a,b,v)=>{const t=Math.max(0,Math.min(1,(v-a)/(b-a)));return t*t*(3-2*t);};
  return [
    (1-smooth(430,535,y))*(1-smooth(420,485,x)),
    smooth(420,545,x)*(1-smooth(450,610,y)),
    (1-smooth(165,215,x))*smooth(230,270,y)*(1-smooth(340,400,y)),
    smooth(260,280,x)*(1-smooth(335,355,x))*(1-smooth(245,282,y)),
    smooth(420,500,x)*(1-smooth(350,440,y)),
    smooth(560,610,x)*(1-smooth(245,290,y)),
  ];
}

export function portraitArmPosition(x,y,seconds) {
  const b=portraitUpperBinding(x,y),g=originalPortraitGestures(seconds);
  let q=[x,y];
  for(const[pivot,angle]of[[[618,255],g.wrist*b[5]],[[540,402],g.elbow*b[4]],[[450,530],g.arm*b[1]]]){
    const dx=q[0]-pivot[0],dy=q[1]-pivot[1];
    q=[pivot[0]+dx*Math.cos(angle)-dy*Math.sin(angle),pivot[1]+dx*Math.sin(angle)+dy*Math.cos(angle)];
  }
  return q;
}

export function originalPortraitPosition(x,y,seconds) {
  const binding=portraitLegPieceBinding(x,y,portraitLegPart(x,y)), angles=originalPortraitAngles(seconds);
  const angle=(angles.far+(angles.near-angles.far)*binding.near)*binding.blend;
  const dx=x-binding.pivot[0],dy=y-binding.pivot[1];
  return [binding.pivot[0]+dx*Math.cos(angle)-dy*Math.sin(angle),
    binding.pivot[1]+dx*Math.sin(angle)+dy*Math.cos(angle)];
}

