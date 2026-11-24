// Isolated UI fixture only. This plugin is never part of metricsPatchRows().
export const inject = ['connection'];
export const fixtureRow = { cycle: 3509, updates: 4, response_nll: 1.4523, mother_response_nll: 1.5,
  ambient_response_nll: null, progress: 0, joint_progress: 0.0237,
  causal_program: { active_relations: 12, evidence_count: 32, mother_prequential: { mse: 0.182 } },
  learned_head_coupling: { per_layer_gradient_norm: [0, 0.0023, 0.0014, 0.008] } };
export const fixtureCard = { schema: 1, identity: { name: '梦漓', individual_id: 'UI_FIXTURE_ONLY' }, branch_id: 'fixture-bound-3509',
  source: { checkpoint: '3509/current.json (UI fixture; not a body)', cycle: 3509 },
  age: { cycles: 3509, byte_events: 456789 },
  memory: { occupied: 213, capacity: 4096, episodes_committed: 702, oldest_since_write_events: 45632 },
  latest: { updates: 4, momentum_norms: [0, 0.1123, 0.0847, 0.1543] } };
export function apply(ctx) {
  ctx.connection.fetch.register({ path: '/api/family/telemetry', methods: ['GET'], requestBody: 'buffered',
    fetch: () => Response.json({ schema: 1,
      source: { ...fixtureCard.source, branch_id: fixtureCard.branch_id, kind: 'saved_snapshot' },
      status: { loaded: false, mode: 'paused', busy: false }, life: { card: fixtureCard, state: 'saved' },
      history: { rows: [fixtureRow], source: '3509/metrics.jsonl (explicit UI fixture)', limited: false }, latest_event: null }) });
  ctx.on('webserver/index-inject', rows => rows.push({ kind: 'script', placement: 'head', text: `
    const fixtureListeners=new Set();
    let fixtureSnapshot={connected:true,state:{loaded:false,mode:'paused',busy:false,cycle:0,events:[]}};
    window.__MENGLI_NATIVE_FAMILY__={controller:{getSnapshot:()=>fixtureSnapshot,subscribe:fn=>{fixtureListeners.add(fn);return()=>fixtureListeners.delete(fn)}}};
    window.metricsFixtureCommit=event=>{fixtureSnapshot={connected:true,state:{...fixtureSnapshot.state,loaded:true,events:[...fixtureSnapshot.state.events,event]}};for(const fn of fixtureListeners)fn()};
  ` }));
}
