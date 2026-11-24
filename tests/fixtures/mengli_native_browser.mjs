// Explicit UI-only fixture. Never imported by the production launcher.
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { FamilyRuntime } from '../../scripts/ib_local/mengli_harness/family_runtime.mjs';
import { createMengliServer } from '../../scripts/ib_local/mengli_harness/mengli_service.mjs';
import { createWorkbenchRuntime } from '../../scripts/ib_local/mengli_harness/workbench_runtime.mjs';

export async function startNativeBrowserFixture() {
  const root = path.resolve('data/harness', `native-browser-fixture-${randomUUID()}`);
  fs.mkdirSync(root, { recursive: true });
  const card = path.join(root, 'life.json');
  fs.writeFileSync(card, JSON.stringify({ schema: 1, identity: { name: '梦漓', individual_id: 'OFFLINE_UI_FIXTURE' }, branch_id: 'offline-ui',
    source: { checkpoint: 'OFFLINE_UI_FIXTURE_NOT_A_BODY', cycle: 0, sha256: '0'.repeat(64) },
    age: { cycles: 0, byte_events: 0, born_at: null, active_seconds: null },
    memory: { occupied: 0, capacity: 4096, episodes_committed: 0, oldest_since_write_events: null },
    latest: { updates: 0, momentum_norms: [0, 0, 0, 0] } }));
  const requests = [];
  let cycle = 0;
  const worker = { request: async (command, value) => {
    requests.push({ command, ...value });
    if (command === 'load') return { branch_id: 'offline-ui', cycle, response_bytes: 128, checkpoint: 'OFFLINE_UI_FIXTURE_NOT_A_BODY' };
    if (command === 'step') {
      cycle++;
      const candidate = Buffer.from((value.text || 'Offline mother fixture') + '\n');
      const bytes = candidate.subarray(0, 128);
      return { branch_id: 'offline-ui', cycle, response_bytes: 128, checkpoint: 'OFFLINE_UI_FIXTURE_NOT_A_BODY',
        caregiver: value.caregiver, caregiver_text: value.text || 'Offline mother fixture',
        utterance: { text: 'Offline pre-input action fixture', spoke: true, timing: 'before_caregiver_input' },
        delivered: { text: bytes.toString('utf8'), hex: bytes.toString('hex'), byte_end: bytes.length,
          candidate_bytes: candidate.length, capacity_bytes: 128, truncated: bytes.length < candidate.length, utf8_complete: true },
        choice: 'generate', updates: 0, plan_version: value.plan.version };
    }
    throw new Error('Unexpected fixture command');
  }, close: async () => {} };
  const family = new FamilyRuntime({ dataRoot: root, lifeCardPath: card, worker });
  const server = createMengliServer({ branchFile: () => path.join(root, 'no-dsh-branch') }, { family, lifeCardPath: card });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const bridge = `http://127.0.0.1:${server.address().port}`;
  const workbench = createWorkbenchRuntime({ familyOrigin: bridge, dataRoot: path.join(root, 'workbench'),
    port: 0, env: { DEEPSEEK_API_KEY: '', MENGLI_INITIAL_LOGIN_KEY: 'offline-native-preview-key' } });
  const info = await workbench.start();
  family.setWorkbench(workbench, info);
  return { bridge, requests, family, loginKey: 'offline-native-preview-key', close: async () => {
    await family.close();
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
    fs.rmSync(root, { recursive: true, force: true });
  } };
}
