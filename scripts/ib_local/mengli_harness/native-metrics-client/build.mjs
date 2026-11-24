import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_PACKAGE_ROOT = 'F:/16、杂货行账册/ledger-dsh/node_modules/@deepseek-ai';

/** Precompiled ModuleLoader plugin. No compiler, install, or dependency copying. */
export function buildMetricsClient({ packageRoot = DEFAULT_PACKAGE_ROOT } = {}) {
  // Fail at build time if the requested official installation lacks the native seats.
  for (const name of ['dsh-client-ui-sidebar-right', 'dsh-client-ui-layout']) {
    fs.accessSync(path.join(packageRoot, name, 'package.json'));
  }
  const model = fs.readFileSync(path.join(root, 'metrics-state.mjs'), 'utf8').replaceAll('export ', '');
  const client = fs.readFileSync(path.join(root, 'client-source.js'), 'utf8');
  const css = fs.readFileSync(path.join(root, 'metrics.css'), 'utf8');
  const lib = path.join(root, 'lib');
  fs.mkdirSync(lib, { recursive: true });
  fs.writeFileSync(path.join(lib, 'index.js'), '// Browser-only, read-only telemetry plugin. No server services.\nexport function apply() {}\n');
  fs.writeFileSync(path.join(lib, 'client.js'), `window.__ModuleLoader__.load({id:'@mengli/native-metrics-client',factory(require){\nconst METRICS_CSS=${JSON.stringify(css)};\n${model}\n${client}\nreturn {apply,inject};\n}});\n`);
  return { root, entry: path.join(lib, 'index.js'), client: path.join(lib, 'client.js') };
}

/** Cordis name is an insert target, never a replacement/match guard. */
export function metricsPatchRows(options = {}) {
  const { entry } = buildMetricsClient(options);
  return [{ insert: [{ id: 'mengli-native-metrics', name: entry }] }];
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  console.log(JSON.stringify(buildMetricsClient({ packageRoot: process.env.MENGLI_DSH_PACKAGE_ROOT || DEFAULT_PACKAGE_ROOT })));
}
