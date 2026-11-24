/** Compatibility launcher: the mother world and dashboard share one service. */
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const service = fileURLToPath(new URL('./mengli_service.mjs', import.meta.url));
const child = spawn(process.execPath, [service, ...process.argv.slice(2)], {
  stdio: 'inherit',
});
child.on('error', error => {
  console.error(`Cannot start MengLi service: ${error.message}`);
  process.exitCode = 1;
});
child.on('exit', (code, signal) => {
  process.exitCode = code ?? (signal ? 1 : 0);
});
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => child.kill(signal));
}
