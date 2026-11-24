/** Retired external-home setup. Validate and print the single local entry point. */
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const service = fileURLToPath(new URL('./mengli_service.mjs', import.meta.url));
fs.accessSync(service, fs.constants.R_OK);
console.log('No setup or session seeding is required. Existing sessions are unchanged.');
console.log(`Start: node "${service}" --port 8900`);
console.log('Dashboard: http://127.0.0.1:8900');
console.log('The service owns local persistence under data/harness/mengli.');
